"""
====================================================================
@file: post_processor.py
@description: Module hậu xử lý văn bản ASR tiếng Việt:
              1. Sửa lỗi lặp từ/âm tiết (fix_duplicates)
              2. Thay thế từ sai bằng từ điển (dictionary correction)
              3. Chỉnh sửa văn bản và tóm tắt bằng LLM (Gemini/OpenAI)
@author: Nguyễn Trí Thượng
@project: VietASR Pro
@email: nguyentrithuong471@gmail.com
@github: CheeseThuong
@version: 2.0.0
====================================================================
"""

import re
import json
import os
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
from threading import Lock


# Đường dẫn mặc định
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _MODULE_DIR / "post_processing_config.json"
_DEFAULT_DICT_PATH = _MODULE_DIR / "correction_dict.json"


# ======================================================================
# Rate Limiter cho Gemini API (module-level, dùng chung toàn app)
# ======================================================================
class GeminiRateLimiter:
    """Giới hạn tần suất gọi Gemini API để tránh 429."""

    def __init__(self, max_calls=15, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def can_call(self) -> bool:
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

    def wait_time(self) -> float:
        with self.lock:
            if not self.calls:
                return 0
            return max(0, self.period - (time.time() - self.calls[0]))


# Singleton rate limiter instance
gemini_limiter = GeminiRateLimiter(max_calls=15, period=60)


# ======================================================================
# Cache kết quả LLM theo MD5 hash
# ======================================================================
class LLMCache:
    """Cache kết quả Gemini/OpenAI theo MD5 hash của (text + prompt)."""

    def __init__(self, max_size=200):
        self.max_size = max_size
        self.cache = {}
        self.lock = Lock()

    def _make_key(self, text: str, prompt: str = "") -> str:
        raw = f"{prompt}|||{text}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, prompt: str = "") -> Optional[str]:
        key = self._make_key(text, prompt)
        with self.lock:
            return self.cache.get(key)

    def put(self, text: str, result: str, prompt: str = ""):
        key = self._make_key(text, prompt)
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Xóa entry cũ nhất (FIFO đơn giản)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = result


# Singleton cache instance
llm_cache = LLMCache(max_size=200)


# ======================================================================
# PostProcessor — Pipeline chính
# ======================================================================
class PostProcessor:
    """Pipeline hậu xử lý văn bản ASR tiếng Việt."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        dict_path: Optional[str] = None,
    ):
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self.dict_path = Path(dict_path) if dict_path else _DEFAULT_DICT_PATH

        # Load config & dictionary
        self.config = self._load_json(self.config_path, self._default_config())
        self.correction_dict = self._load_json(self.dict_path, {})

        # Loại bỏ entries rỗng (chưa có từ đúng) và key "_comment"
        self.correction_dict = {
            k: v
            for k, v in self.correction_dict.items()
            if v and not k.startswith("_")
        }

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_config():
        return {
            "pipeline": {
                "fix_duplicates": True,
                "use_dictionary": True,
                "use_llm": False,  # Tắt mặc định — Gemini chỉ gọi qua /api/summarize
            },
            "llm": {
                "provider": "gemini",
                "api_key": "",  # Không lưu key trong code — dùng env GEMINI_API_KEY
                "model": "gemini-2.5-flash-lite-preview-06-17",
                "max_tokens": 2048,
                "temperature": 0.3,
            },
        }

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Không đọc được {path}: {e}")
        return default

    def save_config(self):
        """Ghi config hiện tại ra file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

    def save_dictionary(self):
        """Ghi từ điển hiện tại ra file."""
        with open(self.dict_path, "w", encoding="utf-8") as f:
            json.dump(self.correction_dict, f, ensure_ascii=False, indent=4)

    def reload(self):
        """Tải lại config và từ điển từ file."""
        self.config = self._load_json(self.config_path, self._default_config())
        raw_dict = self._load_json(self.dict_path, {})
        self.correction_dict = {
            k: v for k, v in raw_dict.items() if v and not k.startswith("_")
        }

    # ------------------------------------------------------------------
    # Pipeline property helpers
    # ------------------------------------------------------------------
    @property
    def pipeline_cfg(self):
        return self.config.get("pipeline", {})

    @property
    def llm_cfg(self):
        return self.config.get("llm", {})

    # ------------------------------------------------------------------
    # Bước 1: Sửa lỗi lặp từ / âm tiết
    # ------------------------------------------------------------------
    @staticmethod
    def fix_duplicate_syllables(text: str) -> str:
        """
        Sửa lỗi lặp từ/âm tiết:
        - Lặp dính liền:  "họhọ" → "họ",  "làlàm" → "làm"
        - Lặp có space:   "bị bị" → "bị"
        """
        if not text:
            return text

        # --- 1a. Lặp dính liền (e.g. "họhọ" → "họ") ---
        vietnamese_chars = (
            r"[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
            r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
            r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨ"
            r"ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
        )

        # Tìm từ dính lặp: ≥2 âm tiết dính nhau mà phần đầu trùng phần sau
        pattern_stuck = re.compile(
            r"(?<!\w)(" + vietnamese_chars + r"{1,8})\1(" + vietnamese_chars + r"*?)(?!\w)"
        )
        text = pattern_stuck.sub(lambda m: m.group(1) + m.group(2), text)

        # --- 1b. Lặp có space: "bị bị" → "bị", "là là" → "là" ---
        pattern_space_dup = re.compile(
            r"\b(" + vietnamese_chars + r"+)\s+\1\b", re.IGNORECASE
        )
        text = pattern_space_dup.sub(r"\1", text)

        # Dọn dẹp: loại bỏ space thừa
        text = re.sub(r"\s{2,}", " ", text).strip()

        return text

    # ------------------------------------------------------------------
    # Bước 2: Thay thế từ điển
    # ------------------------------------------------------------------
    def apply_dictionary(self, text: str) -> str:
        """Thay thế từ sai theo từ điển correction_dict."""
        if not text or not self.correction_dict:
            return text

        for wrong, correct in self.correction_dict.items():
            if wrong and correct:
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                text = pattern.sub(correct, text)

        return text

    # ------------------------------------------------------------------
    # Bước 3: LLM chỉnh sửa
    # ------------------------------------------------------------------
    def apply_llm_correction(self, text: str, system_prompt: str = None) -> str:
        """Gọi LLM API để chỉnh sửa ngữ pháp/chính tả tiếng Việt."""
        if not text:
            return text

        provider = self.llm_cfg.get("provider", "gemini")

        # Ƭu tiên đọc API key từ biến môi trường (bảo mật hơn)
        api_key = os.environ.get("GEMINI_API_KEY", "") or self.llm_cfg.get("api_key", "")

        if not api_key:
            print("⚠ LLM API key chưa được cấu hình — bỏ qua bước LLM")
            return text

        # Kiểm tra cache trước
        cached = llm_cache.get(text, system_prompt or "")
        if cached is not None:
            print("✓ LLM cache hit")
            return cached

        try:
            if provider == "gemini":
                result = self._call_gemini(text, api_key, system_prompt)
            elif provider == "openai":
                result = self._call_openai(text, api_key)
            else:
                print(f"⚠ Provider '{provider}' không được hỗ trợ")
                return text

            # Lưu cache nếu thành công
            if result and result != text:
                llm_cache.put(text, result, system_prompt or "")

            return result
        except Exception as e:
            print(f"⚠ LLM correction failed: {e}")
            return str(e) if "Rate limit" in str(e) else text

    # ------------------------------------------------------------------
    # Gemini API (REST, retry 3 lần, rate limiter, cache)
    # ------------------------------------------------------------------
    def _call_gemini(self, text: str, api_key: str, system_prompt: str = None) -> str:
        """Gọi Google Gemini API qua REST (không cần SDK)."""
        model = self.llm_cfg.get("model", "gemini-2.0-flash")
        temperature = self.llm_cfg.get("temperature", 0.3)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )

        prompt = system_prompt
        if not prompt:
            prompt = (
                "Bạn là trợ lý chỉnh sửa văn bản tiếng Việt. "
                "Hãy chỉnh sửa đoạn văn bản sau đây từ hệ thống nhận dạng giọng nói (ASR). "
                "CHỈ sửa lỗi chính tả, ngữ pháp và từ ngữ sai. "
                "KHÔNG thay đổi ý nghĩa gốc, KHÔNG thêm bớt nội dung, KHÔNG dịch sang ngôn ngữ khác. "
                "KHÔNG giải thích, chỉ trả về văn bản đã chỉnh sửa.\n\n"
            )

        prompt += f"\n\nVăn bản:\n{text}"

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.llm_cfg.get("max_tokens", 4096),
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
            method="POST",
        )

        # Rate limiter check
        if not gemini_limiter.can_call():
            wait = gemini_limiter.wait_time()
            raise Exception(f"Rate limit: thử lại sau {wait:.0f} giây")

        # Retry logic: 3 lần với exponential backoff
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                # Parse response
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        corrected = parts[0].get("text", "").strip()
                        if corrected:
                            return corrected

                print("⚠ Gemini trả về kết quả rỗng — giữ nguyên văn bản gốc")
                return text

            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:
                    wait_time = (2 ** attempt) * 5
                    print(f"⚠ Gemini 429 Rate Limit. Đợi {wait_time}s... (lần {attempt + 1}/3)")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"⚠ Gemini HTTP {e.code}: {e.reason}")
                    raise
            except Exception as e:
                last_error = e
                if "429" in str(e):
                    wait_time = (2 ** attempt) * 5
                    print(f"⚠ Gemini Rate Limit. Đợi {wait_time}s... (lần {attempt + 1}/3)")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        raise Exception(f"Gemini không phản hồi sau 3 lần thử: {last_error}")

    # ------------------------------------------------------------------
    # OpenAI API
    # ------------------------------------------------------------------
    def _call_openai(self, text: str, api_key: str) -> str:
        """Gọi OpenAI ChatCompletion API qua REST."""
        model = self.llm_cfg.get("model", "gpt-3.5-turbo")

        url = "https://api.openai.com/v1/chat/completions"

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý chỉnh sửa văn bản tiếng Việt từ hệ thống ASR. "
                        "CHỈ sửa lỗi chính tả, ngữ pháp. KHÔNG thay đổi ý nghĩa. "
                        "KHÔNG giải thích, chỉ trả về văn bản đã sửa."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": self.llm_cfg.get("temperature", 0.3),
            "max_tokens": self.llm_cfg.get("max_tokens", 4096),
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        choices = result.get("choices", [])
        if choices:
            corrected = choices[0].get("message", {}).get("content", "").strip()
            if corrected:
                return corrected

        return text

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------
    def process(self, text: str, system_prompt: str = None) -> dict:
        """
        Chạy pipeline hậu xử lý.

        Nếu system_prompt được truyền, LLM sẽ sử dụng prompt này thay vì mặc định.

        Returns:
            dict: {
                "original": str,        # Văn bản gốc
                "processed": str,       # Văn bản đã xử lý
                "steps_applied": list,  # Danh sách bước đã áp dụng
            }
        """
        if not text or not text.strip():
            return {
                "original": text or "",
                "processed": text or "",
                "steps_applied": [],
            }

        original = text.strip()
        result = original
        steps = []

        # Bước 1: Sửa lỗi lặp
        if self.pipeline_cfg.get("fix_duplicates", True):
            before = result
            result = self.fix_duplicate_syllables(result)
            if result != before:
                steps.append("fix_duplicates")

        # Bước 2: Từ điển thay thế
        if self.pipeline_cfg.get("use_dictionary", True):
            before = result
            result = self.apply_dictionary(result)
            if result != before:
                steps.append("dictionary")

        # Bước 3: LLM chỉnh sửa
        # Always use LLM if system_prompt is provided (e.g., summarize triggers),
        # otherwise adhere to use_llm config.
        use_llm = self.pipeline_cfg.get("use_llm", False) or system_prompt is not None
        if use_llm:
            before = result
            llm_res = self.apply_llm_correction(result, system_prompt=system_prompt)
            if "Rate limit" in llm_res:
                raise Exception(llm_res)

            if llm_res != before:
                steps.append("llm")
                result = llm_res

        return {
            "original": original,
            "processed": result,
            "steps_applied": steps,
        }


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    pp = PostProcessor()

    test_text = (
        "nếu tổng thống trump gọi là chuẩn phê đó thì những nhà tại việt hay "
        "những nhà chính trị của hồng công có thể bị bị gọi là đống bằng hay là "
        "bị giới hạn bởi lật hoa kỳ để gọi là du lịch hay làlàm ăn minh bên "
        "hoa kỳ đó là những điều mà những nhà đầu tư họ chỉ lo tiệp tì tại có "
        "thôi họng có lo họ không lo vì dân chủ họhọ bất an trong tương lai "
        "chuyện đã xảy ra nếu có tạm thời giải quyết thì có thể là tạn thời "
        "như mà không có nghĩa là trong tương lai sẽ tải diện"
    )

    print("=" * 70)
    print("[POST-PROCESSOR] Vietnamese ASR Post-Processing Pipeline")
    print("=" * 70)
    print(f"\n[INPUT]:\n{test_text}")

    result = pp.process(test_text)

    print(f"\n[OUTPUT]:\n{result['processed']}")
    print(f"\n[STEPS]: {result['steps_applied']}")
    print(f"\n[CONFIG]: {json.dumps(pp.pipeline_cfg, ensure_ascii=False)}")
