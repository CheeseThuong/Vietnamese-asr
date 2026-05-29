"""
====================================================================
@file: gemini_client.py
@description: Module quản lý tập trung việc gọi Google Gemini API.
              Sử dụng SDK google-genai chính thức.
              Cung cấp:
              - Đọc API key từ biến môi trường GEMINI_API_KEY (qua .env)
              - Đọc tên model từ GEMINI_MODEL (mặc định: gemini-2.5-flash-lite)
              - Khóa request đơn (threading.Lock) để tránh gọi song song
              - Retry với exponential backoff cho lỗi 429
              - Xử lý lỗi 403 rõ ràng (không retry)
              - Timeout handling cho request chậm
              - Cache đơn giản theo hash để không tóm tắt lại cùng 1 transcript
              - Chunking transcript dài thành nhiều phần rồi tổng hợp
@author: Nguyễn Trí Thượng
@project: VietASR Pro
@version: 3.0.0
====================================================================
"""

import hashlib
import logging
import os
import random
import re
import time
import threading
from pathlib import Path
from typing import Optional

# Tải biến môi trường từ .env (nếu có)
try:
    from dotenv import load_dotenv
    # Tìm .env ở thư mục gốc project (2 cấp trên file này)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # python-dotenv không bắt buộc, dùng env system

logger = logging.getLogger("VietASR.GeminiClient")

# ======================================================================
# Cấu hình toàn cục — đọc từ biến môi trường
# ======================================================================

# Tên model mặc định: dùng gemini-2.5-flash-lite (rẻ, đủ nhanh cho ASR)
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Số ký tự tối đa mỗi chunk trước khi tách (≈ 6000 từ tiếng Việt)
MAX_CHUNK_CHARS = 8000

# Số lần retry tối đa khi gặp lỗi 429
MAX_RETRIES = 4

# Timeout mỗi request (giây)
REQUEST_TIMEOUT = 60

# Số giây chờ tối đa khi backoff
MAX_BACKOFF_SECONDS = 64

# ======================================================================
# Khóa đơn: chỉ 1 yêu cầu Gemini được xử lý tại một thời điểm
# (tránh gửi hàng chục request song song gây 429)
# ======================================================================
_gemini_lock = threading.Lock()


# ======================================================================
# Cache kết quả theo SHA-256 hash của (system_prompt + text)
# Tránh gọi Gemini lại cho cùng 1 transcript
# ======================================================================
class _GeminiCache:
    """Cache đơn giản lưu kết quả Gemini theo hash SHA-256 của đầu vào."""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._store: dict = {}  # {key: result}
        self._lock = threading.Lock()

    def _key(self, text: str, prompt: str) -> str:
        """Tạo khóa cache từ SHA-256 của prompt + text."""
        raw = f"{prompt}|||{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, prompt: str = "") -> Optional[str]:
        """Lấy kết quả đã cache. Trả về None nếu chưa có."""
        with self._lock:
            return self._store.get(self._key(text, prompt))

    def put(self, text: str, result: str, prompt: str = "") -> None:
        """Lưu kết quả vào cache. Xóa entry cũ nhất nếu đầy (FIFO)."""
        key = self._key(text, prompt)
        with self._lock:
            if len(self._store) >= self.max_size:
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[key] = result


# Singleton cache dùng chung toàn app
_cache = _GeminiCache(max_size=256)


# ======================================================================
# Khởi tạo Gemini client (google-genai SDK)
# ======================================================================
def _get_client(api_key: str):
    """
    Tạo google.genai.Client với API key được cung cấp.
    Lazy import để không lỗi nếu google-genai chưa cài.
    """
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError:
        raise ImportError(
            "Thiếu thư viện google-genai. "
            "Hãy chạy: pip install google-genai"
        )


# ======================================================================
# Hàm nội bộ: gọi Gemini 1 lần với retry + backoff
# ======================================================================
def _call_gemini_once(
    client,
    model: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """
    Gọi Gemini API 1 lần, tự động retry khi gặp lỗi 429.
    Raise PermissionError nếu 403, RuntimeError nếu hết retry.

    Args:
        client: google.genai.Client đã khởi tạo
        model: Tên model Gemini
        prompt: Nội dung prompt hoàn chỉnh
        temperature: Độ sáng tạo (0–1)
        max_tokens: Số token tối đa trong phản hồi

    Returns:
        Văn bản phản hồi từ Gemini

    Raises:
        PermissionError: Lỗi 403 (API key không hợp lệ)
        TimeoutError: Vượt quá thời gian chờ
        RuntimeError: Hết số lần retry
    """
    # pyrefly: ignore [missing-import]
    from google.genai import types as genai_types

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # Gọi Gemini với timeout qua SDK
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    # Timeout xử lý phía client (giây)
                    http_options=genai_types.HttpOptions(
                        timeout=REQUEST_TIMEOUT * 1000  # milliseconds
                    ),
                ),
            )

            # Lấy text từ response
            if response and response.text:
                return response.text.strip()

            # Response rỗng — không retry, trả về rỗng
            logger.warning("[Gemini] Phản hồi rỗng từ Gemini")
            return ""

        except Exception as err:
            last_error = err
            err_str = str(err).lower()

            # Xử lý lỗi 403 Forbidden (không retry)
            if "403" in str(err) or "permission" in err_str or "forbidden" in err_str:
                logger.error(f"[Gemini] 403 Forbidden — API key không hợp lệ: {err}")
                raise PermissionError(
                    f"Gemini 403 Forbidden: API key không hợp lệ hoặc bị thu hồi quyền. "
                    f"Chi tiết: {err}"
                )

            # Xử lý lỗi 429 Rate Limit (retry với backoff)
            if "429" in str(err) or "quota" in err_str or "rate" in err_str or "resource_exhausted" in err_str:
                # Tính thời gian chờ: exponential backoff + jitter
                base_wait = min((2 ** attempt) * 5, MAX_BACKOFF_SECONDS)
                jitter = random.uniform(0, 3)
                wait_secs = base_wait + jitter

                logger.warning(
                    f"[Gemini] 429 Rate Limit — đợi {wait_secs:.1f}s "
                    f"(lần thử {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait_secs)
                continue

            # Lỗi timeout
            if "timeout" in err_str or "deadline" in err_str or "timed out" in err_str:
                wait_secs = min((2 ** attempt) * 3, 30) + random.uniform(0, 2)
                logger.warning(
                    f"[Gemini] Timeout — đợi {wait_secs:.1f}s "
                    f"(lần thử {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait_secs)
                continue

            # Lỗi khác — không retry, raise ngay
            logger.error(f"[Gemini] Lỗi không xác định: {err}")
            raise

    # Hết số lần retry
    raise RuntimeError(
        f"Gemini API không phản hồi sau {MAX_RETRIES} lần thử. "
        f"Lỗi cuối: {last_error}"
    )


# ======================================================================
# Hàm công khai chính: call_gemini
# ======================================================================
def call_gemini(
    text: str,
    system_prompt: str = "",
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
) -> str:
    """
    Gọi Google Gemini API an toàn với khóa đơn, cache và retry.

    QUAN TRỌNG:
    - Hàm này dùng threading.Lock để đảm bảo chỉ 1 yêu cầu tại một thời điểm.
    - API key được đọc từ env GEMINI_API_KEY, KHÔNG nhận từ frontend.
    - Kết quả được cache theo SHA-256 hash của (prompt + text).

    Args:
        text: Văn bản cần xử lý
        system_prompt: Hướng dẫn cho Gemini
        model: Tên model (mặc định: giá trị env GEMINI_MODEL)
        temperature: Nhiệt độ sinh văn bản (0.0–1.0)
        max_tokens: Số token tối đa trong phản hồi
        api_key: API key (mặc định: đọc từ env GEMINI_API_KEY)

    Returns:
        Văn bản đã được Gemini xử lý

    Raises:
        ValueError: Khi thiếu API key
        PermissionError: Khi Gemini trả về 403 Forbidden
        RuntimeError: Khi vượt quá số lần retry
    """
    # Đọc API key — ưu tiên tham số > env var
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Thiếu GEMINI_API_KEY. "
            "Vui lòng đặt trong file .env hoặc biến môi trường."
        )

    # Chọn model — ưu tiên tham số > env var > mặc định
    chosen_model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    # Xây dựng prompt đầy đủ
    if system_prompt:
        full_prompt = f"{system_prompt}\n\nVăn bản:\n{text}"
    else:
        full_prompt = text

    # Kiểm tra cache trước khi gọi API
    cached = _cache.get(text, system_prompt)
    if cached is not None:
        logger.info("[Gemini] Cache hit — bỏ qua gọi API")
        return cached

    logger.info(
        f"[Gemini] Gọi API: model={chosen_model}, "
        f"text_len={len(text)} ký tự"
    )

    # Khởi tạo client
    client = _get_client(key)

    # Dùng lock: chỉ 1 yêu cầu tại một thời điểm
    with _gemini_lock:
        result = _call_gemini_once(
            client=client,
            model=chosen_model,
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Lưu vào cache nếu có kết quả
    if result:
        _cache.put(text, result, system_prompt)
        logger.info(f"[Gemini] Thành công, kết quả: {len(result)} ký tự")

    return result or text


# ======================================================================
# Hàm tách transcript dài thành chunks
# ======================================================================
def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Tách transcript dài thành danh sách chunks không vượt quá max_chars.
    Cố gắng tách tại ranh giới câu (dấu . ! ?) để giữ ngữ nghĩa.

    Args:
        text: Văn bản cần tách
        max_chars: Số ký tự tối đa mỗi chunk

    Returns:
        Danh sách các chunk văn bản
    """
    if len(text) <= max_chars:
        return [text]

    # Tách tại dấu câu kết thúc câu
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Nếu câu đơn lẻ đã vượt max_chars, tách theo khoảng trắng
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk += (" " + word) if current_chunk else word
        else:
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence) if current_chunk else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c]


# ======================================================================
# Hàm tiện ích: summarize_transcript (hỗ trợ chunking cho audio dài)
# ======================================================================
def summarize_transcript(
    transcript: str,
    mode: str = "summary",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Tóm tắt transcript hoàn chỉnh bằng Gemini.
    Hỗ trợ transcript dài bằng cách:
    1. Tách thành chunks
    2. Tóm tắt từng chunk
    3. Tổng hợp các bản tóm tắt thành 1 bản cuối

    HÀM NÀY chỉ nên được gọi SAU KHI transcript đã hoàn tất,
    KHÔNG gọi cho từng audio chunk riêng lẻ.

    Args:
        transcript: Toàn bộ văn bản transcript
        mode: Chế độ xử lý ('summary', 'meeting', 'notes', 'translate')
        api_key: Gemini API key (mặc định từ env GEMINI_API_KEY)
        model: Tên model Gemini (mặc định từ env GEMINI_MODEL)

    Returns:
        dict với các trường:
          - summary: Văn bản tóm tắt cuối cùng
          - chunks_count: Số chunks đã xử lý
          - model: Tên model đã dùng
    """
    # Bảng prompt theo chế độ (bằng tiếng Việt)
    chunk_prompts = {
        "summary": (
            "Tóm tắt ngắn gọn đoạn văn bản nhận dạng giọng nói (ASR) tiếng Việt sau đây. "
            "Chỉ giữ lại thông tin quan trọng nhất, bằng tiếng Việt:"
        ),
        "meeting": (
            "Tóm tắt các điểm chính từ đoạn biên bản cuộc họp (ASR) tiếng Việt sau. "
            "Ghi chú ngắn gọn về chủ đề, quyết định, hành động, bằng tiếng Việt:"
        ),
        "notes": (
            "Tạo ghi chú học tập ngắn gọn từ đoạn văn bản ASR tiếng Việt sau. "
            "Dùng bullet points, bằng tiếng Việt:"
        ),
        "translate": (
            "Dịch đoạn văn bản tiếng Việt sau sang tiếng Anh. "
            "Chỉ trả về bản dịch, không giải thích:"
        ),
    }

    # Prompt tổng hợp (khi có nhiều chunks)
    final_prompts = {
        "summary": (
            "Dựa trên các bản tóm tắt từng phần dưới đây của một cuộc ghi âm tiếng Việt, "
            "hãy tạo một bản tóm tắt tổng hợp duy nhất, mạch lạc, đầy đủ thông tin chính, bằng tiếng Việt:"
        ),
        "meeting": (
            "Dựa trên các ghi chú tóm tắt từng phần dưới đây, "
            "hãy tạo một biên bản cuộc họp hoàn chỉnh bao gồm: chủ đề, nội dung chính, "
            "quyết định và hành động tiếp theo, bằng tiếng Việt:"
        ),
        "notes": (
            "Dựa trên các ghi chú từng phần dưới đây, "
            "hãy tổng hợp thành ghi chú học tập đầy đủ, dạng bullet points, bằng tiếng Việt:"
        ),
        "translate": (
            "Kết hợp các bản dịch từng phần dưới đây thành bản dịch tiếng Anh hoàn chỉnh, "
            "mạch lạc và tự nhiên:"
        ),
    }

    chosen_model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    chunk_prompt = chunk_prompts.get(mode, chunk_prompts["summary"])
    final_prompt = final_prompts.get(mode, final_prompts["summary"])

    # Tách transcript thành chunks nếu dài
    chunks = _split_into_chunks(transcript, max_chars=MAX_CHUNK_CHARS)
    chunks_count = len(chunks)

    logger.info(
        f"[Summarize] Bắt đầu tóm tắt: mode={mode}, "
        f"total_chars={len(transcript)}, chunks={chunks_count}"
    )

    # Nếu chỉ có 1 chunk, tóm tắt trực tiếp
    if chunks_count == 1:
        summary = call_gemini(
            text=transcript,
            system_prompt=chunk_prompt,
            model=chosen_model,
            api_key=api_key,
        )
        return {
            "summary": summary,
            "chunks_count": 1,
            "model": chosen_model,
        }

    # Nhiều chunks: tóm tắt từng chunk, rồi tổng hợp
    logger.info(f"[Summarize] Xử lý {chunks_count} chunks...")
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        logger.info(f"[Summarize] Chunk {i + 1}/{chunks_count} ({len(chunk)} ký tự)")
        try:
            chunk_sum = call_gemini(
                text=chunk,
                system_prompt=chunk_prompt,
                model=chosen_model,
                api_key=api_key,
            )
            if chunk_sum:
                chunk_summaries.append(chunk_sum)
        except Exception as chunk_err:
            # Nếu 1 chunk thất bại, ghi log nhưng tiếp tục các chunk còn lại
            logger.warning(f"[Summarize] Chunk {i + 1} thất bại: {chunk_err}")
            # Dùng phần đầu của chunk gốc làm fallback
            chunk_summaries.append(chunk[:200] + "...")

    if not chunk_summaries:
        raise RuntimeError("Không tóm tắt được bất kỳ chunk nào.")

    # Nếu chỉ có 1 kết quả (các chunk khác thất bại), trả luôn
    if len(chunk_summaries) == 1:
        return {
            "summary": chunk_summaries[0],
            "chunks_count": chunks_count,
            "model": chosen_model,
        }

    # Tổng hợp các bản tóm tắt chunk thành 1 bản cuối
    combined_text = "\n\n---\n\n".join(
        f"[Phần {i + 1}]:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    logger.info(f"[Summarize] Tổng hợp {len(chunk_summaries)} bản tóm tắt chunk...")
    final_summary = call_gemini(
        text=combined_text,
        system_prompt=final_prompt,
        model=chosen_model,
        api_key=api_key,
        max_tokens=8192,  # Cho phép tóm tắt cuối dài hơn
    )

    return {
        "summary": final_summary or combined_text,
        "chunks_count": chunks_count,
        "model": chosen_model,
    }


# ======================================================================
# Hàm kiểm tra kết nối Gemini (dùng cho /api/gemini-test)
# ======================================================================
def test_gemini_connection(api_key: Optional[str] = None) -> dict:
    """
    Gửi prompt ngắn để kiểm tra kết nối Gemini API.

    Args:
        api_key: API key (mặc định từ env GEMINI_API_KEY)

    Returns:
        dict với: success, model, response, error
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    chosen_model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    if not key:
        return {
            "success": False,
            "model": chosen_model,
            "response": None,
            "error": "Thiếu GEMINI_API_KEY trong biến môi trường.",
        }

    try:
        client = _get_client(key)
        test_prompt = "Trả lời ngắn gọn: API Gemini đang hoạt động."

        with _gemini_lock:
            response = _call_gemini_once(
                client=client,
                model=chosen_model,
                prompt=test_prompt,
                temperature=0.1,
                max_tokens=64,
            )

        return {
            "success": True,
            "model": chosen_model,
            "response": response,
            "error": None,
        }

    except PermissionError as perm_err:
        return {
            "success": False,
            "model": chosen_model,
            "response": None,
            "error": f"403 Forbidden: {perm_err}",
        }
    except Exception as err:
        return {
            "success": False,
            "model": chosen_model,
            "response": None,
            "error": str(err),
        }
