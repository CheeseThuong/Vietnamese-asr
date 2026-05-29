"""
launcher.py — VietASR Pro Cross-Platform Launcher
===================================================
Launcher Python đa nền tảng (Windows / macOS / Linux).
Sử dụng bởi run_vietasr.bat và run_vietasr.sh làm entry point.

Chức năng:
  1. Kiểm tra Python version >= 3.8
  2. Cài dependencies từ requirements.txt
  3. Kiểm tra / tạo file .env từ .env.example
  4. Xác định entry point (app/app.py)
  5. Khởi động Flask server trong subprocess
  6. Poll /api/health cho đến khi server sẵn sàng
  7. Tự động mở trình duyệt
  8. Giữ terminal và xử lý Ctrl+C sạch sẽ

KHÔNG gọi script này từ bên trong venv — hãy để run_vietasr.bat/sh
kích hoạt venv rồi gọi: python launcher.py
"""

import subprocess
import sys
import os
import time
import signal
import webbrowser
import shutil
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# ── Cấu hình ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 8)
DEFAULT_PORT = 5000
HEALTH_TIMEOUT_S = 120  # giây chờ server sẵn sàng
HEALTH_POLL_S   = 1.5   # khoảng cách giữa mỗi lần poll

# ── Màu terminal ───────────────────────────────────────────────
class C:
    R  = '\033[0;31m'
    G  = '\033[0;32m'
    Y  = '\033[1;33m'
    B  = '\033[0;34m'
    CY = '\033[0;36m'
    BD = '\033[1m'
    NC = '\033[0m'
    @staticmethod
    def ok(s): return f"{C.G}{s}{C.NC}"
    @staticmethod
    def warn(s): return f"{C.Y}{s}{C.NC}"
    @staticmethod
    def err(s): return f"{C.R}{s}{C.NC}"
    @staticmethod
    def info(s): return f"{C.CY}{s}{C.NC}"

# ── Helper: print step ─────────────────────────────────────────
def step(n: int, total: int, msg: str):
    print(f"  {C.CY}[{n}/{total}]{C.NC} {msg}")

# ── Kiểm tra Python version ────────────────────────────────────
def check_python():
    if sys.version_info < MIN_PYTHON:
        print(C.err(f"  ✗ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ yêu cầu. Bạn đang dùng {sys.version}"))
        sys.exit(1)
    print(f"     {C.ok(f'Python {sys.version.split()[0]} ✓')}")

# ── Đọc port từ .env ──────────────────────────────────────────
def read_port() -> int:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return DEFAULT_PORT
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip().upper()
        v = v.strip().strip('"').strip("'")
        if k in ("FLASK_PORT", "PORT") and v.isdigit():
            return int(v)
    return DEFAULT_PORT

# ── Kiểm tra và tạo .env ──────────────────────────────────────
def ensure_env():
    env = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env.exists():
        if example.exists():
            shutil.copy(example, env)
            print(C.warn("     ⚠ Đã tạo .env từ .env.example"))
            print(C.warn("       Mở .env và điền GEMINI_API_KEY nếu muốn dùng Gemini."))
        else:
            print(C.warn("     ⚠ Không tìm thấy .env — tiếp tục không có API key"))
    else:
        print(f"     {C.ok('.env đã tồn tại ✓')}")

# ── Cài dependencies ──────────────────────────────────────────
def install_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        print(C.warn("     ⚠ Không tìm thấy requirements.txt — bỏ qua"))
        return
    print("     Đang cài đặt...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req),
         "--quiet", "--no-warn-script-location"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(C.err("  ✗ Cài dependencies thất bại. Kiểm tra kết nối internet."))
        sys.exit(1)
    print(f"     {C.ok('Dependencies đã cài ✓')}")

# ── Xác định entry point ──────────────────────────────────────
def find_entry_point() -> Path:
    candidates = [
        ROOT / "app" / "app.py",     # Ưu tiên Flask app chính
        ROOT / "app.py",
    ]
    # run_server.py chỉ dùng nếu nó chứa Flask
    rsp = ROOT / "run_server.py"
    if rsp.exists():
        content = rsp.read_text(encoding="utf-8", errors="ignore").lower()
        if "flask" in content or "socketio" in content:
            candidates.insert(0, rsp)

    for c in candidates:
        if c.exists():
            return c

    print(C.err("  ✗ Không tìm thấy entry point Flask. Kiểm tra app/app.py."))
    sys.exit(1)

# ── Poll /api/health ──────────────────────────────────────────
def wait_for_server(port: int, timeout: float) -> bool:
    urls = [
        f"http://127.0.0.1:{port}/api/health",
        f"http://127.0.0.1:{port}/health",
    ]
    deadline = time.time() + timeout
    attempts = 0
    while time.time() < deadline:
        for url in urls:
            try:
                with urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (URLError, OSError):
                pass
        time.sleep(HEALTH_POLL_S)
        attempts += 1
        print(f"\r     Chờ server... ({attempts * HEALTH_POLL_S:.0f}s / {timeout}s)", end="", flush=True)
    print()
    return False

# ── Main ──────────────────────────────────────────────────────
def main():
    print()
    print(f"  {C.BD}{C.CY}╔══════════════════════════════════════╗{C.NC}")
    print(f"  {C.BD}{C.CY}║        VietASR Pro  v2.1.0          ║{C.NC}")
    print(f"  {C.BD}{C.CY}║   Nhận Dạng Giọng Nói Tiếng Việt   ║{C.NC}")
    print(f"  {C.BD}{C.CY}╚══════════════════════════════════════╝{C.NC}")
    print()

    TOTAL = 5

    step(1, TOTAL, "Kiểm tra Python...")
    check_python()

    step(2, TOTAL, "Cài đặt dependencies...")
    install_requirements()

    step(3, TOTAL, "Kiểm tra file .env...")
    ensure_env()

    port = read_port()
    entry = find_entry_point()

    step(4, TOTAL, f"Khởi động server: {entry.relative_to(ROOT)}")
    print()
    print(f"  {C.B}──────────────────────────────────────────{C.NC}")
    print(f"  Server đang khởi động tại port {port}...")
    print(f"  {C.B}──────────────────────────────────────────{C.NC}")
    print()

    # Thiết lập biến môi trường
    env = {**os.environ, "FLASK_PORT": str(port)}

    server_proc = subprocess.Popen(
        [sys.executable, str(entry)],
        cwd=ROOT,
        env=env,
    )

    # Xử lý Ctrl+C
    def _signal_handler(sig, frame):
        print(f"\n\n  {C.Y}Đang dừng server...{C.NC}")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print(f"  {C.ok('Server đã dừng. Tạm biệt!')}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    step(5, TOTAL, f"Chờ server sẵn sàng (tối đa {HEALTH_TIMEOUT_S}s)...")
    url = f"http://127.0.0.1:{port}"
    ready = wait_for_server(port, HEALTH_TIMEOUT_S)

    # Kiểm tra xem tiến trình server có thoát đột ngột không
    if server_proc.poll() is not None:
        print(C.err(f"\n  ✗ Server đã dừng đột ngột với mã thoát {server_proc.returncode}."))
        print(C.err("    Vui lòng kiểm tra logs/app.log hoặc console để tìm lỗi."))
        sys.exit(1)

    if ready:
        print(f"\n  {C.ok('✓ Server sẵn sàng!')} Đang mở trình duyệt...")
        webbrowser.open(url)
    else:
        print(f"\n  {C.warn('⚠ Server chưa phản hồi trong 120 giây — không mở trình duyệt.')}")
        print(f"    Có thể truy cập thủ công tại: {url}")

    print()
    print(f"  {C.BD}{C.G}════════════════════════════════════════════{C.NC}")
    print(f"  {C.BD}{C.G}  VietASR Pro đang chạy tại:{C.NC}")
    print(f"  {C.BD}{C.G}  {url}{C.NC}")
    print()
    print(f"  {C.Y}  Nhấn Ctrl+C để dừng server.{C.NC}")
    print(f"  {C.BD}{C.G}════════════════════════════════════════════{C.NC}")
    print()

    # Chờ server kết thúc
    server_proc.wait()


if __name__ == "__main__":
    main()
