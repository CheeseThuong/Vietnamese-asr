#!/usr/bin/env bash
# ============================================================
#  run_vietasr.sh — VietASR Pro One-Click Launcher (macOS/Linux)
#  Tự động tạo môi trường, cài thư viện, khởi động server
#  và mở trình duyệt khi sẵn sàng.
# ============================================================

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN} ╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN} ║        VietASR Pro  v2.1.0          ║${NC}"
echo -e "${BOLD}${CYAN} ║   Nhận Dạng Giọng Nói Tiếng Việt   ║${NC}"
echo -e "${BOLD}${CYAN} ╚══════════════════════════════════════╝${NC}"
echo ""

# ── Kiểm tra Python ─────────────────────────────────────────
echo -e "${CYAN}[1/6]${NC} Kiểm tra Python..."
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "${RED}  ✗ Không tìm thấy Python. Cài Python 3.8+ từ: https://python.org${NC}"
    exit 1
fi
PYTHON_CMD=$(command -v python3 || command -v python)
PYVER=$($PYTHON_CMD --version 2>&1)
echo -e "   ${GREEN}${PYVER} ✓${NC}"

# ── Tạo virtual environment ──────────────────────────────────
echo -e "${CYAN}[2/6]${NC} Kiểm tra virtual environment..."
if [ ! -f "$ROOT/venv/bin/activate" ]; then
    echo "   Tạo venv mới..."
    $PYTHON_CMD -m venv venv
    echo -e "   ${GREEN}venv đã tạo ✓${NC}"
else
    echo -e "   ${GREEN}venv đã tồn tại ✓${NC}"
fi

# ── Kích hoạt venv ──────────────────────────────────────────
source "$ROOT/venv/bin/activate"
echo -e "${CYAN}[3/6]${NC} ${GREEN}Đã kích hoạt virtual environment ✓${NC}"

# ── Cài dependencies ────────────────────────────────────────
echo -e "${CYAN}[4/6]${NC} Kiểm tra và cài đặt dependencies..."
if [ -f "$ROOT/requirements.txt" ]; then
    pip install -r "$ROOT/requirements.txt" --quiet --no-warn-script-location
    echo -e "   ${GREEN}Dependencies đã cài ✓${NC}"
else
    echo -e "   ${YELLOW}⚠ Không tìm thấy requirements.txt — bỏ qua${NC}"
fi

# ── Kiểm tra .env ───────────────────────────────────────────
echo -e "${CYAN}[5/6]${NC} Kiểm tra cấu hình .env..."
if [ ! -f "$ROOT/.env" ]; then
    if [ -f "$ROOT/.env.example" ]; then
        cp "$ROOT/.env.example" "$ROOT/.env"
        echo -e "   ${YELLOW}⚠ Đã tạo .env từ .env.example${NC}"
        echo -e "   ${YELLOW}  Mở .env và điền GEMINI_API_KEY nếu muốn dùng Gemini AI.${NC}"
    else
        echo -e "   ${YELLOW}⚠ Không tìm thấy .env — server sẽ chạy không có API key${NC}"
    fi
else
    echo -e "   ${GREEN}.env đã tồn tại ✓${NC}"
fi

# ── Đọc port từ .env ────────────────────────────────────────
PORT=5000
if [ -f "$ROOT/.env" ]; then
    _P=$(grep -E "^FLASK_PORT=" "$ROOT/.env" | cut -d= -f2 | tr -d '[:space:]')
    [ -n "$_P" ] && PORT="$_P"
    _P=$(grep -E "^PORT=" "$ROOT/.env" | cut -d= -f2 | tr -d '[:space:]')
    [ -n "$_P" ] && PORT="$_P"
fi

# ── Xác định entry point ─────────────────────────────────────
ENTRY=""
if [ -f "$ROOT/launcher.py" ]; then
    ENTRY="launcher.py"
elif [ -f "$ROOT/run_server.py" ] && grep -qi "flask\|socketio\|app.py" "$ROOT/run_server.py" 2>/dev/null; then
    ENTRY="run_server.py"
elif [ -f "$ROOT/app/app.py" ]; then
    ENTRY="app/app.py"
else
    echo -e "${RED}  ✗ Không tìm thấy entry point. Kiểm tra app/app.py tồn tại.${NC}"
    exit 1
fi

# ── Khởi động server ────────────────────────────────────────
echo -e "${CYAN}[6/6]${NC} Khởi động server (${ENTRY})..."
echo ""
echo -e " ──────────────────────────────────────────"
echo -e "  Server đang khởi động... chờ vài giây."
echo -e " ──────────────────────────────────────────"
echo ""

python "$ROOT/$ENTRY" &
SERVER_PID=$!

# Polling /api/health
echo -n "  Đang chờ server sẵn sàng tại http://127.0.0.1:${PORT} "
for i in $(seq 1 30); do
    sleep 1
    echo -n "."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/api/health" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        break
    fi
    # Fallback /health
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        break
    fi
done
echo ""

# Mở trình duyệt
URL="http://127.0.0.1:${PORT}"
echo -e "${GREEN}  ✓ Server sẵn sàng! Đang mở trình duyệt...${NC}"
if command -v xdg-open &>/dev/null; then
    xdg-open "$URL" &>/dev/null &
elif command -v open &>/dev/null; then
    open "$URL"
fi

echo ""
echo -e "${BOLD}${GREEN} ════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  VietASR Pro đang chạy tại:${NC}"
echo -e "${BOLD}${GREEN}  ${URL}${NC}"
echo ""
echo -e "${YELLOW}  Nhấn Ctrl+C để dừng server.${NC}"
echo -e "${BOLD}${GREEN} ════════════════════════════════════════════${NC}"
echo ""

# Chờ server process
wait $SERVER_PID
