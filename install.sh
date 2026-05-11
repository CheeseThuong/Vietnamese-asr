#!/usr/bin/env bash
# =====================================================================
# VietASR Pro - Auto Installer (Linux/macOS)
# Cài đặt tự động: Python check, venv, dependencies, model, Flask
# =====================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║        VietASR Pro - Cai Dat Tu Dong (Linux/macOS)      ║"
echo "  ║        Vietnamese ASR - Automatic Installer              ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =====================================================================
# STEP 1: Kiem tra Python
# =====================================================================
echo -e "${BOLD}[1/6] Kiem tra Python...${NC}"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        PYTHON_VERSION=$("$cmd" --version 2>&1 | awk '{print $2}')
        MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
        MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
        
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}  [LOI] Khong tim thay Python >= 3.9!${NC}"
    echo ""
    echo "  Cai dat Python:"
    echo "    Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    echo "    macOS:         brew install python@3.11"
    echo "    Hoac tai tu:   https://www.python.org/downloads/"
    echo ""
    exit 1
fi

echo -e "  ${GREEN}[OK]${NC} Python $PYTHON_VERSION ($PYTHON_CMD)"

# =====================================================================
# STEP 2: Tao Virtual Environment
# =====================================================================
echo ""
echo -e "${BOLD}[2/6] Tao virtual environment...${NC}"

VENV_DIR="$SCRIPT_DIR/.venv"

if [ -f "$VENV_DIR/bin/activate" ]; then
    echo -e "  ${GREEN}[OK]${NC} venv da ton tai, su dung lai."
else
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "  ${GREEN}[OK]${NC} Da tao venv tai: $VENV_DIR"
fi

# =====================================================================
# STEP 3: Activate venv
# =====================================================================
echo ""
echo -e "${BOLD}[3/6] Kich hoat virtual environment...${NC}"

source "$VENV_DIR/bin/activate"
echo -e "  ${GREEN}[OK]${NC} venv da kich hoat"

# Upgrade pip
pip install --upgrade pip --quiet 2>/dev/null
echo -e "  ${GREEN}[OK]${NC} pip da cap nhat"

# =====================================================================
# STEP 4: Cai dat dependencies
# =====================================================================
echo ""
echo -e "${BOLD}[4/6] Cai dat thu vien (co the mat vai phut)...${NC}"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt" --quiet 2>/dev/null || {
        echo -e "  ${YELLOW}[CANH BAO]${NC} Mot so thu vien co the khong cai duoc."
        echo "  Dang thu cai tung thu vien..."
        while IFS= read -r line; do
            # Skip comments and empty lines
            [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
            pip install "$line" --quiet 2>/dev/null || true
        done < "$SCRIPT_DIR/requirements.txt"
    }
    echo -e "  ${GREEN}[OK]${NC} Da cai dat dependencies"
else
    echo -e "  ${YELLOW}[CANH BAO]${NC} Khong tim thay requirements.txt"
    echo "  Dang cai dat thu vien co ban..."
    pip install torch torchaudio transformers datasets librosa soundfile flask flask-cors pydub jiwer --quiet
    echo -e "  ${GREEN}[OK]${NC} Da cai dat thu vien co ban"
fi

# Cài thêm Flask nếu thiếu
pip install flask flask-cors pydub --quiet 2>/dev/null || true

# =====================================================================
# STEP 5: Kiem tra model
# =====================================================================
echo ""
echo -e "${BOLD}[5/6] Kiem tra model ASR...${NC}"

MODEL_DIR="$SCRIPT_DIR/final_model"

if [ -d "$MODEL_DIR" ]; then
    echo -e "  ${GREEN}[OK]${NC} Da tim thay model tai: $MODEL_DIR"
else
    echo ""
    echo "  Khong tim thay model local tai: $MODEL_DIR"
    echo "  He thong se tu dong tai model tu Hugging Face khi chay."
    echo ""
    echo "  Neu ban co model rieng, hay copy vao thu muc:"
    echo "    $MODEL_DIR"
    echo ""
    read -p "  Nhap duong dan model (Enter de bo qua): " USER_MODEL
    
    if [ -n "$USER_MODEL" ] && [ -d "$USER_MODEL" ]; then
        cp -r "$USER_MODEL" "$MODEL_DIR"
        echo -e "  ${GREEN}[OK]${NC} Da copy model tu: $USER_MODEL"
    else
        echo -e "  ${GREEN}[OK]${NC} Se su dung model tu Hugging Face"
    fi
fi

# =====================================================================
# STEP 6: Khoi dong server
# =====================================================================
echo ""
echo -e "${BOLD}[6/6] Khoi dong VietASR Pro...${NC}"
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                 CAI DAT HOAN TAT!                       ║"
echo "  ║                                                          ║"
echo "  ║  Server se khoi dong tai: http://localhost:5000          ║"
echo "  ║  Nhan Ctrl+C de dung server                              ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Mở trình duyệt sau 3 giây
(sleep 3 && {
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:5000" 2>/dev/null
    elif command -v open &> /dev/null; then
        open "http://localhost:5000"
    fi
}) &

# Chạy server
cd "$SCRIPT_DIR"
python -m app.app
