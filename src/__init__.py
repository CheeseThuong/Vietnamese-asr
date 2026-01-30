"""
Vietnamese ASR Package
Nhận dạng giọng nói tiếng Việt
"""

__version__ = "1.0.0"
__author__ = "Nguyễn Trí Thượng"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent.parent

# Project directories
DATA_DIR = PACKAGE_ROOT / "Data"
MODELS_DIR = PACKAGE_ROOT / "models"
RESULTS_DIR = PACKAGE_ROOT / "results"
CONFIGS_DIR = PACKAGE_ROOT / "configs"

# Ensure directories exist
for directory in [MODELS_DIR, RESULTS_DIR, CONFIGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)
