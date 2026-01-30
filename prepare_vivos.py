"""
Main entry point - Prepare VIVOS dataset only
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.prepare_vivos_only import main

if __name__ == "__main__":
    main()
