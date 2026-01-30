"""
Main entry point - Training
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.train_wav2vec2 import main

if __name__ == "__main__":
    main()
