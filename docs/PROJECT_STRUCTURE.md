# Vietnamese ASR Project Structure

```
vietnamese-asr/
│
├── src/                          # Source code chính
│   ├── __init__.py
│   │
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── prepare_dataset.py   # Gộp VIVOS + VinBigData
│   │   ├── prepare_vivos_only.py # Chỉ VIVOS
│   │   ├── preprocessing.py     # Preprocessing pipeline
│   │   └── normalize_audio.py   # Audio normalization
│   │
│   ├── models/                   # Model definitions (reserved)
│   │   └── __init__.py
│   │
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   ├── train_wav2vec2.py    # Training pipeline
│   │   └── language_model.py    # Language model
│   │
│   ├── evaluation/               # Evaluation scripts
│   │   ├── __init__.py
│   │   └── evaluate.py          # WER/CER evaluation
│   │
│   ├── api/                      # API server
│   │   ├── __init__.py
│   │   └── server.py            # FastAPI backend
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── optimization.py      # Model optimization
│       ├── profiling.py         # Profiling tools
│       └── demo.py              # Demo script
│
├── scripts/                      # Utility scripts
│   ├── setup/                    # Setup scripts
│   │   ├── install_dependencies.bat
│   │   └── quick_start.bat
│   │
│   ├── profiling/                # Profiling utilities
│   │   └── flamegraph_guide.py
│   │
│   ├── check_dependencies.py
│   ├── run_pipeline.bat
│   └── run_pipeline.sh
│
├── configs/                      # Configuration files
│   └── (training configs, model configs)
│
├── notebooks/                    # Jupyter notebooks
│   └── (exploratory analysis, demos)
│
├── tests/                        # Unit tests
│   └── (test files)
│
├── docs/                         # Documentation
│   └── (additional documentation)
│
├── Data/                         # Raw datasets
│   ├── vivos/
│   └── Data/ (VinBigData)
│
├── processed_data/               # Processed datasets
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── models/                       # Trained models
│   └── wav2vec2-vietnamese-asr/
│
├── language_models/              # Language models
│   └── vietnamese_5gram.bin
│
├── results/                      # Evaluation results
│   └── (predictions, metrics)
│
├── static/                       # Web UI assets
│   └── index.html
│
├── requirements.txt              # Main dependencies
├── requirements-core.txt         # Core dependencies
├── requirements-optional.txt     # Optional dependencies
├── .gitignore
└── README.md
```

## 📂 Directory Purposes

### Source Code (`src/`)
- **data/**: All data processing, loading, and preprocessing
- **models/**: Custom model definitions (reserved for future)
- **training/**: Training pipelines and language models
- **evaluation/**: Evaluation metrics and analysis
- **api/**: Web API server
- **utils/**: Shared utilities (optimization, profiling, demo)

### Scripts (`scripts/`)
- **setup/**: Installation and quick start scripts
- **profiling/**: Performance profiling tools
- Root level: Pipeline and dependency management

### Configuration (`configs/`)
- Training configurations
- Model hyperparameters
- Dataset configurations

### Data Directories
- **Data/**: Raw, unprocessed datasets (gitignored)
- **processed_data/**: Cleaned and prepared data
- **models/**: Saved model checkpoints
- **language_models/**: N-gram or neural LMs
- **results/**: Evaluation outputs and predictions

### Development
- **notebooks/**: Jupyter notebooks for exploration
- **tests/**: Unit and integration tests
- **docs/**: Additional documentation

## 🔄 Import Examples

```python
# Data processing
from src.data import VietnameseASRDataset, prepare_dataset, normalize_audio

# Training
from src.training import train_model, create_model, LanguageModelDecoder

# Evaluation
from src.evaluation import ASREvaluator

# Utils
from src.utils import CPUProfiler, optimize_model_for_inference

# API
from src.api import app
```

## 🎯 Benefits

1. **Modularity**: Each component has its own directory
2. **Scalability**: Easy to add new features
3. **Testability**: Clear separation for unit tests
4. **Maintainability**: Find code quickly
5. **Professional**: Standard Python project structure
6. **Collaboration**: Easy for team members to navigate
