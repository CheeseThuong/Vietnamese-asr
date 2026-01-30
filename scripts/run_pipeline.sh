#!/bin/bash
# Script để chạy toàn bộ pipeline từ đầu đến cuối

echo "=========================================="
echo "Vietnamese ASR - Complete Pipeline"
echo "=========================================="

# 1. Prepare dataset
echo -e "\n[1/7] Preparing dataset..."
python prepare_dataset.py
if [ $? -ne 0 ]; then
    echo "Error: Dataset preparation failed!"
    exit 1
fi

# 2. Data preprocessing
echo -e "\n[2/7] Preprocessing data..."
python data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Error: Data preprocessing failed!"
    exit 1
fi

# 3. Train model
echo -e "\n[3/7] Training Wav2Vec2 model..."
python train_wav2vec2.py
if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

# 4. Build language model
echo -e "\n[4/7] Building language model..."
python language_model.py
if [ $? -ne 0 ]; then
    echo "Warning: Language model building failed, continuing..."
fi

# 5. Evaluate
echo -e "\n[5/7] Evaluating model..."
python run_evaluation.py
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

# 6. Optimize
echo -e "\n[6/7] Optimizing model..."
python optimization.py
if [ $? -ne 0 ]; then
    echo "Warning: Optimization failed, continuing..."
fi

# 7. Done
echo -e "\n[7/7] Pipeline complete!"
echo "=========================================="
echo "All steps completed successfully!"
echo "To start the web server, run:"
echo "  python api_server.py"
echo "=========================================="
