"""
Demo 2: Neural Network Training với PyFlame
Minh họa cách xây dựng và train một neural network đơn giản
"""

import numpy as np

print("=" * 70)
print("DEMO 2: NEURAL NETWORK TRAINING")
print("=" * 70)

try:
    import pyflame as pf
    from pyflame import nn, optim
    
    print("\n[1] Định nghĩa Model")
    print("-" * 70)
    
    # Tạo một simple neural network
    model = nn.Sequential(
        nn.Linear(784, 256),    # Input layer
        nn.ReLU(),              # Activation
        nn.Linear(256, 128),    # Hidden layer 1
        nn.ReLU(),
        nn.Linear(128, 10)      # Output layer
    )
    
    print("Model Architecture:")
    print("  Input:  784 features")
    print("  Hidden: 256 → ReLU → 128 → ReLU")
    print("  Output: 10 classes")
    
    print("\n[2] Setup Optimizer và Loss Function")
    print("-" * 70)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Optimizer: Adam (lr=0.001)")
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    print(f"Loss: CrossEntropyLoss")
    
    print("\n[3] Tạo Dummy Data")
    print("-" * 70)
    
    # Tạo batch dữ liệu giả
    batch_size = 32
    x = pf.randn([batch_size, 784])  # Input data
    y = pf.randint(0, 10, [batch_size])  # Labels
    
    print(f"Batch size: {batch_size}")
    print(f"Input shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    
    print("\n[4] Training Loop (5 steps)")
    print("-" * 70)
    
    for step in range(5):
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluate loss
        loss_value = pf.eval(loss).numpy()
        print(f"Step {step+1}/5 - Loss: {loss_value:.4f}")
    
    print("\n[5] Kiến trúc Model Components")
    print("-" * 70)
    
    # Các layers khác
    print("\nAvailable Layers:")
    print("  - Linear: Fully connected layer")
    print("  - Conv1d, Conv2d: Convolutional layers")
    print("  - BatchNorm1d, BatchNorm2d: Batch normalization")
    print("  - LayerNorm: Layer normalization")
    print("  - Dropout: Dropout regularization")
    print("  - MaxPool2d, AvgPool2d: Pooling layers")
    print("  - MultiheadAttention: Self-attention mechanism")
    
    print("\nActivation Functions:")
    print("  - ReLU, LeakyReLU, PReLU")
    print("  - Sigmoid, Tanh")
    print("  - GELU, SiLU, Softmax")
    
    print("\n[6] Other Loss Functions")
    print("-" * 70)
    
    print("Available Loss Functions:")
    print("  - MSELoss: Mean Squared Error")
    print("  - L1Loss: Mean Absolute Error")
    print("  - CrossEntropyLoss: Classification")
    print("  - BCELoss: Binary Cross-Entropy")
    print("  - NLLLoss: Negative Log Likelihood")
    print("  - KLDivLoss: KL Divergence")
    
    print("\n[7] Other Optimizers")
    print("-" * 70)
    
    print("Available Optimizers:")
    print("  - SGD: Stochastic Gradient Descent")
    print("  - Adam: Adaptive Moment Estimation")
    print("  - AdamW: Adam with Weight Decay")
    print("  - RMSprop: Root Mean Square Propagation")
    
    print("\n[8] Learning Rate Schedulers")
    print("-" * 70)
    
    print("Available Schedulers:")
    print("  - StepLR: Decay LR every N steps")
    print("  - CosineAnnealingLR: Cosine schedule")
    print("  - ReduceLROnPlateau: Adaptive LR")
    print("  - OneCycleLR: One-cycle policy")
    
    print("\n" + "=" * 70)
    print("✓ Demo 2 completed successfully!")
    print("=" * 70)

except ImportError as e:
    print(f"\n✗ PyFlame chưa được cài đặt!")
    print(f"Error: {e}")
    
except Exception as e:
    print(f"\n✗ Lỗi: {e}")
    import traceback
    traceback.print_exc()
    print("\nNOTE: Một số features cần Cerebras hardware.")
