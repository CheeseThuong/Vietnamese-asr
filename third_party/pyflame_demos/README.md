# PyFlame Demo Package
**Bộ demo cho thư viện PyFlame - Deep Learning Framework for Cerebras WSE**

---

## 📋 Tổng Quan

PyFlame là framework deep learning được thiết kế tối ưu cho **Cerebras Wafer-Scale Engine (WSE)** - một kiến trúc phần cứng đột phá với 850,000+ Processing Elements trên một chip wafer.

### Đặc điểm chính:
- 🔥 **API giống PyTorch** - dễ học, dễ migrate
- 🚀 **Lazy evaluation** - tối ưu computation graph
- 🎯 **Automatic CSL generation** - compile xuống Cerebras hardware
- 🌐 **2D mesh layouts** - tận dụng kiến trúc WSE
- 💻 **CPU reference mode** - development không cần hardware

---

## 📂 Cấu Trúc Demo

### Demo 1: Basic Tensor Operations
**File:** `01_basic_tensor_operations.py`

**Nội dung:**
- Tensor creation (zeros, ones, randn, from_numpy)
- Arithmetic operations (+, -, *, /)
- Matrix multiplication (@)
- Activation functions (ReLU, Sigmoid, Tanh)
- Reduction operations (sum, mean, max, min)
- Reshape và transpose
- **Lazy evaluation** với `pf.eval()`

**Chạy demo:**
```bash
python 01_basic_tensor_operations.py
```

**Expected output:**
```
==================================================
PYFLAME DEMO 1: BASIC TENSOR OPERATIONS
==================================================

[1] Tensor Creation
--------------------------------------------------
Zeros tensor: Tensor([[0., 0., 0., 0.], ...])
Ones tensor: Tensor([[1., 1., 1., 1.], ...])
...
```

---

### Demo 2: Neural Network Training
**File:** `02_neural_network_training.py`

**Nội dung:**
- Neural network definition với `nn.Sequential`
- Optimizer setup (Adam)
- Loss function (CrossEntropyLoss)
- Training loop (5 epochs demo)
- Danh sách tất cả layers, activations, losses, optimizers

**Chạy demo:**
```bash
python 02_neural_network_training.py
```

**Expected output:**
```
==================================================
PYFLAME DEMO 2: NEURAL NETWORK TRAINING
==================================================

[1] Model Architecture
--------------------------------------------------
Input: 784 (28x28 image)
  ↓ Linear(784 → 256)
  ↓ ReLU
  ↓ Linear(256 → 128)
...
Step 1: Loss = 2.3456
Step 2: Loss = 2.1234
...
```

---

### Demo 3: Cerebras WSE Mesh Layout
**File:** `03_mesh_layout_demo.py`

**Nội dung:**
- Giới thiệu kiến trúc WSE
- Single PE layout
- Row partition layout
- Column partition layout
- **2D Grid layout** (optimal cho matmul)
- Communication patterns
- Performance benefits

**Chạy demo:**
```bash
python 03_mesh_layout_demo.py
```

**Học được gì:**
- Cách PyFlame tối ưu cho Cerebras hardware
- Distributed tensor layouts
- Massive parallelism
- So sánh GPU vs WSE

---

### Demo 4: PyFlame vs PyTorch
**File:** `04_pytorch_comparison.py`

**Nội dung:**
- Side-by-side code comparison
- API similarities
- Key differences (eager vs lazy)
- Hardware differences (GPU vs WSE)
- When to use each framework
- Migration guide
- Summary table

**Chạy demo:**
```bash
python 04_pytorch_comparison.py
```

**Rất hữu ích để:**
- Hiểu mối quan hệ PyFlame ↔ PyTorch
- Quyết định khi nào dùng framework nào
- Migrate code từ PyTorch

---

## 🛠️ Yêu Cầu Hệ Thống

### Tối thiểu:
- **Python:** 3.8+
- **CMake:** 3.18+
- **Compiler:** C++17 support (GCC 8+, Clang 7+, MSVC 2019+)
- **Dependencies:** pybind11, NumPy

### Tùy chọn (cho full features):
- **Cerebras SDK** (proprietary - for actual WSE execution)
- Access to Cerebras CS-2/CS-3 hardware hoặc Cerebras Cloud

---

## 📥 Cài Đặt PyFlame

### Option 1: Build từ source (Recommended)

```bash
# Clone repository
git clone https://github.com/CTO92/PyFlame.git
cd PyFlame

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (Windows)
cmake --build . --config Release

# Build (Linux/Mac)
make -j$(nproc)

# Install
cd ..
pip install -e .
```

### Option 2: Với Cerebras SDK (nếu có access)

```bash
# Setup Cerebras environment
source /path/to/cerebras/sdk/setup.sh

# Build với SDK support
cmake .. -DCMAKE_BUILD_TYPE=Release -DCEREBRAS_SDK_PATH=/path/to/sdk
cmake --build . --config Release
pip install -e .
```

---

## ✅ Kiểm Tra Cài Đặt

```bash
python -c "import pyflame as pf; print(f'PyFlame version: {pf.__version__}')"
```

**Expected output:**
```
PyFlame version: 2.0.0-alpha
```

**Nếu lỗi:**
- Check Python version: `python --version`
- Check CMake: `cmake --version`
- Check build errors in `build/` folder
- Đọc `INSTALLATION_GUIDE.md`

---

## 🚀 Quick Start

```python
import pyflame as pf
import pyflame.nn as nn

# Create tensors
x = pf.randn([32, 784])
y = pf.ones([32, 10])

# Define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Forward pass
output = model(x)

# Evaluate (execute computation graph)
result = pf.eval(output)
print(result.shape)  # [32, 10]
```

---

## 🐛 Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'pyflame'`
**Solution:**
```bash
cd PyFlame
pip install -e .
```

### Issue 2: `CMake Error: Could not find pybind11`
**Solution:**
```bash
pip install pybind11
# Hoặc
conda install pybind11 -c conda-forge
```

### Issue 3: Build fails với C++ errors
**Solution:**
- Check compiler hỗ trợ C++17
- Update CMake lên 3.18+
- Windows: Dùng Visual Studio 2019+

### Issue 4: Demos chạy nhưng warnings
**Note:** CPU reference mode có thể show warnings - bình thường!
```
WARNING: Running on CPU reference implementation
```

---

## 📊 So Sánh Frameworks

| Feature | PyTorch | TensorFlow | **PyFlame** |
|---------|---------|------------|-------------|
| **API Style** | Pythonic | Keras/Functional | **PyTorch-like** |
| **Execution** | Eager | Eager/Graph | **Lazy (Graph)** |
| **Hardware** | CPU/CUDA | CPU/CUDA/TPU | **Cerebras WSE** |
| **Scalability** | Multi-GPU | Multi-GPU/TPU | **Massive Parallel** |
| **Best For** | Research | Production | **Large-Scale Training** |

---

## 📚 Tài Liệu Tham Khảo

- **PyFlame GitHub:** https://github.com/CTO92/PyFlame
- **Cerebras WSE:** https://www.cerebras.net/product-system/
- **CSL Documentation:** https://sdk.cerebras.net/csl/
- **PyFlame Examples:** `PyFlame/examples/` folder

---

## 📝 Notes cho Báo Cáo

### Điểm nhấn khi trình bày:

1. **PyFlame không phải thư viện ASR**
   - Là framework deep learning tổng quát (như PyTorch)
   - Tối ưu cho Cerebras WSE hardware

2. **Cerebras WSE là gì?**
   - 850,000+ Processing Elements trên 1 chip wafer
   - 44GB on-wafer SRAM
   - 20 Tb/s fabric bandwidth
   - Lớn nhất thế giới (46,225 mm²)

3. **Tại sao PyFlame quan trọng?**
   - Train được models cực lớn (LLMs)
   - Linear scalability
   - Không bị GPU memory bottleneck
   - Automatic optimization cho WSE

4. **CPU Reference Mode**
   - Development/testing không cần hardware
   - Giống như CUDA emulation trên CPU
   - Đủ để học và demo

5. **Migration từ PyTorch**
   - Rất dễ (API gần như giống hệt)
   - Chỉ cần change imports
   - Thêm `pf.eval()` cho lazy evaluation

---

## 🎯 Kết Luận

PyFlame là framework mạnh mẽ cho large-scale deep learning training, đặc biệt khi có access vào Cerebras WSE hardware. API tương thích PyTorch giúp migration dễ dàng, trong khi lazy evaluation và automatic CSL generation tối ưu hóa performance trên kiến trúc wafer-scale.

**Demos này cung cấp:**
- ✅ Hiểu biết cơ bản về PyFlame
- ✅ Hands-on experience với API
- ✅ So sánh với PyTorch
- ✅ Insight về Cerebras WSE architecture

**Phù hợp cho:** Research, learning, và proof-of-concept cho large-scale AI training.

---

**Author:** Demo package for academic presentation  
**Version:** 1.0  
**Last Updated:** 2024
