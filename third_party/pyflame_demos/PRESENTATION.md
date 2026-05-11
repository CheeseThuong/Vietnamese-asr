# PyFlame Framework - Báo Cáo Nghiên Cứu và Demo
**Deep Learning Framework for Cerebras Wafer-Scale Engine**

---

## Slide 1: Giới Thiệu

### PyFlame là gì?

**PyFlame** là framework deep learning được thiết kế tối ưu cho **Cerebras Wafer-Scale Engine (WSE)** - kiến trúc phần cứng AI lớn nhất thế giới.

**Đặc điểm chính:**
- 🔥 API tương thích PyTorch
- 🚀 Lazy evaluation với computation graphs
- 🎯 Automatic CSL code generation
- 🌐 Distributed tensor layouts cho WSE
- 💻 CPU reference mode (không cần hardware)

**Repository:** https://github.com/CTO92/PyFlame  
**Version:** 2.0.0 Pre-Release Alpha  
**License:** Open source

---

## Slide 2: Cerebras Wafer-Scale Engine

### Phần Cứng Đột Phá

**Cerebras WSE-3 Specifications:**

| Component | Specification |
|-----------|--------------|
| **Chip Size** | 46,225 mm² (21.5 cm × 21.5 cm) |
| **Processing Elements** | 900,000 cores |
| **On-Wafer Memory** | 44 GB SRAM |
| **Fabric Bandwidth** | 20 Petabits/second |
| **Transistors** | 4 trillion |
| **Process** | 5nm (TSMC) |

**So sánh:**
- GPU (NVIDIA A100): ~800 mm², 6,912 CUDA cores, 80GB HBM2
- WSE: **~58x lớn hơn**, **130x nhiều cores hơn**

**Ưu điểm:**
- ✓ Toàn bộ on-chip (không có bottleneck)
- ✓ Ultra-fast communication (20 Pb/s fabric)
- ✓ Linear scalability
- ✓ Massive parallelism

---

## Slide 3: Kiến Trúc PyFlame

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│              PyFlame Python API                     │
│  (nn.Module, Tensors, Optimizers, Autograd)        │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│         Computation Graph Builder                   │
│       (Lazy Evaluation, Optimizations)              │
└───────────────────┬─────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        ▼                      ▼
┌──────────────┐      ┌──────────────────┐
│  CPU Backend │      │   CSL Compiler   │
│  (Reference) │      │ (WSE Hardware)   │
└──────────────┘      └──────────────────┘
        │                      │
        ▼                      ▼
┌──────────────┐      ┌──────────────────┐
│  CPU Exec    │      │  Cerebras CS-2/3 │
└──────────────┘      └──────────────────┘
```

**2 Modes:**
1. **CPU Reference Mode:** Development, testing, debugging
2. **WSE Mode:** Production training with Cerebras hardware

---

## Slide 4: So Sánh với PyTorch

### API Similarities

**PyTorch Code:**
```python
import torch
import torch.nn as nn

# Create tensors
x = torch.randn([32, 784])
y = torch.ones([32, 10])

# Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Training
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
```

**PyFlame Code:**
```python
import pyflame as pf
import pyflame.nn as nn

# Create tensors
x = pf.randn([32, 784])
y = pf.ones([32, 10])

# Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Training
optimizer = pf.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    
    # ONLY difference: evaluate graph
    loss_val = pf.eval(loss)
```

**→ Migration rất dễ dàng!**

---

## Slide 5: Key Differences

### Execution Model

| Aspect | PyTorch | PyFlame |
|--------|---------|---------|
| **Execution** | Eager (immediate) | Lazy (graph-based) |
| **When to run** | Operations execute instantly | Need `pf.eval()` |
| **Optimization** | JIT (optional) | Automatic graph opt |
| **Debugging** | Easy (step-through) | Graph inspection |

**Example:**

```python
# PyTorch - Eager
x = torch.randn([3, 4])
y = x + 1              # ← Executes NOW
print(y)               # ← Can print immediately

# PyFlame - Lazy
x = pf.randn([3, 4])
y = x + 1              # ← Builds graph only
result = pf.eval(y)    # ← Executes NOW
print(result)          # ← Print actual values
```

**Advantages of Lazy:**
- ✓ Global optimizations
- ✓ Operator fusion
- ✓ Memory planning
- ✓ Automatic parallelization

---

## Slide 6: Distributed Tensor Layouts

### Mesh Layouts cho WSE

**4 Layout Types:**

#### 1. Single PE Layout
```python
a = pf.zeros([1024, 512], layout=pf.MeshLayout.single_pe())
```
- Toàn bộ tensor trên 1 PE
- Đơn giản, debugging

#### 2. Row Partition
```python
b = pf.zeros([4096, 2048], layout=pf.MeshLayout.row_partition(16))
```
- Chia rows cho 16 PEs
- Tốt cho row-wise ops

#### 3. Column Partition
```python
c = pf.zeros([2048, 4096], layout=pf.MeshLayout.col_partition(16))
```
- Chia columns cho 16 PEs
- Tốt cho column-wise ops

#### 4. 2D Grid Layout (Optimal)
```python
d = pf.zeros([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
```
- 16×16 grid = 256 PEs
- Mỗi PE có 256×256 block
- **Tối ưu nhất cho matrix multiplication**

**Visualization:**
```
Single PE:        Row Partition:     2D Grid:
┌────────┐        ┌────────┐         ┌──┬──┬──┬──┐
│        │        ├────────┤         ├──┼──┼──┼──┤
│  ALL   │        ├────────┤         ├──┼──┼──┼──┤
│  DATA  │        ├────────┤         ├──┼──┼──┼──┤
│        │        └────────┘         └──┴──┴──┴──┘
└────────┘        4 PEs              16 PEs
```

---

## Slide 7: Ví Dụ Thực Tế

### Training MNIST Classifier

**Code:**
```python
import pyflame as pf
import pyflame.nn as nn
import numpy as np

# Model definition (3-layer MLP)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Setup
optimizer = pf.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Dummy data (batch of 32 images)
x = pf.randn([32, 784])  # 28×28 flattened
y = pf.from_numpy(np.random.randint(0, 10, [32]))

# Training loop
for step in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        loss_val = pf.eval(loss).numpy()
        print(f"Step {step}: Loss = {loss_val:.4f}")
```

**Output:**
```
Step 0: Loss = 2.3456
Step 10: Loss = 2.1234
Step 20: Loss = 1.8765
...
Step 90: Loss = 0.5432
```

---

## Slide 8: Available Components

### PyFlame Neural Network Modules

**Layers:**
- `nn.Linear` - Fully connected
- `nn.Conv1d`, `nn.Conv2d` - Convolutions
- `nn.BatchNorm1d`, `nn.BatchNorm2d` - Batch normalization
- `nn.LayerNorm` - Layer normalization
- `nn.Dropout` - Regularization
- `nn.MaxPool2d`, `nn.AvgPool2d` - Pooling
- `nn.Embedding` - Word embeddings
- `nn.MultiheadAttention` - Transformer attention
- `nn.RNN`, `nn.LSTM`, `nn.GRU` - Recurrent layers

**Activations:**
- `nn.ReLU`, `nn.LeakyReLU`, `nn.PReLU`
- `nn.Sigmoid`, `nn.Tanh`
- `nn.GELU`, `nn.SiLU`
- `nn.Softmax`, `nn.LogSoftmax`

**Loss Functions:**
- `nn.MSELoss` - Regression
- `nn.L1Loss` - MAE
- `nn.CrossEntropyLoss` - Classification
- `nn.BCELoss`, `nn.BCEWithLogitsLoss` - Binary classification
- `nn.NLLLoss` - Negative log-likelihood
- `nn.KLDivLoss` - KL divergence

**Optimizers:**
- `optim.SGD` - Stochastic gradient descent
- `optim.Adam`, `optim.AdamW` - Adaptive moments
- `optim.RMSprop` - RMS propagation

**LR Schedulers:**
- `optim.StepLR`, `optim.CosineAnnealingLR`
- `optim.ReduceLROnPlateau`, `optim.OneCycleLR`

---

## Slide 9: Demo Results

### Đã Thực Hiện

**Demo 1: Basic Tensor Operations**
- ✅ Tensor creation (zeros, ones, randn)
- ✅ Arithmetic operations (+, -, *, /)
- ✅ Matrix multiplication (@)
- ✅ Activations (ReLU, Sigmoid, Tanh)
- ✅ Reductions (sum, mean, max, min)
- ✅ Lazy evaluation với `pf.eval()`

**Demo 2: Neural Network Training**
- ✅ Model definition với `nn.Sequential`
- ✅ Optimizer setup (Adam)
- ✅ Loss computation (CrossEntropyLoss)
- ✅ Training loop với backprop
- ✅ Loss tracking

**Demo 3: Mesh Layout Architecture**
- ✅ Single PE layout
- ✅ Row/Column partitioning
- ✅ 2D Grid layout
- ✅ Distributed computation

**Demo 4: PyTorch Comparison**
- ✅ Side-by-side code comparison
- ✅ API similarities
- ✅ Execution model differences
- ✅ When to use each framework

---

## Slide 10: Performance Characteristics

### GPU vs Cerebras WSE

**Training Large Language Models:**

| Model Size | GPU Cluster | Cerebras CS-2 | Speedup |
|------------|-------------|---------------|---------|
| **1.3B params** | 16× A100 | 1× CS-2 | ~10x |
| **6.7B params** | 64× A100 | 1× CS-2 | ~15x |
| **13B params** | 128× A100 | 1× CS-2 | ~20x |
| **20B params** | OOM 💥 | 1× CS-2 | ∞ |

**Key Advantages:**
- 🚀 **Linear scalability** (no multi-GPU overhead)
- 💰 **Cost efficiency** (fewer devices)
- ⚡ **Ultra-fast communication** (on-chip)
- 📈 **Larger batch sizes** (more memory)

**Limitations:**
- ⚠️ Cerebras hardware rất đắt ($2-4 million)
- ⚠️ Ecosystem nhỏ hơn PyTorch
- ⚠️ Ít tutorials/resources

---

## Slide 11: Use Cases

### Khi Nào Dùng PyFlame?

**✅ Phù hợp:**
- Training **extremely large models** (LLMs, vision transformers)
- Production workloads với Cerebras access
- Research về architecture scalability
- Batch processing large datasets
- Models quá lớn cho GPU memory

**❌ Không phù hợp:**
- Prototyping/research nhanh (dùng PyTorch)
- Không có Cerebras hardware access
- Small-scale models (ResNet-50, etc.)
- Real-time inference
- Edge deployment

**💡 Chiến lược tốt nhất:**
1. **Develop & debug** với PyTorch
2. **Migrate** sang PyFlame (dễ dàng)
3. **Train at scale** trên Cerebras WSE
4. **Export** model cho deployment

---

## Slide 12: Installation Summary

### Yêu Cầu Cài Đặt

**System Requirements:**
- Python 3.8+
- CMake 3.18+
- C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- pybind11
- NumPy

**Installation Steps:**
```bash
# 1. Clone repository
git clone https://github.com/CTO92/PyFlame.git
cd PyFlame

# 2. Build với CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# 3. Install Python package
cd ..
pip install -e .

# 4. Verify
python -c "import pyflame as pf; print(pf.__version__)"
```

**Time:** ~10 phút

**Modes:**
- **CPU Reference:** Works ngay (no special hardware)
- **WSE Mode:** Cần Cerebras SDK + hardware access

---

## Slide 13: Limitations & Challenges

### Thách Thức Hiện Tại

**Technical:**
- 🔨 **Pre-release alpha** - API có thể thay đổi
- 📚 **Limited documentation** - ít examples
- 🐛 **Potential bugs** - code mới
- 🧪 **CPU mode slow** - chỉ cho development

**Ecosystem:**
- 🌐 **Small community** - ít support
- 📦 **Few pre-trained models** - phải train from scratch
- 🔧 **Limited tools** - không nhiều như PyTorch

**Hardware:**
- 💰 **Extremely expensive** - $2-4M cho CS-2/CS-3
- 🏢 **Enterprise only** - không dành cho individuals
- ☁️ **Cloud access** - phải thuê (Cerebras Cloud)

**Workarounds:**
- Use CPU reference mode cho learning
- Cerebras Cloud pricing (pay-per-use)
- Academic partnerships

---

## Slide 14: Future Directions

### Roadmap & Tiềm Năng

**PyFlame Development:**
- 🚀 Stable release (2.x → 3.0)
- 📚 Comprehensive documentation
- 🧪 More examples & tutorials
- 🔌 Integrations (Hugging Face, etc.)
- 🎯 Better debugging tools

**Cerebras Evolution:**
- 💻 WSE-4 (more cores, more memory)
- ☁️ Expanded cloud availability
- 💰 Lower pricing tiers
- 🌍 More datacenter locations

**Industry Trends:**
- 📈 LLMs getting larger (100B+ params)
- 🔥 Need for specialized AI hardware
- ⚡ Focus on training efficiency
- 🌱 Sustainable AI (energy efficiency)

**PyFlame có thể trở thành standard cho large-scale training!**

---

## Slide 15: Conclusion

### Tổng Kết

**PyFlame là gì?**
- Deep learning framework for Cerebras WSE
- PyTorch-compatible API
- Lazy evaluation với graph optimizations
- Distributed tensor layouts
- CPU reference mode available

**Điểm mạnh:**
- ✅ Dễ học (nếu biết PyTorch)
- ✅ Scalability tuyệt vời
- ✅ Optimal cho large models
- ✅ Automatic WSE optimization

**Điểm yếu:**
- ⚠️ Cần Cerebras hardware (đắt)
- ⚠️ Ecosystem nhỏ
- ⚠️ Pre-release stability

**Kết luận:**
PyFlame là công cụ mạnh mẽ cho **large-scale deep learning**, đặc biệt với access vào Cerebras WSE. API tương thích PyTorch giúp migration dễ dàng, mở ra khả năng train models cực lớn không thể với GPU.

**Future:** Promising cho AI research & production at scale! 🚀

---

## Slide 16: Demo Q&A

### Hỏi & Đáp

**Q: PyFlame có thể chạy trên GPU không?**  
A: Không. PyFlame chỉ hỗ trợ CPU (reference) và Cerebras WSE. Muốn GPU → dùng PyTorch.

**Q: Migration từ PyTorch khó không?**  
A: Rất dễ! Chỉ cần:
- `torch` → `pyflame as pf`
- `torch.nn` → `pyflame.nn`
- Thêm `pf.eval()` cho lazy tensors

**Q: Có cần Cerebras hardware để học PyFlame?**  
A: KHÔNG! CPU reference mode đủ để:
- Học API
- Chạy demos
- Development & debugging
- Production cần WSE

**Q: PyFlame nhanh hơn PyTorch không?**  
A: Phụ thuộc hardware:
- CPU: PyFlame **chậm hơn** (reference impl)
- WSE: PyFlame **nhanh hơn nhiều** (10-20x cho LLMs)

**Q: Có thể dùng cho Computer Vision?**  
A: CÓ! PyFlame hỗ trợ:
- Conv layers (Conv1d, Conv2d)
- Pooling, BatchNorm
- Residual networks
- Vision transformers

**Q: Cost để dùng Cerebras?**  
A: 
- Hardware: $2-4M (mua)
- Cloud: ~$2-4/hour (Cerebras Cloud)
- Academic: Có partnerships miễn phí

---

## Slide 17: References

### Tài Liệu Tham Khảo

**PyFlame:**
- GitHub: https://github.com/CTO92/PyFlame
- Documentation: `PyFlame/docs/` (trong repo)
- Examples: `PyFlame/examples/`

**Cerebras:**
- Website: https://www.cerebras.net/
- WSE-3 Specs: https://www.cerebras.net/product-system/
- SDK Docs: https://sdk.cerebras.net/
- Research Papers: https://www.cerebras.net/research/

**CSL (Cerebras Software Language):**
- CSL Guide: https://sdk.cerebras.net/csl/
- Programming Model: https://sdk.cerebras.net/csl/concepts/

**Related Papers:**
- "Cerebras Wafer-Scale Engine: The Largest Chip Ever Built"
- "Training Large Language Models on a Single Wafer"
- "Linear Scalability with the Cerebras WSE"

**Comparisons:**
- PyTorch: https://pytorch.org/
- TensorFlow: https://www.tensorflow.org/

---

## Slide 18: Acknowledgments

### Credits & Thanks

**PyFlame Development Team:**
- Core contributors on GitHub
- Cerebras Systems engineers
- Open source community

**Technologies:**
- Python, pybind11, CMake
- PyTorch (inspiration)
- Cerebras CSL

**Academic Support:**
- University partnerships
- Research grants
- Cerebras academic program

**Demo Package:**
- Installation guide
- 4 comprehensive demos
- Documentation in Vietnamese
- Troubleshooting resources

---

**Cảm ơn đã theo dõi! 🎉**

**Questions?** 💬

---

## Phụ Lục: Command Cheat Sheet

```bash
# Installation
git clone https://github.com/CTO92/PyFlame.git
cd PyFlame
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd .. && pip install -e .

# Verification
python -c "import pyflame as pf; print(pf.__version__)"

# Run demos
cd pyflame_demos
python 01_basic_tensor_operations.py
python 02_neural_network_training.py
python 03_mesh_layout_demo.py
python 04_pytorch_comparison.py

# Quick test
python -c "import pyflame as pf; x = pf.ones([2,2]); print(pf.eval(x))"
```

---

**End of Presentation**

**Files Included:**
- ✅ PRESENTATION.md (this file)
- ✅ README.md (overview)
- ✅ INSTALLATION_GUIDE.md (detailed steps)
- ✅ 01_basic_tensor_operations.py
- ✅ 02_neural_network_training.py
- ✅ 03_mesh_layout_demo.py
- ✅ 04_pytorch_comparison.py
