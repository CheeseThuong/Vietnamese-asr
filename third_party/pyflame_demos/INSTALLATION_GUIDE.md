# Hướng Dẫn Cài Đặt PyFlame
**PyFlame - Deep Learning Framework for Cerebras Wafer-Scale Engine**

---

## 📋 Mục Lục
1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Chuẩn Bị Môi Trường](#chuẩn-bị-môi-trường)
3. [Cài Đặt PyFlame](#cài-đặt-pyflame)
4. [Kiểm Tra Cài Đặt](#kiểm-tra-cài-đặt)
5. [Chạy Demo](#chạy-demo)
6. [Troubleshooting](#troubleshooting)

---

## 🖥️ Yêu Cầu Hệ Thống

### Tối Thiểu (CPU Reference Mode)

| Component | Requirement |
|-----------|------------|
| **OS** | Windows 10+, Linux (Ubuntu 20.04+), macOS 11+ |
| **Python** | 3.8, 3.9, 3.10, hoặc 3.11 |
| **CMake** | 3.18 trở lên |
| **Compiler** | C++17 support |
| **RAM** | 8GB+ (khuyến nghị 16GB) |
| **Disk** | 2GB free space |

### Compiler Requirements

**Windows:**
- Visual Studio 2019 hoặc 2022
- MSVC với C++17 support
- Hoặc: MinGW-w64 (GCC 8+)

**Linux:**
- GCC 8+ hoặc Clang 7+
- Build essentials: `sudo apt install build-essential`

**macOS:**
- Xcode Command Line Tools
- Clang 7+

### Optional (Full Features)

| Component | Purpose |
|-----------|---------|
| **Cerebras SDK** | Execute trên WSE hardware |
| **CS-2/CS-3 Access** | Actual Cerebras hardware |
| **Cerebras Cloud** | Cloud-based WSE access |

> **Note:** Demos có thể chạy hoàn toàn trên **CPU reference mode** - không cần hardware đắt tiền!

---

## 🔧 Chuẩn Bị Môi Trường

### Step 1: Check Python Version

```bash
python --version
```

**Expected:** Python 3.8.x, 3.9.x, 3.10.x, hoặc 3.11.x

**Nếu không đúng:**
```bash
# Windows: Download từ python.org
# Linux:
sudo apt install python3.10 python3.10-dev
# macOS:
brew install python@3.10
```

### Step 2: Check CMake

```bash
cmake --version
```

**Expected:** cmake version 3.18.0 hoặc cao hơn

**Nếu chưa có hoặc phiên bản cũ:**

**Windows:**
```bash
# Download từ https://cmake.org/download/
# Hoặc dùng chocolatey:
choco install cmake
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cmake

# Hoặc build từ source nếu cần version mới:
wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0.tar.gz
tar -xzf cmake-3.27.0.tar.gz
cd cmake-3.27.0
./bootstrap
make -j$(nproc)
sudo make install
```

**macOS:**
```bash
brew install cmake
```

### Step 3: Cài Đặt Dependencies

```bash
# Install pybind11 (REQUIRED)
pip install pybind11

# Install NumPy
pip install numpy

# Optional: Git (nếu chưa có)
# Windows: https://git-scm.com/download/win
# Linux: sudo apt install git
# macOS: brew install git
```

### Step 4: Kiểm Tra Compiler

**Windows (Visual Studio):**
```bash
# Mở "Developer Command Prompt for VS 2019/2022"
cl
```
Should show Microsoft C/C++ Compiler version

**Linux/macOS:**
```bash
gcc --version
# hoặc
clang --version
```
Should show version 8+ (GCC) hoặc 7+ (Clang)

---

## 📥 Cài Đặt PyFlame

### Method 1: Build từ Source (Recommended)

#### Step 1: Clone Repository

```bash
# Vào thư mục projects
cd "D:\Projects\do_an_tri_tue_nhan_tao"

# Clone PyFlame
git clone https://github.com/CTO92/PyFlame.git
cd PyFlame
```

#### Step 2: Create Build Directory

```bash
mkdir build
cd build
```

#### Step 3: Configure với CMake

**Windows (Visual Studio):**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Linux/macOS:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Expected output:**
```
-- The C compiler identification is MSVC 19.x.x
-- The CXX compiler identification is MSVC 19.x.x
-- Found Python: 3.10.x
-- Found pybind11: 2.x.x
-- Configuring done
-- Generating done
-- Build files have been written to: .../build
```

**Nếu gặp errors:** Xem [Troubleshooting](#troubleshooting)

#### Step 4: Build

**Windows:**
```bash
cmake --build . --config Release
```

**Linux/macOS:**
```bash
make -j$(nproc)
```

Build process có thể mất **5-15 phút** tùy máy.

**Expected output (kết thúc):**
```
[100%] Built target pyflame
```

#### Step 5: Install Python Package

```bash
# Quay lại root directory
cd ..

# Install in editable mode
pip install -e .
```

**Expected output:**
```
Successfully installed pyflame-2.0.0a2
```

---

### Method 2: Với Cerebras SDK (Advanced)

> **Note:** Chỉ áp dụng nếu có access vào Cerebras SDK

```bash
# Setup Cerebras environment
source /opt/cerebras/sdk/setup.sh

# Configure với SDK
cd PyFlame
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCEREBRAS_SDK_PATH=/opt/cerebras/sdk

# Build và install
cmake --build . --config Release
cd ..
pip install -e .
```

---

## ✅ Kiểm Tra Cài Đặt

### Test 1: Import PyFlame

```bash
python -c "import pyflame as pf; print(f'PyFlame version: {pf.__version__}')"
```

**Expected output:**
```
PyFlame version: 2.0.0-alpha
```

**Nếu lỗi `ModuleNotFoundError`:**
- Check `pip list | grep pyflame`
- Re-run: `pip install -e .` trong PyFlame directory

### Test 2: Create Tensor

```bash
python -c "import pyflame as pf; x = pf.zeros([2, 3]); print(x)"
```

**Expected output:**
```
Tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

### Test 3: Lazy Evaluation

```bash
python -c "import pyflame as pf; x = pf.ones([2, 2]); y = x + 1; result = pf.eval(y); print(result)"
```

**Expected output:**
```
[[2. 2.]
 [2. 2.]]
```

### Test 4: Neural Network

```python
# Save as test_nn.py
import pyflame as pf
import pyflame.nn as nn

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

x = pf.randn([4, 10])
output = model(x)
result = pf.eval(output)
print(f"Output shape: {result.shape}")
print("✓ Neural network test passed!")
```

Run:
```bash
python test_nn.py
```

**Expected:**
```
Output shape: (4, 2)
✓ Neural network test passed!
```

---

## 🚀 Chạy Demo

### Demo 1: Basic Operations

```bash
cd "D:\Projects\do_an_tri_tue_nhan_tao\pyflame_demos"
python 01_basic_tensor_operations.py
```

**Expected:** In ra các tensor operations, activations, reductions, etc.

### Demo 2: Neural Network Training

```bash
python 02_neural_network_training.py
```

**Expected:** Training loop với loss giảm dần

### Demo 3: Mesh Layout (WSE Architecture)

```bash
python 03_mesh_layout_demo.py
```

**Expected:** Giải thích về distributed layouts

### Demo 4: PyFlame vs PyTorch

```bash
python 04_pytorch_comparison.py
```

**Expected:** Side-by-side comparison table

---

## 🐛 Troubleshooting

### Issue 1: CMake không tìm thấy Python

**Error:**
```
CMake Error: Could not find Python
```

**Solution:**
```bash
# Specify Python explicitly
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$(which python3)

# Windows:
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="python"
```

### Issue 2: pybind11 not found

**Error:**
```
CMake Error: Could not find pybind11
```

**Solution:**
```bash
pip install pybind11

# Nếu vẫn lỗi, specify path:
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
```

### Issue 3: C++ Compiler Error

**Error:**
```
error: C++17 standard requested but CXX17 is not supported
```

**Solution:**

**Windows:** Install Visual Studio 2019+ với C++ workload

**Linux:**
```bash
sudo apt install gcc-10 g++-10
export CC=gcc-10
export CXX=g++-10
```

**macOS:**
```bash
xcode-select --install
```

### Issue 4: Build fails với "undefined reference"

**Solution:**
```bash
# Clean và rebuild
cd build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Issue 5: ImportError khi chạy demos

**Error:**
```
ImportError: DLL load failed while importing _pyflame_core
```

**Solution (Windows):**
- Cài Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Hoặc copy DLLs từ build folder

**Solution (Linux):**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/PyFlame/build
```

### Issue 6: Demos chạy nhưng warnings

**Warning:**
```
WARNING: Running on CPU reference implementation
PyFlame execution on actual WSE requires Cerebras SDK
```

**Note:** Đây là **bình thường**! CPU reference mode works perfectly cho demos.

### Issue 7: Performance rất chậm

**Lý do:** CPU reference mode không optimize như actual hardware

**Expected:**
- Operations thực thi chậm hơn PyTorch trên CPU
- Đủ để demo và học
- Production cần Cerebras hardware

---

## 🎯 Verification Checklist

Trước khi báo cáo, check list này:

- [ ] `python --version` → 3.8+
- [ ] `cmake --version` → 3.18+
- [ ] `pip list | grep pyflame` → pyflame 2.0.0a2
- [ ] `python -c "import pyflame"` → No errors
- [ ] `python 01_basic_tensor_operations.py` → Success
- [ ] `python 02_neural_network_training.py` → Success
- [ ] `python 03_mesh_layout_demo.py` → Success
- [ ] `python 04_pytorch_comparison.py` → Success

---

## 📞 Hỗ Trợ

**Nếu vẫn gặp vấn đề:**

1. Check PyFlame GitHub Issues: https://github.com/CTO92/PyFlame/issues
2. Đọc official docs (nếu có): `PyFlame/docs/`
3. Check build logs trong `build/` folder
4. Try clean rebuild:
   ```bash
   cd PyFlame
   rm -rf build
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

---

## 📚 Next Steps

Sau khi cài đặt thành công:

1. ✅ Đọc `README.md` trong `pyflame_demos/`
2. ✅ Chạy hết 4 demos
3. ✅ Đọc `PRESENTATION.md` (nếu có)
4. ✅ Chuẩn bị slides cho báo cáo
5. ✅ Thực hành giải thích concepts

**Good luck với presentation! 🚀**

---

**Author:** Installation Guide for Academic Presentation  
**Version:** 1.0  
**Last Updated:** 2024
