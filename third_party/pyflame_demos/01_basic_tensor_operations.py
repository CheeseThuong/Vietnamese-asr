"""
Demo 1: Basic Tensor Operations với PyFlame
Minh họa các phép toán tensor cơ bản
"""

import numpy as np

print("=" * 70)
print("DEMO 1: BASIC TENSOR OPERATIONS")
print("=" * 70)

try:
    import pyflame as pf
    
    print("\n[1] Tạo Tensors")
    print("-" * 70)
    
    # Tạo tensor từ zeros, ones
    a = pf.zeros([3, 4])
    print(f"Zeros tensor (3x4): {a.shape}")
    
    b = pf.ones([3, 4])
    print(f"Ones tensor (3x4): {b.shape}")
    
    # Tạo tensor random
    c = pf.randn([3, 4])
    print(f"Random normal tensor (3x4): {c.shape}")
    
    # Tạo từ NumPy array
    np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    d = pf.from_numpy(np_array)
    print(f"From NumPy (2x3): {d.shape}")
    
    print("\n[2] Phép toán số học")
    print("-" * 70)
    
    # Các phép toán cơ bản
    x = pf.randn([2, 3])
    y = pf.randn([2, 3])
    
    # Cộng, trừ, nhân, chia
    add_result = x + y
    sub_result = x - y
    mul_result = x * y
    div_result = x / y
    
    print(f"Addition: {add_result.shape}")
    print(f"Subtraction: {sub_result.shape}")
    print(f"Multiplication: {mul_result.shape}")
    print(f"Division: {div_result.shape}")
    
    print("\n[3] Ma trận multiplication")
    print("-" * 70)
    
    # Matrix multiplication
    m1 = pf.randn([4, 3])
    m2 = pf.randn([3, 5])
    result = m1 @ m2
    
    print(f"Matrix A: {m1.shape}")
    print(f"Matrix B: {m2.shape}")
    print(f"A @ B: {result.shape}")
    
    print("\n[4] Activation Functions")
    print("-" * 70)
    
    t = pf.randn([100])
    
    relu_out = pf.relu(t)
    sigmoid_out = pf.sigmoid(t)
    tanh_out = pf.tanh(t)
    
    print(f"ReLU: {relu_out.shape}")
    print(f"Sigmoid: {sigmoid_out.shape}")
    print(f"Tanh: {tanh_out.shape}")
    
    print("\n[5] Reduction Operations")
    print("-" * 70)
    
    tensor = pf.randn([10, 20])
    
    sum_result = tensor.sum()
    mean_result = tensor.mean()
    max_result = tensor.max()
    min_result = tensor.min()
    
    print(f"Original tensor: {tensor.shape}")
    print(f"Sum: scalar")
    print(f"Mean: scalar")
    print(f"Max: scalar")
    print(f"Min: scalar")
    
    print("\n[6] Reshape và Transpose")
    print("-" * 70)
    
    original = pf.randn([6, 8])
    reshaped = original.reshape([4, 12])
    transposed = original.transpose()
    
    print(f"Original: {original.shape}")
    print(f"Reshaped to (4, 12): {reshaped.shape}")
    print(f"Transposed: {transposed.shape}")
    
    print("\n[7] Lazy Evaluation Demo")
    print("-" * 70)
    print("PyFlame uses LAZY EVALUATION:")
    print("- Operations build a computation graph")
    print("- Actual computation happens only when pf.eval() is called")
    
    # Build graph without execution
    lazy_a = pf.randn([100, 100])
    lazy_b = pf.randn([100, 100])
    lazy_c = lazy_a @ lazy_b
    lazy_d = pf.relu(lazy_c)
    lazy_e = lazy_d.sum()
    
    print("\nComputation graph built (not executed yet)")
    print("Calling pf.eval() to execute...")
    
    # Execute the graph
    result = pf.eval(lazy_e)
    print(f"Result computed: {result}")
    
    # Convert to NumPy for inspection
    np_result = result.numpy()
    print(f"NumPy value: {np_result}")
    
    print("\n" + "=" * 70)
    print("✓ Demo 1 completed successfully!")
    print("=" * 70)

except ImportError as e:
    print(f"\n✗ PyFlame chưa được cài đặt!")
    print(f"Error: {e}")
    print("\nHướng dẫn cài đặt:")
    print("1. cd PyFlame")
    print("2. mkdir build && cd build")
    print("3. cmake .. -DCMAKE_BUILD_TYPE=Release")
    print("4. cmake --build .")
    print("5. pip install -e .")
    
except Exception as e:
    print(f"\n✗ Lỗi: {e}")
    print("\nNOTE: Một số features có thể cần Cerebras hardware.")
    print("Demo này chạy trên CPU reference implementation.")
