"""
Demo 3: Cerebras WSE Architecture - Mesh Layout
Minh họa cách PyFlame tận dụng kiến trúc 2D mesh của Cerebras WSE
"""

import numpy as np

print("=" * 70)
print("DEMO 3: CEREBRAS WSE MESH LAYOUT")
print("=" * 70)

try:
    import pyflame as pf
    
    print("\n[1] Giới thiệu Cerebras WSE")
    print("-" * 70)
    print("Cerebras Wafer-Scale Engine (WSE):")
    print("  • 850,000+ Processing Elements (PEs)")
    print("  • 2D mesh topology")
    print("  • Massive parallelism")
    print("  • On-chip communication (ultra-fast)")
    
    print("\n[2] Single PE Layout (Default)")
    print("-" * 70)
    
    # Tất cả data trên 1 PE
    a = pf.zeros([1024, 512], layout=pf.MeshLayout.single_pe())
    print(f"Tensor shape: {a.shape}")
    print(f"Layout: ALL data on single PE")
    print(f"Use case: Small tensors, debugging")
    
    print("\n[3] Row Partition Layout")
    print("-" * 70)
    
    # Phân chia theo rows
    b = pf.zeros([4096, 2048], layout=pf.MeshLayout.row_partition(16))
    print(f"Tensor shape: {b.shape}")
    print(f"Layout: Rows split across 16 PEs")
    print(f"  • Each PE gets {4096//16} rows")
    print(f"  • Good for: Row-wise operations")
    
    print("\n[4] Column Partition Layout")
    print("-" * 70)
    
    # Phân chia theo columns
    c = pf.zeros([2048, 4096], layout=pf.MeshLayout.col_partition(16))
    print(f"Tensor shape: {c.shape}")
    print(f"Layout: Columns split across 16 PEs")
    print(f"  • Each PE gets {4096//16} columns")
    print(f"  • Good for: Column-wise operations")
    
    print("\n[5] 2D Grid Layout (Most Powerful)")
    print("-" * 70)
    
    # 2D tiling
    d = pf.zeros([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
    print(f"Tensor shape: {d.shape}")
    print(f"Layout: 16x16 grid of PEs (256 PEs total)")
    print(f"  • Each PE gets {4096//16} x {4096//16} block")
    print(f"  • Block size: 256 x 256")
    print(f"  • Optimal for: Matrix multiplication")
    
    print("\n[6] Matrix Multiplication với Grid Layout")
    print("-" * 70)
    
    # Large matrix multiplication
    x = pf.zeros([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
    y = pf.zeros([4096, 4096], layout=pf.MeshLayout.grid(16, 16))
    z = x @ y
    
    print(f"Matrix A: {x.shape} on 16x16 grid")
    print(f"Matrix B: {y.shape} on 16x16 grid")
    print(f"Result C: {z.shape}")
    print(f"\nComputation distributed across 256 PEs!")
    print(f"Each PE computes local block multiplication")
    
    print("\n[7] Communication Patterns")
    print("-" * 70)
    print("WSE 2D Mesh Communication:")
    print("  • East-West: Row-wise data flow")
    print("  • North-South: Column-wise data flow")
    print("  • All-reduce: Collective operations")
    print("  • Wavelet: Efficient broadcast/reduce")
    
    print("\n[8] Performance Benefits")
    print("-" * 70)
    print("Advantages of WSE Architecture:")
    print("  ✓ No GPU memory bottleneck")
    print("  ✓ On-chip communication (20 Tb/s)")
    print("  ✓ Massive parallelism")
    print("  ✓ Linear scalability")
    print("  ✓ Perfect for large models")
    
    print("\n[9] Comparison: GPU vs WSE")
    print("-" * 70)
    print("GPU (NVIDIA A100):")
    print("  • 80 GB HBM memory")
    print("  • ~100 SM (Streaming Multiprocessors)")
    print("  • Off-chip memory access")
    print("")
    print("Cerebras WSE-3:")
    print("  • 44 GB on-wafer SRAM")
    print("  • 900,000 Processing Elements")
    print("  • On-chip everything!")
    print("  • 20 Tb/s fabric bandwidth")
    
    print("\n" + "=" * 70)
    print("✓ Demo 3 completed successfully!")
    print("=" * 70)
    print("\nNOTE: Execution on actual WSE hardware requires:")
    print("  1. Cerebras SDK (proprietary)")
    print("  2. Access to CS-2/CS-3 or Cerebras Cloud")

except ImportError as e:
    print(f"\n✗ PyFlame chưa được cài đặt!")
    print(f"Error: {e}")
    
except Exception as e:
    print(f"\n✗ Lỗi: {e}")
    print("\nNOTE: Demo này chạy trên CPU reference implementation.")
    print("Actual execution cần Cerebras hardware.")
