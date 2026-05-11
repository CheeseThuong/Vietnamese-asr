"""
Demo 4: PyFlame vs PyTorch - Side-by-Side Comparison
So sánh code giữa PyFlame và PyTorch để thấy sự tương đồng
"""

print("=" * 70)
print("DEMO 4: PYFLAME VS PYTORCH COMPARISON")
print("=" * 70)

print("\n[1] TENSOR CREATION")
print("-" * 70)

print("\nPyTorch Code:")
print("""
import torch
x = torch.zeros([3, 4])
y = torch.ones([3, 4])
z = torch.randn([3, 4])
""")

print("\nPyFlame Code:")
print("""
import pyflame as pf
x = pf.zeros([3, 4])
y = pf.ones([3, 4])
z = pf.randn([3, 4])
""")

print("\n✓ Syntax GIỐNG NHAU!")

print("\n" + "=" * 70)
print("\n[2] NEURAL NETWORK DEFINITION")
print("-" * 70)

print("\nPyTorch Code:")
print("""
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
""")

print("\nPyFlame Code:")
print("""
import pyflame.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
""")

print("\n✓ API GIỐNG HỆT!")

print("\n" + "=" * 70)
print("\n[3] TRAINING LOOP")
print("-" * 70)

print("\nPyTorch Code:")
print("""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
""")

print("\nPyFlame Code:")
print("""
optimizer = pf.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
""")

print("\n✓ WORKFLOW GIỐNG NHAU!")

print("\n" + "=" * 70)
print("\n[4] KEY DIFFERENCES - EXECUTION MODEL")
print("-" * 70)

print("\nPyTorch: EAGER Execution")
print("  • Operations execute immediately")
print("  • Easy debugging")
print("  • Results available instantly")
print("""
  x = torch.randn([3, 4])
  y = x + 1  # ← Executes NOW
  print(y)   # ← Can print immediately
""")

print("\nPyFlame: LAZY Evaluation")
print("  • Builds computation graph first")
print("  • Optimizes before execution")
print("  • Need pf.eval() to execute")
print("""
  x = pf.randn([3, 4])
  y = x + 1        # ← Builds graph only
  result = pf.eval(y)  # ← Executes NOW
  print(result)    # ← Print actual result
""")

print("\n" + "=" * 70)
print("\n[5] KEY DIFFERENCES - TARGET HARDWARE")
print("-" * 70)

print("\nPyTorch:")
print("  • CPU, CUDA (GPU), ROCm (AMD), MPS (Apple)")
print("  • Multi-GPU via DDP, FSDP")
print("  • Mature ecosystem")

print("\nPyFlame:")
print("  • Cerebras Wafer-Scale Engine (WSE)")
print("  • 850,000+ Processing Elements")
print("  • 2D mesh topology")
print("  • Automatic CSL code generation")
print("  • CPU reference mode (for development)")

print("\n" + "=" * 70)
print("\n[6] KEY DIFFERENCES - SCALABILITY")
print("-" * 70)

print("\nPyTorch GPU Limitations:")
print("  • Memory: 80GB (A100) ← Model must fit")
print("  • Multi-GPU needs sharding")
print("  • Communication overhead")

print("\nPyFlame WSE Advantages:")
print("  • Memory: 44GB on-wafer SRAM")
print("  • Compute: 900,000 cores")
print("  • Bandwidth: 20 Tb/s fabric")
print("  • No PCIe bottleneck!")
print("  • Linear scaling")

print("\n" + "=" * 70)
print("\n[7] WHEN TO USE EACH")
print("-" * 70)

print("\nUse PyTorch when:")
print("  ✓ General-purpose deep learning")
print("  ✓ Research & prototyping")
print("  ✓ Have GPU infrastructure")
print("  ✓ Need mature ecosystem")
print("  ✓ Want eager execution")

print("\nUse PyFlame when:")
print("  ✓ Have Cerebras hardware access")
print("  ✓ Training massive models (LLMs)")
print("  ✓ Need extreme scalability")
print("  ✓ Want optimal WSE utilization")
print("  ✓ Production large-scale training")

print("\n" + "=" * 70)
print("\n[8] PRACTICAL EXAMPLE - MNIST TRAINING")
print("-" * 70)

try:
    import pyflame as pf
    import pyflame.nn as nn
    import numpy as np
    
    print("\nDefining model in PyFlame...")
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    optimizer = pf.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Dummy data
    x = pf.randn([32, 784])
    y = pf.from_numpy(np.random.randint(0, 10, [32]))
    
    print("\nTraining for 3 steps...")
    for step in range(3):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        loss_val = pf.eval(loss).numpy()
        print(f"  Step {step+1}: Loss = {loss_val:.4f}")
    
    print("\n✓ Training successful on PyFlame!")

except ImportError:
    print("\nNOTE: PyFlame not installed - showing example code only")
    print("      Install PyFlame to run actual training")

except Exception as e:
    print(f"\nNOTE: {e}")
    print("      Running on CPU reference mode")

print("\n" + "=" * 70)
print("\n[9] SUMMARY TABLE")
print("-" * 70)

print("""
╔═══════════════════╦══════════════════════╦═════════════════════════╗
║ Feature           ║ PyTorch              ║ PyFlame                 ║
╠═══════════════════╬══════════════════════╬═════════════════════════╣
║ API Style         ║ Pythonic, intuitive  ║ PyTorch-like            ║
║ Execution         ║ Eager                ║ Lazy (graph-based)      ║
║ Hardware          ║ CPU, CUDA, ROCm      ║ Cerebras WSE, CPU ref   ║
║ Scalability       ║ Multi-GPU (complex)  ║ Native massive parallel ║
║ Memory            ║ GPU RAM limited      ║ 44GB on-wafer SRAM      ║
║ Best For          ║ Research, general ML ║ Large-scale training    ║
║ Ecosystem         ║ Mature, huge         ║ Emerging, specialized   ║
║ Learning Curve    ║ Easy                 ║ Easy (if know PyTorch)  ║
╚═══════════════════╩══════════════════════╩═════════════════════════╝
""")

print("\n" + "=" * 70)
print("✓ Demo 4 completed!")
print("=" * 70)
print("\nConclusion:")
print("  • PyFlame API rất giống PyTorch")
print("  • Dễ migration từ PyTorch sang PyFlame")
print("  • Khác biệt chính: execution model và target hardware")
print("  • PyFlame tối ưu cho Cerebras WSE architecture")
