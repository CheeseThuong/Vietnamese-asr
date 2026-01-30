"""
CPU-optimized profiling tools với Flame Graph support
Hỗ trợ Windows và không cần GPU
"""
import cProfile
import pstats
import io
from pathlib import Path
import time
import psutil
import os
from functools import wraps

class CPUProfiler:
    """
    CPU Profiler sử dụng cProfile (built-in Python)
    """
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = None
    
    def start(self):
        """Bắt đầu profiling"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        print("⏱ CPU Profiling started...")
    
    def stop(self, output_file=None):
        """Dừng profiling và lưu kết quả"""
        if self.profiler is None:
            print("⚠ Profiler not started")
            return
        
        self.profiler.disable()
        
        # Generate report
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        report = s.getvalue()
        
        # Save to file
        if output_file is None:
            output_file = self.output_dir / f"profile_{int(time.time())}.txt"
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"✓ Profiling report saved to: {output_file}")
        
        # Try to generate flamegraph if flameprof is available
        try:
            self._generate_flamegraph(output_file)
        except Exception as e:
            print(f"⚠ Flamegraph generation skipped: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("TOP 10 SLOWEST FUNCTIONS")
        print("="*60)
        ps_summary = pstats.Stats(self.profiler)
        ps_summary.strip_dirs()
        ps_summary.sort_stats('cumulative')
        ps_summary.print_stats(10)
        print("="*60)
        
        return report
    
    def _generate_flamegraph(self, profile_file):
        """
        Tạo flamegraph từ profiling data (nếu có flameprof)
        
        Install: pip install flameprof
        """
        try:
            import subprocess
            
            # Save profiler stats
            stats_file = self.output_dir / "profile.stats"
            ps = pstats.Stats(self.profiler)
            ps.dump_stats(str(stats_file))
            
            # Generate flamegraph
            svg_file = self.output_dir / "flamegraph.svg"
            
            # Sử dụng flameprof
            cmd = f"flameprof {stats_file} > {svg_file}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            
            print(f"🔥 Flamegraph generated: {svg_file}")
            print(f"   Open in browser to visualize")
            
        except ImportError:
            print("💡 Tip: Install flameprof for flamegraph visualization:")
            print("   pip install flameprof")
        except Exception as e:
            # Silent fail - flamegraph is optional
            pass

def profile_function(func):
    """
    Decorator để profile một function cụ thể
    
    Usage:
        @profile_function
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        profiler.disable()
        
        print(f"\n{'='*60}")
        print(f"Function: {func.__name__}")
        print(f"Time: {elapsed:.4f}s")
        print(f"{'='*60}")
        
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())
        
        return result
    
    return wrapper

class MemoryProfiler:
    """
    Memory profiler để theo dõi RAM usage
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = 0
    
    def start(self):
        """Bắt đầu theo dõi memory"""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 Memory tracking started: {self.start_memory:.2f} MB")
    
    def checkpoint(self, label=""):
        """Checkpoint memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        delta = current_memory - self.start_memory
        print(f"📊 Memory {label}: {current_memory:.2f} MB (Δ {delta:+.2f} MB)")
        return current_memory
    
    def stop(self):
        """Dừng và hiển thị memory usage"""
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        delta = final_memory - self.start_memory
        
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        print(f"Start:  {self.start_memory:.2f} MB")
        print(f"Final:  {final_memory:.2f} MB")
        print(f"Delta:  {delta:+.2f} MB")
        print(f"Peak:   {self.process.memory_info().peak_wset / 1024 / 1024:.2f} MB")
        print("="*60)
        
        return final_memory

def measure_inference_speed(model, processor, audio_path, num_runs=10, warmup=3):
    """
    Đo tốc độ inference chi tiết
    
    Args:
        model: ASR model
        processor: Processor
        audio_path: Path to test audio file
        num_runs: Số lần chạy để benchmark
        warmup: Số lần warm-up
    """
    import torch
    import torchaudio
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"INFERENCE SPEED BENCHMARK")
    print(f"{'='*60}")
    print(f"Audio: {audio_path}")
    print(f"Runs: {num_runs} (after {warmup} warmup)")
    
    # Load audio
    audio_array, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_array = resampler(audio_array)
    audio_array = audio_array.squeeze().numpy()
    audio_duration = len(audio_array) / 16000
    
    print(f"Audio duration: {audio_duration:.2f}s")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    print("\nWarming up...")
    for _ in range(warmup):
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            _ = model(input_values)
    
    # Benchmark
    print(f"Benchmarking {num_runs} runs...")
    timings = {
        'preprocessing': [],
        'inference': [],
        'decoding': [],
        'total': []
    }
    
    for _ in range(num_runs):
        # Preprocessing
        t0 = time.time()
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        t1 = time.time()
        timings['preprocessing'].append(t1 - t0)
        
        # Inference
        with torch.no_grad():
            logits = model(input_values).logits
        t2 = time.time()
        timings['inference'].append(t2 - t1)
        
        # Decoding
        pred_ids = torch.argmax(logits, dim=-1)
        _ = processor.batch_decode(pred_ids)
        t3 = time.time()
        timings['decoding'].append(t3 - t2)
        
        timings['total'].append(t3 - t0)
    
    # Statistics
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for stage, times in timings.items():
        times_np = np.array(times)
        print(f"\n{stage.upper()}:")
        print(f"  Mean:   {times_np.mean()*1000:.2f} ms")
        print(f"  Std:    {times_np.std()*1000:.2f} ms")
        print(f"  Min:    {times_np.min()*1000:.2f} ms")
        print(f"  Max:    {times_np.max()*1000:.2f} ms")
        print(f"  Median: {np.median(times_np)*1000:.2f} ms")
    
    # Real-time factor
    rtf = np.mean(timings['total']) / audio_duration
    print(f"\nReal-Time Factor (RTF): {rtf:.3f}")
    print(f"  (Lower is better, <1.0 means faster than real-time)")
    
    print("="*60)
    
    return timings

def main():
    """
    Demo usage of profiling tools
    """
    print("="*60)
    print("CPU PROFILING TOOLS DEMO")
    print("="*60)
    
    # Demo CPU profiler
    profiler = CPUProfiler()
    profiler.start()
    
    # Simulate some work
    import time
    import numpy as np
    
    def simulate_work():
        data = np.random.randn(1000, 1000)
        result = np.dot(data, data.T)
        time.sleep(0.1)
        return result
    
    for i in range(5):
        simulate_work()
    
    profiler.stop()
    
    # Demo memory profiler
    print("\n" + "="*60)
    mem_profiler = MemoryProfiler()
    mem_profiler.start()
    
    # Allocate some memory
    large_data = [np.random.randn(100, 100) for _ in range(10)]
    mem_profiler.checkpoint("after allocation")
    
    del large_data
    mem_profiler.checkpoint("after deletion")
    
    mem_profiler.stop()
    
    print("\n✓ Profiling demo complete!")
    print("\n" + "="*60)
    print("AVAILABLE PROFILING TOOLS")
    print("="*60)
    print("\n1. Built-in cProfile (included)")
    print("   - Text-based profiling")
    print("   - Fast and reliable")
    
    print("\n2. Flamegraph visualization (optional)")
    print("   Install: pip install flameprof")
    print("   - Visual flamegraph (SVG)")
    print("   - Interactive analysis")
    
    print("\n3. py-spy (recommended for production)")
    print("   Install: pip install py-spy")
    print("   Usage: py-spy record -o profile.svg -- python your_script.py")
    print("   - Low overhead")
    print("   - Sampling profiler")
    print("   - No code changes needed")
    
    print("\n4. line_profiler (line-by-line)")
    print("   Install: pip install line_profiler")
    print("   - Detailed line analysis")
    print("   - Use @profile decorator")
    
    print("\n" + "="*60)
    print("\nUsage in your code:")
    print("  from profiling_cpu import CPUProfiler, MemoryProfiler, profile_function")
    print("  profiler = CPUProfiler()")
    print("  profiler.start()")
    print("  # your code here")
    print("  profiler.stop()")
    print("="*60)

if __name__ == "__main__":
    main()
