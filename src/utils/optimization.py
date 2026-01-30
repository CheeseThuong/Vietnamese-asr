"""
Performance optimization và profiling với PyFlame và các công cụ khác
"""
import time
import torch
import functools
from pathlib import Path
from typing import Callable, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """
    Performance profiler để theo dõi thời gian thực thi
    """
    def __init__(self):
        self.timings = {}
    
    def profile(self, name: str):
        """
        Decorator để profile function execution time
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(elapsed)
                
                logger.info(f"{name}: {elapsed:.4f}s")
                return result
            return wrapper
        return decorator
    
    def get_stats(self, name: str = None):
        """
        Lấy thống kê timing
        """
        if name:
            timings = self.timings.get(name, [])
            if not timings:
                return None
            return {
                'mean': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'count': len(timings)
            }
        else:
            return {
                name: self.get_stats(name)
                for name in self.timings.keys()
            }
    
    def print_stats(self):
        """
        In ra thống kê performance
        """
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS")
        print("="*60)
        
        for name, stats in self.get_stats().items():
            if stats:
                print(f"\n{name}:")
                print(f"  Mean: {stats['mean']:.4f}s")
                print(f"  Min:  {stats['min']:.4f}s")
                print(f"  Max:  {stats['max']:.4f}s")
                print(f"  Count: {stats['count']}")

# Global profiler instance
profiler = PerformanceProfiler()

def optimize_model_for_inference(model, use_quantization: bool = True, device: str = "cpu"):
    """
    Tối ưu hóa model cho inference (CPU hoặc GPU)
    
    Args:
        model: PyTorch model
        use_quantization: Có sử dụng quantization không
        device: "cpu" hoặc "cuda"
    """
    logger.info(f"Optimizing model for inference ({device})...")
    
    # Set to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply quantization if requested
    if use_quantization:
        try:
            if device == "cpu":
                # CPU: Dynamic quantization (rất hiệu quả!)
                logger.info("Applying CPU-optimized quantization...")
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("✓ Applied int8 dynamic quantization (CPU)")
                logger.info("  - Model size: ~75% reduction")
                logger.info("  - Speed: ~2-3x faster on CPU")
            else:
                # GPU: Static quantization hoặc bitsandbytes
                logger.info("Applying GPU-optimized quantization...")
                # Keep FP16 for GPU
                model = model.half()
                logger.info("✓ Applied FP16 quantization (GPU)")
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
    
    # Use torch.compile if available (PyTorch 2.0+)
    # Note: Chỉ hiệu quả với PyTorch 2.0+
    try:
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            logger.info("Applying torch.compile optimization...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("✓ Applied torch.compile optimization")
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}")
    
    return model

def export_to_onnx(
    model,
    processor,
    output_path: str,
    opset_version: int = 14
):
    """
    Export model sang ONNX format để tối ưu hóa inference
    
    Args:
        model: Wav2Vec2 model
        processor: Wav2Vec2Processor
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    import torch.onnx
    
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 16000 * 5)  # 5 seconds of audio
    input_values = processor(
        dummy_input.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values
    
    # Export
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_values,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_values'],
            output_names=['logits'],
            dynamic_axes={
                'input_values': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )
    
    logger.info(f"✓ Model exported to: {output_path}")

class ONNXInferenceEngine:
    """
    ONNX inference engine để tăng tốc độ inference
    """
    def __init__(self, onnx_model_path: str, processor):
        import onnxruntime as ort
        
        self.processor = processor
        
        # Create ONNX runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info(f"✓ ONNX model loaded with providers: {self.session.get_providers()}")
    
    @profiler.profile("onnx_inference")
    def transcribe(self, audio_array, sampling_rate: int = 16000):
        """
        Transcribe audio using ONNX model
        """
        # Process audio
        input_values = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_values.numpy()
        
        # Run inference
        outputs = self.session.run(
            None,
            {'input_values': input_values}
        )
        
        logits = torch.from_numpy(outputs[0])
        
        # Decode
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(pred_ids)[0]
        
        return transcription

def profile_with_pyflame(func: Callable, *args, **kwargs):
    """
    Profile function với PyFlame
    
    Note: PyFlame yêu cầu cài đặt riêng và chỉ hoạt động trên Linux
    """
    try:
        import py_flame_graph
        
        # Profile function
        result = py_flame_graph.profile(func, *args, **kwargs)
        
        return result
        
    except ImportError:
        logger.warning("py-flamegraph not available, skipping flame graph profiling")
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Flame graph profiling failed: {e}")
        return func(*args, **kwargs)

def benchmark_inference(model, processor, test_audio_paths: list, num_runs: int = 10):
    """
    Benchmark inference performance
    
    Args:
        model: ASR model
        processor: Processor
        test_audio_paths: List of test audio file paths
        num_runs: Number of benchmark runs
    """
    import torchaudio
    
    logger.info(f"Running benchmark with {len(test_audio_paths)} audio files, {num_runs} runs each...")
    
    device = next(model.parameters()).device
    model.eval()
    
    timings = []
    
    for audio_path in test_audio_paths:
        # Load audio
        audio_array, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_array = resampler(audio_array)
        audio_array = audio_array.squeeze().numpy()
        
        # Warm up
        input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            _ = model(input_values)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.time()
            
            input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            _ = processor.batch_decode(pred_ids)
            
            elapsed = time.time() - start
            timings.append(elapsed)
    
    # Statistics
    import numpy as np
    timings = np.array(timings)
    
    print("\n" + "="*60)
    print("INFERENCE BENCHMARK RESULTS")
    print("="*60)
    print(f"Total runs: {len(timings)}")
    print(f"Mean: {timings.mean():.4f}s")
    print(f"Std:  {timings.std():.4f}s")
    print(f"Min:  {timings.min():.4f}s")
    print(f"Max:  {timings.max():.4f}s")
    print(f"P50:  {np.percentile(timings, 50):.4f}s")
    print(f"P95:  {np.percentile(timings, 95):.4f}s")
    print(f"P99:  {np.percentile(timings, 99):.4f}s")
    print("="*60)
    
    return timings

def optimize_batch_inference(model, processor, audio_arrays: list, batch_size: int = 8):
    """
    Tối ưu hóa batch inference
    """
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    
    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i:i+batch_size]
        
        # Process batch
        inputs = processor(
            batch,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
        
        # Inference
        with torch.no_grad():
            if attention_mask is not None:
                logits = model(input_values, attention_mask=attention_mask).logits
            else:
                logits = model(input_values).logits
        
        # Decode
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = processor.batch_decode(pred_ids)
        
        results.extend(transcriptions)
    
    return results

def main():
    """
    Main function để test optimization
    """
    base_dir = Path(__file__).parent
    model_dir = base_dir / 'models' / 'wav2vec2-vietnamese-asr' / 'final_model'
    
    if not model_dir.exists():
        logger.error(f"Model not found at: {model_dir}")
        return
    
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    
    # Optimize model
    logger.info("\nOptimizing model...")
    model = optimize_model_for_inference(model, use_quantization=True, device=device)
    
    # Export to ONNX (optional)
    onnx_path = base_dir / 'models' / 'wav2vec2-vietnamese-asr' / 'model.onnx'
    logger.info("\nExporting to ONNX...")
    try:
        # Move model to CPU for ONNX export
        model_cpu = model.cpu() if device == "cuda" else model
        export_to_onnx(model_cpu, processor, str(onnx_path))
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
    
    # Print profiler stats
    profiler.print_stats()
    
    logger.info("\n✓ Optimization complete!")
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*60)
    
    if device == "cpu":
        logger.info("\n🖥️ CPU OPTIMIZATIONS APPLIED:")
        logger.info("  ✓ Dynamic int8 quantization (~75% size reduction)")
        logger.info("  ✓ Gradient disabled (faster inference)")
        logger.info("  ✓ Eval mode enabled")
        logger.info("\n💡 CPU Performance Tips:")
        logger.info("  1. Use quantized model for ~2-3x speedup")
        logger.info("  2. Use ONNX Runtime for additional speedup")
        logger.info("  3. Batch processing when possible")
        logger.info("  4. Consider Intel MKL optimization")
    else:
        logger.info("\n🚀 GPU OPTIMIZATIONS APPLIED:")
        logger.info("  ✓ FP16 precision (faster on modern GPUs)")
        logger.info("  ✓ Gradient disabled")
        logger.info("  ✓ Eval mode enabled")
        logger.info("\n💡 GPU Performance Tips:")
        logger.info("  1. Use larger batch sizes")
        logger.info("  2. Enable mixed precision training")
        logger.info("  3. Use torch.compile() for ~20-30% speedup")
    
    logger.info("\n" + "="*60)

if __name__ == "__main__":
    main()
