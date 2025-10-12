#!/usr/bin/env python3
"""
Benchmark script to test different GPU optimization strategies for embeddings.
"""

import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Test data - representative sentences
test_texts = [
    "This is a test sentence for embedding generation performance.",
    "Machine learning models require optimization for best performance.",
    "GPU acceleration can significantly improve throughput for neural networks.",
    "PyTorch supports various optimization techniques including mixed precision.",
    "The AMD Radeon 8060S provides 128GB of unified memory for ML workloads.",
] * 50  # 250 sentences

def benchmark_baseline(model, texts, batch_size=256):
    """Baseline: FP32, no optimizations"""
    print(f"\n{'='*60}")
    print(f"BASELINE: FP32, batch_size={batch_size}")
    print(f"{'='*60}")

    # Warmup
    _ = model.encode(texts[:5], batch_size=batch_size, show_progress_bar=False)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    start = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    throughput = len(texts) / elapsed
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} sentences/s")
    print(f"Embedding shape: {embeddings.shape}")

    return elapsed, throughput


def benchmark_fp16(model, texts, batch_size=256):
    """Test with FP16 mixed precision"""
    print(f"\n{'='*60}")
    print(f"FP16 MIXED PRECISION, batch_size={batch_size}")
    print(f"{'='*60}")

    try:
        # Enable FP16
        if hasattr(model, '_first_module'):
            model._first_module().auto_model.half()

        # Warmup
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _ = model.encode(texts[:5], batch_size=batch_size, show_progress_bar=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Benchmark
        start = time.time()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        throughput = len(texts) / elapsed
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.1f} sentences/s")
        print(f"Speedup: {throughput / (len(texts) / elapsed):.2f}x")

        # Restore FP32
        if hasattr(model, '_first_module'):
            model._first_module().auto_model.float()

        return elapsed, throughput
    except Exception as e:
        print(f"FP16 failed: {e}")
        return None, None


def benchmark_bfloat16(model, texts, batch_size=256):
    """Test with BF16 mixed precision"""
    print(f"\n{'='*60}")
    print(f"BFLOAT16 MIXED PRECISION, batch_size={batch_size}")
    print(f"{'='*60}")

    try:
        # Warmup
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model.encode(texts[:5], batch_size=batch_size, show_progress_bar=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Benchmark
        start = time.time()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        throughput = len(texts) / elapsed
        print(f"Time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.1f} sentences/s")

        return elapsed, throughput
    except Exception as e:
        print(f"BF16 failed: {e}")
        return None, None


def benchmark_batch_sizes(model, texts):
    """Test different batch sizes"""
    print(f"\n{'='*60}")
    print(f"BATCH SIZE EXPERIMENTS")
    print(f"{'='*60}")

    batch_sizes = [64, 128, 256, 512]
    results = {}

    for bs in batch_sizes:
        try:
            # Warmup
            _ = model.encode(texts[:5], batch_size=bs, show_progress_bar=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # Benchmark
            start = time.time()
            _ = model.encode(texts, batch_size=bs, show_progress_bar=False, convert_to_numpy=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start

            throughput = len(texts) / elapsed
            results[bs] = (elapsed, throughput)
            print(f"Batch {bs:3d}: {elapsed:.2f}s ({throughput:.1f} sent/s)")
        except Exception as e:
            print(f"Batch {bs} failed: {e}")
            results[bs] = (None, None)

    return results


def main():
    print("="*60)
    print("GPU EMBEDDING OPTIMIZATION BENCHMARK")
    print("="*60)

    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"ROCm Version: {torch.version.hip}")

    print(f"\nTest dataset: {len(test_texts)} sentences")
    print(f"Avg length: {np.mean([len(t) for t in test_texts]):.0f} chars")

    # Load model
    print(f"\nLoading model: all-mpnet-base-v2...")
    model = SentenceTransformer('all-mpnet-base-v2')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model device: {model.device}")

    # Run benchmarks
    baseline_time, baseline_throughput = benchmark_baseline(model, test_texts, batch_size=256)

    fp16_time, fp16_throughput = benchmark_fp16(model, test_texts, batch_size=256)

    bf16_time, bf16_throughput = benchmark_bfloat16(model, test_texts, batch_size=256)

    batch_results = benchmark_batch_sizes(model, test_texts)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline (FP32):     {baseline_throughput:.1f} sent/s")
    if fp16_throughput:
        speedup = fp16_throughput / baseline_throughput
        print(f"FP16:                {fp16_throughput:.1f} sent/s ({speedup:.2f}x)")
    if bf16_throughput:
        speedup = bf16_throughput / baseline_throughput
        print(f"BF16:                {bf16_throughput:.1f} sent/s ({speedup:.2f}x)")

    print(f"\nBest batch size:")
    best_bs = max(batch_results.items(), key=lambda x: x[1][1] if x[1][1] else 0)
    print(f"  Batch {best_bs[0]}: {best_bs[1][1]:.1f} sent/s")


if __name__ == "__main__":
    main()
