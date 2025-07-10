#!/usr/bin/env python3
"""
Benchmark GPU vs CPU performance for semantic search indexing.
"""

import time
import os
import sys
from pathlib import Path
import tempfile
import shutil
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator, EmbeddingConfig
from src.chunker import Chunk

def create_test_chunks(num_chunks: int = 100) -> list:
    """Create test chunks with realistic text."""
    chunks = []
    base_text = """
    This is a test chunk for benchmarking semantic search performance. 
    It contains enough text to simulate a real conversation chunk.
    The model needs to process this text and generate embeddings.
    This helps us understand the performance characteristics of GPU vs CPU.
    """
    
    for i in range(num_chunks):
        chunk = Chunk(
            id=f"test_{i}",
            text=f"{base_text} This is chunk number {i}.",
            metadata={"test": True, "index": i}
        )
        chunks.append(chunk)
    
    return chunks

def benchmark_embeddings(use_gpu: bool, chunks: list, batch_sizes: list = None) -> dict:
    """Benchmark embedding generation with GPU or CPU."""
    if batch_sizes is None:
        batch_sizes = [8, 16, 32, 64]
    
    results = {}
    
    # Test different batch sizes
    for batch_size in batch_sizes:
        print(f"\n{'GPU' if use_gpu else 'CPU'} - Batch size: {batch_size}")
        
        # Create fresh embedding generator for each test
        config = EmbeddingConfig(
            model_name="all-mpnet-base-v2",
            batch_size=batch_size,
            use_gpu=use_gpu,
            auto_batch_size=False,  # We want to control batch size manually
            show_progress=True
        )
        
        embedder = EmbeddingGenerator(config)
        
        # Load model (includes loading time)
        model_load_start = time.time()
        embedder.load_model()
        model_load_time = time.time() - model_load_start
        print(f"Model load time: {model_load_time:.2f}s")
        
        # Get model info
        model_info = embedder.get_model_info()
        print(f"Device: {model_info.get('device', 'unknown')}")
        
        # Warmup run (exclude from timing)
        print("Warming up...")
        warmup_chunks = chunks[:min(10, len(chunks))]
        embedder.generate_embeddings(warmup_chunks)
        
        # Actual benchmark
        print(f"Benchmarking {len(chunks)} chunks...")
        start_time = time.time()
        embeddings = embedder.generate_embeddings(chunks)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(chunks) / processing_time
        
        results[f"batch_{batch_size}"] = {
            "model_load_time": model_load_time,
            "processing_time": processing_time,
            "throughput": throughput,
            "device": model_info.get('device', 'unknown'),
            "gpu_info": model_info.get('gpu_info', {})
        }
        
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Throughput: {throughput:.1f} chunks/s")
        
        # Clean up to free memory
        del embedder
    
    return results

def main():
    """Run the benchmark."""
    print("🚀 Semantic Search GPU vs CPU Benchmark")
    print("=" * 50)
    
    # Test with different chunk counts
    chunk_counts = [100, 500, 1000]
    
    for num_chunks in chunk_counts:
        print(f"\n\n📊 Testing with {num_chunks} chunks")
        print("-" * 50)
        
        chunks = create_test_chunks(num_chunks)
        
        # Benchmark CPU
        print("\n🖥️  CPU Benchmark")
        cpu_results = benchmark_embeddings(use_gpu=False, chunks=chunks)
        
        # Benchmark GPU
        print("\n🎮 GPU Benchmark")
        gpu_results = benchmark_embeddings(use_gpu=True, chunks=chunks)
        
        # Compare results
        print(f"\n\n📈 Results Summary for {num_chunks} chunks:")
        print("-" * 50)
        
        for batch_size in [8, 16, 32, 64]:
            key = f"batch_{batch_size}"
            if key in cpu_results and key in gpu_results:
                cpu_time = cpu_results[key]["processing_time"]
                gpu_time = gpu_results[key]["processing_time"]
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                print(f"\nBatch size {batch_size}:")
                print(f"  CPU: {cpu_time:.2f}s ({cpu_results[key]['throughput']:.1f} chunks/s)")
                print(f"  GPU: {gpu_time:.2f}s ({gpu_results[key]['throughput']:.1f} chunks/s)")
                print(f"  Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()