#!/usr/bin/env python3
"""
Debug GPU performance issues - measure actual timings.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingGenerator, EmbeddingConfig
from src.chunker import Chunk
from src.gpu_utils import assess_gpu_capability, log_gpu_status
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_chunks(num_chunks: int = 100) -> list:
    """Create test chunks."""
    chunks = []
    base_text = "This is a test chunk " * 50  # ~250 chars
    
    for i in range(num_chunks):
        chunk = Chunk(
            id=f"test_{i}",
            text=f"{base_text} Number {i}.",
            metadata={"test": True}
        )
        chunks.append(chunk)
    
    return chunks

def measure_gpu_overhead():
    """Measure specific GPU operations to find bottlenecks."""
    print("\n🔍 Debugging GPU Performance Issues")
    print("=" * 60)
    
    # 1. Check GPU capability
    print("\n1. GPU Capability Assessment:")
    gpu_cap = assess_gpu_capability()
    log_gpu_status(gpu_cap)
    
    # 2. Measure model loading time
    print("\n2. Model Loading Time:")
    
    # CPU model loading
    print("\n   CPU Model Loading:")
    cpu_config = EmbeddingConfig(use_gpu=False, show_progress=False)
    cpu_embedder = EmbeddingGenerator(cpu_config)
    
    cpu_load_start = time.time()
    cpu_embedder.load_model()
    cpu_load_time = time.time() - cpu_load_start
    print(f"   CPU load time: {cpu_load_time:.2f}s")
    
    # GPU model loading
    print("\n   GPU Model Loading:")
    gpu_config = EmbeddingConfig(use_gpu=True, show_progress=False)
    gpu_embedder = EmbeddingGenerator(gpu_config)
    
    gpu_load_start = time.time()
    gpu_embedder.load_model()
    gpu_load_time = time.time() - gpu_load_start
    print(f"   GPU load time: {gpu_load_time:.2f}s")
    print(f"   GPU overhead: {gpu_load_time - cpu_load_time:.2f}s")
    
    # 3. Measure single embedding generation
    print("\n3. Single Embedding Generation:")
    test_text = "This is a test sentence for embedding generation."
    
    # CPU single embedding
    cpu_single_start = time.time()
    cpu_embedder.generate_single_embedding(test_text)
    cpu_single_time = time.time() - cpu_single_start
    print(f"   CPU: {cpu_single_time*1000:.1f}ms")
    
    # GPU single embedding
    gpu_single_start = time.time()
    gpu_embedder.generate_single_embedding(test_text)
    gpu_single_time = time.time() - gpu_single_start
    print(f"   GPU: {gpu_single_time*1000:.1f}ms")
    print(f"   GPU overhead: {(gpu_single_time - cpu_single_time)*1000:.1f}ms")
    
    # 4. Measure batch processing with different sizes
    print("\n4. Batch Processing Performance:")
    batch_sizes = [1, 10, 50, 100, 500]
    
    for batch_size in batch_sizes:
        print(f"\n   Batch size: {batch_size}")
        chunks = create_test_chunks(batch_size)
        
        # CPU batch
        cpu_batch_start = time.time()
        cpu_embedder.generate_embeddings(chunks)
        cpu_batch_time = time.time() - cpu_batch_start
        cpu_throughput = batch_size / cpu_batch_time
        
        # GPU batch
        gpu_batch_start = time.time()
        gpu_embedder.generate_embeddings(chunks)
        gpu_batch_time = time.time() - gpu_batch_start
        gpu_throughput = batch_size / gpu_batch_time
        
        print(f"   CPU: {cpu_batch_time:.2f}s ({cpu_throughput:.1f} chunks/s)")
        print(f"   GPU: {gpu_batch_time:.2f}s ({gpu_throughput:.1f} chunks/s)")
        print(f"   Speedup: {cpu_batch_time/gpu_batch_time:.2f}x")
    
    # 5. Check device transfer overhead
    print("\n5. Device Information:")
    print(f"   CPU device: {cpu_embedder.model.device if cpu_embedder.model else 'N/A'}")
    print(f"   GPU device: {gpu_embedder.model.device if gpu_embedder.model else 'N/A'}")
    
    # 6. Memory pressure test
    print("\n6. Memory Pressure Test:")
    import psutil
    process = psutil.Process()
    
    print(f"   Memory before: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    
    # Process larger batch
    large_chunks = create_test_chunks(1000)
    start = time.time()
    gpu_embedder.generate_embeddings(large_chunks)
    elapsed = time.time() - start
    
    print(f"   Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"   Time for 1000 chunks: {elapsed:.2f}s ({1000/elapsed:.1f} chunks/s)")

if __name__ == "__main__":
    measure_gpu_overhead()