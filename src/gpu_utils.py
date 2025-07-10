#!/usr/bin/env python3
"""
GPU detection and utilities for semantic search.

This module provides utilities for detecting GPU availability, checking
FAISS GPU support, and managing GPU resources for optimal performance.
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GPUCapability:
    """GPU capability assessment."""

    torch_cuda_available: bool = False
    faiss_gpu_available: bool = False
    gpu_count: int = 0
    gpu_memory_total: Optional[int] = None
    gpu_memory_free: Optional[int] = None
    gpu_names: list = None
    recommended_batch_size: int = 16
    can_use_gpu: bool = False
    status_message: str = ""


def detect_torch_gpu() -> Tuple[bool, Dict[str, Any]]:
    """Detect PyTorch GPU availability (CUDA or MPS) and GPU info."""
    try:
        import torch

        # Check for CUDA first
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Get CUDA GPU information
                gpu_info = {
                    "backend": "cuda",
                    "gpu_count": gpu_count,
                    "current_device": torch.cuda.current_device(),
                    "devices": [],
                }

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory
                    memory_free = memory_total - torch.cuda.memory_allocated(i)

                    device_info = {
                        "id": i,
                        "name": props.name,
                        "memory_total": memory_total,
                        "memory_free": memory_free,
                        "memory_total_gb": memory_total / (1024**3),
                        "memory_free_gb": memory_free / (1024**3),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                    gpu_info["devices"].append(device_info)

                return True, gpu_info

        # Check for Apple MPS (Metal Performance Shaders)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Get system memory as proxy for GPU memory on Apple Silicon (unified memory)
            import psutil

            memory_total = psutil.virtual_memory().total
            memory_available = psutil.virtual_memory().available

            gpu_info = {
                "backend": "mps",
                "gpu_count": 1,  # MPS is single-device
                "devices": [
                    {
                        "id": 0,
                        "name": "Apple Silicon GPU (MPS)",
                        "memory_total": memory_total,
                        "memory_free": memory_available,
                        "memory_total_gb": memory_total / (1024**3),
                        "memory_free_gb": memory_available / (1024**3),
                        "compute_capability": "MPS",
                    }
                ],
            }

            return True, gpu_info

        # No GPU acceleration available
        reasons = []
        if not torch.cuda.is_available():
            reasons.append("CUDA not available")
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            reasons.append("MPS not available")

        return False, {"reason": f"No GPU acceleration available: {'; '.join(reasons)}"}

    except ImportError as e:
        return False, {"reason": f"PyTorch not installed: {str(e)}"}
    except Exception as e:
        return False, {"reason": f"Error detecting PyTorch GPU: {str(e)}"}


def detect_faiss_gpu() -> Tuple[bool, Dict[str, Any]]:
    """Detect FAISS GPU availability (CUDA only - MPS not supported)."""
    try:
        # Try importing faiss with GPU support
        import faiss

        # Check if GPU resources are available
        try:
            # Try to create GPU resources (CUDA only)
            resources = faiss.StandardGpuResources()
            gpu_count = faiss.get_num_gpus()

            if gpu_count == 0:
                return False, {"reason": "No CUDA GPUs detected by FAISS"}

            return True, {
                "gpu_count": gpu_count,
                "faiss_version": (
                    faiss.__version__ if hasattr(faiss, "__version__") else "unknown"
                ),
            }

        except AttributeError:
            # FAISS CPU version doesn't have GPU functions
            return False, {
                "reason": "FAISS GPU functions not available (using faiss-cpu or MPS not supported)"
            }
        except Exception as e:
            return False, {"reason": f"Error creating GPU resources: {str(e)}"}

    except ImportError:
        return False, {"reason": "FAISS not installed"}


def estimate_gpu_memory_requirements(
    num_chunks: int, embedding_dim: int = 768
) -> Dict[str, float]:
    """Estimate GPU memory requirements for given dataset size."""
    # Memory calculations (in GB)

    # FAISS index memory (float32 vectors)
    index_memory = (num_chunks * embedding_dim * 4) / (1024**3)

    # Model memory (sentence transformer - roughly 500MB for all-mpnet-base-v2)
    model_memory = 0.5

    # Working memory for batch processing (assume 10% overhead)
    working_memory = (index_memory + model_memory) * 0.1

    # Total estimated memory
    total_memory = index_memory + model_memory + working_memory

    return {
        "index_memory_gb": index_memory,
        "model_memory_gb": model_memory,
        "working_memory_gb": working_memory,
        "total_memory_gb": total_memory,
        "recommended_gpu_memory_gb": total_memory * 1.2,  # 20% safety margin
    }


def calculate_optimal_batch_size(
    available_memory_gb: float, embedding_dim: int = 768, backend: str = "cuda"
) -> int:
    """Calculate optimal batch size based on available GPU memory."""
    # Reserve memory for model and overhead (1GB)
    working_memory = available_memory_gb - 1.0

    if working_memory <= 0:
        return 8  # Minimal batch size

    # Each embedding uses 4 bytes (float32) * embedding_dim
    # Plus overhead for intermediate computations (factor of 4)
    memory_per_item = (embedding_dim * 4 * 4) / (1024**3)  # Convert to GB

    batch_size = int(working_memory / memory_per_item)

    # Clamp to reasonable values
    # MPS has different performance characteristics - cap at 64 for optimal performance
    if backend == "mps":
        batch_size = max(8, min(batch_size, 64))
    else:
        batch_size = max(8, min(batch_size, 256))

    return batch_size


def assess_gpu_capability(
    target_chunks: int = 10000, embedding_dim: int = 768
) -> GPUCapability:
    """Assess overall GPU capability for semantic search."""
    capability = GPUCapability()

    # Check PyTorch GPU (CUDA or MPS)
    torch_available, torch_info = detect_torch_gpu()
    capability.torch_cuda_available = torch_available

    # Check FAISS GPU (CUDA only)
    faiss_available, faiss_info = detect_faiss_gpu()
    capability.faiss_gpu_available = faiss_available

    if torch_available and "devices" in torch_info:
        # Use the first GPU for calculations
        primary_gpu = torch_info["devices"][0]
        capability.gpu_count = torch_info["gpu_count"]
        capability.gpu_memory_total = primary_gpu["memory_total"]
        capability.gpu_memory_free = primary_gpu["memory_free"]
        capability.gpu_names = [device["name"] for device in torch_info["devices"]]

        # Calculate optimal batch size
        memory_gb = primary_gpu["memory_free_gb"]
        capability.recommended_batch_size = calculate_optimal_batch_size(
            memory_gb, embedding_dim
        )

    # Determine if GPU can be used
    backend = torch_info.get("backend", "unknown") if torch_available else "none"

    if torch_available:
        if backend == "mps":
            # Apple Silicon MPS: embeddings accelerated, FAISS on CPU
            capability.can_use_gpu = True
            capability.status_message = (
                "✅ Apple Silicon MPS ready (embeddings accelerated, search on CPU)"
            )
        elif backend == "cuda" and faiss_available:
            # CUDA: both embeddings and FAISS accelerated
            if capability.gpu_memory_free:
                memory_req = estimate_gpu_memory_requirements(
                    target_chunks, embedding_dim
                )
                available_gb = capability.gpu_memory_free / (1024**3)

                if available_gb >= memory_req["recommended_gpu_memory_gb"]:
                    capability.can_use_gpu = True
                    capability.status_message = f"✅ CUDA GPU ready (Free: {available_gb:.1f}GB, Required: {memory_req['recommended_gpu_memory_gb']:.1f}GB)"
                else:
                    capability.can_use_gpu = False
                    capability.status_message = f"⚠️ Insufficient GPU memory (Free: {available_gb:.1f}GB, Required: {memory_req['recommended_gpu_memory_gb']:.1f}GB)"
            else:
                capability.can_use_gpu = True
                capability.status_message = "✅ CUDA GPU ready"
        elif backend == "cuda" and not faiss_available:
            # CUDA available but FAISS GPU not installed
            capability.can_use_gpu = True
            capability.status_message = "⚠️ CUDA available for embeddings only (install faiss-gpu for full acceleration)"
        else:
            capability.can_use_gpu = False
            capability.status_message = "❌ Unknown GPU backend"
    else:
        capability.can_use_gpu = False

        # Determine why GPU can't be used
        reasons = []
        if not torch_available:
            reasons.append(f"PyTorch GPU: {torch_info.get('reason', 'unavailable')}")

        capability.status_message = f"❌ GPU unavailable: {'; '.join(reasons)}"

    return capability


def get_gpu_installation_guide() -> str:
    """Get installation guide for GPU support."""
    return """
🚀 GPU Support Installation Guide

For optimal performance, install GPU-enabled packages:

1. Install GPU dependencies:
   uv sync --extra gpu

2. Verify CUDA installation:
   nvidia-smi

3. Test GPU detection:
   uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

4. For conda users:
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   conda install -c conda-forge faiss-gpu

Requirements:
- NVIDIA GPU with CUDA compute capability ≥ 3.5
- CUDA toolkit 11.4+ or 12.1+
- Sufficient GPU memory (≥2GB recommended)

Troubleshooting:
- Ensure nvidia-smi shows CUDA version
- Check GPU memory with: nvidia-smi
- Verify PyTorch CUDA: torch.cuda.is_available()
"""


def log_gpu_status(capability: GPUCapability, logger: logging.Logger = None) -> None:
    """Log detailed GPU status information."""
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"GPU Status: {capability.status_message}")

    if capability.torch_cuda_available:
        logger.info(f"PyTorch CUDA: ✅ Available ({capability.gpu_count} GPUs)")
        if capability.gpu_names:
            for i, name in enumerate(capability.gpu_names):
                memory_gb = (
                    capability.gpu_memory_total / (1024**3)
                    if capability.gpu_memory_total
                    else 0
                )
                logger.info(f"  GPU {i}: {name} ({memory_gb:.1f}GB)")
    else:
        logger.info("PyTorch CUDA: ❌ Unavailable")

    if capability.faiss_gpu_available:
        logger.info("FAISS GPU: ✅ Available")
    else:
        logger.info("FAISS GPU: ❌ Unavailable")

    if capability.can_use_gpu:
        logger.info(f"Recommended batch size: {capability.recommended_batch_size}")


# Convenience function for CLI usage
def quick_gpu_check() -> bool:
    """Quick check if GPU can be used. Returns True if GPU is available and ready."""
    capability = assess_gpu_capability()
    return capability.can_use_gpu


def get_gpu_summary() -> str:
    """Get a summary of GPU status for display."""
    capability = assess_gpu_capability()

    if capability.can_use_gpu:
        gpu_info = (
            f"GPU: {capability.gpu_names[0]}"
            if capability.gpu_names
            else "GPU: Available"
        )
        memory_info = (
            f" ({capability.gpu_memory_free / (1024**3):.1f}GB free)"
            if capability.gpu_memory_free
            else ""
        )
        return f"✅ {gpu_info}{memory_info}"
    else:
        return f"❌ GPU: Unavailable"
