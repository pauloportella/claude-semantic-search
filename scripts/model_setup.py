#!/usr/bin/env python3
"""
Model setup script for Claude Semantic Search.

Downloads and caches the all-mpnet-base-v2 model locally.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def get_model_cache_dir() -> Path:
    """Get the directory for caching models."""
    current_dir = Path(__file__).parent.parent
    cache_dir = current_dir / "data" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model(model_name: str = "all-mpnet-base-v2", force_download: bool = False) -> Path:
    """
    Download and cache the sentence transformer model.
    
    Args:
        model_name: Name of the model to download
        force_download: Force re-download even if model exists
        
    Returns:
        Path to the cached model directory
    """
    cache_dir = get_model_cache_dir()
    model_path = cache_dir / model_name
    
    if model_path.exists() and not force_download:
        print(f"Model {model_name} already exists at {model_path}")
        return model_path
    
    print(f"Downloading {model_name} model...")
    print(f"This will download approximately 420MB")
    
    try:
        # Download with progress bar
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
        
        # Save to our specific cache directory
        model.save(str(model_path))
        
        print(f"✅ Model downloaded and cached at {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)


def verify_model(model_path: Path) -> bool:
    """
    Verify the downloaded model works correctly.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        True if model loads and works correctly
    """
    try:
        print("Verifying model...")
        
        # Load model
        model = SentenceTransformer(str(model_path))
        
        # Test embedding generation
        test_sentences = [
            "Hello world",
            "This is a test sentence for embedding generation",
            "Claude conversations about code and debugging"
        ]
        
        embeddings = model.encode(test_sentences)
        
        # Verify embedding properties
        assert embeddings.shape[0] == len(test_sentences), "Wrong number of embeddings"
        assert embeddings.shape[1] == 768, "Wrong embedding dimension (should be 768)"
        # Note: embeddings are numpy arrays by default, not torch tensors
        import numpy as np
        assert embeddings.dtype == np.float32, "Wrong embedding dtype"
        
        print(f"✅ Model verification successful")
        print(f"   - Embedding dimension: {embeddings.shape[1]}")
        print(f"   - Model parameters: ~110M")
        print(f"   - Model size on disk: ~420MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False


def get_system_info() -> dict:
    """Get system information for debugging."""
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "platform": sys.platform,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
    
    return info


def main():
    """Main entry point for model setup."""
    print("Claude Semantic Search - Model Setup")
    print("=" * 40)
    
    # Print system info
    system_info = get_system_info()
    print(f"Python: {system_info['python_version'].split()[0]}")
    print(f"PyTorch: {system_info['torch_version']}")
    print(f"CUDA Available: {system_info['cuda_available']}")
    print()
    
    # Download model
    model_path = download_model()
    
    # Verify model
    if verify_model(model_path):
        print("🎉 Model setup complete!")
        print(f"Model cached at: {model_path}")
        print()
        print("Next steps:")
        print("1. Run tests: uv run pytest tests/test_environment_setup.py")
        print("2. Start implementing Phase 1: JSONL Parser")
    else:
        print("❌ Model setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()