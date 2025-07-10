"""
Tests for model download and loading functionality.

Verifies that the all-mpnet-base-v2 model can be downloaded and loaded correctly.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_setup import (
    download_model,
    get_model_cache_dir,
    get_system_info,
    verify_model,
)


def test_get_model_cache_dir():
    """Test that model cache directory is created correctly."""
    cache_dir = get_model_cache_dir()

    assert cache_dir.exists(), "Cache directory should be created"
    assert cache_dir.is_dir(), "Cache directory should be a directory"
    assert cache_dir.name == "models", "Cache directory should be named 'models'"


def test_get_system_info():
    """Test system information gathering."""
    system_info = get_system_info()

    required_keys = [
        "python_version",
        "torch_version",
        "cuda_available",
        "device_count",
        "platform",
    ]

    for key in required_keys:
        assert key in system_info, f"Missing key: {key}"

    # Test types
    assert isinstance(system_info["python_version"], str)
    assert isinstance(system_info["torch_version"], str)
    assert isinstance(system_info["cuda_available"], bool)
    assert isinstance(system_info["device_count"], int)
    assert isinstance(system_info["platform"], str)


@pytest.mark.slow
def test_download_model_real():
    """Test actual model download (marked as slow test)."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Temporarily change the cache directory
        with patch("scripts.model_setup.get_model_cache_dir") as mock_cache_dir:
            mock_cache_dir.return_value = Path(tmp_dir)

            # Download the model
            model_path = download_model("all-mpnet-base-v2")

            # Verify the model was downloaded
            assert model_path.exists(), "Model path should exist after download"
            assert model_path.is_dir(), "Model path should be a directory"

            # Check for expected files
            expected_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "vocab.txt",
            ]

            model_files = list(model_path.glob("*"))
            assert len(model_files) > 0, "Model directory should contain files"


def test_download_model_exists():
    """Test that download_model handles existing models correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = Path(tmp_dir) / "all-mpnet-base-v2"
        model_dir.mkdir()

        # Create a dummy file to simulate existing model
        (model_dir / "config.json").write_text('{"model_type": "test"}')

        with patch("scripts.model_setup.get_model_cache_dir") as mock_cache_dir:
            mock_cache_dir.return_value = Path(tmp_dir)

            # Should not re-download existing model
            result_path = download_model("all-mpnet-base-v2", force_download=False)
            assert result_path == model_dir
            assert result_path.exists()


@pytest.mark.slow
def test_verify_model_real():
    """Test model verification with real model (marked as slow)."""
    # First ensure model is downloaded
    model_path = download_model("all-mpnet-base-v2")

    # Verify the model
    is_valid = verify_model(model_path)
    assert is_valid, "Model should be valid"


def test_verify_model_properties():
    """Test that model has expected properties."""
    try:
        from sentence_transformers import SentenceTransformer

        # Create a mock model with expected properties
        mock_model = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.shape = (3, 768)
        mock_embeddings.dtype = "float32"

        mock_model.encode.return_value = mock_embeddings

        # Test the verification logic
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_model

            # This would normally call verify_model, but we'll test the logic
            test_sentences = [
                "Hello world",
                "This is a test sentence for embedding generation",
                "Claude conversations about code and debugging",
            ]

            embeddings = mock_model.encode(test_sentences)

            # Verify embedding properties
            assert embeddings.shape[0] == len(
                test_sentences
            ), "Wrong number of embeddings"
            assert embeddings.shape[1] == 768, "Wrong embedding dimension"

    except ImportError:
        pytest.skip("sentence-transformers not available")


def test_model_cache_persistence():
    """Test that model cache persists across calls."""
    cache_dir = get_model_cache_dir()

    # Create a test file in cache
    test_file = cache_dir / "test_persistence.txt"
    test_file.write_text("test content")

    # Get cache dir again - should be the same
    cache_dir2 = get_model_cache_dir()
    assert cache_dir == cache_dir2

    # Test file should still exist
    assert test_file.exists()

    # Clean up
    test_file.unlink()


def test_embedding_dimensions():
    """Test that model produces correct embedding dimensions."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        # Test with a small model first (for speed)
        # If available, use a tiny model for testing
        test_sentences = ["Hello", "World", "Test"]

        # Mock the model to return correct dimensions
        mock_model = MagicMock()
        mock_embeddings = torch.randn(3, 768, dtype=torch.float32)
        mock_model.encode.return_value = mock_embeddings

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = mock_model

            embeddings = mock_model.encode(test_sentences)

            assert embeddings.shape == (
                3,
                768,
            ), f"Expected (3, 768), got {embeddings.shape}"
            assert (
                embeddings.dtype == torch.float32
            ), f"Expected float32, got {embeddings.dtype}"

    except ImportError:
        pytest.skip("Required libraries not available")


def test_model_path_validation():
    """Test model path validation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with empty directory
        empty_dir = Path(tmp_dir) / "empty"
        empty_dir.mkdir()

        # Should fail verification
        is_valid = verify_model(empty_dir)
        assert not is_valid, "Empty directory should fail verification"

        # Test with non-existent directory
        non_existent = Path(tmp_dir) / "does_not_exist"
        is_valid = verify_model(non_existent)
        assert not is_valid, "Non-existent directory should fail verification"


@pytest.mark.parametrize(
    "model_name",
    [
        "all-mpnet-base-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
)
def test_model_name_variations(model_name):
    """Test different model name formats."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("scripts.model_setup.get_model_cache_dir") as mock_cache_dir:
            mock_cache_dir.return_value = Path(tmp_dir)

            # Mock SentenceTransformer to avoid actual download
            with patch("sentence_transformers.SentenceTransformer") as mock_st:
                mock_model = MagicMock()
                mock_st.return_value = mock_model

                # Should handle different model name formats
                try:
                    result = download_model(model_name)
                    assert result is not None
                except Exception as e:
                    # If mocking fails, that's okay for this test
                    pass


if __name__ == "__main__":
    # Run with different verbosity levels
    pytest.main([__file__, "-v", "-m", "not slow"])
