"""
Tests for environment setup and dependencies.

Verifies that all required dependencies are installed and working correctly.
"""

import sys
from pathlib import Path

import pytest


# Test imports for all required dependencies
def test_python_version():
    """Test that Python version is 3.11+."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


def test_import_sentence_transformers():
    """Test that sentence-transformers can be imported."""
    try:
        import sentence_transformers

        assert hasattr(sentence_transformers, "SentenceTransformer")
    except ImportError:
        pytest.fail("sentence-transformers not installed")


def test_import_faiss():
    """Test that faiss can be imported."""
    try:
        import faiss

        assert hasattr(faiss, "IndexFlatIP")
    except ImportError:
        pytest.fail("faiss-cpu not installed")


def test_import_torch():
    """Test that torch can be imported."""
    try:
        import torch

        assert hasattr(torch, "tensor")
    except ImportError:
        pytest.fail("torch not installed")


def test_import_numpy():
    """Test that numpy can be imported."""
    try:
        import numpy as np

        assert hasattr(np, "array")
    except ImportError:
        pytest.fail("numpy not installed")


def test_import_pandas():
    """Test that pandas can be imported."""
    try:
        import pandas as pd

        assert hasattr(pd, "DataFrame")
    except ImportError:
        pytest.fail("pandas not installed")


def test_import_tqdm():
    """Test that tqdm can be imported."""
    try:
        import tqdm

        assert hasattr(tqdm, "tqdm")
    except ImportError:
        pytest.fail("tqdm not installed")


def test_import_transformers():
    """Test that transformers can be imported."""
    try:
        import transformers

        assert hasattr(transformers, "AutoTokenizer")
    except ImportError:
        pytest.fail("transformers not installed")


def test_import_huggingface_hub():
    """Test that huggingface-hub can be imported."""
    try:
        import huggingface_hub

        assert hasattr(huggingface_hub, "hf_hub_download")
    except ImportError:
        pytest.fail("huggingface-hub not installed")


def test_import_sklearn():
    """Test that scikit-learn can be imported."""
    try:
        import sklearn

        assert hasattr(sklearn, "metrics")
    except ImportError:
        pytest.fail("scikit-learn not installed")


def test_directory_structure():
    """Test that required directories exist."""
    project_root = Path(__file__).parent.parent

    required_dirs = [
        "src",
        "tests",
        "scripts",
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


def test_project_files():
    """Test that required project files exist."""
    project_root = Path(__file__).parent.parent

    required_files = [
        "pyproject.toml",
        ".gitignore",
        "scripts/model_setup.py",
        "src/__init__.py",
        "tests/__init__.py",
    ]

    for file_name in required_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"File {file_name} does not exist"
        assert file_path.is_file(), f"{file_name} is not a file"


def test_model_setup_script():
    """Test that model setup script is executable."""
    project_root = Path(__file__).parent.parent
    model_setup_path = project_root / "scripts" / "model_setup.py"

    assert model_setup_path.exists()

    # Test that the script can be imported
    import sys

    sys.path.insert(0, str(project_root))

    try:
        from scripts.model_setup import get_model_cache_dir, get_system_info

        # Test basic function calls
        cache_dir = get_model_cache_dir()
        assert cache_dir.exists()

        system_info = get_system_info()
        assert "python_version" in system_info
        assert "torch_version" in system_info

    except ImportError as e:
        pytest.fail(f"Cannot import model_setup script: {e}")


def test_torch_functionality():
    """Test basic torch functionality."""
    import torch

    # Test tensor creation
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == (3,)
    assert x.dtype == torch.float32

    # Test basic operations
    y = x * 2
    assert torch.allclose(y, torch.tensor([2.0, 4.0, 6.0]))


def test_numpy_functionality():
    """Test basic numpy functionality."""
    import numpy as np

    # Test array creation
    arr = np.array([1, 2, 3])
    assert arr.shape == (3,)
    assert arr.dtype == np.int64 or arr.dtype == np.int32

    # Test basic operations
    result = arr * 2
    expected = np.array([2, 4, 6])
    assert np.array_equal(result, expected)


def test_faiss_functionality():
    """Test basic faiss functionality."""
    import faiss
    import numpy as np

    # Test index creation
    dimension = 128
    index = faiss.IndexFlatIP(dimension)
    assert index.d == dimension
    assert index.ntotal == 0

    # Test adding vectors
    vectors = np.random.random((10, dimension)).astype(np.float32)
    index.add(vectors)
    assert index.ntotal == 10

    # Test search
    query = vectors[0:1]  # First vector as query
    distances, indices = index.search(query, k=5)
    assert distances.shape == (1, 5)
    assert indices.shape == (1, 5)
    assert indices[0][0] == 0  # First result should be the query itself


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
