"""Test project name partial matching functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.storage import HybridStorage, SearchConfig, StorageConfig
from src.chunker import Chunk


class TestProjectFilter:
    """Test cases for project name partial matching."""

    @pytest.fixture
    def storage(self):
        """Create a storage instance with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(data_dir=tmpdir, use_gpu=False)
            storage = HybridStorage(config)
            storage.initialize()
            
            # Add test chunks with different project names
            chunks = [
                Chunk(
                    id="chunk1",
                    text="Test persistence implementation",
                    metadata={
                        "project_name": "-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine",
                        "has_code": True
                    },
                    embedding=np.random.rand(768).astype(np.float32)
                ),
                Chunk(
                    id="chunk2", 
                    text="Another daisy feature",
                    metadata={
                        "project_name": "-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine",
                        "has_code": False
                    },
                    embedding=np.random.rand(768).astype(np.float32)
                ),
                Chunk(
                    id="chunk3",
                    text="Different project content",
                    metadata={
                        "project_name": "semantic-search",
                        "has_code": True
                    },
                    embedding=np.random.rand(768).astype(np.float32)
                ),
                Chunk(
                    id="chunk4",
                    text="DAISY uppercase test",
                    metadata={
                        "project_name": "-Users-jrbaron-DAISY-HFT-ENGINE",
                        "has_code": True
                    },
                    embedding=np.random.rand(768).astype(np.float32)
                )
            ]
            
            storage.add_chunks(chunks)
            yield storage

    def test_partial_project_name_match(self, storage):
        """Test that partial project names match correctly."""
        query_embedding = np.random.rand(768).astype(np.float32)
        config = SearchConfig(top_k=10)
        
        # Test partial match
        filters = {"project_name": "daisy-hft-engine"}
        results = storage.search(query_embedding, config, filters)
        
        # Should match chunks 1, 2, and 4 (all contain "daisy-hft-engine")
        assert len(results) == 3
        assert all("daisy-hft-engine" in r.metadata["project_name"].lower() for r in results)

    def test_case_insensitive_project_match(self, storage):
        """Test that project name matching is case-insensitive."""
        query_embedding = np.random.rand(768).astype(np.float32)
        config = SearchConfig(top_k=10)
        
        # Test case-insensitive match
        filters = {"project_name": "DAISY-hft-ENGINE"}
        results = storage.search(query_embedding, config, filters)
        
        # Should match all daisy chunks (1, 2, and 4)
        assert len(results) == 3

    def test_exact_project_name_still_works(self, storage):
        """Test that exact project names still work."""
        query_embedding = np.random.rand(768).astype(np.float32)
        config = SearchConfig(top_k=10)
        
        # Test exact match
        filters = {"project_name": "semantic-search"}
        results = storage.search(query_embedding, config, filters)
        
        # Should only match chunk 3
        assert len(results) == 1
        assert results[0].metadata["project_name"] == "semantic-search"

    def test_project_filter_with_other_filters(self, storage):
        """Test project filter combined with other filters."""
        query_embedding = np.random.rand(768).astype(np.float32)
        config = SearchConfig(top_k=10)
        
        # Test combined filters
        filters = {
            "project_name": "daisy",
            "has_code": True
        }
        results = storage.search(query_embedding, config, filters)
        
        # Should match chunks 1 and 4 (daisy projects with code, not chunk 2 which has no code)
        assert len(results) == 2
        assert all(r.metadata["has_code"] for r in results)
        assert all("daisy" in r.metadata["project_name"].lower() for r in results)

    def test_no_match_project_filter(self, storage):
        """Test when no projects match the filter."""
        query_embedding = np.random.rand(768).astype(np.float32)
        config = SearchConfig(top_k=10)
        
        # Test no match
        filters = {"project_name": "nonexistent-project"}
        results = storage.search(query_embedding, config, filters)
        
        assert len(results) == 0