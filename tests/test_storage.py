"""
Tests for the storage module.
"""

import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import faiss
import numpy as np
import pytest

from src.chunker import Chunk
from src.storage import HybridStorage, SearchConfig, SearchResult, StorageConfig


class TestStorageConfig:
    """Test suite for StorageConfig."""

    def test_config_defaults(self):
        """Test StorageConfig default values."""
        config = StorageConfig()
        assert config.data_dir == "./data"
        assert config.db_name == "metadata.db"
        assert config.index_name == "embeddings.faiss"
        assert config.embedding_dim == 768
        assert config.index_type == "flat"
        assert config.ivf_nlist == 100
        assert config.hnsw_m == 16
        assert config.normalize_embeddings is True
        assert config.auto_save is True
        assert config.backup_enabled is True

    def test_config_custom_values(self):
        """Test StorageConfig with custom values."""
        config = StorageConfig(
            data_dir="/tmp/test_data",
            db_name="test.db",
            index_name="test.faiss",
            embedding_dim=384,
            index_type="ivf",
            ivf_nlist=50,
            hnsw_m=32,
            normalize_embeddings=False,
            auto_save=False,
            backup_enabled=False,
        )

        assert config.data_dir == "/tmp/test_data"
        assert config.db_name == "test.db"
        assert config.index_name == "test.faiss"
        assert config.embedding_dim == 384
        assert config.index_type == "ivf"
        assert config.ivf_nlist == 50
        assert config.hnsw_m == 32
        assert config.normalize_embeddings is False
        assert config.auto_save is False
        assert config.backup_enabled is False


class TestSearchConfig:
    """Test suite for SearchConfig."""

    def test_config_defaults(self):
        """Test SearchConfig default values."""
        config = SearchConfig()
        assert config.top_k == 10
        assert config.similarity_threshold == 0.0
        assert config.include_metadata is True
        assert config.include_text is True
        assert config.max_results == 100


class TestSearchResult:
    """Test suite for SearchResult."""

    def test_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(chunk_id="test_chunk", similarity=0.85)

        assert result.chunk_id == "test_chunk"
        assert result.similarity == 0.85
        assert result.chunk is None
        assert result.metadata is None
        assert result.text is None


class TestHybridStorage:
    """Test suite for HybridStorage."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = StorageConfig(
            data_dir=self.temp_dir,
            embedding_dim=4,  # Small dimension for testing
            auto_save=False,  # Disable auto-save for tests
        )
        self.storage = HybridStorage(self.config)

        # Create test chunks
        self.test_chunks = [
            Chunk(
                id="chunk_001",
                text="This is about machine learning and AI.",
                metadata={
                    "session_id": "session_1",
                    "project_name": "test_project",
                    "chunk_type": "qa_pair",
                    "timestamp": "2024-01-15T10:00:00",
                    "has_code": False,
                    "has_tools": False,
                    "message_count": 2,
                    "char_count": 38,
                    "word_count": 8,
                },
                embedding=[0.1, 0.2, 0.3, 0.4],
            ),
            Chunk(
                id="chunk_002",
                text="Python programming and data science topics.",
                metadata={
                    "session_id": "session_1",
                    "project_name": "test_project",
                    "chunk_type": "code_block",
                    "timestamp": "2024-01-15T10:01:00",
                    "has_code": True,
                    "has_tools": True,
                    "message_count": 1,
                    "char_count": 43,
                    "word_count": 6,
                },
                embedding=[0.5, 0.6, 0.7, 0.8],
            ),
            Chunk(
                id="chunk_003",
                text="Natural language processing techniques.",
                metadata={
                    "session_id": "session_2",
                    "project_name": "other_project",
                    "chunk_type": "tool_usage",
                    "timestamp": "2024-01-15T11:00:00",
                    "has_code": False,
                    "has_tools": True,
                    "message_count": 3,
                    "char_count": 37,
                    "word_count": 4,
                },
                embedding=[0.9, 0.1, 0.2, 0.3],
            ),
        ]

    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, "storage") and self.storage:
            try:
                self.storage.close()
            except:
                pass

        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_storage_initialization(self):
        """Test storage initialization."""
        assert self.storage.config == self.config
        assert self.storage.data_dir == Path(self.temp_dir)
        assert self.storage.db is None
        assert self.storage.faiss_index is None
        assert self.storage.total_chunks == 0
        assert self.storage.embedding_dim == 4

    def test_initialize_storage(self):
        """Test storage initialization."""
        self.storage.initialize()

        # Check SQLite was initialized
        assert self.storage.db is not None
        assert self.storage.db_path.exists()

        # Check FAISS was initialized
        assert self.storage.faiss_index is not None
        assert isinstance(self.storage.faiss_index, faiss.IndexFlatIP)

        # Check tables were created
        cursor = self.storage.db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "chunks" in tables
        assert "files" in tables

    def test_initialize_with_ivf_index(self):
        """Test initialization with IVF index."""
        config = StorageConfig(
            data_dir=self.temp_dir,
            embedding_dim=4,
            index_type="ivf",
            ivf_nlist=2,  # Small value for testing
        )
        storage = HybridStorage(config)
        storage.initialize()

        assert isinstance(storage.faiss_index, faiss.IndexIVFFlat)
        storage.close()

    def test_initialize_with_hnsw_index(self):
        """Test initialization with HNSW index."""
        config = StorageConfig(
            data_dir=self.temp_dir,
            embedding_dim=4,
            index_type="hnsw",
            hnsw_m=4,  # Small value for testing
        )
        storage = HybridStorage(config)
        storage.initialize()

        assert isinstance(storage.faiss_index, faiss.IndexHNSWFlat)
        storage.close()

    def test_initialize_with_invalid_index_type(self):
        """Test initialization with invalid index type."""
        config = StorageConfig(
            data_dir=self.temp_dir, embedding_dim=4, index_type="invalid"
        )
        storage = HybridStorage(config)

        with pytest.raises(ValueError, match="Unknown index type"):
            storage.initialize()

    def test_add_chunks(self):
        """Test adding chunks to storage."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Check FAISS index
        assert self.storage.faiss_index.ntotal == 3
        assert self.storage.total_chunks == 3

        # Check SQLite database
        cursor = self.storage.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        assert cursor.fetchone()[0] == 3

        # Check ID mappings
        assert len(self.storage.chunk_id_to_faiss_id) == 3
        assert len(self.storage.faiss_id_to_chunk_id) == 3

    def test_add_chunks_without_embeddings(self):
        """Test adding chunks without embeddings."""
        self.storage.initialize()

        # Create chunks without embeddings
        chunks_no_embeddings = [
            Chunk(id="no_emb_1", text="Test", metadata={}),
            Chunk(id="no_emb_2", text="Test", metadata={}),
        ]

        self.storage.add_chunks(chunks_no_embeddings)

        # Should not add any chunks
        assert self.storage.faiss_index.ntotal == 0
        assert self.storage.total_chunks == 0

    def test_add_empty_chunks_list(self):
        """Test adding empty chunks list."""
        self.storage.initialize()
        self.storage.add_chunks([])

        assert self.storage.faiss_index.ntotal == 0
        assert self.storage.total_chunks == 0

    def test_search_basic(self):
        """Test basic search functionality."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Search with first chunk's embedding
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = self.storage.search(query_embedding)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

        # First result should be most similar
        assert results[0].chunk_id == "chunk_001"
        assert results[0].similarity > 0.8  # Should be high similarity

    def test_search_with_config(self):
        """Test search with custom configuration."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        config = SearchConfig(
            top_k=2, similarity_threshold=0.5, include_metadata=True, include_text=True
        )

        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = self.storage.search(query_embedding, config)

        assert len(results) <= 2
        assert all(r.similarity >= 0.5 for r in results)
        assert all(r.metadata is not None for r in results)
        assert all(r.text is not None for r in results)

    def test_search_with_filters(self):
        """Test search with filters."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Filter by project
        filters = {"project_name": "test_project"}
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = self.storage.search(query_embedding, filters=filters)

        assert len(results) == 2  # Only chunks from test_project
        assert all(r.metadata["project_name"] == "test_project" for r in results)

    def test_search_with_range_filters(self):
        """Test search with range filters."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Filter by word count
        filters = {"word_count": {"gte": 5}}
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = self.storage.search(query_embedding, filters=filters)

        assert len(results) == 2  # Only chunks with word_count >= 5

    def test_search_with_list_filters(self):
        """Test search with list filters."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Filter by chunk type
        filters = {"chunk_type": ["qa_pair", "code_block"]}
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = self.storage.search(query_embedding, filters=filters)

        assert len(results) == 2  # Only qa_pair and code_block chunks

    def test_get_chunk_by_id(self):
        """Test getting chunk by ID."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        chunk = self.storage.get_chunk_by_id("chunk_001")

        assert chunk is not None
        assert chunk.id == "chunk_001"
        assert chunk.text == "This is about machine learning and AI."
        assert chunk.metadata["session_id"] == "session_1"

    def test_get_chunk_by_id_not_found(self):
        """Test getting non-existent chunk."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        chunk = self.storage.get_chunk_by_id("nonexistent")
        assert chunk is None

    def test_get_chunks_by_session(self):
        """Test getting chunks by session ID."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        chunks = self.storage.get_chunks_by_session("session_1")

        assert len(chunks) == 2
        assert all(c.metadata["session_id"] == "session_1" for c in chunks)

    def test_get_chunks_by_project(self):
        """Test getting chunks by project name."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        chunks = self.storage.get_chunks_by_project("test_project")

        assert len(chunks) == 2
        assert all(c.metadata["project_name"] == "test_project" for c in chunks)

    def test_delete_chunk(self):
        """Test deleting a chunk."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Delete a chunk
        result = self.storage.delete_chunk("chunk_001")
        assert result is True

        # Check it's gone
        chunk = self.storage.get_chunk_by_id("chunk_001")
        assert chunk is None

        # Check total count
        assert self.storage.total_chunks == 2

    def test_delete_chunk_not_found(self):
        """Test deleting non-existent chunk."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        result = self.storage.delete_chunk("nonexistent")
        assert result is False

    def test_delete_chunks_by_session(self):
        """Test deleting chunks by session."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        deleted_count = self.storage.delete_chunks_by_session("session_1")

        assert deleted_count == 2
        assert self.storage.total_chunks == 1

        # Check remaining chunk
        remaining_chunks = self.storage.get_chunks_by_session("session_2")
        assert len(remaining_chunks) == 1

    def test_get_stats(self):
        """Test getting storage statistics."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        stats = self.storage.get_stats()

        assert stats["total_chunks"] == 3
        assert stats["total_sessions"] == 2
        assert stats["total_projects"] == 2
        assert "chunk_types" in stats
        assert stats["chunk_types"]["qa_pair"] == 1
        assert stats["chunk_types"]["code_block"] == 1
        assert stats["chunk_types"]["tool_usage"] == 1
        assert stats["embedding_dimension"] == 4
        assert stats["index_type"] == "flat"
        # Check that projects list is included
        assert "projects" in stats
        assert len(stats["projects"]) == 2
        assert "test_project" in stats["projects"]
        assert "other_project" in stats["projects"]

    def test_get_all_projects(self):
        """Test getting all project names."""
        self.storage.initialize()
        
        # Test with empty storage
        projects = self.storage.get_all_projects()
        assert projects == []
        
        # Add chunks with different projects
        test_chunks = [
            Chunk(
                id="chunk_p1_1",
                text="Project 1 content",
                metadata={
                    "session_id": "session_p1",
                    "project_name": "Project Alpha",
                    "timestamp": "2024-01-01T10:00:00Z",
                },
                embedding=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            ),
            Chunk(
                id="chunk_p1_2",
                text="More Project 1 content",
                metadata={
                    "session_id": "session_p1",
                    "project_name": "Project Alpha",
                    "timestamp": "2024-01-01T11:00:00Z",
                },
                embedding=np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            ),
            Chunk(
                id="chunk_p2_1",
                text="Project 2 content",
                metadata={
                    "session_id": "session_p2",
                    "project_name": "Project Beta",
                    "timestamp": "2024-01-02T10:00:00Z",
                },
                embedding=np.array([0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            ),
            Chunk(
                id="chunk_p3_1",
                text="Project 3 content",
                metadata={
                    "session_id": "session_p3",
                    "project_name": "Project Gamma",
                    "timestamp": "2024-01-03T10:00:00Z",
                },
                embedding=np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32),
            ),
            Chunk(
                id="chunk_no_project",
                text="Content without project",
                metadata={
                    "session_id": "session_no_project",
                    "timestamp": "2024-01-04T10:00:00Z",
                },
                embedding=np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            ),
        ]
        
        self.storage.add_chunks(test_chunks)
        
        # Get all projects
        projects = self.storage.get_all_projects()
        
        # Should return sorted list of unique project names
        assert len(projects) == 3
        assert projects == ["Project Alpha", "Project Beta", "Project Gamma"]
        
        # Test that empty project names are not included
        self.storage.add_chunks([
            Chunk(
                id="chunk_empty_project",
                text="Content with empty project",
                metadata={
                    "session_id": "session_empty",
                    "project_name": "",
                    "timestamp": "2024-01-05T10:00:00Z",
                },
                embedding=np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float32),
            )
        ])
        
        projects_after = self.storage.get_all_projects()
        assert len(projects_after) == 3  # Empty project name should not be included
        assert projects_after == ["Project Alpha", "Project Beta", "Project Gamma"]

    def test_get_all_projects_error_handling(self):
        """Test get_all_projects error handling."""
        # Test without initialization
        with pytest.raises(RuntimeError, match="Database not initialized"):
            self.storage.get_all_projects()

    def test_save_and_load_index(self):
        """Test saving and loading FAISS index."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Save index
        self.storage.save_index()
        assert self.storage.index_path.exists()

        # Create new storage instance
        new_storage = HybridStorage(self.config)
        new_storage.initialize()

        # Should load existing index
        assert new_storage.faiss_index.ntotal == 3
        assert new_storage.total_chunks == 3

        new_storage.close()

    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Create backup
        backup_dir = os.path.join(self.temp_dir, "backup")
        self.storage.backup(backup_dir)

        # Verify backup files exist
        backup_path = Path(backup_dir)
        assert (
            backup_path / self.config.index_name
        ).exists()  # Should exist since we added chunks
        assert (backup_path / self.config.db_name).exists()

        # Clear current storage
        self.storage.close()
        if self.storage.index_path.exists():
            self.storage.index_path.unlink()
        if self.storage.db_path.exists():
            self.storage.db_path.unlink()

        # Restore from backup
        new_storage = HybridStorage(self.config)
        new_storage.initialize()
        new_storage.restore(backup_dir)

        # Verify data is restored
        assert new_storage.total_chunks == 3
        chunk = new_storage.get_chunk_by_id("chunk_001")
        assert chunk is not None

        new_storage.close()

    def test_context_manager(self):
        """Test using storage as context manager."""
        with HybridStorage(self.config) as storage:
            storage.add_chunks(self.test_chunks)
            assert storage.total_chunks == 3

        # Storage should be closed after context
        # (Can't easily test this without internal state inspection)

    def test_normalize_embeddings_disabled(self):
        """Test storage with embedding normalization disabled."""
        config = StorageConfig(
            data_dir=self.temp_dir, embedding_dim=4, normalize_embeddings=False
        )
        storage = HybridStorage(config)
        storage.initialize()

        # Create index without normalization
        assert isinstance(storage.faiss_index, faiss.IndexFlatL2)

        storage.close()

    def test_matches_filters(self):
        """Test filter matching logic."""
        self.storage.initialize()

        chunk_data = {
            "project_name": "test_project",
            "word_count": 10,
            "has_code": True,
            "chunk_type": "qa_pair",
        }

        # Test exact match
        assert self.storage._matches_filters(
            chunk_data, {"project_name": "test_project"}
        )
        assert not self.storage._matches_filters(
            chunk_data, {"project_name": "other_project"}
        )

        # Test range filters
        assert self.storage._matches_filters(chunk_data, {"word_count": {"gte": 5}})
        assert self.storage._matches_filters(chunk_data, {"word_count": {"lte": 15}})
        assert not self.storage._matches_filters(chunk_data, {"word_count": {"gt": 10}})

        # Test list filters
        assert self.storage._matches_filters(
            chunk_data, {"chunk_type": ["qa_pair", "code_block"]}
        )
        assert not self.storage._matches_filters(
            chunk_data, {"chunk_type": ["tool_usage"]}
        )

    def test_rebuild_id_mappings(self):
        """Test ID mapping rebuilding."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Clear mappings
        self.storage.chunk_id_to_faiss_id.clear()
        self.storage.faiss_id_to_chunk_id.clear()
        self.storage.total_chunks = 0

        # Rebuild
        self.storage._rebuild_id_mappings()

        # Check mappings are restored
        assert len(self.storage.chunk_id_to_faiss_id) == 3
        assert len(self.storage.faiss_id_to_chunk_id) == 3
        assert self.storage.total_chunks == 3

    def test_optimize_storage(self):
        """Test storage optimization."""
        self.storage.initialize()
        self.storage.add_chunks(self.test_chunks)

        # Run optimization
        self.storage.optimize()

        # Check database was vacuumed (hard to test directly)
        # The main thing is that it doesn't crash
        assert self.storage.total_chunks == 3

    def test_storage_with_auto_save(self):
        """Test storage with auto-save enabled."""
        config = StorageConfig(data_dir=self.temp_dir, embedding_dim=4, auto_save=True)
        storage = HybridStorage(config)
        storage.initialize()

        # Add chunks (should auto-save)
        storage.add_chunks(self.test_chunks)

        # Check index was saved
        assert storage.index_path.exists()

        storage.close()

    def test_search_empty_storage(self):
        """Test search on empty storage."""
        self.storage.initialize()

        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = self.storage.search(query_embedding)

        assert len(results) == 0
