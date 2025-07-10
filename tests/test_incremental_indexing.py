"""
Test incremental indexing functionality.
"""

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from src.chunker import Chunk
from src.storage import HybridStorage, StorageConfig


class TestIncrementalIndexing:
    """Test incremental indexing features."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StorageConfig(data_dir=temp_dir, auto_save=False)
            storage = HybridStorage(config)
            storage.initialize()
            yield storage

    def test_file_modification_detection(self, temp_storage):
        """Test file modification detection."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            # First time - should be modified
            assert temp_storage.is_file_modified(temp_file) is True

            # Update file info
            temp_storage.update_file_info(temp_file, 10)

            # Should not be modified now
            assert temp_storage.is_file_modified(temp_file) is False

            # Touch the file to update modification time
            time.sleep(0.1)  # Ensure time difference
            Path(temp_file).touch()

            # Should be modified again
            assert temp_storage.is_file_modified(temp_file) is True

        finally:
            os.unlink(temp_file)

    def test_update_file_info(self, temp_storage):
        """Test updating file information."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            # Update file info
            temp_storage.update_file_info(temp_file, 5)

            # Check database
            cursor = temp_storage.db.cursor()
            result = cursor.execute(
                "SELECT * FROM files WHERE path = ?", (temp_file,)
            ).fetchone()

            assert result is not None
            assert result["path"] == temp_file
            assert result["chunk_count"] == 5
            assert result["last_modified"] is not None
            assert result["last_indexed"] is not None

        finally:
            os.unlink(temp_file)

    def test_remove_chunks_for_file(self, temp_storage):
        """Test removing chunks for a file."""
        # Add some chunks
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                text=f"Test text {i}",
                metadata={"file_path": "/test/file1.jsonl", "test": True},
                embedding=[0.1] * 768,
            )
            for i in range(3)
        ]

        # Add another chunk from different file
        chunks.append(
            Chunk(
                id="chunk_other",
                text="Other file text",
                metadata={"file_path": "/test/file2.jsonl", "test": True},
                embedding=[0.2] * 768,
            )
        )

        temp_storage.add_chunks(chunks)

        # Verify all chunks are added
        cursor = temp_storage.db.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 4

        # Remove chunks for file1
        removed = temp_storage.remove_chunks_for_file("/test/file1.jsonl")
        assert removed == 3

        # Verify only file2 chunk remains
        count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count == 1

        remaining = cursor.execute("SELECT * FROM chunks").fetchone()
        assert remaining["id"] == "chunk_other"

    def test_clear_all_data(self, temp_storage):
        """Test clearing all data."""
        # Add some chunks
        chunks = [
            Chunk(
                id=f"chunk_{i}",
                text=f"Test text {i}",
                metadata={"file_path": f"/test/file{i}.jsonl"},
                embedding=[0.1 * i] * 768,
            )
            for i in range(5)
        ]
        temp_storage.add_chunks(chunks)

        # Add file info
        for i in range(3):
            temp_storage.update_file_info(f"/test/file{i}.jsonl", i + 1)

        # Verify data exists
        cursor = temp_storage.db.cursor()
        chunk_count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        file_count = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]

        assert chunk_count == 5
        assert file_count == 3
        assert temp_storage.faiss_index.ntotal == 5

        # Clear all data
        temp_storage.clear_all_data()

        # Verify all data is cleared
        chunk_count = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        file_count = cursor.execute("SELECT COUNT(*) FROM files").fetchone()[0]

        assert chunk_count == 0
        assert file_count == 0
        assert temp_storage.faiss_index.ntotal == 0
        assert len(temp_storage.chunk_id_to_faiss_id) == 0
        assert len(temp_storage.faiss_id_to_chunk_id) == 0

    def test_nonexistent_file_is_modified(self, temp_storage):
        """Test that nonexistent files are considered modified."""
        assert temp_storage.is_file_modified("/nonexistent/file.jsonl") is True

    def test_remove_chunks_for_nonexistent_file(self, temp_storage):
        """Test removing chunks for file that has no chunks."""
        removed = temp_storage.remove_chunks_for_file("/nonexistent/file.jsonl")
        assert removed == 0
