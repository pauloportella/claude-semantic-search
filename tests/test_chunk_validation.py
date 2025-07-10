"""
Test chunk validation and text sanitization.
"""

from datetime import datetime, timezone

import pytest

from src.chunker import Chunk, ChunkingConfig, ConversationChunker
from src.embeddings import EmbeddingConfig, EmbeddingGenerator
from src.parser import Conversation, Message


class TestChunkValidation:
    """Test chunk validation and text sanitization."""

    def test_create_chunk_with_none_text(self):
        """Test that chunks with None text are not created."""
        chunker = ConversationChunker()

        # Try to create chunk with None text
        chunk = chunker._create_chunk(
            text=None, chunk_type="test", conversation=None, messages=[]
        )

        assert chunk is None

    def test_create_chunk_with_empty_text(self):
        """Test that chunks with empty text are not created."""
        chunker = ConversationChunker()

        # Try to create chunk with empty string
        chunk = chunker._create_chunk(
            text="", chunk_type="test", conversation=None, messages=[]
        )

        assert chunk is None

        # Try with whitespace only
        chunk = chunker._create_chunk(
            text="   \n\t  ", chunk_type="test", conversation=None, messages=[]
        )

        assert chunk is None

    def test_create_chunk_with_non_string_text(self):
        """Test that non-string text is converted to string."""
        chunker = ConversationChunker()

        # Try to create chunk with list (will be converted to string)
        chunk = chunker._create_chunk(
            text=["some", "list"], chunk_type="test", conversation=None, messages=[]
        )

        # Should create chunk with string representation
        assert chunk is not None
        assert chunk.text == "['some', 'list']"

    def test_chunk_conversation_filters_invalid_chunks(self):
        """Test that chunk_conversation filters out invalid chunks."""
        # Create messages with empty content
        messages = [
            Message(
                uuid="1",
                content="Valid user message",
                timestamp=datetime.now(timezone.utc),
                role="user",
            ),
            Message(
                uuid="2",
                content="",  # Empty content
                timestamp=datetime.now(timezone.utc),
                role="assistant",
            ),
            Message(
                uuid="3",
                content="Another valid message",
                timestamp=datetime.now(timezone.utc),
                role="user",
            ),
            Message(
                uuid="4",
                content="Valid assistant response",
                timestamp=datetime.now(timezone.utc),
                role="assistant",
            ),
        ]

        conversation = Conversation(
            session_id="test",
            messages=messages,
            project_name="test",
            file_path="test.jsonl",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            total_messages=4,
        )

        chunker = ConversationChunker()
        chunks = chunker.chunk_conversation(conversation)

        # Should only create chunks with valid text
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.text
            assert chunk.text.strip()
            assert isinstance(chunk.text, str)


class TestEmbeddingValidation:
    """Test embedding generation with text validation."""

    def test_generate_embeddings_with_invalid_texts(self):
        """Test that embedding generation handles invalid texts gracefully."""
        config = EmbeddingConfig(show_progress=False)
        embedder = EmbeddingGenerator(config)
        embedder.load_model()

        # Create chunks with various invalid texts
        chunks = [
            Chunk(id="1", text="Valid text"),
            Chunk(id="2", text=None),  # None text
            Chunk(id="3", text=""),  # Empty text
            Chunk(id="4", text="   "),  # Whitespace only
            Chunk(id="5", text="Another valid text"),
        ]

        # Should handle invalid texts without error
        embeddings = embedder.generate_embeddings(chunks)

        assert len(embeddings) == len(chunks)

        # All embeddings should be valid numpy arrays
        for embedding in embeddings:
            assert embedding is not None
            assert len(embedding) == 768  # all-mpnet-base-v2 dimension

    def test_batch_processing_with_mixed_valid_invalid_texts(self):
        """Test batch processing with mix of valid and invalid texts."""
        config = EmbeddingConfig(show_progress=False, batch_size=2)
        embedder = EmbeddingGenerator(config)
        embedder.load_model()

        # Create many chunks to test batching
        chunks = []
        for i in range(10):
            if i % 3 == 0:
                # Insert invalid text
                text = None if i % 2 == 0 else ""
            else:
                text = f"Valid text chunk number {i}"

            chunks.append(Chunk(id=str(i), text=text))

        # Should process all chunks without error
        embeddings = embedder.generate_embeddings(chunks)

        assert len(embeddings) == len(chunks)

        # All embeddings should be valid
        for embedding in embeddings:
            assert embedding is not None
            assert len(embedding) == 768
