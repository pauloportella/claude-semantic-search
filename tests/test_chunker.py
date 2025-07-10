"""
Tests for the chunker module.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from src.chunker import Chunk, ChunkingConfig, ConversationChunker
from src.parser import Conversation, Message


class TestConversationChunker:
    """Test suite for ConversationChunker."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = ChunkingConfig(
            max_chunk_size=1000, context_window=2, min_chunk_size=50
        )
        self.chunker = ConversationChunker(self.config)

    def create_test_conversation(self) -> Conversation:
        """Create a test conversation with various message types."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        messages = [
            Message(
                uuid="msg-001",
                content="Hello, can you help me with Python?",
                timestamp=base_time,
                role="user",
            ),
            Message(
                uuid="msg-002",
                content="Of course! I'd be happy to help you with Python. What specific topic would you like assistance with?",
                timestamp=base_time + timedelta(seconds=30),
                role="assistant",
            ),
            Message(
                uuid="msg-003",
                content="I need to write a function that sorts a list of dictionaries",
                timestamp=base_time + timedelta(minutes=1),
                role="user",
            ),
            Message(
                uuid="msg-004",
                content="Here's a Python function for that:\n\n```python\ndef sort_dicts(data, key):\n    return sorted(data, key=lambda x: x[key])\n\n# Example usage:\npeople = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]\nsorted_people = sort_dicts(people, 'age')\nprint(sorted_people)\n```\n\nThis function sorts the list by the specified key.",
                timestamp=base_time + timedelta(minutes=1, seconds=30),
                role="assistant",
                has_code=True,
                tool_calls=[
                    {
                        "name": "python",
                        "input": "def sort_dicts(data, key):\n    return sorted(data, key=lambda x: x[key])",
                    }
                ],
            ),
            Message(
                uuid="msg-005",
                content="Perfect! Thank you for the explanation.",
                timestamp=base_time + timedelta(minutes=2),
                role="user",
            ),
        ]

        return Conversation(
            session_id="test-session",
            messages=messages,
            project_name="test-project",
            file_path="/test/path.jsonl",
            created_at=base_time,
            updated_at=base_time + timedelta(minutes=2),
            total_messages=len(messages),
            has_tool_usage=True,
            has_code_blocks=True,
        )

    def test_chunking_config_defaults(self):
        """Test ChunkingConfig default values."""
        config = ChunkingConfig()
        assert config.max_chunk_size == 2000
        assert config.context_window == 2
        assert config.overlap_size == 200
        assert config.min_chunk_size == 100
        assert config.include_tool_results is True
        assert config.preserve_context is True

    def test_chunker_initialization(self):
        """Test chunker initializes correctly."""
        chunker = ConversationChunker()
        assert chunker.config.max_chunk_size == 2000
        assert chunker.chunk_counter == 0

        custom_config = ChunkingConfig(max_chunk_size=500)
        chunker_custom = ConversationChunker(custom_config)
        assert chunker_custom.config.max_chunk_size == 500

    def test_chunk_conversation_basic(self):
        """Test basic conversation chunking."""
        conversation = self.create_test_conversation()
        chunks = self.chunker.chunk_conversation(conversation)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.id.startswith("chunk_") for chunk in chunks)
        assert all(len(chunk.text) >= self.config.min_chunk_size for chunk in chunks)

    def test_create_qa_chunks(self):
        """Test Q&A chunk creation."""
        conversation = self.create_test_conversation()
        chunks = self.chunker._create_qa_chunks(conversation)

        assert len(chunks) >= 2  # At least 2 Q&A pairs

        # Check first chunk
        first_chunk = chunks[0]
        assert "User: Hello, can you help me with Python?" in first_chunk.text
        assert "Assistant: Of course!" in first_chunk.text
        assert first_chunk.metadata["chunk_type"] == "qa_pair"
        assert first_chunk.metadata["message_count"] == 2
        assert first_chunk.metadata["has_code"] is False

        # Check code chunk
        code_chunk = next((c for c in chunks if c.metadata.get("has_code")), None)
        assert code_chunk is not None
        assert "```python" in code_chunk.text
        assert code_chunk.metadata["has_tools"] is True

    def test_create_code_chunks(self):
        """Test code-focused chunk creation."""
        conversation = self.create_test_conversation()
        chunks = self.chunker._create_code_chunks(conversation)

        # Should have at least one code chunk
        assert len(chunks) >= 1

        code_chunk = chunks[0]
        assert code_chunk.metadata["chunk_type"] == "code_block"
        assert "language" in code_chunk.metadata
        assert "code_lines" in code_chunk.metadata
        assert "```python" in code_chunk.text
        assert "def sort_dicts" in code_chunk.text

    def test_create_tool_chunks(self):
        """Test tool usage chunk creation."""
        conversation = self.create_test_conversation()
        chunks = self.chunker._create_tool_chunks(conversation)

        assert len(chunks) >= 1

        tool_chunk = chunks[0]
        assert tool_chunk.metadata["chunk_type"] == "tool_usage"
        assert "tools_used" in tool_chunk.metadata
        assert "has_results" in tool_chunk.metadata
        assert "python" in tool_chunk.metadata["tools_used"]

    def test_format_qa_pair(self):
        """Test Q&A pair formatting."""
        user_msg = Message(
            uuid="u1",
            content="What is Python?",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            role="user",
        )

        assistant_msg = Message(
            uuid="a1",
            content="Python is a programming language.",
            timestamp=datetime(2024, 1, 15, 10, 0, 30),
            role="assistant",
        )

        formatted = self.chunker._format_qa_pair(user_msg, assistant_msg)

        assert "[2024-01-15 10:00] User: What is Python?" in formatted
        assert "Assistant: Python is a programming language." in formatted

    def test_extract_code_blocks(self):
        """Test code block extraction."""
        content = """Here's some code:

```python
def hello():
    print("Hello, world!")
```

And some inline code: `print("test")` here.

Another block:
```javascript
console.log("JS code");
```"""

        code_blocks = self.chunker._extract_code_blocks(content)

        assert len(code_blocks) >= 2  # At least 2 fenced blocks

        python_block = next((b for b in code_blocks if b["language"] == "python"), None)
        assert python_block is not None
        assert "def hello():" in python_block["code"]

        js_block = next((b for b in code_blocks if b["language"] == "javascript"), None)
        assert js_block is not None
        assert "console.log" in js_block["code"]

    def test_get_context(self):
        """Test context extraction."""
        conversation = self.create_test_conversation()
        messages = conversation.messages

        # Test context from middle of conversation
        context = self.chunker._get_context(messages, 3, 2)

        assert "[Context]" in context
        assert "User:" in context
        assert "Assistant:" in context

        # Test no context at beginning
        context = self.chunker._get_context(messages, 0, 2)
        assert context == ""

    def test_identify_context_segments(self):
        """Test context segment identification."""
        # Create conversation with time gaps
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        messages = [
            Message(uuid="1", content="First topic", timestamp=base_time, role="user"),
            Message(
                uuid="2",
                content="Response 1",
                timestamp=base_time + timedelta(seconds=30),
                role="assistant",
            ),
            Message(
                uuid="3",
                content="Follow up",
                timestamp=base_time + timedelta(minutes=1),
                role="user",
            ),
            Message(
                uuid="4",
                content="Response 2",
                timestamp=base_time + timedelta(minutes=1, seconds=30),
                role="assistant",
            ),
            # Large time gap
            Message(
                uuid="5",
                content="Now let's discuss something different",
                timestamp=base_time + timedelta(hours=1),
                role="user",
            ),
            Message(
                uuid="6",
                content="New topic response",
                timestamp=base_time + timedelta(hours=1, seconds=30),
                role="assistant",
            ),
        ]

        segments = self.chunker._identify_context_segments(messages)

        # Should identify segments based on time gaps
        assert len(segments) >= 1
        assert all(isinstance(seg, tuple) and len(seg) == 2 for seg in segments)

    def test_is_segment_boundary(self):
        """Test segment boundary detection."""
        conversation = self.create_test_conversation()
        messages = conversation.messages

        # First message is always a boundary
        assert self.chunker._is_segment_boundary(messages, 0) is True

        # Normal conversation flow should not be boundary
        assert self.chunker._is_segment_boundary(messages, 1) is False

        # Test with time gap
        messages[2].timestamp = messages[1].timestamp + timedelta(hours=1)
        assert self.chunker._is_segment_boundary(messages, 2) is True

    def test_split_large_chunk(self):
        """Test splitting of large chunks."""
        user_msg = Message(
            uuid="u1", content="Long question", timestamp=datetime.now(), role="user"
        )

        assistant_msg = Message(
            uuid="a1",
            content="Very long response that exceeds the maximum chunk size limit. "
            * 50,
            timestamp=datetime.now(),
            role="assistant",
        )

        # Create text that exceeds max_chunk_size and can be split by words
        long_text = (
            "word " * 250
        )  # 250 * 5 = 1250 chars, exceeds max_chunk_size of 1000

        chunks = self.chunker._split_large_chunk(long_text, user_msg, assistant_msg)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= self.config.max_chunk_size for chunk in chunks)
        assert all(chunk.metadata["chunk_type"] == "qa_pair_split" for chunk in chunks)

    def test_create_chunk_metadata(self):
        """Test chunk metadata creation."""
        conversation = self.create_test_conversation()
        message = conversation.messages[0]

        chunk = self.chunker._create_chunk(
            "Test chunk text",
            "test_type",
            conversation,
            [message],
            {"extra": "metadata"},
        )

        assert chunk.id.startswith("chunk_")
        assert chunk.text == "Test chunk text"
        assert chunk.metadata["chunk_type"] == "test_type"
        assert chunk.metadata["message_count"] == 1
        assert chunk.metadata["session_id"] == "test-session"
        assert chunk.metadata["project_name"] == "test-project"
        assert chunk.metadata["extra"] == "metadata"
        assert "char_count" in chunk.metadata
        assert "word_count" in chunk.metadata

    def test_deduplicate_chunks(self):
        """Test chunk deduplication."""
        # Create duplicate chunks
        chunk1 = Chunk(id="1", text="Same text", metadata={})
        chunk2 = Chunk(id="2", text="Same text", metadata={})
        chunk3 = Chunk(id="3", text="Different text", metadata={})

        chunks = [chunk1, chunk2, chunk3]
        unique_chunks = self.chunker._deduplicate_chunks(chunks)

        assert len(unique_chunks) == 2
        assert any(chunk.text == "Same text" for chunk in unique_chunks)
        assert any(chunk.text == "Different text" for chunk in unique_chunks)

    def test_get_chunk_stats(self):
        """Test chunk statistics generation."""
        conversation = self.create_test_conversation()
        chunks = self.chunker.chunk_conversation(conversation)

        stats = self.chunker.get_chunk_stats(chunks)

        assert "total_chunks" in stats
        assert "chunk_types" in stats
        assert "total_characters" in stats
        assert "total_words" in stats
        assert "avg_chunk_size" in stats
        assert "avg_words_per_chunk" in stats

        assert stats["total_chunks"] == len(chunks)
        assert stats["total_characters"] > 0
        assert stats["avg_chunk_size"] > 0

    def test_empty_conversation(self):
        """Test handling of empty conversation."""
        empty_conversation = Conversation(
            session_id="empty",
            messages=[],
            project_name="test",
            file_path="/test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        chunks = self.chunker.chunk_conversation(empty_conversation)
        assert len(chunks) == 0

        stats = self.chunker.get_chunk_stats(chunks)
        assert stats == {}

    def test_single_message_conversation(self):
        """Test conversation with single message."""
        single_msg_conversation = Conversation(
            session_id="single",
            messages=[
                Message(
                    uuid="solo",
                    content="Single message",
                    timestamp=datetime.now(),
                    role="user",
                )
            ],
            project_name="test",
            file_path="/test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        chunks = self.chunker.chunk_conversation(single_msg_conversation)

        # Should handle gracefully (no Q&A pairs possible)
        assert len(chunks) == 0 or all(
            chunk.metadata["chunk_type"] != "qa_pair" for chunk in chunks
        )

    def test_context_preservation_disabled(self):
        """Test chunking with context preservation disabled."""
        config = ChunkingConfig(preserve_context=False)
        chunker = ConversationChunker(config)

        conversation = self.create_test_conversation()
        chunks = chunker._create_qa_chunks(conversation)

        # Chunks should not contain context
        for chunk in chunks:
            assert "[Context]" not in chunk.text

    def test_tool_results_disabled(self):
        """Test chunking with tool results disabled."""
        config = ChunkingConfig(include_tool_results=False)
        chunker = ConversationChunker(config)

        # Add tool results to test message
        msg_with_tools = Message(
            uuid="tool-msg",
            content="Using tools",
            timestamp=datetime.now(),
            role="assistant",
            tool_calls=[{"name": "test"}],
            tool_results=[{"output": "result"}],
        )

        chunk_text = chunker._format_tool_chunk(msg_with_tools)

        # Should not include tool results
        assert "Tool Results:" not in chunk_text
        assert "Tool Calls:" in chunk_text  # But calls should be included

    def test_min_chunk_size_filtering(self):
        """Test that chunks below minimum size are filtered out."""
        config = ChunkingConfig(min_chunk_size=100)
        chunker = ConversationChunker(config)

        # Create conversation with very short messages
        short_conversation = Conversation(
            session_id="short",
            messages=[
                Message(uuid="1", content="Hi", timestamp=datetime.now(), role="user"),
                Message(
                    uuid="2",
                    content="Hello",
                    timestamp=datetime.now(),
                    role="assistant",
                ),
            ],
            project_name="test",
            file_path="/test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        chunks = chunker._create_qa_chunks(short_conversation)

        # Very short Q&A should be filtered out
        assert len(chunks) == 0 or all(
            len(chunk.text) >= config.min_chunk_size for chunk in chunks
        )
