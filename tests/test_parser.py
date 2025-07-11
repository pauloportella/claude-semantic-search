"""
Tests for the JSONL parser module.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from src.parser import Conversation, JSONLParser, Message


class TestJSONLParser:
    """Test suite for JSONLParser."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = JSONLParser()
        self.test_data_dir = Path(__file__).parent.parent / "data" / "test_fixtures"
        self.sample_file = self.test_data_dir / "sample_conversation.jsonl"

    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        assert self.parser.supported_formats == ["claude-conversation-v1"]

    def test_parse_message_basic(self):
        """Test parsing a basic message."""
        data = {
            "uuid": "msg-001",
            "role": "user",
            "content": "Hello, world!",
            "timestamp": "2024-01-15T10:00:00Z",
            "sessionId": "session-123",
        }

        message = self.parser._parse_message(data)

        assert message is not None
        assert message.uuid == "msg-001"
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.parent_uuid is None
        assert message.has_code is False
        assert len(message.tool_calls) == 0
        assert len(message.tool_results) == 0

    def test_parse_message_with_code(self):
        """Test parsing message with code blocks."""
        data = {
            "uuid": "msg-002",
            "role": "assistant",
            "content": "Here's a Python function:\n\n```python\ndef hello():\n    print('Hello!')\n```",
            "timestamp": "2024-01-15T10:00:30Z",
            "parentUuid": "msg-001",
        }

        message = self.parser._parse_message(data)

        assert message is not None
        assert message.has_code is True
        assert message.parent_uuid == "msg-001"
        assert "```python" in message.content

    def test_parse_message_with_tools(self):
        """Test parsing message with tool calls."""
        data = {
            "uuid": "msg-003",
            "role": "assistant",
            "content": "I'll run some code for you.",
            "timestamp": "2024-01-15T10:01:00Z",
            "tool_calls": [{"name": "python", "input": "print('Hello!')"}],
            "tool_results": [{"output": "Hello!"}],
        }

        message = self.parser._parse_message(data)

        assert message is not None
        assert len(message.tool_calls) == 1
        assert len(message.tool_results) == 1
        assert message.tool_calls[0]["name"] == "python"
        assert message.tool_results[0]["output"] == "Hello!"

    def test_extract_content_from_blocks(self):
        """Test extracting content from content blocks."""
        blocks = [{"text": "First block"}, {"text": "Second block"}]

        content = self.parser._extract_from_content_blocks(blocks)
        assert content == "First block\nSecond block"

    def test_extract_content_from_dict(self):
        """Test extracting content from content dictionary."""
        content_dict = {"text": "This is the content"}

        content = self.parser._extract_from_content_dict(content_dict)
        assert content == "This is the content"

    def test_extract_timestamp_iso_format(self):
        """Test extracting timestamp in ISO format."""
        data = {"timestamp": "2024-01-15T10:00:00Z"}

        timestamp = self.parser._extract_timestamp(data)
        assert isinstance(timestamp, datetime)
        assert timestamp.year == 2024
        assert timestamp.month == 1
        assert timestamp.day == 15

    def test_extract_timestamp_milliseconds(self):
        """Test extracting timestamp in milliseconds."""
        data = {"timestamp": 1705312800000}  # 2024-01-15T10:00:00Z in ms

        timestamp = self.parser._extract_timestamp(data)
        assert isinstance(timestamp, datetime)
        assert timestamp.year == 2024

    def test_extract_timestamp_fallback(self):
        """Test timestamp extraction fallback to current time."""
        data = {"no_timestamp": "here"}

        timestamp = self.parser._extract_timestamp(data)
        assert isinstance(timestamp, datetime)
        # Should be close to current time (both should be timezone-aware)
        assert abs((datetime.now(timezone.utc) - timestamp).total_seconds()) < 2

    def test_has_code_blocks(self):
        """Test code block detection."""
        assert self.parser._has_code_blocks("```python\nprint('hello')\n```") is True
        assert self.parser._has_code_blocks("Use `print()` function") is True
        assert self.parser._has_code_blocks("<code>hello</code>") is True
        assert self.parser._has_code_blocks("No code here") is False

    def test_extract_session_id(self):
        """Test session ID extraction."""
        data = {"sessionId": "session-123"}
        assert self.parser._extract_session_id(data) == "session-123"

        data = {"session_id": "session-456"}
        assert self.parser._extract_session_id(data) == "session-456"

        data = {"no_session": "here"}
        assert self.parser._extract_session_id(data) is None

    def test_extract_project_name(self):
        """Test project name extraction from file path."""
        file_path = "/path/to/projects/my_project/conversation.jsonl"
        project_name = self.parser._extract_project_name(file_path)
        assert project_name == "my_project"

        file_path = "/conversation.jsonl"
        project_name = self.parser._extract_project_name(file_path)
        assert project_name == "conversation"

    def test_build_conversation(self):
        """Test building conversation from messages."""
        messages = [
            Message(
                uuid="msg-001",
                content="Hello",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                role="user",
            ),
            Message(
                uuid="msg-002",
                content="Hi there!",
                timestamp=datetime(2024, 1, 15, 10, 0, 30),
                role="assistant",
                parent_uuid="msg-001",
            ),
        ]

        conversation = self.parser._build_conversation(
            messages, "session-123", "/path/to/projects/test_project/conv.jsonl"
        )

        assert conversation.session_id == "session-123"
        assert conversation.project_name == "test_project"
        assert conversation.total_messages == 2
        assert conversation.has_tool_usage is False
        assert conversation.has_code_blocks is False
        assert len(conversation.messages) == 2
        assert (
            conversation.messages[0].uuid == "msg-001"
        )  # Should be sorted by timestamp

    def test_parse_file_success(self):
        """Test parsing a valid JSONL file."""
        # Mock sample conversation data
        sample_data = """{"uuid": "msg-001", "role": "user", "content": "Hello", "timestamp": "2024-01-15T10:00:00Z", "sessionId": "session-123"}
{"uuid": "msg-002", "role": "assistant", "content": "Hi there!", "timestamp": "2024-01-15T10:00:01Z", "sessionId": "session-123"}
{"uuid": "msg-003", "role": "user", "content": "Can you write code?", "timestamp": "2024-01-15T10:00:02Z", "sessionId": "session-123"}
{"uuid": "msg-004", "role": "assistant", "content": [{"type": "text", "text": "Sure, here's an example:"}, {"type": "code", "language": "python", "text": "print('Hello, World!')"}], "timestamp": "2024-01-15T10:00:03Z", "sessionId": "session-123"}
{"uuid": "msg-005", "role": "user", "content": "Thanks!", "timestamp": "2024-01-15T10:00:04Z", "sessionId": "session-123", "toolCalls": [{"id": "tool-1", "name": "test_tool", "arguments": {}}]}"""

        with patch("builtins.open", mock_open(read_data=sample_data)):
            with patch("pathlib.Path.exists", return_value=True):
                conversation = self.parser.parse_file("/fake/sample.jsonl")

                assert conversation is not None
                assert conversation.session_id == "session-123"
                assert conversation.total_messages == 5
                assert conversation.has_code_blocks is True
                assert conversation.has_tool_usage is True
                assert len(conversation.messages) == 5

                # Check message order
                assert conversation.messages[0].role == "user"
                assert conversation.messages[1].role == "assistant"
                assert conversation.messages[3].has_code is True

    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        conversation = self.parser.parse_file("/non/existent/file.jsonl")
        assert conversation is None

    def test_parse_file_invalid_json(self):
        """Test parsing file with invalid JSON."""
        invalid_json = 'invalid json line\n{"uuid": "msg-001", "role": "user", "content": "valid message", "timestamp": "2024-01-15T10:00:00Z"}'

        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with patch("pathlib.Path.exists", return_value=True):
                # This should handle the invalid JSON gracefully and process the valid line
                conversation = self.parser.parse_file("/fake/path.jsonl")
                # Should process the valid JSON line and skip the invalid one
                assert conversation is not None
                assert len(conversation.messages) == 1
                assert conversation.messages[0].uuid == "msg-001"

    def test_scan_directory(self):
        """Test scanning directory for JSONL files."""
        # Mock conversation data
        conv1_data = """{"uuid": "msg-001", "role": "user", "content": "Hello", "timestamp": "2024-01-15T10:00:00Z", "sessionId": "session-123"}"""
        conv2_data = """{"uuid": "msg-002", "role": "assistant", "content": "Hi", "timestamp": "2024-01-15T10:00:01Z", "sessionId": "session-456"}"""
        
        # Mock the rglob to return specific paths
        mock_jsonl_files = [Path("/fake/dir/conv1.jsonl"), Path("/fake/dir/conv2.jsonl")]
        
        def mock_open_side_effect(file_path, *args, **kwargs):
            file_path_str = str(file_path)
            if "conv1.jsonl" in file_path_str:
                return mock_open(read_data=conv1_data)()
            elif "conv2.jsonl" in file_path_str:
                return mock_open(read_data=conv2_data)()
            raise FileNotFoundError(f"No such file: {file_path}")
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.rglob") as mock_rglob:
                # Return mock files for *.jsonl pattern, empty for *.json
                def rglob_side_effect(pattern):
                    if pattern == "*.jsonl":
                        return mock_jsonl_files
                    return []
                
                mock_rglob.side_effect = rglob_side_effect
                
                with patch("builtins.open", side_effect=mock_open_side_effect):
                    conversations = list(self.parser.scan_directory("/fake/dir"))
                    
                    assert len(conversations) == 2
                    assert all(isinstance(conv, Conversation) for conv in conversations)
                    assert any(conv.session_id == "session-123" for conv in conversations)
                    assert any(conv.session_id == "session-456" for conv in conversations)

    def test_scan_directory_not_found(self):
        """Test scanning non-existent directory."""
        with pytest.raises(FileNotFoundError):
            list(self.parser.scan_directory("/non/existent/directory"))

    def test_message_dataclass(self):
        """Test Message dataclass functionality."""
        message = Message(
            uuid="test-uuid",
            content="Test content",
            timestamp=datetime.now(),
            role="user",
        )

        assert message.uuid == "test-uuid"
        assert message.content == "Test content"
        assert message.role == "user"
        assert message.parent_uuid is None
        assert message.tool_calls == []
        assert message.tool_results == []
        assert message.has_code is False
        assert message.raw_data == {}

    def test_conversation_dataclass(self):
        """Test Conversation dataclass functionality."""
        now = datetime.now()
        conversation = Conversation(
            session_id="test-session",
            messages=[],
            project_name="test-project",
            file_path="/test/path.jsonl",
            created_at=now,
            updated_at=now,
        )

        assert conversation.session_id == "test-session"
        assert conversation.project_name == "test-project"
        assert conversation.file_path == "/test/path.jsonl"
        assert conversation.total_messages == 0
        assert conversation.has_tool_usage is False
        assert conversation.has_code_blocks is False
        assert len(conversation.messages) == 0

    def test_extract_timestamp_mixed_timezones(self):
        """Test handling mixed timezone formats (the issue we fixed)."""
        # Test offset-aware ISO format
        data1 = {"timestamp": "2024-01-15T10:00:00+00:00"}
        timestamp1 = self.parser._extract_timestamp(data1)
        assert timestamp1.tzinfo is not None
        assert timestamp1.tzinfo == timezone.utc

        # Test offset-naive ISO format (should be converted to UTC)
        data2 = {"timestamp": "2024-01-15T10:00:00"}
        timestamp2 = self.parser._extract_timestamp(data2)
        assert timestamp2.tzinfo is not None
        assert timestamp2.tzinfo == timezone.utc

        # Test Z suffix format
        data3 = {"timestamp": "2024-01-15T10:00:00Z"}
        timestamp3 = self.parser._extract_timestamp(data3)
        assert timestamp3.tzinfo is not None
        assert timestamp3.tzinfo == timezone.utc

        # All should be comparable without error
        assert timestamp1 == timestamp2  # Same time, both UTC
        assert timestamp1 == timestamp3  # Same time, both UTC

    def test_extract_timestamp_various_formats(self):
        """Test various timestamp formats found in Claude conversations."""
        # Test different field names
        for field in ["timestamp", "created_at", "createdAt", "time"]:
            data = {field: "2024-01-15T10:00:00Z"}
            timestamp = self.parser._extract_timestamp(data)
            assert timestamp.tzinfo == timezone.utc
            assert timestamp.year == 2024

        # Test milliseconds as integer
        data = {"timestamp": 1705312800000}
        timestamp = self.parser._extract_timestamp(data)
        assert timestamp.tzinfo == timezone.utc

        # Test seconds as integer
        data = {"timestamp": 1705312800}
        timestamp = self.parser._extract_timestamp(data)
        assert timestamp.tzinfo == timezone.utc

        # Test with timezone offset
        data = {"timestamp": "2024-01-15T10:00:00-05:00"}
        timestamp = self.parser._extract_timestamp(data)
        assert timestamp.tzinfo is not None
        # The timestamp preserves its original timezone
        assert timestamp.hour == 10  # Original hour is preserved
        # But it should be comparable with UTC times
        utc_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        assert timestamp == utc_time  # Same moment in time

    def test_build_conversation_with_mixed_timezones(self):
        """Test building conversation with messages having different timezone formats."""
        messages = [
            Message(
                uuid="msg-1",
                content="First message",
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                role="user",
            ),
            Message(
                uuid="msg-2",
                content="Second message",
                timestamp=datetime(2024, 1, 15, 10, 1, 0, tzinfo=timezone.utc),
                role="assistant",
            ),
            Message(
                uuid="msg-3",
                content="Third message",
                timestamp=datetime(2024, 1, 15, 10, 2, 0, tzinfo=timezone.utc),
                role="user",
            ),
        ]

        # This should not raise any timezone comparison errors
        conversation = self.parser._build_conversation(
            messages, "test-session", "/test/path.jsonl"
        )

        assert conversation is not None
        assert len(conversation.messages) == 3
        # Messages should be sorted by timestamp
        assert conversation.messages[0].uuid == "msg-1"
        assert conversation.messages[1].uuid == "msg-2"
        assert conversation.messages[2].uuid == "msg-3"

    def test_parse_real_world_timestamps(self):
        """Test parsing real-world timestamp formats from Claude conversations."""
        # Test data mimicking actual Claude conversation format
        test_cases = [
            # Format from the error message
            {"createdAt": "2024-07-01T21:34:26.262000+00:00"},
            {"createdAt": "2024-07-01T21:34:26.262Z"},
            {"createdAt": "2024-07-01T21:34:26"},
            {"timestamp": "2025-01-01T00:00:00Z"},
            {"timestamp": "2025-01-01T00:00:00+00:00"},
            {"created_at": 1705312800000},  # milliseconds
            {"time": 1705312800},  # seconds
        ]

        for data in test_cases:
            timestamp = self.parser._extract_timestamp(data)
            assert timestamp.tzinfo is not None, f"Failed for data: {data}"
            # All timestamps should be timezone-aware
            # This ensures they can be compared without errors

    def test_parse_file_with_mixed_timestamp_formats(self):
        """Test parsing a file with messages containing different timestamp formats."""
        jsonl_content = "\n".join(
            [
                json.dumps(
                    {
                        "uuid": "msg-1",
                        "role": "user",
                        "content": "First message",
                        "timestamp": "2024-01-15T10:00:00Z",
                        "sessionId": "test-session",
                    }
                ),
                json.dumps(
                    {
                        "uuid": "msg-2",
                        "role": "assistant",
                        "content": "Second message",
                        "createdAt": "2024-01-15T10:01:00+00:00",
                        "sessionId": "test-session",
                    }
                ),
                json.dumps(
                    {
                        "uuid": "msg-3",
                        "role": "user",
                        "content": "Third message",
                        "timestamp": 1705312920000,  # milliseconds
                        "sessionId": "test-session",
                    }
                ),
            ]
        )

        with patch("builtins.open", mock_open(read_data=jsonl_content)):
            with patch("pathlib.Path.exists", return_value=True):
                conversation = self.parser.parse_file("/fake/path.jsonl")

                assert conversation is not None
                assert len(conversation.messages) == 3
                # All messages should have been parsed successfully
                # and sorting by timestamp should work without errors

    def test_extract_content_edge_cases(self):
        """Test content extraction edge cases."""
        # Test with nested list content
        data = {"content": [{"text": "Part 1"}, "Part 2"]}
        content = self.parser._extract_content(data)
        assert "Part 1" in content
        assert "Part 2" in content

        # Test with nested dict content
        data = {"content": {"message": "Nested message"}}
        content = self.parser._extract_content(data)
        assert content == "Nested message"

        # Test fallback to str() for unknown content structure
        data = {"content": {"unknown": "structure"}}
        content = self.parser._extract_content(data)
        assert "unknown" in content

        # Test empty content extraction
        data = {"no_content": "here"}
        content = self.parser._extract_content(data)
        assert content == ""

    def test_timestamp_parsing_errors(self):
        """Test timestamp parsing error handling."""
        # Test invalid timestamp string
        data = {"timestamp": "invalid-timestamp-format"}
        timestamp = self.parser._extract_timestamp(data)
        # Should fallback to current time
        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo == timezone.utc

        # Test invalid millisecond value
        data = {"timestamp": "not-a-number"}
        timestamp = self.parser._extract_timestamp(data)
        assert isinstance(timestamp, datetime)
        assert timestamp.tzinfo == timezone.utc

    def test_parse_message_error_handling(self):
        """Test message parsing error handling."""
        # Test with data that causes an exception (non-dict type)
        data = "this_will_cause_error"
        message = self.parser._parse_message(data)
        # Should return None on error
        assert message is None

    def test_scan_directory_error_handling(self):
        """Test directory scanning error handling."""
        # Create a mock directory with a file that causes parsing error
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True

            with patch("pathlib.Path.rglob") as mock_rglob:
                mock_rglob.return_value = [Path("/fake/error.jsonl")]

                with patch.object(self.parser, "parse_file") as mock_parse:
                    # First call succeeds, second raises exception
                    mock_parse.side_effect = [None, Exception("Parse error")]

                    conversations = list(self.parser.scan_directory("/fake/dir"))
                    # Should handle the error gracefully
                    assert len(conversations) == 0
