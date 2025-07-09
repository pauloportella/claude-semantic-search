"""
Tests for the JSONL parser module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open

from src.parser import JSONLParser, Message, Conversation


class TestJSONLParser:
    """Test suite for JSONLParser."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.parser = JSONLParser()
        self.test_data_dir = Path(__file__).parent.parent / "data" / "test_fixtures"
        self.sample_file = self.test_data_dir / "sample_conversation.jsonl"
    
    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        assert self.parser.supported_formats == ['claude-conversation-v1']
    
    def test_parse_message_basic(self):
        """Test parsing a basic message."""
        data = {
            "uuid": "msg-001",
            "role": "user",
            "content": "Hello, world!",
            "timestamp": "2024-01-15T10:00:00Z",
            "sessionId": "session-123"
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
            "parentUuid": "msg-001"
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
            "tool_calls": [
                {"name": "python", "input": "print('Hello!')"}
            ],
            "tool_results": [
                {"output": "Hello!"}
            ]
        }
        
        message = self.parser._parse_message(data)
        
        assert message is not None
        assert len(message.tool_calls) == 1
        assert len(message.tool_results) == 1
        assert message.tool_calls[0]["name"] == "python"
        assert message.tool_results[0]["output"] == "Hello!"
    
    def test_extract_content_from_blocks(self):
        """Test extracting content from content blocks."""
        blocks = [
            {"text": "First block"},
            {"text": "Second block"}
        ]
        
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
        # Should be close to current time
        assert abs((datetime.now() - timestamp).total_seconds()) < 2
    
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
                role="user"
            ),
            Message(
                uuid="msg-002",
                content="Hi there!",
                timestamp=datetime(2024, 1, 15, 10, 0, 30),
                role="assistant",
                parent_uuid="msg-001"
            )
        ]
        
        conversation = self.parser._build_conversation(
            messages, 
            "session-123", 
            "/path/to/projects/test_project/conv.jsonl"
        )
        
        assert conversation.session_id == "session-123"
        assert conversation.project_name == "test_project"
        assert conversation.total_messages == 2
        assert conversation.has_tool_usage is False
        assert conversation.has_code_blocks is False
        assert len(conversation.messages) == 2
        assert conversation.messages[0].uuid == "msg-001"  # Should be sorted by timestamp
    
    def test_parse_file_success(self):
        """Test parsing a valid JSONL file."""
        # Ensure the sample file exists
        assert self.sample_file.exists(), f"Sample file not found: {self.sample_file}"
        
        conversation = self.parser.parse_file(str(self.sample_file))
        
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
        invalid_json = "invalid json line\n{\"uuid\": \"msg-001\", \"role\": \"user\", \"content\": \"valid message\", \"timestamp\": \"2024-01-15T10:00:00Z\"}"
        
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
        # Create a generator and convert to list
        conversations = list(self.parser.scan_directory(str(self.test_data_dir)))
        
        assert len(conversations) >= 1
        assert all(isinstance(conv, Conversation) for conv in conversations)
        assert any(conv.session_id == "session-123" for conv in conversations)
    
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
            role="user"
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
            updated_at=now
        )
        
        assert conversation.session_id == "test-session"
        assert conversation.project_name == "test-project"
        assert conversation.file_path == "/test/path.jsonl"
        assert conversation.total_messages == 0
        assert conversation.has_tool_usage is False
        assert conversation.has_code_blocks is False
        assert len(conversation.messages) == 0