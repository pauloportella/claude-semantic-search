"""
JSONL Parser for Claude conversations.

This module parses JSONL files containing Claude conversation data and extracts
structured conversation objects with metadata.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field


@dataclass
class Message:
    """Represents a single message in a conversation."""
    uuid: str
    content: str
    timestamp: datetime
    role: str  # 'user' or 'assistant'
    parent_uuid: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    has_code: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Represents a complete conversation thread."""
    session_id: str
    messages: List[Message]
    project_name: str
    file_path: str
    created_at: datetime
    updated_at: datetime
    total_messages: int = 0
    has_tool_usage: bool = False
    has_code_blocks: bool = False
    

class JSONLParser:
    """Parser for Claude conversation JSONL files."""
    
    def __init__(self):
        self.supported_formats = ['claude-conversation-v1']
    
    def parse_file(self, file_path: str) -> Optional[Conversation]:
        """Parse a single JSONL file and return a Conversation object."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            messages = []
            session_id = None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        message = self._parse_message(data)
                        if message:
                            messages.append(message)
                            if session_id is None:
                                session_id = self._extract_session_id(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                        continue
            
            if not messages:
                return None
            
            return self._build_conversation(messages, session_id, file_path)
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _parse_message(self, data: Dict[str, Any]) -> Optional[Message]:
        """Parse a single message from JSONL data."""
        try:
            # Extract basic message info
            uuid = data.get('uuid', '')
            content = self._extract_content(data)
            timestamp = self._extract_timestamp(data)
            role = data.get('role', 'unknown')
            parent_uuid = data.get('parentUuid')
            
            # Extract tool information
            tool_calls = self._extract_tool_calls(data)
            tool_results = self._extract_tool_results(data)
            
            # Check for code blocks
            has_code = self._has_code_blocks(content)
            
            return Message(
                uuid=uuid,
                content=content,
                timestamp=timestamp,
                role=role,
                parent_uuid=parent_uuid,
                tool_calls=tool_calls,
                tool_results=tool_results,
                has_code=has_code,
                raw_data=data
            )
        
        except Exception as e:
            print(f"Error parsing message: {e}")
            return None
    
    def _extract_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from message data."""
        # Try different possible content fields
        content_fields = ['content', 'text', 'message', 'body']
        
        for field in content_fields:
            if field in data:
                content = data[field]
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle content as array of blocks
                    return self._extract_from_content_blocks(content)
                elif isinstance(content, dict):
                    # Handle nested content structure
                    return self._extract_from_content_dict(content)
        
        return ""
    
    def _extract_from_content_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Extract text from content blocks array."""
        text_parts = []
        for block in blocks:
            if isinstance(block, dict):
                if 'text' in block:
                    text_parts.append(block['text'])
                elif 'content' in block:
                    text_parts.append(str(block['content']))
            elif isinstance(block, str):
                text_parts.append(block)
        return '\n'.join(text_parts)
    
    def _extract_from_content_dict(self, content: Dict[str, Any]) -> str:
        """Extract text from nested content dictionary."""
        if 'text' in content:
            return content['text']
        elif 'message' in content:
            return content['message']
        else:
            return str(content)
    
    def _extract_timestamp(self, data: Dict[str, Any]) -> datetime:
        """Extract timestamp from message data."""
        timestamp_fields = ['timestamp', 'created_at', 'createdAt', 'time']
        
        for field in timestamp_fields:
            if field in data:
                timestamp = data[field]
                if isinstance(timestamp, str):
                    try:
                        # Try ISO format first
                        return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        try:
                            # Try parsing as milliseconds
                            return datetime.fromtimestamp(int(timestamp) / 1000)
                        except (ValueError, TypeError):
                            continue
                elif isinstance(timestamp, (int, float)):
                    try:
                        # Assume milliseconds if > 1e10, otherwise seconds
                        if timestamp > 1e10:
                            return datetime.fromtimestamp(timestamp / 1000)
                        else:
                            return datetime.fromtimestamp(timestamp)
                    except (ValueError, TypeError):
                        continue
        
        # Default to current time if no timestamp found
        return datetime.now()
    
    def _extract_tool_calls(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from message data."""
        tool_calls = []
        
        # Check for tool_calls field
        if 'tool_calls' in data:
            tool_calls.extend(data['tool_calls'])
        
        # Check for function calls in content
        if 'function_call' in data:
            tool_calls.append(data['function_call'])
        
        return tool_calls
    
    def _extract_tool_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool results from message data."""
        tool_results = []
        
        # Check for tool_results field
        if 'tool_results' in data:
            tool_results.extend(data['tool_results'])
        
        # Check for function results
        if 'function_result' in data:
            tool_results.append(data['function_result'])
        
        return tool_results
    
    def _has_code_blocks(self, content: str) -> bool:
        """Check if content contains code blocks."""
        return '```' in content or '<code>' in content or '`' in content
    
    def _extract_session_id(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract session ID from message data."""
        session_fields = ['sessionId', 'session_id', 'conversation_id', 'thread_id']
        
        for field in session_fields:
            if field in data:
                return str(data[field])
        
        return None
    
    def _build_conversation(self, messages: List[Message], session_id: Optional[str], 
                           file_path: str) -> Conversation:
        """Build a Conversation object from parsed messages."""
        # Sort messages by timestamp
        messages.sort(key=lambda m: m.timestamp)
        
        # Extract metadata
        project_name = self._extract_project_name(file_path)
        created_at = messages[0].timestamp if messages else datetime.now()
        updated_at = messages[-1].timestamp if messages else datetime.now()
        
        # Calculate statistics
        total_messages = len(messages)
        has_tool_usage = any(m.tool_calls or m.tool_results for m in messages)
        has_code_blocks = any(m.has_code for m in messages)
        
        return Conversation(
            session_id=session_id or f"session_{created_at.isoformat()}",
            messages=messages,
            project_name=project_name,
            file_path=file_path,
            created_at=created_at,
            updated_at=updated_at,
            total_messages=total_messages,
            has_tool_usage=has_tool_usage,
            has_code_blocks=has_code_blocks
        )
    
    def _extract_project_name(self, file_path: str) -> str:
        """Extract project name from file path."""
        path = Path(file_path)
        
        # Look for project name in path components
        parts = path.parts
        if len(parts) >= 2 and parts[-2] != '/':
            # Assume project name is the parent directory name
            return parts[-2]
        
        # Fallback to filename without extension
        return path.stem
    
    def scan_directory(self, directory: str) -> Generator[Conversation, None, None]:
        """Scan directory for JSONL files and yield Conversation objects."""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all JSONL files
        jsonl_files = []
        for pattern in ['*.jsonl', '*.json']:
            jsonl_files.extend(directory.rglob(pattern))
        
        for file_path in jsonl_files:
            try:
                conversation = self.parse_file(str(file_path))
                if conversation:
                    yield conversation
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue