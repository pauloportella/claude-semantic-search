"""
Smart chunking strategies for Claude conversations.

This module provides various chunking strategies to break down conversations
into semantic units optimized for embedding and search.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .parser import Conversation, Message


@dataclass
class Chunk:
    """Represents a semantic chunk for embedding."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    max_chunk_size: int = 2000  # Maximum characters per chunk
    context_window: int = 2  # Number of previous messages to include
    overlap_size: int = 200  # Character overlap between chunks
    min_chunk_size: int = 100  # Minimum characters per chunk
    code_block_threshold: int = 5  # Minimum lines for separate code chunks
    include_tool_results: bool = True
    preserve_context: bool = True
    

class ConversationChunker:
    """Creates semantic chunks from conversations."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.chunk_counter = 0
    
    def chunk_conversation(self, conversation: Conversation) -> List[Chunk]:
        """Create chunks from a conversation using multiple strategies."""
        chunks = []
        
        # Strategy 1: Question-Answer pairs
        qa_chunks = self._create_qa_chunks(conversation)
        chunks.extend(qa_chunks)
        
        # Strategy 2: Extended context chunks for complex discussions
        context_chunks = self._create_context_chunks(conversation)
        chunks.extend(context_chunks)
        
        # Strategy 3: Code-focused chunks
        code_chunks = self._create_code_chunks(conversation)
        chunks.extend(code_chunks)
        
        # Strategy 4: Tool usage chunks
        tool_chunks = self._create_tool_chunks(conversation)
        chunks.extend(tool_chunks)
        
        return self._deduplicate_chunks(chunks)
    
    def _filter_messages(self, messages: List[Message]) -> List[Message]:
        """Filter out unwanted messages from conversation."""
        filtered = []
        
        for message in messages:
            # Skip messages with unknown role (system messages)
            if message.role == 'unknown':
                continue
            
            # Skip Claude Code hook-related messages
            if self._is_hook_message(message):
                continue
            
            # Skip system tool messages
            if self._is_system_tool_message(message):
                continue
            
            filtered.append(message)
        
        return filtered
    
    def _is_hook_message(self, message: Message) -> bool:
        """Check if message is related to Claude Code hooks."""
        content = message.content.lower()
        
        # Official Claude Code hook events
        hook_events = [
            'pretooluse',
            'posttooluse', 
            'notification',
            'stop',
            'subagentstop'
        ]
        
        # Hook-related file paths and patterns
        hook_patterns = [
            '.claude/hooks/',
            'hook:',
            'hooks.json',
            'claude code hook'
        ]
        
        # Tool matchers that hooks commonly use
        tool_matchers = [
            'task', 'bash', 'glob', 'grep', 'read', 'edit', 
            'write', 'webfetch', 'websearch'
        ]
        
        # Check for any hook indicators
        all_indicators = hook_events + hook_patterns
        
        return any(indicator in content for indicator in all_indicators)
    
    def _is_system_tool_message(self, message: Message) -> bool:
        """Check if message is a system tool operation message."""
        content = message.content.lower()
        
        # Check for system tool patterns
        system_patterns = [
            'pretooluse:',
            'posttooluse:',
            'completed successfully:',
            'tool use:',
            'system:'
        ]
        
        return any(pattern in content for pattern in system_patterns)
    
    def _create_qa_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Create chunks from question-answer pairs."""
        chunks = []
        messages = self._filter_messages(conversation.messages)
        
        for i in range(len(messages) - 1):
            if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                # Create chunk text
                chunk_text = self._format_qa_pair(user_msg, assistant_msg)
                
                # Add context if enabled
                if self.config.preserve_context:
                    context = self._get_context(messages, i, self.config.context_window)
                    if context:
                        chunk_text = f"{context}\n\n{chunk_text}"
                
                # Check size limits
                if len(chunk_text) > self.config.max_chunk_size:
                    # Split large chunks
                    sub_chunks = self._split_large_chunk(chunk_text, user_msg, assistant_msg)
                    chunks.extend(sub_chunks)
                elif len(chunk_text) >= self.config.min_chunk_size:
                    chunk = self._create_chunk(
                        chunk_text,
                        chunk_type="qa_pair",
                        conversation=conversation,
                        messages=[user_msg, assistant_msg]
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _create_context_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Create chunks with extended context for complex discussions."""
        chunks = []
        messages = self._filter_messages(conversation.messages)
        
        # Find conversation segments that benefit from extended context
        segments = self._identify_context_segments(messages)
        
        for segment in segments:
            start_idx, end_idx = segment
            segment_messages = messages[start_idx:end_idx + 1]
            
            chunk_text = self._format_message_sequence(segment_messages)
            
            if len(chunk_text) <= self.config.max_chunk_size and len(chunk_text) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_text,
                    chunk_type="context_segment",
                    conversation=conversation,
                    messages=segment_messages
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_code_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Create chunks focused on code blocks and technical content."""
        chunks = []
        
        for message in self._filter_messages(conversation.messages):
            if message.has_code:
                code_blocks = self._extract_code_blocks(message.content)
                
                for code_block in code_blocks:
                    if len(code_block['code'].split('\n')) >= self.config.code_block_threshold:
                        # Create chunk with code and surrounding context
                        chunk_text = self._format_code_chunk(message, code_block)
                        
                        chunk = self._create_chunk(
                            chunk_text,
                            chunk_type="code_block",
                            conversation=conversation,
                            messages=[message],
                            extra_metadata={
                                "language": code_block.get('language', 'unknown'),
                                "code_lines": len(code_block['code'].split('\n'))
                            }
                        )
                        chunks.append(chunk)
        
        return chunks
    
    def _create_tool_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Create chunks focused on tool usage and results."""
        chunks = []
        
        for message in self._filter_messages(conversation.messages):
            if message.tool_calls or message.tool_results:
                chunk_text = self._format_tool_chunk(message)
                
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunk = self._create_chunk(
                        chunk_text,
                        chunk_type="tool_usage",
                        conversation=conversation,
                        messages=[message],
                        extra_metadata={
                            "tools_used": [tool.get('name', 'unknown') for tool in message.tool_calls],
                            "has_results": bool(message.tool_results)
                        }
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _format_qa_pair(self, user_msg: Message, assistant_msg: Message) -> str:
        """Format a question-answer pair into chunk text."""
        timestamp = user_msg.timestamp.strftime("%Y-%m-%d %H:%M")
        
        text = f"[{timestamp}] User: {user_msg.content}\n\n"
        text += f"Assistant: {assistant_msg.content}"
        
        return text
    
    def _format_message_sequence(self, messages: List[Message]) -> str:
        """Format a sequence of messages into chunk text."""
        parts = []
        
        for msg in messages:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            parts.append(f"[{timestamp}] {msg.role.title()}: {msg.content}")
        
        return "\n\n".join(parts)
    
    def _format_code_chunk(self, message: Message, code_block: Dict[str, Any]) -> str:
        """Format a code block with surrounding context."""
        timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M")
        
        # Extract context around the code block
        content = message.content
        code_start = content.find(code_block['raw'])
        
        # Get text before and after code block
        before_text = content[:code_start].strip()
        after_text = content[code_start + len(code_block['raw']):].strip()
        
        text = f"[{timestamp}] {message.role.title()}:\n"
        
        if before_text:
            text += f"{before_text}\n\n"
        
        text += f"```{code_block.get('language', '')}\n{code_block['code']}\n```"
        
        if after_text:
            text += f"\n\n{after_text}"
        
        return text
    
    def _format_tool_chunk(self, message: Message) -> str:
        """Format tool usage and results into chunk text."""
        timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M")
        
        text = f"[{timestamp}] {message.role.title()}: {message.content}\n\n"
        
        if message.tool_calls:
            text += "Tool Calls:\n"
            for i, tool_call in enumerate(message.tool_calls, 1):
                text += f"{i}. {tool_call.get('name', 'unknown')}\n"
                if 'input' in tool_call:
                    text += f"   Input: {tool_call['input']}\n"
        
        if message.tool_results and self.config.include_tool_results:
            text += "\nTool Results:\n"
            for i, result in enumerate(message.tool_results, 1):
                text += f"{i}. {result.get('output', 'No output')}\n"
        
        return text
    
    def _get_context(self, messages: List[Message], current_idx: int, context_size: int) -> str:
        """Get context from previous messages."""
        if current_idx == 0 or context_size == 0:
            return ""
        
        start_idx = max(0, current_idx - context_size)
        context_messages = messages[start_idx:current_idx]
        
        if not context_messages:
            return ""
        
        context_parts = []
        for msg in context_messages:
            # Truncate long messages in context
            content = msg.content
            if len(content) > 200:
                content = content[:200] + "..."
            context_parts.append(f"[Context] {msg.role.title()}: {content}")
        
        return "\n".join(context_parts)
    
    def _identify_context_segments(self, messages: List[Message]) -> List[Tuple[int, int]]:
        """Identify conversation segments that benefit from extended context."""
        segments = []
        current_segment = []
        
        for i, message in enumerate(messages):
            # Start new segment on topic changes or time gaps
            if self._is_segment_boundary(messages, i):
                if len(current_segment) >= 3:  # Only segments with multiple exchanges
                    segments.append((current_segment[0], current_segment[-1]))
                current_segment = [i]
            else:
                current_segment.append(i)
        
        # Add final segment
        if len(current_segment) >= 3:
            segments.append((current_segment[0], current_segment[-1]))
        
        return segments
    
    def _is_segment_boundary(self, messages: List[Message], idx: int) -> bool:
        """Determine if a message starts a new conversation segment."""
        if idx == 0:
            return True
        
        current_msg = messages[idx]
        prev_msg = messages[idx - 1]
        
        # Time gap boundary (>30 minutes)
        time_gap = current_msg.timestamp - prev_msg.timestamp
        if time_gap > timedelta(minutes=30):
            return True
        
        # Topic change detection (simple heuristic)
        if current_msg.role == 'user' and len(current_msg.content) > 100:
            # Check for topic change keywords
            topic_keywords = ['now', 'next', 'different', 'instead', 'change', 'new topic']
            content_lower = current_msg.content.lower()
            if any(keyword in content_lower for keyword in topic_keywords):
                return True
        
        return False
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from message content."""
        code_blocks = []
        
        # Regex for fenced code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2)
            
            code_blocks.append({
                'language': language,
                'code': code,
                'raw': match.group(0)
            })
        
        # Also check for inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.finditer(inline_pattern, content)
        
        for match in inline_matches:
            code = match.group(1)
            if len(code) > 20:  # Only longer inline code
                code_blocks.append({
                    'language': 'text',
                    'code': code,
                    'raw': match.group(0)
                })
        
        return code_blocks
    
    def _split_large_chunk(self, chunk_text: str, user_msg: Message, assistant_msg: Message) -> List[Chunk]:
        """Split large chunks into smaller ones."""
        chunks = []
        
        # Try splitting by paragraphs first
        paragraphs = chunk_text.split('\n\n')
        
        # If no paragraphs or single paragraph is too large, split by words
        if len(paragraphs) == 1 or any(len(p) > self.config.max_chunk_size for p in paragraphs):
            words = chunk_text.split()
            current_chunk = ""
            
            for word in words:
                if len(current_chunk) + len(word) + 1 <= self.config.max_chunk_size:
                    current_chunk += word + " "
                else:
                    if current_chunk.strip():
                        chunk = self._create_chunk(
                            current_chunk.strip(),
                            chunk_type="qa_pair_split",
                            conversation=None,
                            messages=[user_msg, assistant_msg]
                        )
                        chunks.append(chunk)
                    current_chunk = word + " "
            
            # Add final chunk
            if current_chunk.strip():
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    chunk_type="qa_pair_split",
                    conversation=None,
                    messages=[user_msg, assistant_msg]
                )
                chunks.append(chunk)
        else:
            # Split by paragraphs
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 <= self.config.max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk.strip():
                        chunk = self._create_chunk(
                            current_chunk.strip(),
                            chunk_type="qa_pair_split",
                            conversation=None,
                            messages=[user_msg, assistant_msg]
                        )
                        chunks.append(chunk)
                    current_chunk = paragraph + "\n\n"
            
            # Add final chunk
            if current_chunk.strip():
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    chunk_type="qa_pair_split",
                    conversation=None,
                    messages=[user_msg, assistant_msg]
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, chunk_type: str, conversation: Optional[Conversation],
                     messages: List[Message], extra_metadata: Optional[Dict[str, Any]] = None) -> Chunk:
        """Create a chunk with metadata."""
        self.chunk_counter += 1
        chunk_id = f"chunk_{self.chunk_counter:06d}"
        
        metadata = {
            "chunk_type": chunk_type,
            "message_count": len(messages),
            "message_uuids": [msg.uuid for msg in messages],
            "has_code": any(msg.has_code for msg in messages),
            "has_tools": any(msg.tool_calls or msg.tool_results for msg in messages),
            "char_count": len(text),
            "word_count": len(text.split()),
        }
        
        if conversation:
            metadata.update({
                "session_id": conversation.session_id,
                "project_name": conversation.project_name,
                "file_path": conversation.file_path,
            })
        
        if messages:
            metadata.update({
                "timestamp": messages[0].timestamp.isoformat(),
                "roles": [msg.role for msg in messages],
            })
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return Chunk(id=chunk_id, text=text, metadata=metadata)
    
    def _deduplicate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Remove duplicate chunks based on similarity."""
        if not chunks:
            return chunks
        
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            # Simple deduplication by text hash
            text_hash = hash(chunk.text)
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about generated chunks."""
        if not chunks:
            return {}
        
        chunk_types = {}
        total_chars = 0
        total_words = 0
        
        for chunk in chunks:
            chunk_type = chunk.metadata.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            total_chars += chunk.metadata.get("char_count", 0)
            total_words += chunk.metadata.get("word_count", 0)
        
        return {
            "total_chunks": len(chunks),
            "chunk_types": chunk_types,
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": total_chars / len(chunks) if chunks else 0,
            "avg_words_per_chunk": total_words / len(chunks) if chunks else 0,
        }