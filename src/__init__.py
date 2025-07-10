"""
Claude Semantic Search - Main package.

This package provides semantic search capabilities for Claude conversations
stored in ~/.claude/projects using sentence transformers and FAISS.
"""

__version__ = "0.1.0"

from .chunker import Chunk, ChunkingConfig, ConversationChunker
from .embeddings import EmbeddingConfig, EmbeddingGenerator, EmbeddingStats

# Package imports
from .parser import Conversation, JSONLParser, Message
from .storage import HybridStorage, SearchConfig, SearchResult, StorageConfig

__all__ = [
    "JSONLParser",
    "Conversation",
    "Message",
    "ConversationChunker",
    "ChunkingConfig",
    "Chunk",
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "EmbeddingStats",
    "HybridStorage",
    "StorageConfig",
    "SearchConfig",
    "SearchResult",
]
