"""
Claude Semantic Search - Main package.

This package provides semantic search capabilities for Claude conversations
stored in ~/.claude/projects using sentence transformers and FAISS.
"""

__version__ = "0.1.0"

# Package imports
from .parser import JSONLParser, Conversation, Message
from .chunker import ConversationChunker, ChunkingConfig, Chunk
from .embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingStats
from .storage import HybridStorage, StorageConfig, SearchConfig, SearchResult

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