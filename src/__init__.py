"""
Claude Semantic Search - Main package.

This package provides semantic search capabilities for Claude conversations
stored in ~/.claude/projects using sentence transformers and FAISS.
"""

__version__ = "0.1.0"

# Package imports
from .embeddings import EmbeddingGenerator
from .parser import ConversationParser
from .chunker import ConversationChunker
from .storage import HybridStorage
from .indexer import IncrementalIndexer

__all__ = [
    "EmbeddingGenerator",
    "ConversationParser", 
    "ConversationChunker",
    "HybridStorage",
    "IncrementalIndexer",
]