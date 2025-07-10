#!/usr/bin/env python3
"""
Command-line interface for Claude Semantic Search.

Provides commands for indexing Claude conversations and searching through them.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

from tqdm import tqdm
import click

from .parser import JSONLParser
from .chunker import ConversationChunker, ChunkingConfig
from .embeddings import EmbeddingGenerator, EmbeddingConfig
from .storage import HybridStorage, StorageConfig, SearchConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SemanticSearchCLI:
    """Main CLI class for semantic search operations."""
    
    def __init__(self, data_dir: str = "./data"):
        """Initialize CLI with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.parser = JSONLParser()
        self.chunker = ConversationChunker(ChunkingConfig())
        
        # Initialize embeddings with local model
        embedding_config = EmbeddingConfig(
            model_name="all-mpnet-base-v2",
            batch_size=8,  # Smaller batch for stability
            cache_dir=str(self.data_dir / "models")
        )
        self.embedder = EmbeddingGenerator(embedding_config)
        
        # Initialize storage
        storage_config = StorageConfig(
            data_dir=str(self.data_dir),
            embedding_dim=768,
            auto_save=True
        )
        self.storage = HybridStorage(storage_config)
        
    def scan_claude_projects(self, base_path: str = "~/.claude/projects") -> List[Path]:
        """Scan for Claude conversation files."""
        base_path = Path(base_path).expanduser()
        
        if not base_path.exists():
            click.echo(f"❌ Claude projects directory not found: {base_path}")
            sys.exit(1)
        
        # Find all JSONL files
        jsonl_files = list(base_path.rglob("*.jsonl"))
        
        if not jsonl_files:
            click.echo("❌ No JSONL files found in Claude projects directory")
            sys.exit(1)
        
        click.echo(f"📁 Found {len(jsonl_files)} conversation files")
        return jsonl_files
    
    def index_conversations(self, files: List[Path], force: bool = False) -> Dict[str, Any]:
        """Index conversation files."""
        click.echo("🚀 Starting conversation indexing...")
        
        # Initialize storage
        self.storage.initialize()
        
        # Load embedding model
        if not self.embedder.is_model_loaded:
            click.echo("📥 Loading embedding model...")
            self.embedder.load_model()
            click.echo("✅ Model loaded successfully")
        
        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": [],
            "start_time": time.time()
        }
        
        # Process files with progress bar
        with tqdm(files, desc="Processing files", unit="file") as pbar:
            for file_path in pbar:
                try:
                    pbar.set_postfix_str(f"Processing {file_path.name}")
                    
                    # Parse conversation
                    conversation = self.parser.parse_file(str(file_path))
                    if not conversation:
                        stats["files_skipped"] += 1
                        continue
                    
                    # Create chunks
                    chunks = self.chunker.chunk_conversation(conversation)
                    stats["chunks_created"] += len(chunks)
                    
                    if not chunks:
                        stats["files_skipped"] += 1
                        continue
                    
                    # Generate embeddings
                    embeddings = self.embedder.generate_embeddings(chunks)
                    
                    # Store in hybrid storage
                    self.storage.add_chunks(chunks)
                    stats["chunks_indexed"] += len(chunks)
                    stats["files_processed"] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)
                    continue
        
        stats["end_time"] = time.time()
        stats["duration"] = stats["end_time"] - stats["start_time"]
        
        return stats
    
    def search_conversations(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                           top_k: int = 10) -> List[Dict[str, Any]]:
        """Search through indexed conversations."""
        # Initialize storage
        self.storage.initialize()
        
        # Load embedding model
        if not self.embedder.is_model_loaded:
            self.embedder.load_model()
        
        # Generate query embedding
        query_embedding = self.embedder.generate_single_embedding(query)
        
        # Search
        search_config = SearchConfig(
            top_k=top_k,
            include_metadata=True,
            include_text=True
        )
        
        results = self.storage.search(query_embedding, search_config, filters)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "chunk_id": result.chunk_id,
                "similarity": float(result.similarity),
                "text": result.text,
                "metadata": result.metadata,
                "project": result.metadata.get("project_name", "unknown"),
                "session": result.metadata.get("session_id", "unknown"),
                "timestamp": result.metadata.get("timestamp", "unknown"),
                "has_code": result.metadata.get("has_code", False),
                "has_tools": result.metadata.get("has_tools", False)
            })
        
        return formatted_results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        self.storage.initialize()
        return self.storage.get_stats()


@click.group()
@click.option('--data-dir', default='./data', help='Data directory for storage')
@click.pass_context
def cli(ctx, data_dir):
    """Claude Semantic Search CLI - Index and search your Claude conversations."""
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir


@cli.command()
@click.option('--claude-dir', default='~/.claude/projects', help='Claude projects directory')
@click.option('--force', is_flag=True, help='Force reindexing of all files')
@click.pass_context
def index(ctx, claude_dir, force):
    """Index Claude conversations for semantic search."""
    cli_instance = SemanticSearchCLI(ctx.obj['data_dir'])
    
    # Scan for files
    files = cli_instance.scan_claude_projects(claude_dir)
    
    # Index conversations
    stats = cli_instance.index_conversations(files, force)
    
    # Display results
    click.echo(f"\n🎉 Indexing complete!")
    click.echo(f"📊 Statistics:")
    click.echo(f"   • Files processed: {stats['files_processed']}")
    click.echo(f"   • Files skipped: {stats['files_skipped']}")
    click.echo(f"   • Chunks created: {stats['chunks_created']}")
    click.echo(f"   • Chunks indexed: {stats['chunks_indexed']}")
    click.echo(f"   • Duration: {stats['duration']:.1f}s")
    
    if stats['errors']:
        click.echo(f"   • Errors: {len(stats['errors'])}")
        for error in stats['errors'][:3]:  # Show first 3 errors
            click.echo(f"     - {error}")


@cli.command()
@click.argument('query')
@click.option('--top-k', default=10, help='Number of results to return')
@click.option('--project', help='Filter by project name')
@click.option('--has-code', is_flag=True, help='Filter for chunks with code')
@click.option('--has-tools', is_flag=True, help='Filter for chunks with tool usage')
@click.option('--json', 'output_json', is_flag=True, help='Output results as JSON')
@click.pass_context
def search(ctx, query, top_k, project, has_code, has_tools, output_json):
    """Search through indexed conversations."""
    cli_instance = SemanticSearchCLI(ctx.obj['data_dir'])
    
    # Build filters
    filters = {}
    if project:
        filters['project_name'] = project
    if has_code:
        filters['has_code'] = True
    if has_tools:
        filters['has_tools'] = True
    
    # Search
    try:
        results = cli_instance.search_conversations(query, filters, top_k)
        
        if output_json:
            # JSON output for Alfred integration
            click.echo(json.dumps({
                "items": [
                    {
                        "uid": result["chunk_id"],
                        "title": result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"],
                        "subtitle": f"Project: {result['project']} | Similarity: {result['similarity']:.3f}",
                        "arg": result["chunk_id"],
                        "text": result["text"],
                        "quicklookurl": "",
                        "variables": {
                            "similarity": result["similarity"],
                            "project": result["project"],
                            "session": result["session"],
                            "timestamp": result["timestamp"]
                        }
                    }
                    for result in results
                ]
            }, indent=2))
        else:
            # Human-readable output
            click.echo(f"🔍 Found {len(results)} results for: '{query}'")
            click.echo()
            
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. [Similarity: {result['similarity']:.3f}] {result['project']}")
                click.echo(f"   {result['text'][:200]}...")
                click.echo(f"   Session: {result['session']} | Time: {result['timestamp']}")
                if result['has_code']:
                    click.echo("   🔧 Contains code")
                if result['has_tools']:
                    click.echo("   🛠️  Contains tools")
                click.echo()
                
    except Exception as e:
        click.echo(f"❌ Search failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show statistics about the current index."""
    cli_instance = SemanticSearchCLI(ctx.obj['data_dir'])
    
    try:
        stats = cli_instance.get_index_stats()
        
        click.echo("📊 Index Statistics:")
        click.echo(f"   • Total chunks: {stats['total_chunks']:,}")
        click.echo(f"   • Total sessions: {stats['total_sessions']:,}")
        click.echo(f"   • Total projects: {stats['total_projects']:,}")
        click.echo(f"   • FAISS index size: {stats['faiss_index_size'] / 1024 / 1024:.1f} MB")
        click.echo(f"   • Database size: {stats['database_size'] / 1024 / 1024:.1f} MB")
        click.echo(f"   • Total storage: {stats['total_storage_size'] / 1024 / 1024:.1f} MB")
        click.echo(f"   • Embedding dimension: {stats['embedding_dimension']}")
        click.echo(f"   • Index type: {stats['index_type']}")
        
        if stats['chunk_types']:
            click.echo(f"   • Chunk types:")
            for chunk_type, count in stats['chunk_types'].items():
                click.echo(f"     - {chunk_type}: {count:,}")
                
    except Exception as e:
        click.echo(f"❌ Failed to get stats: {str(e)}")
        sys.exit(1)


# Legacy function names for pyproject.toml compatibility
def index_command():
    """Entry point for claude-index command."""
    sys.argv = ['claude-index'] + sys.argv[1:]
    cli(['index'] + sys.argv[1:])


def search_command():
    """Entry point for claude-search command."""
    sys.argv = ['claude-search'] + sys.argv[1:]
    cli(['search'] + sys.argv[1:])


def stats_command():
    """Entry point for claude-stats command."""
    sys.argv = ['claude-stats'] + sys.argv[1:]
    cli(['stats'] + sys.argv[1:])


if __name__ == '__main__':
    cli()