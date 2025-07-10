#!/usr/bin/env python3
"""
Command-line interface for Claude Semantic Search.

Provides commands for indexing Claude conversations and searching through them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
from tqdm import tqdm

from .chunker import Chunk, ChunkingConfig, ConversationChunker
from .embeddings import EmbeddingConfig, EmbeddingGenerator
from .parser import JSONLParser
from .storage import HybridStorage, SearchConfig, StorageConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


class SemanticSearchCLI:
    """Main CLI class for semantic search operations."""

    def __init__(self, data_dir: str = "./data", use_gpu: bool = False) -> None:
        """Initialize CLI with data directory and GPU option."""
        self.data_dir: Path = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.use_gpu: bool = use_gpu

        # Initialize components
        self.parser: JSONLParser = JSONLParser()
        self.chunker: ConversationChunker = ConversationChunker(ChunkingConfig())

        # Initialize embeddings with local model
        embedding_config: EmbeddingConfig = EmbeddingConfig(
            model_name="all-mpnet-base-v2",
            batch_size=8,  # Will be auto-adjusted for GPU
            cache_dir=str(self.data_dir / "models"),
            use_gpu=use_gpu,
            auto_batch_size=True,
        )
        self.embedder: EmbeddingGenerator = EmbeddingGenerator(embedding_config)

        # Initialize storage
        storage_config: StorageConfig = StorageConfig(
            data_dir=str(self.data_dir),
            embedding_dim=768,
            auto_save=True,
            use_gpu=use_gpu,
        )
        self.storage: HybridStorage = HybridStorage(storage_config)

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

    def index_conversations(
        self, files: List[Path], force: bool = False
    ) -> Dict[str, Any]:
        """Index conversation files with retry support."""
        click.echo("🚀 Starting conversation indexing...")

        # Initialize storage
        self.storage.initialize()

        # Handle --force option
        if force:
            click.echo("🗑️  Force flag detected - clearing all existing data...")
            self.storage.clear_all_data()
            click.echo("✅ All data cleared")

        # Load embedding model
        if not self.embedder.is_model_loaded:
            click.echo("📥 Loading embedding model...")
            self.embedder.load_model()
            click.echo("✅ Model loaded successfully")

        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "files_unchanged": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "chunks_removed": 0,
            "errors": [],
            "start_time": time.time(),
        }

        failed_files = []

        # Process files with progress bar
        with tqdm(files, desc="Processing files", unit="file") as pbar:
            for file_path in pbar:
                try:
                    pbar.set_postfix_str(f"Checking {file_path.name}")

                    # Check if file needs indexing (unless --force)
                    if not force and not self.storage.is_file_modified(str(file_path)):
                        stats["files_unchanged"] += 1
                        pbar.set_postfix_str(f"Skipped (unchanged): {file_path.name}")
                        continue

                    pbar.set_postfix_str(f"Processing {file_path.name}")

                    # Remove old chunks for this file (if any)
                    removed_count = self.storage.remove_chunks_for_file(str(file_path))
                    if removed_count > 0:
                        stats["chunks_removed"] += removed_count

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

                    # Update file info
                    self.storage.update_file_info(str(file_path), len(chunks))

                    stats["chunks_indexed"] += len(chunks)
                    stats["files_processed"] += 1

                except Exception as e:
                    error_msg = f"Error processing {file_path}: {str(e)}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)
                    failed_files.append(file_path)
                    continue

        # Retry failed files once
        if failed_files:
            click.echo(f"\n🔄 Retrying {len(failed_files)} failed files...")
            retry_success = 0

            with tqdm(failed_files, desc="Retrying failed files", unit="file") as pbar:
                for file_path in pbar:
                    try:
                        pbar.set_postfix_str(f"Retrying {file_path.name}")

                        # Remove old chunks for this file (if any)
                        removed_count = self.storage.remove_chunks_for_file(
                            str(file_path)
                        )
                        if removed_count > 0:
                            stats["chunks_removed"] += removed_count

                        # Parse conversation
                        conversation = self.parser.parse_file(str(file_path))
                        if not conversation:
                            continue

                        # Create chunks
                        chunks = self.chunker.chunk_conversation(conversation)

                        if not chunks:
                            continue

                        # Generate embeddings
                        embeddings = self.embedder.generate_embeddings(chunks)

                        # Store in hybrid storage
                        self.storage.add_chunks(chunks)

                        # Update file info
                        self.storage.update_file_info(str(file_path), len(chunks))

                        # Update stats
                        stats["chunks_created"] += len(chunks)
                        stats["chunks_indexed"] += len(chunks)
                        stats["files_processed"] += 1
                        retry_success += 1

                        # Remove from errors list
                        stats["errors"] = [
                            err for err in stats["errors"] if file_path.name not in err
                        ]

                    except Exception as e:
                        # Failed again, keep in errors
                        logger.error(f"Retry failed for {file_path}: {str(e)}")
                        continue

            if retry_success > 0:
                click.echo(f"✅ Successfully retried {retry_success} files")

        stats["end_time"] = time.time()
        stats["duration"] = stats["end_time"] - stats["start_time"]

        return stats

    def search_conversations(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10
    ) -> List[Dict[str, Any]]:
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
            top_k=top_k, include_metadata=True, include_text=True
        )

        results = self.storage.search(query_embedding, search_config, filters)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "chunk_id": result.chunk_id,
                    "similarity": float(result.similarity),
                    "text": result.text,
                    "metadata": result.metadata,
                    "project": result.metadata.get("project_name", "unknown"),
                    "session": result.metadata.get("session_id", "unknown"),
                    "timestamp": result.metadata.get("timestamp", "unknown"),
                    "has_code": result.metadata.get("has_code", False),
                }
            )

        return formatted_results

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        self.storage.initialize()
        return self.storage.get_stats()


@click.group()
@click.option("--data-dir", default="./data", help="Data directory for storage")
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Claude Semantic Search CLI - Index and search your Claude conversations."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@cli.command()
@click.option(
    "--claude-dir", default="~/.claude/projects", help="Claude projects directory"
)
@click.option("--force", is_flag=True, help="Force reindexing of all files")
@click.option("--gpu", is_flag=True, help="Use GPU acceleration for faster indexing")
@click.pass_context
def index(ctx: click.Context, claude_dir: str, force: bool, gpu: bool) -> None:
    """Index Claude conversations for semantic search."""
    cli_instance = SemanticSearchCLI(ctx.obj["data_dir"], use_gpu=gpu)

    # Scan for files
    files = cli_instance.scan_claude_projects(claude_dir)

    # Index conversations
    stats = cli_instance.index_conversations(files, force)

    # Display results
    click.echo(f"\n🎉 Indexing complete!")
    click.echo(f"📊 Statistics:")
    click.echo(f"   • Files processed: {stats['files_processed']}")
    click.echo(f"   • Files unchanged: {stats.get('files_unchanged', 0)}")
    click.echo(f"   • Files skipped: {stats['files_skipped']}")
    click.echo(f"   • Chunks created: {stats['chunks_created']}")
    click.echo(f"   • Chunks indexed: {stats['chunks_indexed']}")
    if stats.get("chunks_removed", 0) > 0:
        click.echo(f"   • Chunks removed: {stats['chunks_removed']}")
    click.echo(f"   • Duration: {stats['duration']:.1f}s")

    if stats["errors"]:
        click.echo(f"   • Errors: {len(stats['errors'])}")
        for error in stats["errors"][:3]:  # Show first 3 errors
            click.echo(f"     - {error}")


@cli.command()
@click.argument("query")
@click.option("--top-k", default=10, help="Number of results to return")
@click.option("--project", help="Filter by project name (supports partial matching)")
@click.option("--has-code", is_flag=True, help="Filter for chunks with code")
@click.option("--after", help="Filter for chunks after date (YYYY-MM-DD)")
@click.option("--before", help="Filter for chunks before date (YYYY-MM-DD)")
@click.option("--session", help="Filter by session ID")
@click.option(
    "--related-to", help="Find chunks related to given chunk ID (same session)"
)
@click.option(
    "--same-session",
    is_flag=True,
    help="Include chunks from same session as --related-to",
)
@click.option(
    "--full-content", is_flag=True, help="Show full content instead of truncated"
)
@click.option(
    "--chunk-id", help="Get specific chunk by ID (ignores query and other filters)"
)
@click.option("--gpu", is_flag=True, help="Use GPU acceleration for faster search")
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    top_k: int,
    project: Optional[str],
    has_code: bool,
    after: Optional[str],
    before: Optional[str],
    session: Optional[str],
    related_to: Optional[str],
    same_session: bool,
    full_content: bool,
    chunk_id: Optional[str],
    gpu: bool,
    output_json: bool,
) -> None:
    """Search through indexed conversations."""
    cli_instance = SemanticSearchCLI(ctx.obj["data_dir"], use_gpu=gpu)

    # Handle direct chunk retrieval
    if chunk_id:
        try:
            cli_instance.storage.initialize()
            chunk = cli_instance.storage.get_chunk_by_id(chunk_id)

            if not chunk:
                click.echo(f"❌ Chunk not found: {chunk_id}")
                sys.exit(1)

            # Get metadata from database for display
            chunk_data = cli_instance.storage._get_chunk_data(chunk_id)

            if output_json:
                click.echo(
                    json.dumps(
                        {
                            "items": [
                                {
                                    "uid": chunk_id,
                                    "title": (
                                        chunk.text[:100] + "..."
                                        if len(chunk.text) > 100
                                        else chunk.text
                                    ),
                                    "subtitle": f"Direct chunk retrieval",
                                    "arg": chunk_id,
                                    "text": chunk.text,
                                    "quicklookurl": "",
                                    "variables": {
                                        "project": (
                                            chunk_data.get("project_name", "unknown")
                                            if chunk_data
                                            else "unknown"
                                        ),
                                        "session": (
                                            chunk_data.get("session_id", "unknown")
                                            if chunk_data
                                            else "unknown"
                                        ),
                                        "timestamp": (
                                            chunk_data.get("timestamp", "unknown")
                                            if chunk_data
                                            else "unknown"
                                        ),
                                    },
                                }
                            ]
                        },
                        indent=2,
                    )
                )
            else:
                click.echo(f"📄 Chunk: {chunk_id}")
                click.echo(
                    f"   Project: {chunk_data.get('project_name', 'unknown') if chunk_data else 'unknown'}"
                )
                click.echo(
                    f"   Session: {chunk_data.get('session_id', 'unknown') if chunk_data else 'unknown'}"
                )
                click.echo(
                    f"   Time: {chunk_data.get('timestamp', 'unknown') if chunk_data else 'unknown'}"
                )
                if chunk_data and chunk_data.get("has_code"):
                    click.echo("   🔧 Contains code")
                click.echo()
                click.echo(chunk.text)

            return
        except Exception as e:
            click.echo(f"❌ Failed to retrieve chunk: {str(e)}")
            sys.exit(1)

    # Handle related chunks
    if related_to:
        try:
            cli_instance.storage.initialize()

            # Get the reference chunk to find its session
            ref_chunk_data = cli_instance.storage._get_chunk_data(related_to)
            if not ref_chunk_data:
                click.echo(f"❌ Reference chunk not found: {related_to}")
                sys.exit(1)

            ref_session_id = ref_chunk_data.get("session_id")
            if not ref_session_id:
                click.echo(f"❌ Reference chunk has no session ID: {related_to}")
                sys.exit(1)

            # If --same-session flag is used, override query and get all chunks from session
            if same_session:
                related_chunks = cli_instance.storage.get_chunks_by_session(
                    ref_session_id
                )

                # Convert to result format for display
                results = []
                for chunk in related_chunks:
                    # Skip the reference chunk itself
                    if chunk.id == related_to:
                        continue

                    chunk_data = cli_instance.storage._get_chunk_data(chunk.id)
                    results.append(
                        {
                            "chunk_id": chunk.id,
                            "similarity": 1.0,  # Perfect similarity for same session
                            "text": chunk.text,
                            "project": (
                                chunk_data.get("project_name", "unknown")
                                if chunk_data
                                else "unknown"
                            ),
                            "session": (
                                chunk_data.get("session_id", "unknown")
                                if chunk_data
                                else "unknown"
                            ),
                            "timestamp": (
                                chunk_data.get("timestamp", "unknown")
                                if chunk_data
                                else "unknown"
                            ),
                            "has_code": (
                                chunk_data.get("has_code", False)
                                if chunk_data
                                else False
                            ),
                        }
                    )

                # Sort by timestamp for chronological order
                results.sort(key=lambda x: x["timestamp"])

                # Display results
                if output_json:
                    click.echo(
                        json.dumps(
                            {
                                "items": [
                                    {
                                        "uid": result["chunk_id"],
                                        "title": (
                                            result["text"][:100] + "..."
                                            if len(result["text"]) > 100
                                            else result["text"]
                                        ),
                                        "subtitle": f"Related to {related_to} | Same session",
                                        "arg": result["chunk_id"],
                                        "text": result["text"],
                                        "quicklookurl": "",
                                        "variables": {
                                            "similarity": result["similarity"],
                                            "project": result["project"],
                                            "session": result["session"],
                                            "timestamp": result["timestamp"],
                                        },
                                    }
                                    for result in results[:top_k]
                                ]
                            },
                            indent=2,
                        )
                    )
                else:
                    click.echo(
                        f"🔗 Found {len(results)} related chunks to {related_to} (same session: {ref_session_id})"
                    )
                    click.echo()

                    for i, result in enumerate(results[:top_k], 1):
                        click.echo(f"{i}. [Related] {result['project']}")

                        # Show full content or truncated based on flag
                        if full_content:
                            click.echo(f"   {result['text']}")
                        else:
                            click.echo(f"   {result['text'][:200]}...")

                        click.echo(
                            f"   Session: {result['session']} | Time: {result['timestamp']}"
                        )
                        if result["has_code"]:
                            click.echo("   🔧 Contains code")
                        click.echo()

                return
            else:
                # For --related-to without --same-session, add session filter to regular search
                session = ref_session_id

        except Exception as e:
            click.echo(f"❌ Failed to find related chunks: {str(e)}")
            sys.exit(1)

    # Build filters
    filters = {}
    if project:
        filters["project_name"] = project
    if has_code:
        filters["has_code"] = True
    if session:
        filters["session_id"] = session

    # Add date filters
    if after or before:
        timestamp_filter = {}
        if after:
            try:
                after_dt = datetime.fromisoformat(f"{after}T00:00:00+00:00")
                timestamp_filter["gte"] = after_dt.isoformat()
            except ValueError:
                click.echo(
                    f"❌ Invalid date format for --after: {after}. Use YYYY-MM-DD format."
                )
                sys.exit(1)
        if before:
            try:
                before_dt = datetime.fromisoformat(f"{before}T23:59:59+00:00")
                timestamp_filter["lte"] = before_dt.isoformat()
            except ValueError:
                click.echo(
                    f"❌ Invalid date format for --before: {before}. Use YYYY-MM-DD format."
                )
                sys.exit(1)
        filters["timestamp"] = timestamp_filter

    # Search
    try:
        results = cli_instance.search_conversations(query, filters, top_k)

        if output_json:
            # JSON output for Alfred integration
            click.echo(
                json.dumps(
                    {
                        "items": [
                            {
                                "uid": result["chunk_id"],
                                "title": (
                                    result["text"][:100] + "..."
                                    if len(result["text"]) > 100
                                    else result["text"]
                                ),
                                "subtitle": f"Project: {result['project']} | Similarity: {result['similarity']:.3f}",
                                "arg": result["chunk_id"],
                                "text": result["text"],
                                "quicklookurl": "",
                                "variables": {
                                    "similarity": result["similarity"],
                                    "project": result["project"],
                                    "session": result["session"],
                                    "timestamp": result["timestamp"],
                                },
                            }
                            for result in results
                        ]
                    },
                    indent=2,
                )
            )
        else:
            # Human-readable output
            click.echo(f"🔍 Found {len(results)} results for: '{query}'")
            click.echo()

            for i, result in enumerate(results, 1):
                click.echo(
                    f"{i}. [Similarity: {result['similarity']:.3f}] {result['project']}"
                )

                # Show full content or truncated based on flag
                if full_content:
                    click.echo(f"   {result['text']}")
                else:
                    click.echo(f"   {result['text'][:200]}...")

                click.echo(
                    f"   Session: {result['session']} | Time: {result['timestamp']}"
                )
                if result["has_code"]:
                    click.echo("   🔧 Contains code")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Search failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option("--gpu", is_flag=True, help="Show GPU information")
@click.pass_context
def stats(ctx: click.Context, gpu: bool) -> None:
    """Show statistics about the current index."""
    cli_instance = SemanticSearchCLI(ctx.obj["data_dir"], use_gpu=gpu)

    try:
        stats = cli_instance.get_index_stats()

        click.echo("📊 Index Statistics:")
        click.echo(f"   • Total chunks: {stats['total_chunks']:,}")
        click.echo(f"   • Total sessions: {stats['total_sessions']:,}")
        click.echo(f"   • Total projects: {stats['total_projects']:,}")
        click.echo(
            f"   • FAISS index size: {stats['faiss_index_size'] / 1024 / 1024:.1f} MB"
        )
        click.echo(f"   • Database size: {stats['database_size'] / 1024 / 1024:.1f} MB")
        click.echo(
            f"   • Total storage: {stats['total_storage_size'] / 1024 / 1024:.1f} MB"
        )
        click.echo(f"   • Embedding dimension: {stats['embedding_dimension']}")
        click.echo(f"   • Index type: {stats['index_type']}")

        # Show GPU information
        if stats.get("use_gpu") or stats.get("is_gpu_index"):
            click.echo(f"   • GPU enabled: {'✅' if stats.get('use_gpu') else '❌'}")
            click.echo(f"   • GPU index: {'✅' if stats.get('is_gpu_index') else '❌'}")

        if stats.get("gpu_info"):
            gpu_info = stats["gpu_info"]
            click.echo(f"   • GPU status: {gpu_info.get('status_message', 'Unknown')}")
            if gpu_info.get("gpu_names"):
                click.echo(f"   • GPU devices: {', '.join(gpu_info['gpu_names'])}")
            if gpu_info.get("gpu_memory_total_gb"):
                click.echo(
                    f"   • GPU memory: {gpu_info['gpu_memory_free_gb']:.1f}GB free / {gpu_info['gpu_memory_total_gb']:.1f}GB total"
                )

        if stats["chunk_types"]:
            click.echo(f"   • Chunk types:")
            for chunk_type, count in stats["chunk_types"].items():
                click.echo(f"     - {chunk_type}: {count:,}")

    except Exception as e:
        click.echo(f"❌ Failed to get stats: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option(
    "--claude-dir",
    default="~/.claude/projects",
    help="Claude projects directory to watch",
)
@click.option("--debounce", default=5, help="Debounce interval in seconds (default: 5)")
@click.option("--daemon", is_flag=True, help="Run as background daemon")
@click.option("--gpu", is_flag=True, help="Use GPU acceleration for indexing")
@click.pass_context
def watch(ctx: click.Context, claude_dir: str, debounce: int, daemon: bool, gpu: bool) -> None:
    """Watch Claude conversations for changes and auto-index them."""
    if daemon:
        from .watcher import start_daemon

        start_daemon(
            data_dir=ctx.obj["data_dir"],
            claude_dir=claude_dir,
            debounce_seconds=debounce,
            use_gpu=gpu,
        )
    else:
        from .watcher import run_watcher

        click.echo(f"🔍 Starting file watcher...")
        click.echo(f"   • Watching: {claude_dir}")
        click.echo(f"   • Data directory: {ctx.obj['data_dir']}")
        click.echo(f"   • Debounce interval: {debounce} seconds")
        click.echo(f"   • Press Ctrl+C to stop")
        click.echo()

        try:
            run_watcher(
                data_dir=ctx.obj["data_dir"],
                claude_dir=claude_dir,
                debounce_seconds=debounce,
                use_gpu=gpu,
            )
        except KeyboardInterrupt:
            click.echo("\n👋 File watcher stopped")
        except Exception as e:
            click.echo(f"❌ Watcher failed: {str(e)}")
            sys.exit(1)


@cli.command()
@click.option(
    "--claude-dir",
    default="~/.claude/projects",
    help="Claude projects directory to watch",
)
@click.option("--debounce", default=5, help="Debounce interval in seconds (default: 5)")
@click.option("--gpu", is_flag=True, help="Use GPU acceleration for indexing")
@click.pass_context
def start(ctx: click.Context, claude_dir: str, debounce: int, gpu: bool) -> None:
    """Start the file watcher daemon."""
    from .watcher import start_daemon

    start_daemon(
        data_dir=ctx.obj["data_dir"],
        claude_dir=claude_dir,
        debounce_seconds=debounce,
        use_gpu=gpu,
    )


@cli.command()
@click.pass_context
def stop(ctx: click.Context) -> None:
    """Stop the file watcher daemon."""
    from .watcher import stop_daemon

    stop_daemon(data_dir=ctx.obj["data_dir"])


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check the status of the file watcher daemon."""
    from .watcher import daemon_status

    daemon_status(data_dir=ctx.obj["data_dir"])


# Legacy function names for pyproject.toml compatibility
def index_command():
    """Entry point for claude-index command."""
    sys.argv = ["claude-index"] + sys.argv[1:]
    cli(["index"] + sys.argv[1:])


def search_command():
    """Entry point for claude-search command."""
    sys.argv = ["claude-search"] + sys.argv[1:]
    cli(["search"] + sys.argv[1:])


def stats_command():
    """Entry point for claude-stats command."""
    sys.argv = ["claude-stats"] + sys.argv[1:]
    cli(["stats"] + sys.argv[1:])


def watch_command():
    """Entry point for claude-watch command."""
    sys.argv = ["claude-watch"] + sys.argv[1:]
    cli(["watch"] + sys.argv[1:])


def start_command():
    """Entry point for claude-start command."""
    sys.argv = ["claude-start"] + sys.argv[1:]
    cli(["start"] + sys.argv[1:])


def stop_command():
    """Entry point for claude-stop command."""
    sys.argv = ["claude-stop"] + sys.argv[1:]
    cli(["stop"] + sys.argv[1:])


def status_command():
    """Entry point for claude-status command."""
    sys.argv = ["claude-status"] + sys.argv[1:]
    cli(["status"] + sys.argv[1:])


if __name__ == "__main__":
    cli()
