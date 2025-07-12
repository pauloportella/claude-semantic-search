"""MCP server interface for Claude semantic search."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil
from mcp import ErrorData, McpError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.cli import SemanticSearchCLI

# Create MCP server instance
server: Server = Server("claude-search")

# We'll initialize search_cli on demand with proper GPU settings
search_cli: Optional[SemanticSearchCLI] = None


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available search tools."""
    return [
        Tool(
            name="claude_semantic_search",
            description="Search Claude conversations using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 20)",
                        "default": 20,
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name (supports partial matching)",
                    },
                    "has_code": {
                        "type": "boolean",
                        "description": "Only show results with code",
                    },
                    "after": {
                        "type": "string",
                        "description": "Filter after date (YYYY-MM-DD)",
                    },
                    "before": {
                        "type": "string",
                        "description": "Filter before date (YYYY-MM-DD)",
                    },
                    "session": {
                        "type": "string",
                        "description": "Filter by session ID",
                    },
                    "related_to": {
                        "type": "string",
                        "description": "Find chunks related to given chunk ID",
                    },
                    "same_session": {
                        "type": "boolean",
                        "description": "Include chunks from same session as related_to",
                    },
                    "full_content": {
                        "type": "boolean",
                        "description": "Show full content instead of truncated",
                        "default": False,
                    },
                    "use_gpu": {
                        "type": "boolean",
                        "description": "Use GPU acceleration for faster search",
                        "default": False,
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "Get specific chunk by ID (ignores query and other filters)",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="get_chunk_by_id",
            description="Get a specific conversation chunk by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID",
                    }
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="list_projects",
            description="List all indexed Claude projects",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_stats",
            description="Get search index statistics",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_status",
            description="Get the status of the indexing daemon and last index update",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


def get_search_cli(use_gpu: bool = False) -> SemanticSearchCLI:
    """Get or create search CLI instance with appropriate settings."""
    global search_cli
    if search_cli is None or search_cli.use_gpu != use_gpu:
        # Use environment variable or default data directory
        data_dir = os.environ.get("CLAUDE_SEARCH_DATA_DIR", "~/.claude-semantic-search/data")
        data_dir = str(Path(data_dir).expanduser())  # Expand ~ to full path
        search_cli = SemanticSearchCLI(data_dir, use_gpu=use_gpu)
    return search_cli


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tool and return results."""
    if name == "claude_semantic_search":
        # Check if this is a chunk_id lookup
        chunk_id: Optional[str] = arguments.get("chunk_id")
        if chunk_id:
            # Handle chunk_id lookup (like --chunk-id in CLI)
            cli = get_search_cli()
            chunk = await asyncio.to_thread(cli.storage.get_chunk_by_id, chunk_id)
            
            if chunk:
                # Get metadata from database for display
                chunk_data = await asyncio.to_thread(cli.storage._get_chunk_data, chunk_id)
                
                return [
                    TextContent(
                        type="text",
                        text=f"**Chunk ID**: {chunk_id}\n"
                        f"**Project**: {chunk_data.get('project_name', 'Unknown') if chunk_data else 'Unknown'}\n"
                        f"**Time**: {chunk_data.get('timestamp', 'Unknown') if chunk_data else 'Unknown'}\n\n"
                        f"{chunk.text}",
                    )
                ]
            else:
                raise McpError(ErrorData(code=-32602, message=f"Chunk not found: {chunk_id}"))
        
        # Regular search
        query: str = arguments.get("query", "")
            
        top_k: int = arguments.get("top_k", 20)
        use_gpu: bool = arguments.get("use_gpu", False)
        filters: Dict[str, Any] = {}

        if arguments.get("project"):
            filters["project_name"] = arguments["project"]
        if arguments.get("has_code"):
            filters["has_code"] = True
        # Handle date filters
        if arguments.get("after") or arguments.get("before"):
            timestamp_filter = {}
            if arguments.get("after"):
                after_dt = f"{arguments['after']}T00:00:00+00:00"
                timestamp_filter["gte"] = after_dt
            if arguments.get("before"):
                before_dt = f"{arguments['before']}T23:59:59+00:00"
                timestamp_filter["lte"] = before_dt
            filters["timestamp"] = timestamp_filter
        if arguments.get("session"):
            filters["session_id"] = arguments["session"]

        # Handle related_to functionality
        if arguments.get("related_to"):
            filters["related_to"] = arguments["related_to"]
            if arguments.get("same_session"):
                filters["same_session"] = True

        # Get CLI instance with appropriate GPU settings
        cli = get_search_cli(use_gpu)

        # Perform search
        results = await asyncio.to_thread(
            cli.search_conversations, query, filters, top_k
        )

        # Handle full_content flag
        full_content: bool = arguments.get("full_content", False)

        # Format results
        output: List[str] = []
        for i, result in enumerate(results, 1):
            # Truncate content unless full_content is requested
            content = result['text']
            if not full_content and len(content) > 500:
                content = content[:500] + "..."

            similarity = float(result['similarity']) if result.get('similarity') is not None else 0.0
            output.append(
                f"### Result {i} [Similarity: {similarity:.3f}]\n"
                f"**Chunk ID**: {result['chunk_id']}\n"
                f"**Project**: {result.get('project', 'Unknown')}\n"
                f"**Time**: {result.get('timestamp', 'Unknown')}\n"
                f"**Session**: {result.get('session', 'Unknown')}\n\n"
                f"{content}\n"
                f"{'🔧 Contains code' if result.get('has_code') else ''}\n"
                f"---\n"
            )

        return [
            TextContent(
                type="text",
                text=f"Found {len(results)} results for: '{query}'\n\n"
                + "\n".join(output),
            )
        ]

    elif name == "get_chunk_by_id":
        chunk_id: Optional[str] = arguments.get("chunk_id")
        cli: SemanticSearchCLI = get_search_cli()
        chunk = await asyncio.to_thread(cli.storage.get_chunk_by_id, chunk_id)

        if chunk:
            # Get metadata from database for display
            chunk_data = await asyncio.to_thread(cli.storage._get_chunk_data, chunk_id)
            
            return [
                TextContent(
                    type="text",
                    text=f"**Chunk ID**: {chunk_id}\n"
                    f"**Project**: {chunk_data.get('project_name', 'Unknown') if chunk_data else 'Unknown'}\n"
                    f"**Time**: {chunk_data.get('timestamp', 'Unknown') if chunk_data else 'Unknown'}\n\n"
                    f"{chunk.text}",
                )
            ]
        else:
            raise McpError(ErrorData(code=-32602, message=f"Chunk not found: {chunk_id}"))

    elif name == "list_projects":
        cli = get_search_cli()
        
        try:
            # Ensure storage is initialized
            cli.storage.initialize()
            
            # Get all projects directly from storage
            projects = await asyncio.to_thread(cli.storage.get_all_projects)
            
            # Format project list
            if projects:
                project_list = "\n".join(f"- {p}" for p in projects)
            else:
                project_list = "*No projects found in the index*"
            
            return [
                TextContent(
                    type="text",
                    text=f"**Indexed Projects ({len(projects)})**:\n\n{project_list}",
                )
            ]
        except Exception as e:
            raise McpError(
                ErrorData(
                    code=-32603,
                    message=f"Failed to retrieve projects: {str(e)}"
                )
            )

    elif name == "get_stats":
        cli = get_search_cli()
        stats = await asyncio.to_thread(cli.get_index_stats)

        return [
            TextContent(
                type="text",
                text=f"**Search Index Statistics**\n\n"
                f"- Total chunks: {stats['total_chunks']:,}\n"
                f"- Total sessions: {stats['total_sessions']:,}\n"
                f"- Total projects: {stats['total_projects']:,}\n"
                f"- Index size: {stats.get('faiss_index_size', 0) / 1024 / 1024:.1f} MB\n"
                f"- Database size: {stats.get('database_size', 0) / 1024 / 1024:.1f} MB\n"
                f"- Total storage: {stats.get('total_storage_size', 0) / 1024 / 1024:.1f} MB\n\n"
                f"**Chunk Types**:\n"
                + "\n".join(f"- {k}: {v:,}" for k, v in stats.get("chunk_types", {}).items()),
            )
        ]

    elif name == "get_status":
        # Check daemon status
        # Note: The current implementation doesn't use a daemon, so this is always false
        # TODO: Implement daemon functionality if needed
        is_running = False
        pid_file = Path("./data/daemon.pid")  # Placeholder path

        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # Check if process is running
                is_running = psutil.pid_exists(pid)
            except Exception:
                is_running = False

        # Get last index time from the actual data directory
        cli = get_search_cli()
        db_path = Path(cli.data_dir) / "metadata.db"
        last_indexed = "Never"
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT MAX(last_indexed) FROM files")
                    result = cursor.fetchone()
                    if result and result[0]:
                        last_indexed = result[0]
            except Exception:
                pass

        return [
            TextContent(
                type="text",
                text=f"**Indexing Status**\n\n"
                f"- Daemon running: {'✅ Yes' if is_running else '❌ No'}\n"
                f"- Last index update: {last_indexed}\n"
                f"- Index location: {cli.data_dir}/\n",
            )
        ]

    else:
        raise McpError(ErrorData(code=-32601, message=f"Unknown tool: {name}"))


async def main() -> None:
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run() -> None:
    """Entry point for the MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    run()