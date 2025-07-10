# MCP Server Interface Implementation Guide

## Overview

This guide covers adding an MCP (Model Context Protocol) server interface to the existing claude-search system. The MCP interface will coexist with the CLI and future Alfred workflow, sharing the same core search functionality.

## Quick Start Implementation

### 1. Install MCP SDK

```bash
uv add mcp==1.10.1
```

### 2. Basic MCP Server Structure

```python
# src/mcp_server.py
import asyncio
from pathlib import Path
from mcp.server import Server, McpError
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent
from typing import Any

from src.cli import SemanticSearchCLI
from src.storage import SearchConfig

# Initialize the semantic search system
search_cli = SemanticSearchCLI()

# Create MCP server instance
server = Server("claude-search")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available search tools"""
    return [
        Tool(
            name="semantic_search",
            description="Search Claude conversations using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 20)",
                        "default": 20
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name"
                    },
                    "has_code": {
                        "type": "boolean",
                        "description": "Only show results with code"
                    },
                    "after": {
                        "type": "string",
                        "description": "Filter after date (YYYY-MM-DD)"
                    },
                    "before": {
                        "type": "string",
                        "description": "Filter before date (YYYY-MM-DD)"
                    },
                    "session": {
                        "type": "string",
                        "description": "Filter by session ID"
                    },
                    "related_to": {
                        "type": "string",
                        "description": "Find chunks related to given chunk ID"
                    },
                    "same_session": {
                        "type": "boolean",
                        "description": "Include chunks from same session as related_to"
                    },
                    "full_content": {
                        "type": "boolean",
                        "description": "Show full content instead of truncated",
                        "default": false
                    },
                    "use_gpu": {
                        "type": "boolean",
                        "description": "Use GPU acceleration for faster search",
                        "default": false
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_chunk_by_id",
            description="Get a specific conversation chunk by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID"
                    }
                },
                "required": ["chunk_id"]
            }
        ),
        Tool(
            name="list_projects",
            description="List all indexed Claude projects",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_stats",
            description="Get search index statistics",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_status",
            description="Get the status of the indexing daemon and last index update",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Execute a tool and return results"""
    
    if name == "semantic_search":
        # Extract search parameters
        query = arguments.get("query")
        top_k = arguments.get("top_k", 20)
        filters = {}
        
        if arguments.get("project"):
            filters["project"] = arguments["project"]
        if arguments.get("has_code"):
            filters["has_code"] = True
        if arguments.get("after"):
            filters["after_date"] = arguments["after"]
        if arguments.get("before"):
            filters["before_date"] = arguments["before"]
        if arguments.get("session"):
            filters["session_id"] = arguments["session"]
        
        # Handle related_to functionality
        if arguments.get("related_to"):
            filters["related_to"] = arguments["related_to"]
            if arguments.get("same_session"):
                filters["same_session"] = True
                
        # Configure search settings
        config = SearchConfig(
            top_k=top_k,
            use_gpu=arguments.get("use_gpu", False)
        )
        
        # Perform search
        results = await asyncio.to_thread(
            search_cli.search_conversations,
            query,
            filters,
            config
        )
        
        # Handle full_content flag
        full_content = arguments.get("full_content", False)
        
        # Format results
        output = []
        for i, result in enumerate(results, 1):
            # Truncate content unless full_content is requested
            content = result.text
            if not full_content and len(content) > 500:
                content = content[:500] + "..."
                
            output.append(
                f"### Result {i} [Similarity: {result.similarity:.3f}]\\n"
                f"**Chunk ID**: {result.chunk_id}\\n"
                f"**Project**: {result.metadata.get('project', 'Unknown')}\\n"
                f"**Time**: {result.metadata.get('timestamp', 'Unknown')}\\n"
                f"**Session**: {result.metadata.get('session_id', 'Unknown')}\\n\\n"
                f"{content}\\n"
                f"{'🔧 Contains code' if result.metadata.get('has_code') else ''}\\n"
                f"---\\n"
            )
        
        return [TextContent(
            type="text",
            text=f"Found {len(results)} results for: '{query}'\\n\\n" + "\\n".join(output)
        )]
        
    elif name == "get_chunk_by_id":
        chunk_id = arguments.get("chunk_id")
        chunk = search_cli.storage.get_chunk_by_id(chunk_id)
        
        if chunk:
            return [TextContent(
                type="text",
                text=f"**Chunk ID**: {chunk_id}\\n"
                     f"**Project**: {chunk.metadata.get('project', 'Unknown')}\\n"
                     f"**Time**: {chunk.metadata.get('timestamp', 'Unknown')}\\n\\n"
                     f"{chunk.text}"
            )]
        else:
            raise McpError(f"Chunk not found: {chunk_id}")
            
    elif name == "list_projects":
        stats = search_cli.get_index_stats()
        projects = stats.get("projects", [])
        
        return [TextContent(
            type="text",
            text=f"**Indexed Projects ({len(projects)})**:\\n\\n" + 
                 "\\n".join(f"- {p}" for p in sorted(projects))
        )]
        
    elif name == "get_stats":
        stats = search_cli.get_index_stats()
        
        return [TextContent(
            type="text",
            text=f"**Search Index Statistics**\\n\\n"
                 f"- Total chunks: {stats['total_chunks']:,}\\n"
                 f"- Total sessions: {stats['total_sessions']:,}\\n"
                 f"- Total projects: {stats['total_projects']:,}\\n"
                 f"- Index size: {stats['faiss_size_mb']:.1f} MB\\n"
                 f"- Database size: {stats['db_size_mb']:.1f} MB\\n"
                 f"- Total storage: {stats['total_size_mb']:.1f} MB\\n\\n"
                 f"**Chunk Types**:\\n" +
                 "\\n".join(f"- {k}: {v:,}" for k, v in stats.get('chunk_types', {}).items())
        )]
        
    elif name == "get_status":
        # Check daemon status
        pid_file = Path.home() / ".claude" / "semantic-search" / "daemon.pid"
        is_running = False
        
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                # Check if process is running
                import psutil
                is_running = psutil.pid_exists(pid)
            except:
                is_running = False
                
        # Get last index time
        db_path = Path.home() / ".claude" / "semantic-search" / "data" / "metadata.db"
        last_indexed = "Never"
        if db_path.exists():
            import sqlite3
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(last_indexed) FROM files")
                result = cursor.fetchone()
                if result and result[0]:
                    last_indexed = result[0]
        
        return [TextContent(
            type="text",
            text=f"**Indexing Status**\\n\\n"
                 f"- Daemon running: {'✅ Yes' if is_running else '❌ No'}\\n"
                 f"- Last index update: {last_indexed}\\n"
                 f"- Index location: ~/.claude/semantic-search/data/\\n"
        )]
        
    else:
        raise McpError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Update pyproject.toml

```toml
[project.scripts]
claude-search-mcp = "src.mcp_server:main"

[project.dependencies]
# ... existing dependencies ...
mcp = ">=1.10.1"
```

### 4. Create MCP Configuration

Create `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "claude-search": {
      "command": "uv",
      "args": ["run", "claude-search-mcp"],
      "cwd": "/Users/jrbaron/dev/pauloportella/2025/cc-hacks/semantic-search"
    }
  }
}
```

### 5. Usage Examples

Once configured, you can use natural language in Claude Desktop:

```
"Search for conversations about GPU performance"
"Find all discussions about incremental indexing"
"Show me conversations in the daisy project with code"
"Get chunk chunk_id_12345"
"List all my indexed projects"
"Show search index statistics"
```

## Advanced Features

### 1. Streaming Results

For large result sets, implement streaming:

```python
@server.call_tool()
async def call_tool_streaming(name: str, arguments: Any):
    if name == "semantic_search":
        # Stream results as they're found
        async for result in search_streaming(arguments):
            yield TextContent(type="text", text=format_result(result))
```

### 2. Resource Providers

Expose conversation history as resources:

```python
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="conversations://recent",
            name="Recent Conversations",
            mimeType="text/plain"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "conversations://recent":
        # Return last 10 conversations
        recent = get_recent_conversations(10)
        return TextContent(type="text", text=format_conversations(recent))
```

### 3. Prompts

Add pre-configured search prompts:

```python
@server.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="debug_search",
            description="Search for debugging-related conversations",
            arguments=[
                PromptArgument(
                    name="error_type",
                    description="Type of error to search for",
                    required=True
                )
            ]
        )
    ]
```

## Testing

```python
# tests/test_mcp_server.py
import pytest
from mcp.client import Client
from src.mcp_server import server

@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic search through MCP"""
    # Create test client
    client = Client()
    
    # Call semantic_search tool
    result = await client.call_tool(
        "semantic_search",
        {"query": "test query", "top_k": 5}
    )
    
    assert len(result) > 0
    assert "Found" in result[0].text
```

## Deployment Checklist

- [ ] Install MCP SDK: `uv add mcp==1.10.1`
- [ ] Create `src/mcp_server.py`
- [ ] Update `pyproject.toml` with new script entry
- [ ] Configure Claude Desktop with MCP server path
- [ ] Test with Claude Desktop
- [ ] Add error handling and logging
- [ ] Document usage examples
- [ ] Create troubleshooting guide

## Troubleshooting

1. **Server not starting**: Check logs in Claude Desktop console
2. **Tools not appearing**: Verify JSON schema is valid
3. **Search errors**: Ensure index is built and accessible
4. **Performance issues**: Consider implementing result pagination

## Next Steps

1. Add more sophisticated filtering options
2. Implement conversation context tracking
3. Add support for code execution results
4. Create visual search result formatting
5. Add caching for frequent queries