# Claude Search MCP Server

## Overview

The MCP (Model Context Protocol) server provides a seamless integration between Claude Desktop and your semantic search index, allowing you to search your Claude conversation history using natural language.

## Installation

### 1. Ensure semantic search is set up

First, make sure you have indexed your conversations:

```bash
# Index your conversations
uv run claude-index

# Verify the index
uv run claude-stats
```

### 2. Configure Claude Desktop

Copy the example configuration to Claude Desktop's config directory:

```bash
# macOS
cp configs/claude_desktop_config.example.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Edit the config to set your correct path
# Replace /Users/YOUR_USERNAME/path/to/semantic-search with your actual path
```

Example configuration:
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

### 3. Restart Claude Desktop

After updating the configuration, restart Claude Desktop for the changes to take effect.

## Usage

Once configured, you can use natural language to search your conversations:

### Basic Search
- "Search for conversations about GPU performance"
- "Find discussions about incremental indexing"
- "Show me conversations about daisy project"

### Filtered Search
- "Search for Python code examples in the semantic-search project"
- "Find conversations from last week about MCP"
- "Show me discussions with code from July 2025"

### Specific Queries
- "Get chunk chunk_12345"
- "Show me all indexed projects"
- "What's the status of my search index?"

## Available Tools

### 1. `semantic_search`
Main search tool with comprehensive filtering options:
- **query**: Search text
- **top_k**: Number of results (default: 20)
- **project**: Filter by project name
- **has_code**: Only show results with code
- **after**: Filter after date (YYYY-MM-DD)
- **before**: Filter before date (YYYY-MM-DD)
- **session**: Filter by session ID
- **related_to**: Find chunks related to a chunk ID
- **same_session**: Include chunks from same session
- **full_content**: Show full content instead of truncated
- **use_gpu**: Use GPU acceleration

### 2. `get_chunk_by_id`
Retrieve a specific conversation chunk by its ID.

### 3. `list_projects`
List all indexed Claude projects.

### 4. `get_stats`
Get comprehensive search index statistics.

### 5. `get_status`
Check the status of the indexing daemon and last update time.

## Examples

### Example 1: Search with filters
```
User: Search for GPU performance discussions in the semantic-search project with code

Claude will use:
semantic_search({
  "query": "GPU performance",
  "project": "semantic-search",
  "has_code": true
})
```

### Example 2: Date range search
```
User: Find conversations about MCP from the last week

Claude will use:
semantic_search({
  "query": "MCP",
  "after": "2025-07-03",
  "before": "2025-07-10"
})
```

### Example 3: Get specific chunk
```
User: Show me the full content of chunk_12345

Claude will use:
get_chunk_by_id({
  "chunk_id": "chunk_12345"
})
```

## Troubleshooting

### MCP server not appearing in Claude
1. Check that the config file is in the correct location
2. Verify the path in the config points to your semantic-search directory
3. Ensure you have restarted Claude Desktop
4. Check Claude Desktop's developer console for errors

### Search returns no results
1. Verify your index is built: `uv run claude-stats`
2. Check that the daemon is running: `uv run claude-status`
3. Try a broader search query

### Permission errors
1. Ensure the semantic-search directory is accessible
2. Check that the virtual environment is activated
3. Verify that `uv` is installed and in your PATH

## Advanced Usage

### GPU Acceleration
If you have GPU support configured, Claude can request GPU-accelerated searches:

```
User: Do a fast GPU search for embedding implementations

Claude will use:
semantic_search({
  "query": "embedding implementations",
  "use_gpu": true
})
```

### Related Chunks
Find chunks from the same conversation:

```
User: Find other chunks related to chunk_67890 in the same session

Claude will use:
semantic_search({
  "query": "",
  "related_to": "chunk_67890",
  "same_session": true
})
```

## Integration with Other Tools

The MCP server coexists with other interfaces:
- **CLI**: Continue using `uv run claude-search` directly
- **Alfred**: Future Alfred workflow will work alongside MCP
- **API**: Could be extended with REST API in the future

## Security Considerations

- The MCP server runs locally and only accesses your indexed conversations
- No data is sent to external servers
- The server respects the same permissions as your user account
- Only Claude Desktop with proper configuration can access the server

## Development

To modify or extend the MCP server:

1. Edit `src/mcp_server.py`
2. Run tests: `uv run python -m pytest tests/test_mcp_server.py`
3. Test manually: `uv run claude-search-mcp` (should wait for input)
4. Check with Claude Desktop after changes

## Support

For issues or questions:
1. Check the main README for general setup
2. Review Claude Desktop logs for MCP errors
3. Run `uv run claude-status` to verify system health
4. Create an issue in the repository with details