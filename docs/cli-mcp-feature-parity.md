# CLI vs MCP Feature Parity

## Complete Feature Mapping

This document shows how all CLI features are exposed through the MCP interface.

### Search Features

| CLI Option | MCP Tool | MCP Parameter | Notes |
|------------|----------|---------------|-------|
| `claude-search QUERY` | `semantic_search` | `query` | Main search query |
| `--top-k` | `semantic_search` | `top_k` | Number of results |
| `--project` | `semantic_search` | `project` | Filter by project |
| `--has-code` | `semantic_search` | `has_code` | Filter chunks with code |
| `--after` | `semantic_search` | `after` | Date filter (YYYY-MM-DD) |
| `--before` | `semantic_search` | `before` | Date filter (YYYY-MM-DD) |
| `--session` | `semantic_search` | `session` | Filter by session ID |
| `--related-to` | `semantic_search` | `related_to` | Find related chunks |
| `--same-session` | `semantic_search` | `same_session` | Used with related_to |
| `--full-content` | `semantic_search` | `full_content` | Show full vs truncated |
| `--chunk-id` | `get_chunk_by_id` | `chunk_id` | Get specific chunk |
| `--gpu` | `semantic_search` | `use_gpu` | GPU acceleration |
| `--json` | N/A | N/A | MCP always returns structured data |

### Additional MCP Tools

| Tool | Purpose | CLI Equivalent |
|------|---------|----------------|
| `list_projects` | List all indexed projects | Part of `claude-stats` |
| `get_stats` | Get index statistics | `claude-stats` command |
| `get_status` | Check daemon and index status | `claude-status` command |

## Usage Examples

### CLI
```bash
# Search with filters
uv run claude-search "GPU performance" --project "semantic-search" --has-code --top-k 5

# Get related chunks
uv run claude-search "" --related-to chunk_12345 --same-session

# Date range search
uv run claude-search "incremental indexing" --after 2025-07-01 --before 2025-07-10
```

### MCP (in Claude Desktop)
```
User: Search for GPU performance in the semantic-search project with code examples

Assistant will call:
semantic_search({
  "query": "GPU performance",
  "project": "semantic-search",
  "has_code": true,
  "top_k": 20
})
```

```
User: Find chunks related to chunk_12345 in the same session

Assistant will call:
semantic_search({
  "query": "",
  "related_to": "chunk_12345",
  "same_session": true
})
```

```
User: Search for incremental indexing discussions from last week

Assistant will call:
semantic_search({
  "query": "incremental indexing",
  "after": "2025-07-01",
  "before": "2025-07-10"
})
```

## Implementation Notes

1. **Single Tool Design**: Instead of separate tools for each feature, `semantic_search` is a comprehensive tool that mirrors all CLI search options.

2. **Chunk ID Access**: The `--chunk-id` CLI flag is implemented as a separate `get_chunk_by_id` tool for cleaner separation of concerns.

3. **Structured Output**: While CLI has `--json` flag, MCP always returns structured data that Claude can parse and present nicely.

4. **GPU Acceleration**: The `use_gpu` parameter allows Claude to request faster searches when needed.

5. **Full Content**: By default, MCP returns truncated content (500 chars) unless `full_content: true` is specified, similar to CLI behavior.

## Benefits

- **Complete feature parity**: All CLI capabilities available through MCP
- **Natural language**: Users can describe what they want in plain English
- **Intelligent defaults**: Claude can choose appropriate parameters based on context
- **Structured results**: Claude can format and present results optimally