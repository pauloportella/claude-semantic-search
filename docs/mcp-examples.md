# MCP Server Examples

This document shows examples of using the semantic search MCP server with Claude Code.

## Partial Project Name Matching

The MCP server now supports partial, case-insensitive project name matching:

```javascript
// Search for "persistence" in any project containing "daisy-hft"
await mcp.use_tool("claude_semantic_search", {
  query: "persistence",
  project: "daisy-hft"  // Will match "-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine"
})

// Case-insensitive matching
await mcp.use_tool("claude_semantic_search", {
  query: "optimization",
  project: "DAISY"  // Will match any project containing "daisy" regardless of case
})
```

## Complete Filter Example

```javascript
// Search with all available filters
await mcp.use_tool("claude_semantic_search", {
  query: "error handling",
  project: "semantic",      // Partial project match
  has_code: true,          // Only chunks with code
  after: "2025-01-01",     // After this date
  before: "2025-12-31",    // Before this date
  session: "abc123",       // Specific session
  top_k: 20,              // Number of results
  use_gpu: true,          // GPU acceleration
  full_content: false     // Truncate long results
})
```

## Working with Results

The search results include:
- **Chunk ID**: Unique identifier for the chunk
- **Project**: Full project name (useful to see what partial match found)
- **Similarity**: Score from 0-1 indicating relevance
- **Time**: When the conversation occurred
- **Session**: Session identifier for grouping related chunks
- **Content**: The actual text (truncated unless full_content=true)

## Related Chunks

Find chunks from the same conversation:

```javascript
// First, search for something
const results = await mcp.use_tool("claude_semantic_search", {
  query: "implementation details",
  project: "daisy"
})

// Then get related chunks from the same session
const chunk_id = results[0].chunk_id
await mcp.use_tool("claude_semantic_search", {
  query: "context",
  related_to: chunk_id,
  same_session: true
})
```

## Direct Chunk Access

Retrieve a specific chunk by ID:

```javascript
await mcp.use_tool("get_chunk_by_id", {
  chunk_id: "chunk_4a7654e4_2025-07-07T20:25:50.501000+00:00_1"
})
```

## Project Discovery

List all indexed projects:

```javascript
await mcp.use_tool("list_projects", {})
// Returns list of all project names, useful for understanding what's been indexed
```

## Index Statistics

Get information about the search index:

```javascript
await mcp.use_tool("get_stats", {})
// Returns total chunks, sessions, projects, storage sizes, etc.
```