# Proposed Folder Structure for MCP Server

## Current Structure vs Proposed Changes

```
semantic-search/
├── src/                         # Core library code
│   ├── __init__.py
│   ├── chunker.py              # Conversation chunking
│   ├── cli.py                  # CLI commands implementation
│   ├── embeddings.py           # Embedding generation
│   ├── gpu_utils.py            # GPU utilities
│   ├── parser.py               # JSONL parser
│   ├── storage.py              # FAISS + SQLite storage
│   ├── watcher.py              # File system watcher
│   └── mcp_server.py           # 🆕 MCP server implementation
│
├── scripts/
│   ├── integration_demo.py
│   ├── model_setup.py
│   └── install_mcp.py          # 🆕 Optional: MCP setup helper
│
├── tests/
│   ├── ... (existing tests)
│   └── test_mcp_server.py      # 🆕 MCP server tests
│
├── docs/
│   ├── cli-mcp-feature-parity.md
│   ├── mcp-implementation.md
│   ├── mcp-server-plan.md
│   └── mcp-folder-structure.md # 🆕 This document
│
├── configs/                     # 🆕 Configuration examples
│   ├── claude_desktop_config.example.json
│   └── mcp_manifest.json
│
├── data/                        # Existing data directory
│   ├── embeddings.faiss
│   ├── metadata.db
│   └── models/
│
├── pyproject.toml              # Updated with MCP dependencies
├── README.md                   # Updated with MCP section
└── uv.lock

```

## Key Design Decisions

### 1. **Single mcp_server.py in src/**
- Keeps all source code together
- MCP server is just another interface alongside CLI
- Easy imports from existing modules

### 2. **No Separate MCP Directory**
- Avoids unnecessary nesting
- MCP is an interface, not a separate subsystem
- Maintains flat, simple structure

### 3. **Configuration Examples**
- `configs/` directory for example configurations
- Users copy and modify for their setup
- Keeps user configs out of version control

## File Descriptions

### New Files

**`src/mcp_server.py`**
- MCP server implementation
- Thin wrapper around SemanticSearchCLI
- Exposes tools: semantic_search, get_chunk_by_id, etc.

**`tests/test_mcp_server.py`**
- Unit tests for MCP server
- Mock MCP client tests
- Tool execution tests

**`configs/claude_desktop_config.example.json`**
```json
{
  "mcpServers": {
    "claude-search": {
      "command": "uv",
      "args": ["run", "claude-search-mcp"],
      "cwd": "/path/to/semantic-search"
    }
  }
}
```

**`configs/mcp_manifest.json`**
- MCP server metadata
- Tool definitions
- Version information

### Modified Files

**`pyproject.toml`**
```toml
[project.scripts]
# ... existing scripts ...
claude-search-mcp = "src.mcp_server:main"

[project.optional-dependencies]
mcp = ["mcp>=1.10.1"]
```

**`README.md`**
- Add MCP installation section
- Usage examples with Claude Desktop
- Link to MCP documentation

## Installation Flow

1. **Basic Installation** (existing)
   ```bash
   uv sync
   uv run setup-models
   uv run claude-index
   ```

2. **MCP Installation** (new)
   ```bash
   uv sync --extra mcp
   cp configs/claude_desktop_config.example.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   # Edit config with correct path
   ```

## Why This Structure?

1. **Minimal Changes**: Only adds necessary files
2. **Clear Separation**: MCP is clearly an interface layer
3. **Easy Testing**: Test files mirror source structure
4. **User-Friendly**: Example configs guide setup
5. **Maintainable**: No deep nesting or complex paths

## Alternative Considered (but rejected)

```
# ❌ Rejected: Separate MCP package
src/
├── mcp/
│   ├── __init__.py
│   ├── server.py
│   ├── tools.py
│   └── resources.py
```

This was rejected because:
- Over-engineering for a simple interface
- Makes imports more complex
- Suggests MCP is separate from core functionality
- Adds unnecessary directory depth