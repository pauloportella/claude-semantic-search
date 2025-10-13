# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# Install dependencies
uv sync --dev

# Install globally (makes commands available system-wide)
uv tool install .

# Download embedding model (~420MB, required)
setup-models

# Index conversations
claude-index

# Search
claude-search "your query"
```

## Development

```bash
# Run tests
uv run pytest
uv run pytest --cov=src                    # with coverage
uv run pytest tests/test_storage.py        # specific file

# Code quality
uv run black src/ tests/                   # format (line-length 88)
uv run isort src/ tests/                   # sort imports
uv run mypy src/                           # type check

# Run from source (without global install)
uv run claude-index
uv run claude-search "query"
```

## Architecture

**Data Flow**: JSONL files → Parser → Chunker (4 strategies: Q&A, code, context, tools) → Embeddings (all-mpnet-base-v2, 768-dim) → FAISS + SQLite hybrid storage

**Key Files**:
- `src/parser.py`: Parses Claude Code JSONL format with nested message structures
- `src/chunker.py`: 4 parallel chunking strategies, filters out hook/system messages
- `src/storage.py`: FAISS IndexFlatIP + SQLite, incremental indexing via file modification tracking
- `src/cli.py`: Click commands, all entry points inject subcommand (e.g., `cli(["index"] + sys.argv[1:])`)
- `src/mcp_server.py`: MCP server for Claude Code/Desktop/Cursor integration

**Data Directory**: `~/.claude-semantic-search/data/` (env: `CLAUDE_SEARCH_DATA_DIR`)
- Always use `Path(data_dir).expanduser()` to handle `~`
- MCP server must read from env var to find index

## Critical Implementation Details

### Incremental Indexing
- Tracks file mtimes in SQLite `files` table
- Default: only processes modified files
- `--force`: clears everything and reindexes

### GPU Support
- **NVIDIA CUDA**: Full acceleration (10x speedup)
- **Apple Silicon MPS**: Embeddings only, FAISS stays CPU (2.5x speedup)
- GPU FAISS indexes must be converted to CPU before saving

### Project Filtering
Storage uses **case-insensitive partial matching**:
```python
if value.lower() in chunk_value.lower():  # Not exact match!
```

## Testing MCP Integration

```bash
# After making changes
uv tool install .

# Verify MCP config
cat ~/.claude.json | jq '.mcpServers["claude-semantic-search"]'

# Restart Claude Code, then ask:
# "Search for rust programming"
```

## Common Issues

1. **`pyproject.toml` must include `"scripts"` in packages** or `setup-models` fails
2. **Always expand `~` paths** via `Path().expanduser()`
3. **Storage methods return Optional** - check for None before accessing
4. **MCP server data dir** must come from `CLAUDE_SEARCH_DATA_DIR` env var

## Debugging

```bash
# Check model downloaded
ls ~/.claude-semantic-search/data/models/

# Check index
sqlite3 ~/.claude-semantic-search/data/metadata.db "SELECT COUNT(*) FROM chunks"

# Test GPU
python -c "from src.gpu_utils import assess_gpu_capability; print(assess_gpu_capability())"
```
