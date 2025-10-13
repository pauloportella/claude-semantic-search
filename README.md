# Claude Semantic Search

Semantic search for Claude conversations. Find what you discussed instantly using natural language queries.

<figure style="text-align: center;">
  <img width="600" alt="Claude Semantic Search interface" src="https://github.com/user-attachments/assets/b5d35184-27b7-44c3-b4aa-9dc7c6e21af6" />
</figure>

## Features

- **Natural language search** across all your Claude conversations
- **Multiple interfaces**: CLI, MCP (Claude Code/Desktop/Cursor), Alfred workflow
- **Smart chunking**: Context-aware splitting optimized for code and Q&A
- **GPU acceleration**: AMD ROCm, NVIDIA CUDA, and Apple Silicon support
- **Auto-indexing**: Background daemon watches for new conversations

## Quick Start

```bash
git clone https://github.com/pauloportella/claude-semantic-search
cd claude-semantic-search
./install.sh
```

The installer will:
- Detect and install GPU support (AMD/NVIDIA/Apple Silicon)
- Configure MCP for Claude Code, Claude Desktop, and Cursor
- Download the embedding model (~420MB)
- Index your conversations

Then restart your Claude apps and search:
```
"Find my GPU performance discussions"
"Show conversations about the daisy project with code"
```

## Usage

### CLI Commands

```bash
# Index conversations (auto-detects GPU)
claude-index

# Force reindex everything
claude-index --force

# Search
claude-search "your query"
claude-search "error handling" --project myapp --has-code

# View stats
claude-stats

# Auto-index daemon
claude-start   # start background daemon
claude-status  # check status
claude-stop    # stop daemon
```

### GPU Support

GPU acceleration is **automatic** - the installer detects your hardware:

| GPU Type | Speedup | Support |
|----------|---------|---------|
| **AMD ROCm** | ~6-8x | Embeddings accelerated, search on CPU |
| **NVIDIA CUDA** | ~10x | Full acceleration (embeddings + search) |
| **Apple Silicon** | ~2-3x | Embeddings accelerated, search on CPU |

Disable GPU: `claude-index --no-gpu`

### Advanced Search

```bash
# Filter by project (partial match)
claude-search "database" --project myapp

# Date range
claude-search "deployment" --after 2024-01-01 --before 2024-12-31

# Code only
claude-search "async function" --has-code

# Related chunks
claude-search --related-to chunk_abc123 --same-session
```

## Architecture

```
JSONL Files → Parser → Chunker → Embeddings (all-mpnet-base-v2) → FAISS + SQLite
                                                                         ↓
                                                    CLI / MCP Server / Alfred
```

**Key Components:**
- **Embeddings**: all-mpnet-base-v2 (768-dim vectors)
- **Search**: FAISS IndexFlatIP + SQLite metadata
- **Chunking**: Q&A pairs, code blocks, context segments
- **Incremental**: Only processes new/modified files

## Development

```bash
# Setup
git clone https://github.com/pauloportella/claude-semantic-search
cd claude-semantic-search
uv sync

# Run tests
uv run pytest
uv run pytest --cov=src

# Format code
uv run black src/ tests/
uv run isort src/ tests/
```

## Configuration

**Data Directory**: `~/.claude-semantic-search/data` (customizable)

```bash
# Environment variable
export CLAUDE_SEARCH_DATA_DIR=/your/path

# Or via flag
claude-index --data-dir /your/path
```

**MCP Configuration** (auto-configured by installer):
- Claude Code: `~/.claude.json`
- Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Cursor: `~/.cursor/mcp.json`

## Troubleshooting

```bash
# Check GPU status
claude-stats

# Verify model
ls ~/.claude-semantic-search/data/models/

# Rebuild index
claude-index --force

# Check daemon
claude-status
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [Click](https://github.com/pallets/click) - CLI framework
