# Claude Semantic Search

A powerful semantic search system for Claude conversations that enables fast, intelligent retrieval through natural language queries. Access your Claude Code sessions from anywhere - Claude Code itself, CLI, Claude Desktop, Cursor IDE, and Alfred (in development)!

## Why This Project?

Ever wished you could instantly find that Claude Code conversation where you solved a tricky bug or implemented a complex feature? This project was born from the need to easily retrieve Claude Code sessions from anywhere - whether you're in Claude Code itself, working in your terminal, using Claude Desktop, coding in Cursor IDE, or even through Alfred on macOS.

Your Claude conversations contain valuable knowledge - code solutions, architectural decisions, debugging sessions, and implementation details. This tool makes that knowledge instantly searchable and accessible across all your development workflows.

## Features

### Multi-Platform Access
- **Claude Code**: Search directly within Claude Code using MCP tools
- **CLI**: Powerful command-line tools for indexing and searching
- **Claude Desktop**: Native integration via MCP server
- **Cursor IDE**: Full MCP tool support for searching within your editor
- **Alfred**: JSON output for workflow integration (in development)

### Core Features
- **Semantic Search**: Uses all-mpnet-base-v2 embeddings for high-quality semantic matching
- **Hybrid Storage**: Combines FAISS vector search with SQLite metadata storage
- **Smart Chunking**: Context-aware chunking optimized for code discussions and Q&A pairs
- **Incremental Indexing**: Only processes new/modified files after initial index
- **GPU Acceleration**: Optional CUDA/MPS support for 5-10x performance improvements
- **Auto-Indexing Daemon**: Background service that automatically indexes new conversations

### Advanced Filtering
- Date ranges (before/after)
- Project filtering with partial matching (case-insensitive)
- Code content filtering
- Session-based search
- Related chunk discovery
- Direct chunk access by ID

## Installation

### Prerequisites

- Python 3.11+
- UV package manager (recommended)

### Setup

1. **Clone and setup the project:**
```bash
git clone https://github.com/pauloportella/claude-semantic-search
cd claude-semantic-search
uv sync
```

2. **(Optional) Install GPU support for 5-10x faster performance:**
```bash
# GPU support requires conda due to Python version constraints
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install faiss-gpu -c conda-forge
```

3. **Download the embedding model:**
```bash
uv run setup-models
```

This downloads the all-mpnet-base-v2 model (~420MB) for high-quality embeddings.

## Quick Start

### 1. Index Your Conversations
```bash
uv run claude-index
```

### 2. Search from Terminal
```bash
uv run claude-search "your search query"
```

### 3. Enable in Claude Desktop/Cursor
```bash
# For Claude Desktop
cp configs/claude_desktop_config.example.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# For Cursor IDE
# Add to ~/.cursor/mcp.json (see MCP_SERVER_README.md for details)
```

Then restart Claude Desktop or Cursor and search naturally:
- "Search for my GPU performance discussions"
- "Find conversations about the daisy project with code"

## Usage

### Basic Commands

#### 1. Index Your Conversations

```bash
# Index all Claude conversations (incremental - only new/modified files)
uv run claude-index

# Index from a specific directory
uv run claude-index --claude-dir /path/to/conversations

# Force reindexing of all files (clears existing data)
uv run claude-index --force

# Use GPU acceleration (5-10x faster)
uv run claude-index --gpu
```

**Incremental Indexing**: By default, `claude-index` only processes files that have been added or modified since the last indexing. This makes subsequent runs much faster. Use `--force` to clear all data and reindex everything from scratch.

#### 2. Search Conversations

```bash
# Basic search
uv run claude-search "rust programming"

# Search with filters (project names support partial matching)
uv run claude-search "error handling" --project "my-project" --has-code
uv run claude-search "persistence" --project "daisy-hft"  # Matches any project containing "daisy-hft"

# Limit results
uv run claude-search "debugging" --top-k 5

# JSON output (for Alfred integration)
uv run claude-search "python testing" --json

# Use GPU acceleration for faster search
uv run claude-search "machine learning" --gpu
```

#### 3. View Statistics

```bash
# Show index statistics
uv run claude-stats
```

#### 4. Watch for Changes (Daemon Mode)

```bash
# Start watcher daemon (uses incremental indexing)
uv run claude-start

# Start with custom settings
uv run claude-start --claude-dir /path/to/conversations --debounce 10

# Check daemon status
uv run claude-status

# Stop daemon
uv run claude-stop

# Interactive watch mode (foreground)
uv run claude-watch

# Run watch as daemon (background)
uv run claude-watch --daemon
```

**Auto-indexing**: The daemon watches for new or modified files and automatically indexes them using incremental indexing. When a file changes, only that file and other files in the same directory are checked for updates.

### Advanced Usage

#### Search Filters

- `--project <name>`: Filter by project name
- `--has-code`: Only show chunks containing code
- `--top-k <n>`: Limit number of results (default: 10)
- `--after <date>`: Filter for chunks after date (YYYY-MM-DD)
- `--before <date>`: Filter for chunks before date (YYYY-MM-DD)
- `--session <id>`: Filter by session ID
- `--related-to <chunk-id>`: Find chunks related to given chunk ID
- `--same-session`: Include chunks from same session (use with --related-to)
- `--full-content`: Show full content instead of truncated
- `--chunk-id <id>`: Get specific chunk by ID

#### Examples

```bash
# Find Rust-related conversations with code
uv run claude-search "rust async" --has-code

# Search in specific project
uv run claude-search "database migration" --project "my-app"

# Get top 20 results about testing
uv run claude-search "unit testing" --top-k 20

# Filter by date range
uv run claude-search "kubernetes deployment" --after 2024-01-01 --before 2024-12-31

# Get specific chunk with full content
uv run claude-search --chunk-id "chunk_abc123" --full-content

# Find related chunks in same session
uv run claude-search --related-to "chunk_abc123" --same-session

# Search within specific session
uv run claude-search "error handling" --session "session_xyz789"
```

## GPU Acceleration

Enable GPU support for 5-10x performance improvements:

### Installation

#### For NVIDIA CUDA Systems
```bash
# Install GPU dependencies (requires conda)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install faiss-gpu -c conda-forge

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS GPU count: {faiss.get_num_gpus()}')"
```

#### For Apple Silicon (M1/M2/M3/M4)
```bash
# PyTorch with MPS support (already included in base installation)
# No additional installation needed! 

# Verify MPS availability
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
uv run claude-stats --gpu
```

### Requirements

#### NVIDIA CUDA (Full GPU acceleration)
- **NVIDIA GPU** with CUDA compute capability ≥ 3.5
- **CUDA toolkit** 11.4+ or 12.1+
- **GPU memory** ≥ 2GB recommended (varies by dataset size)

#### Apple Silicon MPS (Embedding acceleration only)
- **Apple M1, M2, M3, or M4** processors
- **macOS 12.3+** with native Python (arm64)
- **Unified memory** ≥ 8GB recommended

### Usage

```bash
# GPU-accelerated indexing (10-100x faster embedding generation)
uv run claude-index --gpu

# GPU-accelerated search (5-10x faster vector search)
uv run claude-search "your query" --gpu

# View GPU status and memory usage
uv run claude-stats --gpu

# Start GPU-enabled daemon
uv run claude-start --gpu
```

### Performance Benefits

| Operation | CPU Time | NVIDIA CUDA | Apple Silicon MPS | 
|-----------|----------|-------------|-------------------|
| Index 1k conversations | ~5 minutes | ~30 seconds (**10x**) | ~2 minutes (**2.5x**) |
| Index 10k conversations | ~50 minutes | ~5 minutes (**10x**) | ~20 minutes (**2.5x**) |
| Search queries | ~200ms | ~20ms (**10x**) | ~200ms (**1x** - CPU) |
| Embedding generation | ~100 texts/sec | ~1000 texts/sec (**10x**) | ~300 texts/sec (**3x**) |

**Note**: Apple Silicon provides 3-5x speedup for embedding generation but search remains on CPU since FAISS GPU doesn't support Metal Performance Shaders.

### Memory Requirements

The system automatically detects available GPU memory and optimizes batch sizes:

- **Small datasets** (< 1k chunks): ~1GB GPU memory
- **Medium datasets** (1k-10k chunks): ~2-4GB GPU memory  
- **Large datasets** (> 10k chunks): ~4-8GB GPU memory

### Troubleshooting

```bash
# Check GPU status
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Test FAISS GPU
python -c "import faiss; print(f'GPU count: {faiss.get_num_gpus()}')"

# Check installation
uv run claude-stats --gpu
```

If GPU support fails, the system automatically falls back to CPU with a warning message.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   JSONL Files   │───▶│   Chunking &    │───▶│   Embeddings    │
│  (~/.claude/)   │    │   Parsing       │    │  (all-mpnet)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Alfred Query   │◀───│  Search Engine  │◀───│ Hybrid Storage  │
│   Interface     │    │   (Semantic +   │    │ FAISS + SQLite  │
└─────────────────┘    │   Metadata)     │    └─────────────────┘
                       └─────────────────┘
```

### Components

1. **Parser**: Extracts structured data from Claude conversation JSONL files
2. **Chunker**: Creates semantic chunks using smart strategies (Q&A pairs, code blocks, context segments)
3. **Embeddings**: Generates 768-dimensional vectors using all-mpnet-base-v2
4. **Storage**: Hybrid FAISS + SQLite for fast semantic and metadata search

## Performance

- **Model**: all-mpnet-base-v2 (420MB, 768 dimensions)
- **Accuracy**: 87-88% semantic similarity
- **Search Speed**: <500ms for typical queries
- **Memory**: <2GB during indexing
- **Storage**: ~1GB for 1000 conversations

## Alfred Integration

The CLI provides JSON output perfect for Alfred workflows:

```bash
uv run claude-search "your query" --json
```

Returns structured JSON with:
- `uid`: Unique chunk identifier
- `title`: Truncated text preview
- `subtitle`: Project name and similarity score
- `arg`: Chunk ID for further processing
- `text`: Full chunk text
- `variables`: Metadata (similarity, project, session, timestamp)

### Sample Alfred Script

```bash
#!/bin/bash
query="$1"
cd /path/to/semantic-search
uv run claude-search "$query" --json
```

## MCP Server Integration

The MCP (Model Context Protocol) server enables native integration with Claude Desktop and Cursor IDE, allowing you to search your conversation history directly within your AI tools.

### Quick Setup

#### Claude Desktop
```bash
cp configs/claude_desktop_config.example.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
# Edit the config and set your correct path
```

#### Cursor IDE
Add to `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "claude-search": {
      "command": "/Users/USERNAME/.local/bin/uv",
      "args": ["run", "--directory", "/path/to/claude-semantic-search", "claude-search-mcp"],
      "cwd": "/path/to/claude-semantic-search"
    }
  }
}
```
**Note**: Replace `/Users/USERNAME/.local/bin/uv` with your actual UV path (find it with `which uv`)

### Available MCP Tools

- `semantic_search`: Full-featured search with all filtering options
- `get_chunk_by_id`: Retrieve specific chunks
- `list_projects`: List all indexed projects
- `get_stats`: View index statistics
- `get_status`: Check daemon status

### Example Usage

In Claude Desktop or Cursor, simply ask:
- "Search for my GPU performance discussions"
- "Find conversations about daisy project with code"
- "Show me all indexed projects"
- "Get chunk chunk_12345"

For detailed MCP setup and troubleshooting, see [MCP_SERVER_README.md](MCP_SERVER_README.md).

## Development

### Project Structure

```
semantic-search/
├── src/
│   ├── parser.py       # JSONL conversation parser
│   ├── chunker.py      # Smart chunking strategies
│   ├── embeddings.py   # Embedding generation
│   ├── storage.py      # Hybrid FAISS + SQLite storage
│   └── cli.py          # Command-line interface
├── tests/              # Comprehensive test suite
├── scripts/            # Utility scripts
└── data/               # Models and index storage
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_parser.py
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/
```

## Configuration

### Environment Variables

- `CLAUDE_PROJECTS_DIR`: Override default Claude projects directory
- `SEMANTIC_SEARCH_DATA_DIR`: Override default data directory
- `EMBEDDING_MODEL`: Override default embedding model

### Storage Configuration

The system uses:
- **FAISS**: IndexFlatIP for exact cosine similarity search
- **SQLite**: Metadata storage with optimized indexes
- **Auto-save**: Indexes are automatically saved after updates

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   ```bash
   # Re-download model
   uv run setup-models
   ```

2. **Empty Search Results**
   ```bash
   # Check if index exists
   uv run claude-stats
   
   # Rebuild index
   uv run claude-index --force
   ```

3. **Permission Errors**
   ```bash
   # Check Claude directory permissions
   ls -la ~/.claude/projects
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size in embeddings.py
   # Use smaller embedding model
   ```

### Performance Optimization

- **Incremental Indexing**: Only new/modified files are reprocessed
  - First indexing: Processes all files
  - Subsequent runs: Only processes changed files (typically 90%+ faster)
  - File modification tracking via SQLite metadata
- **Batch Processing**: Embeddings generated in optimized batches
- **Memory Mapping**: Large indexes use memory-mapped files
- **Compression**: Efficient storage formats reduce disk usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) for embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Click](https://github.com/pallets/click) for CLI framework