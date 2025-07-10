# Claude Semantic Search

A powerful semantic search system for Claude conversations that enables fast, intelligent retrieval through natural language queries.

## Features

- **Semantic Search**: Uses all-mpnet-base-v2 embeddings for high-quality semantic matching
- **Hybrid Storage**: Combines FAISS vector search with SQLite metadata storage
- **Smart Chunking**: Context-aware chunking optimized for code discussions and Q&A pairs
- **CLI Interface**: Simple command-line tools for indexing and searching
- **Alfred Integration**: JSON output format ready for Alfred workflows
- **Incremental Updates**: Efficient indexing of new conversations
- **Date-based Filtering**: Filter search results by date ranges
- **Session Threading**: Search within specific conversation sessions
- **Related Chunks**: Find related chunks within the same conversation session
- **Direct Chunk Access**: Retrieve specific chunks by ID with full content
- **Auto-Indexing Daemon**: Background service that automatically indexes new conversations
- **Service Management**: Start/stop/status commands for managing the watcher daemon

## Installation

### Prerequisites

- Python 3.11+
- UV package manager (recommended)

### Setup

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd semantic-search
uv sync
```

2. **Download the embedding model:**
```bash
uv run setup-models
```

This downloads the all-mpnet-base-v2 model (~420MB) for high-quality embeddings.

## Usage

### Basic Commands

#### 1. Index Your Conversations

```bash
# Index all Claude conversations (default: ~/.claude/projects)
uv run claude-index

# Index from a specific directory
uv run claude-index --claude-dir /path/to/conversations

# Force reindexing of all files
uv run claude-index --force
```

#### 2. Search Conversations

```bash
# Basic search
uv run claude-search "rust programming"

# Search with filters
uv run claude-search "error handling" --project "my-project" --has-code

# Limit results
uv run claude-search "debugging" --top-k 5

# JSON output (for Alfred integration)
uv run claude-search "python testing" --json
```

#### 3. View Statistics

```bash
# Show index statistics
uv run claude-stats
```

#### 4. Watch for Changes (Daemon Mode)

```bash
# Start watcher daemon
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