# Claude Conversations Semantic Search - Implementation Plan

## Project Overview

Build a semantic search system for Claude conversations stored in `~/.claude/projects` to enable fast, intelligent retrieval through Alfred workflows.

**Key Features:**
- Semantic search using all-mpnet-base-v2 embeddings (420MB, 768-dim)
- Hybrid storage: FAISS for vectors + SQLite for metadata
- Incremental indexing for performance
- Context-aware chunking for code discussions
- Alfred-ready API interface

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

## Implementation Phases

### Phase 0: Environment Setup

**Objective**: Initialize project structure and dependencies

**Tasks:**
1. Initialize uv project with Python 3.11+
2. Create directory structure
3. Setup dependencies in pyproject.toml
4. Create model download script
5. Setup git repository

**Dependencies:**
```toml
[project]
dependencies = [
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.0",
    "sqlite3",
    "tqdm>=4.64.0",
    "numpy>=1.21.0",
    "pandas>=1.5.0",
    "pathlib",
    "json",
    "datetime",
    "uuid",
    "hashlib"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0"
]
```

**Tests:**
- `test_environment_setup.py`: Verify all dependencies install correctly
- `test_model_download.py`: Verify model downloads and loads
- `test_directory_structure.py`: Verify all required directories exist

**Success Criteria:**
- All dependencies install without errors
- Model downloads successfully (420MB)
- Model loads and can generate embeddings
- Tests pass: `uv run pytest tests/test_environment_setup.py`

---

### Phase 1: JSONL Parser

**Objective**: Parse Claude conversation files and extract structured data

**Implementation:**

```python
# src/parser.py
class ConversationParser:
    def parse_jsonl_file(self, file_path: str) -> List[Conversation]:
        """Parse a JSONL file and return structured conversations"""
        
    def extract_metadata(self, conversation: Conversation) -> Dict:
        """Extract metadata: project, timestamps, tools, code presence"""
        
    def build_conversation_threads(self, messages: List[Dict]) -> List[Conversation]:
        """Group messages by sessionId and build conversation threads"""
```

**Key Features:**
- Handle nested message structures
- Extract tool results and code blocks
- Build conversation threads using parentUuid
- Extract metadata (project, timestamps, tools used)
- Handle malformed JSON gracefully

**Tests:**
- `test_jsonl_parsing.py`: Parse various JSONL formats
- `test_conversation_threading.py`: Verify conversation threads are built correctly
- `test_metadata_extraction.py`: Verify all metadata is extracted
- `test_error_handling.py`: Handle malformed JSON, missing fields
- `test_edge_cases.py`: Empty files, single messages, long conversations

**Test Fixtures:**
- `data/test_fixtures/simple_conversation.jsonl`
- `data/test_fixtures/complex_conversation.jsonl`
- `data/test_fixtures/malformed_conversation.jsonl`
- `data/test_fixtures/code_heavy_conversation.jsonl`

**Success Criteria:**
- Parse 100+ real conversation files without errors
- Extract all metadata fields correctly
- Build conversation threads accurately
- Handle edge cases gracefully
- Tests pass: `uv run pytest tests/test_parser.py`

---

### Phase 2: Smart Chunking

**Objective**: Convert conversations into semantic chunks optimized for search

**Implementation:**

```python
# src/chunker.py
class ConversationChunker:
    def create_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Create semantic chunks from conversation"""
        
    def create_qa_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Primary strategy: User question + Assistant response"""
        
    def create_context_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Extended chunks with 1-2 previous exchanges"""
        
    def create_code_chunks(self, conversation: Conversation) -> List[Chunk]:
        """Separate chunks for significant code blocks"""
```

**Chunking Strategies:**
1. **Q&A Pairs**: User message + Assistant response + Tool results
2. **Context Chunks**: Include 1-2 previous exchanges for continuity
3. **Code Chunks**: Separate chunks for code blocks >20 lines
4. **Summary Chunks**: For very long conversations, create summary chunks

**Features:**
- Preserve context across message boundaries
- Handle code blocks intelligently
- Maintain metadata for each chunk
- Optimize for 768-dim embeddings
- Respect token limits (384 tokens max)

**Tests:**
- `test_chunking_strategies.py`: Test all chunking strategies
- `test_chunk_boundaries.py`: Verify correct chunk boundaries
- `test_context_preservation.py`: Verify context is preserved
- `test_code_handling.py`: Verify code blocks are handled correctly
- `test_chunk_metadata.py`: Verify chunk metadata is complete
- `test_token_limits.py`: Verify chunks respect token limits

**Success Criteria:**
- No data loss during chunking
- Chunks are semantically meaningful
- Context is preserved appropriately
- Code blocks are handled intelligently
- Tests pass: `uv run pytest tests/test_chunker.py`

---

### Phase 3: Embedding Generation

**Objective**: Convert text chunks to semantic vectors using all-mpnet-base-v2

**Implementation:**

```python
# src/embeddings.py
class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """Initialize with cached model"""
        
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks in batches"""
        
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate single embedding for queries"""
        
    def batch_process(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Process texts in batches for efficiency"""
```

**Features:**
- Use cached all-mpnet-base-v2 model (420MB)
- Batch processing for efficiency (16-32 chunks)
- Progress tracking with tqdm
- Normalized embeddings for cosine similarity
- Handle token limits gracefully
- Memory-efficient processing

**Tests:**
- `test_model_loading.py`: Verify model loads correctly
- `test_embedding_generation.py`: Test embedding generation
- `test_batch_processing.py`: Test batch processing efficiency
- `test_embedding_quality.py`: Test semantic similarity
- `test_normalization.py`: Verify embeddings are normalized
- `test_memory_usage.py`: Monitor memory usage during processing

**Performance Benchmarks:**
- Embedding generation speed (chunks/second)
- Memory usage during batch processing
- Model loading time
- Batch size optimization

**Success Criteria:**
- Model loads without errors
- Embeddings are high quality (semantic similarity tests)
- Batch processing is efficient
- Memory usage is reasonable
- Tests pass: `uv run pytest tests/test_embeddings.py`

---

### Phase 4: Hybrid Storage Layer

**Objective**: Implement FAISS + SQLite storage for fast retrieval

**Implementation:**

```python
# src/storage.py
class HybridStorage:
    def __init__(self, data_dir: str = "./data"):
        """Initialize FAISS index and SQLite database"""
        
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and embeddings to storage"""
        
    def search_semantic(self, query_embedding: np.ndarray, k: int = 50) -> List[int]:
        """Search FAISS index for similar vectors"""
        
    def search_metadata(self, filters: Dict) -> List[int]:
        """Search SQLite for metadata filters"""
        
    def hybrid_search(self, query_embedding: np.ndarray, filters: Dict, k: int = 20) -> List[SearchResult]:
        """Combine semantic and metadata search"""
```

**Storage Schema:**

```sql
-- SQLite Schema
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    project TEXT,
    timestamp DATETIME,
    chunk_type TEXT,
    has_code BOOLEAN,
    has_tools BOOLEAN,
    text TEXT,
    vector_id INTEGER,
    message_uuids TEXT,
    file_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE files (
    path TEXT PRIMARY KEY,
    last_modified DATETIME,
    last_indexed DATETIME,
    chunk_count INTEGER
);

-- Indexes for fast filtering
CREATE INDEX idx_project ON chunks(project);
CREATE INDEX idx_timestamp ON chunks(timestamp);
CREATE INDEX idx_session ON chunks(session_id);
CREATE INDEX idx_has_code ON chunks(has_code);
CREATE INDEX idx_has_tools ON chunks(has_tools);
```

**Features:**
- FAISS IndexFlatIP for exact search (768 dimensions)
- SQLite for metadata and filtering
- Hybrid queries combining both
- Efficient batch operations
- Index persistence and loading
- Memory-mapped files for large datasets

**Tests:**
- `test_faiss_operations.py`: Test FAISS index operations
- `test_sqlite_operations.py`: Test SQLite operations
- `test_hybrid_search.py`: Test combined search
- `test_persistence.py`: Test index saving/loading
- `test_batch_operations.py`: Test batch add/update operations
- `test_search_accuracy.py`: Test search result accuracy

**Success Criteria:**
- FAISS index operations work correctly
- SQLite operations are efficient
- Hybrid search returns relevant results
- Index persistence works
- Tests pass: `uv run pytest tests/test_storage.py`

---

### Phase 5: Incremental Indexing

**Objective**: Efficiently update index when files change

**Implementation:**

```python
# src/indexer.py
class IncrementalIndexer:
    def __init__(self, storage: HybridStorage, parser: ConversationParser, 
                 chunker: ConversationChunker, embedder: EmbeddingGenerator):
        """Initialize with all components"""
        
    def scan_for_changes(self, base_path: str) -> List[str]:
        """Scan for new/modified files since last index"""
        
    def index_files(self, file_paths: List[str]) -> IndexResult:
        """Index new or changed files"""
        
    def remove_old_chunks(self, file_path: str):
        """Remove chunks for files that were reindexed"""
        
    def run_incremental_update(self) -> IndexResult:
        """Run full incremental update"""
```

**Features:**
- Track file modification times
- Detect new and changed files
- Remove old chunks before adding new ones
- Batch processing for efficiency
- Progress tracking and logging
- Error recovery and partial updates
- Configurable scan intervals

**Tests:**
- `test_change_detection.py`: Test file change detection
- `test_incremental_updates.py`: Test incremental indexing
- `test_chunk_removal.py`: Test old chunk removal
- `test_batch_indexing.py`: Test batch processing
- `test_error_recovery.py`: Test error handling
- `test_progress_tracking.py`: Test progress reporting

**Success Criteria:**
- Only changed files are reprocessed
- Old chunks are removed correctly
- Incremental updates are efficient
- Error recovery works
- Tests pass: `uv run pytest tests/test_indexer.py`

---

### Phase 6: Integration and Optimization

**Objective**: Complete end-to-end system with performance optimizations

**Implementation:**

```python
# src/search_engine.py
class SemanticSearchEngine:
    def __init__(self, data_dir: str = "./data"):
        """Initialize complete search system"""
        
    def search(self, query: str, filters: Dict = None, k: int = 20) -> List[SearchResult]:
        """Main search interface"""
        
    def rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results using additional signals"""
        
    def build_index(self, force_rebuild: bool = False) -> IndexResult:
        """Build complete index from scratch"""
        
    def update_index(self) -> IndexResult:
        """Incremental index update"""
```

**Features:**
- Complete search pipeline
- Result reranking with multiple signals
- Performance monitoring
- Configuration management
- Error handling and recovery
- Logging and debugging
- Memory optimization
- Alfred-ready JSON API

**Optimizations:**
- Memory-mapped FAISS index
- Connection pooling for SQLite
- Batch operations
- Caching frequently accessed data
- Parallel processing where possible

**Tests:**
- `test_end_to_end.py`: Complete workflow tests
- `test_search_quality.py`: Search result quality
- `test_performance.py`: Performance benchmarks
- `test_configuration.py`: Configuration management
- `test_error_handling.py`: Error recovery
- `test_api_interface.py`: Alfred API compatibility

**Performance Benchmarks:**
- Index build time for 1000 conversations
- Search latency (<500ms target)
- Memory usage during operation
- Index size efficiency

**Success Criteria:**
- End-to-end search works correctly
- Performance meets targets
- Error handling is robust
- API is Alfred-ready
- Tests pass: `uv run pytest tests/test_integration.py`

---

## Git Workflow

Each phase follows this workflow:

1. **Implement**: Write code for the phase
2. **Test**: Write comprehensive tests
3. **Validate**: Run tests and ensure they pass
4. **Commit**: Commit with descriptive message

```bash
# Example workflow for Phase 1
git add -A
git commit -m "feat: implement JSONL parser with comprehensive tests

- Parse conversation files and extract structured data
- Handle nested message structures and tool results
- Build conversation threads using parentUuid
- Extract metadata (project, timestamps, tools used)
- Handle malformed JSON gracefully
- Add comprehensive test suite with fixtures

Tests: 25 passed, 0 failed"
```

## Testing Strategy

**Test Categories:**
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interactions
- **End-to-End Tests**: Complete workflow
- **Performance Tests**: Speed and memory benchmarks
- **Edge Case Tests**: Error handling and unusual inputs

**Test Data:**
- Real conversation samples (anonymized)
- Synthetic edge cases
- Performance stress tests
- Various conversation patterns

**Coverage Requirements:**
- Minimum 90% code coverage
- All public APIs tested
- Error paths covered
- Performance benchmarks included

## Success Metrics

**Functionality:**
- Index 1000+ conversations without errors
- Search latency <500ms
- Search accuracy >85% for relevant queries
- Handle incremental updates efficiently

**Performance:**
- Index build: <30 minutes for 1000 conversations
- Memory usage: <2GB during indexing
- Index size: <1GB for 1000 conversations
- Search throughput: >10 queries/second

**Reliability:**
- Handle malformed files gracefully
- Recover from partial failures
- Maintain index consistency
- Log errors for debugging

## Deployment

**Alfred Integration:**
- JSON API endpoint for search queries
- Result formatting for Alfred display
- Configuration file for customization
- Installation script for dependencies

**Maintenance:**
- Automated index updates
- Log rotation and cleanup
- Performance monitoring
- Index optimization tools

This plan ensures systematic development with comprehensive testing at each phase, leading to a robust and performant semantic search system for Claude conversations.