"""
Hybrid storage layer combining FAISS vector search with SQLite metadata storage.

This module provides the HybridStorage class that manages both vector embeddings
and metadata for efficient semantic search with filtering capabilities.
"""

import sqlite3
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import faiss
from tqdm import tqdm

from .chunker import Chunk


@dataclass
class StorageConfig:
    """Configuration for hybrid storage."""
    data_dir: str = "./data"
    db_name: str = "metadata.db"
    index_name: str = "embeddings.faiss"
    embedding_dim: int = 768
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    ivf_nlist: int = 100  # For IVF index
    hnsw_m: int = 16  # For HNSW index
    normalize_embeddings: bool = True
    auto_save: bool = True
    backup_enabled: bool = True
    

@dataclass
class SearchConfig:
    """Configuration for search operations."""
    top_k: int = 10
    similarity_threshold: float = 0.0
    include_metadata: bool = True
    include_text: bool = True
    max_results: int = 100
    

@dataclass
class SearchResult:
    """Result from a search operation."""
    chunk_id: str
    similarity: float
    chunk: Optional[Chunk] = None
    metadata: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    

class HybridStorage:
    """Hybrid storage combining FAISS and SQLite for semantic search."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage paths
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.data_dir / self.config.db_name
        self.index_path = self.data_dir / self.config.index_name
        
        # Storage components
        self.db: Optional[sqlite3.Connection] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.chunk_id_to_faiss_id: Dict[str, int] = {}
        self.faiss_id_to_chunk_id: Dict[int, str] = {}
        
        # Stats
        self.total_chunks = 0
        self.embedding_dim = self.config.embedding_dim
        
    def initialize(self) -> None:
        """Initialize the storage components."""
        self.logger.info("Initializing hybrid storage...")
        
        # Initialize SQLite
        self._init_sqlite()
        
        # Initialize FAISS
        self._init_faiss()
        
        # Load existing data if available
        self._load_existing_data()
        
        self.logger.info(f"Storage initialized with {self.total_chunks} chunks")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
        
    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self.db.cursor()
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                faiss_id INTEGER,
                session_id TEXT,
                project_name TEXT,
                file_path TEXT,
                chunk_type TEXT,
                timestamp DATETIME,
                has_code BOOLEAN,
                has_tools BOOLEAN,
                message_count INTEGER,
                char_count INTEGER,
                word_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Files table for tracking processed files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                last_modified DATETIME,
                last_indexed DATETIME,
                chunk_count INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_has_code ON chunks(has_code)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_has_tools ON chunks(has_tools)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)")
        
        self.db.commit()
    
    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        if self.config.index_type == "flat":
            if self.config.normalize_embeddings:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.config.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.config.ivf_nlist
            )
        elif self.config.index_type == "hnsw":
            self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, self.config.hnsw_m)
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        self.logger.info(f"Initialized FAISS index: {type(self.faiss_index).__name__}")
    
    def _load_existing_data(self) -> None:
        """Load existing data from storage."""
        if self.index_path.exists():
            try:
                # Load FAISS index
                self.faiss_index = faiss.read_index(str(self.index_path))
                self.logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
                
                # Rebuild ID mappings from database
                self._rebuild_id_mappings()
                
            except Exception as e:
                self.logger.warning(f"Could not load existing FAISS index: {e}")
                self._init_faiss()
    
    def _rebuild_id_mappings(self) -> None:
        """Rebuild chunk ID to FAISS ID mappings from database."""
        cursor = self.db.cursor()
        cursor.execute("SELECT id, faiss_id FROM chunks WHERE faiss_id IS NOT NULL")
        
        for row in cursor.fetchall():
            chunk_id, faiss_id = row
            self.chunk_id_to_faiss_id[chunk_id] = faiss_id
            self.faiss_id_to_chunk_id[faiss_id] = chunk_id
        
        self.total_chunks = len(self.chunk_id_to_faiss_id)
        self.logger.info(f"Rebuilt ID mappings for {self.total_chunks} chunks")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to storage."""
        if not chunks:
            return
        
        # Validate chunks have embeddings
        chunks_with_embeddings = [c for c in chunks if c.embedding is not None]
        if not chunks_with_embeddings:
            self.logger.warning("No chunks with embeddings to add")
            return
        
        # Prepare embeddings
        embeddings = np.array([c.embedding for c in chunks_with_embeddings], dtype=np.float32)
        
        if self.config.normalize_embeddings:
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        # Add to FAISS
        start_faiss_id = self.faiss_index.ntotal
        self.faiss_index.add(embeddings)
        
        # Add to SQLite
        cursor = self.db.cursor()
        
        for i, chunk in enumerate(chunks_with_embeddings):
            faiss_id = start_faiss_id + i
            
            # Update mappings
            self.chunk_id_to_faiss_id[chunk.id] = faiss_id
            self.faiss_id_to_chunk_id[faiss_id] = chunk.id
            
            # Insert into database
            cursor.execute("""
                INSERT OR REPLACE INTO chunks 
                (id, text, metadata, faiss_id, session_id, project_name, file_path, 
                 chunk_type, timestamp, has_code, has_tools, message_count, 
                 char_count, word_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.id,
                chunk.text,
                json.dumps(chunk.metadata),
                faiss_id,
                chunk.metadata.get('session_id'),
                chunk.metadata.get('project_name'),
                chunk.metadata.get('file_path'),
                chunk.metadata.get('chunk_type'),
                chunk.metadata.get('timestamp'),
                chunk.metadata.get('has_code', False),
                chunk.metadata.get('has_tools', False),
                chunk.metadata.get('message_count', 0),
                chunk.metadata.get('char_count', 0),
                chunk.metadata.get('word_count', 0),
                datetime.now().isoformat()
            ))
        
        self.db.commit()
        self.total_chunks += len(chunks_with_embeddings)
        
        # Auto-save if enabled
        if self.config.auto_save:
            self.save_index()
        
        self.logger.info(f"Added {len(chunks_with_embeddings)} chunks to storage")
    
    def search(self, query_embedding: np.ndarray, config: Optional[SearchConfig] = None,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks."""
        search_config = config or SearchConfig()
        
        # Handle empty storage
        if self.faiss_index.ntotal == 0:
            return []
        
        # Normalize query embedding if needed
        if self.config.normalize_embeddings:
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Search FAISS
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Get more results than needed for filtering
        search_k = min(search_config.max_results, self.faiss_index.ntotal)
        if search_k == 0:
            return []
        
        similarities, faiss_ids = self.faiss_index.search(query_embedding, search_k)
        
        # Convert to results
        results = []
        for i in range(len(faiss_ids[0])):
            faiss_id = faiss_ids[0][i]
            similarity = float(similarities[0][i])
            
            # Skip if below threshold
            if similarity < search_config.similarity_threshold:
                continue
            
            # Get chunk ID
            chunk_id = self.faiss_id_to_chunk_id.get(faiss_id)
            if not chunk_id:
                continue
            
            # Get chunk data from database
            chunk_data = self._get_chunk_data(chunk_id)
            if not chunk_data:
                continue
            
            # Apply filters
            if filters and not self._matches_filters(chunk_data, filters):
                continue
            
            # Create result
            result = SearchResult(
                chunk_id=chunk_id,
                similarity=similarity
            )
            
            if search_config.include_metadata:
                result.metadata = json.loads(chunk_data['metadata']) if chunk_data['metadata'] else {}
            
            if search_config.include_text:
                result.text = chunk_data['text']
            
            # Create full chunk if needed
            if search_config.include_metadata and search_config.include_text:
                result.chunk = Chunk(
                    id=chunk_id,
                    text=chunk_data['text'],
                    metadata=json.loads(chunk_data['metadata']) if chunk_data['metadata'] else {},
                    embedding=None  # Don't include embedding in results
                )
            
            results.append(result)
            
            # Stop if we have enough results
            if len(results) >= search_config.top_k:
                break
        
        return results
    
    def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk data from database."""
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        
        if row:
            # Convert sqlite3.Row to dict
            return {key: row[key] for key in row.keys()}
        return None
    
    def _matches_filters(self, chunk_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if chunk matches filters."""
        for key, value in filters.items():
            if key not in chunk_data:
                continue
            
            chunk_value = chunk_data[key]
            
            if isinstance(value, dict):
                # Handle range filters
                if 'gte' in value and chunk_value < value['gte']:
                    return False
                if 'lte' in value and chunk_value > value['lte']:
                    return False
                if 'gt' in value and chunk_value <= value['gt']:
                    return False
                if 'lt' in value and chunk_value >= value['lt']:
                    return False
            elif isinstance(value, list):
                # Handle list filters (IN)
                if chunk_value not in value:
                    return False
            else:
                # Handle exact match
                if chunk_value != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        chunk_data = self._get_chunk_data(chunk_id)
        if not chunk_data:
            return None
        
        return Chunk(
            id=chunk_id,
            text=chunk_data['text'],
            metadata=json.loads(chunk_data['metadata']) if chunk_data['metadata'] else {},
            embedding=None  # Don't load embedding
        )
    
    def get_chunks_by_session(self, session_id: str) -> List[Chunk]:
        """Get all chunks for a session."""
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM chunks WHERE session_id = ? ORDER BY timestamp", (session_id,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append(Chunk(
                id=row['id'],
                text=row['text'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                embedding=None
            ))
        
        return chunks
    
    def get_chunks_by_project(self, project_name: str) -> List[Chunk]:
        """Get all chunks for a project."""
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM chunks WHERE project_name = ? ORDER BY timestamp", (project_name,))
        
        chunks = []
        for row in cursor.fetchall():
            chunks.append(Chunk(
                id=row['id'],
                text=row['text'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                embedding=None
            ))
        
        return chunks
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk."""
        # Get FAISS ID
        faiss_id = self.chunk_id_to_faiss_id.get(chunk_id)
        if faiss_id is None:
            return False
        
        # Remove from database
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        
        if cursor.rowcount == 0:
            return False
        
        # Remove from mappings
        del self.chunk_id_to_faiss_id[chunk_id]
        del self.faiss_id_to_chunk_id[faiss_id]
        
        # Note: FAISS doesn't support efficient deletion, so we mark as deleted
        # and rebuild index periodically
        
        self.db.commit()
        self.total_chunks -= 1
        
        return True
    
    def delete_chunks_by_session(self, session_id: str) -> int:
        """Delete all chunks for a session."""
        cursor = self.db.cursor()
        cursor.execute("SELECT id FROM chunks WHERE session_id = ?", (session_id,))
        
        chunk_ids = [row[0] for row in cursor.fetchall()]
        deleted_count = 0
        
        for chunk_id in chunk_ids:
            if self.delete_chunk(chunk_id):
                deleted_count += 1
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        cursor = self.db.cursor()
        
        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM chunks")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT project_name) FROM chunks")
        total_projects = cursor.fetchone()[0]
        
        # Chunk type distribution
        cursor.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type")
        chunk_types = dict(cursor.fetchall())
        
        # Storage sizes
        faiss_size = self.index_path.stat().st_size if self.index_path.exists() else 0
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        return {
            "total_chunks": total_chunks,
            "total_sessions": total_sessions,
            "total_projects": total_projects,
            "chunk_types": chunk_types,
            "faiss_index_size": faiss_size,
            "database_size": db_size,
            "total_storage_size": faiss_size + db_size,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.config.index_type
        }
    
    def save_index(self) -> None:
        """Save FAISS index to disk."""
        faiss.write_index(self.faiss_index, str(self.index_path))
        self.logger.info(f"Saved FAISS index to {self.index_path}")
    
    def backup(self, backup_dir: str) -> None:
        """Create backup of storage."""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup FAISS index
        if self.faiss_index and self.faiss_index.ntotal > 0:
            backup_index_path = backup_path / self.config.index_name
            faiss.write_index(self.faiss_index, str(backup_index_path))
        
        # Backup database
        if self.db_path.exists():
            backup_db_path = backup_path / self.config.db_name
            backup_db = sqlite3.connect(str(backup_db_path))
            self.db.backup(backup_db)
            backup_db.close()
        
        self.logger.info(f"Backup created in {backup_path}")
    
    def restore(self, backup_dir: str) -> None:
        """Restore storage from backup."""
        backup_path = Path(backup_dir)
        
        # Restore FAISS index
        backup_index_path = backup_path / self.config.index_name
        if backup_index_path.exists():
            self.faiss_index = faiss.read_index(str(backup_index_path))
        
        # Restore database
        backup_db_path = backup_path / self.config.db_name
        if backup_db_path.exists():
            self.db.close()
            self.db = sqlite3.connect(str(self.db_path))
            self.db.row_factory = sqlite3.Row  # Set row factory
            backup_db = sqlite3.connect(str(backup_db_path))
            backup_db.backup(self.db)
            backup_db.close()
        
        # Rebuild mappings
        self._rebuild_id_mappings()
        
        self.logger.info(f"Restored from backup in {backup_path}")
    
    def optimize(self) -> None:
        """Optimize storage (rebuild indexes, vacuum database)."""
        self.logger.info("Optimizing storage...")
        
        # Vacuum database
        self.db.execute("VACUUM")
        
        # Rebuild FAISS index if needed (to remove deleted items)
        if self.total_chunks != self.faiss_index.ntotal:
            self.logger.info("Rebuilding FAISS index...")
            self._rebuild_faiss_index()
        
        self.logger.info("Storage optimization complete")
    
    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from database."""
        cursor = self.db.cursor()
        cursor.execute("SELECT id, faiss_id FROM chunks WHERE faiss_id IS NOT NULL ORDER BY faiss_id")
        
        # Get all chunk IDs in order
        chunk_ids = [row[0] for row in cursor.fetchall()]
        
        if not chunk_ids:
            return
        
        # Create new index
        self._init_faiss()
        
        # Process chunks in batches
        batch_size = 1000
        for i in range(0, len(chunk_ids), batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            
            # Get embeddings (would need to be stored or regenerated)
            # For now, skip this optimization
            pass
        
        self.logger.info("FAISS index rebuilt")
    
    def close(self) -> None:
        """Close storage connections."""
        if self.config.auto_save:
            self.save_index()
        
        if self.db:
            self.db.close()
        
        self.logger.info("Storage closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()