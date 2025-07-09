"""
Embedding generation using sentence transformers.

This module provides the EmbeddingGenerator class that converts text chunks
into dense vector embeddings using pre-trained sentence transformer models.
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from .chunker import Chunk


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-mpnet-base-v2"
    batch_size: int = 16
    max_seq_length: int = 384
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    normalize_embeddings: bool = True
    show_progress: bool = True
    cache_dir: Optional[str] = None
    

@dataclass
class EmbeddingStats:
    """Statistics for embedding generation."""
    total_chunks: int = 0
    total_tokens: int = 0
    generation_time: float = 0.0
    average_chunk_length: float = 0.0
    throughput_chunks_per_second: float = 0.0
    model_info: Dict[str, Any] = field(default_factory=dict)
    

class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence transformers."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model: Optional[SentenceTransformer] = None
        self.logger = logging.getLogger(__name__)
        self._embedding_dim: Optional[int] = None
        
    def load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading model: {self.config.model_name}")
            
            # Set cache directory if specified
            cache_dir = self.config.cache_dir
            if cache_dir:
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
            
            # Load model
            self.model = SentenceTransformer(
                self.config.model_name,
                cache_folder=cache_dir
            )
            
            # Set device after loading
            if self.config.device != "auto":
                self.model.to(self.config.device)
            
            # Configure model parameters
            self.model.max_seq_length = self.config.max_seq_length
            
            # Get embedding dimension
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self._embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise
    
    def generate_embeddings(self, chunks: List[Chunk]) -> List[np.ndarray]:
        """Generate embeddings for a list of chunks."""
        if not self.model:
            self.load_model()
        
        if not chunks:
            return []
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self._generate_embeddings_batch(texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        return embeddings
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            self.load_model()
        
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False
        )
        
        return embedding
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        start_time = time.time()
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=self.config.show_progress,
            convert_to_numpy=True
        )
        
        generation_time = time.time() - start_time
        
        # Log statistics
        if self.config.show_progress:
            throughput = len(texts) / generation_time if generation_time > 0 else 0
            avg_length = np.mean([len(text) for text in texts])
            
            self.logger.info(
                f"Generated {len(texts)} embeddings in {generation_time:.2f}s "
                f"({throughput:.1f} chunks/s, avg length: {avg_length:.0f} chars)"
            )
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def compute_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute similarity matrix for a list of embeddings."""
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                similarity = self.compute_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def find_similar_chunks(self, query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray], 
                          top_k: int = 5) -> List[tuple]:
        """Find most similar chunks to a query embedding."""
        similarities = []
        
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_stats(self, chunks: List[Chunk]) -> EmbeddingStats:
        """Get statistics about embeddings."""
        if not chunks:
            return EmbeddingStats()
        
        total_chunks = len(chunks)
        total_tokens = sum(len(chunk.text.split()) for chunk in chunks)
        avg_chunk_length = np.mean([len(chunk.text) for chunk in chunks])
        
        model_info = {}
        if self.model:
            model_info = {
                "model_name": self.config.model_name,
                "embedding_dimension": self._embedding_dim,
                "max_seq_length": self.config.max_seq_length,
                "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown"
            }
        
        return EmbeddingStats(
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            average_chunk_length=avg_chunk_length,
            model_info=model_info
        )
    
    def save_embeddings(self, chunks: List[Chunk], file_path: str) -> None:
        """Save embeddings to file."""
        embeddings_data = []
        
        for chunk in chunks:
            if chunk.embedding:
                embeddings_data.append({
                    "chunk_id": chunk.id,
                    "embedding": chunk.embedding,
                    "text": chunk.text,
                    "metadata": chunk.metadata
                })
        
        # Save as numpy archive
        np.savez_compressed(file_path, embeddings=embeddings_data)
        self.logger.info(f"Saved {len(embeddings_data)} embeddings to {file_path}")
    
    def load_embeddings(self, file_path: str) -> List[Chunk]:
        """Load embeddings from file."""
        data = np.load(file_path, allow_pickle=True)
        embeddings_data = data["embeddings"]
        
        chunks = []
        for item in embeddings_data:
            chunk = Chunk(
                id=item["chunk_id"],
                text=item["text"],
                metadata=item["metadata"],
                embedding=item["embedding"]
            )
            chunks.append(chunk)
        
        self.logger.info(f"Loaded {len(chunks)} embeddings from {file_path}")
        return chunks
    
    def validate_embeddings(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate generated embeddings."""
        results = {
            "total_chunks": len(chunks),
            "chunks_with_embeddings": 0,
            "embedding_dimension": None,
            "embedding_stats": {},
            "issues": []
        }
        
        embeddings = []
        for chunk in chunks:
            if chunk.embedding:
                results["chunks_with_embeddings"] += 1
                embeddings.append(np.array(chunk.embedding))
                
                # Check embedding dimension
                if results["embedding_dimension"] is None:
                    results["embedding_dimension"] = len(chunk.embedding)
                elif results["embedding_dimension"] != len(chunk.embedding):
                    results["issues"].append(f"Inconsistent embedding dimension for chunk {chunk.id}")
            else:
                results["issues"].append(f"Missing embedding for chunk {chunk.id}")
        
        if embeddings:
            # Check if all embeddings have the same dimension
            if len(set(len(emb) for emb in embeddings)) == 1:
                # All embeddings have same dimension, can create array
                embeddings_array = np.array(embeddings)
                results["embedding_stats"] = {
                    "mean": np.mean(embeddings_array, axis=0).tolist(),
                    "std": np.std(embeddings_array, axis=0).tolist(),
                    "min": np.min(embeddings_array, axis=0).tolist(),
                    "max": np.max(embeddings_array, axis=0).tolist(),
                    "norm_mean": np.mean(np.linalg.norm(embeddings_array, axis=1)),
                    "norm_std": np.std(np.linalg.norm(embeddings_array, axis=1))
                }
            else:
                # Different dimensions, compute stats individually
                norms = [np.linalg.norm(emb) for emb in embeddings]
                results["embedding_stats"] = {
                    "norm_mean": np.mean(norms),
                    "norm_std": np.std(norms),
                    "note": "Embeddings have different dimensions, limited stats computed"
                }
        
        return results
    
    def benchmark_model(self, test_texts: List[str], warmup_runs: int = 3) -> Dict[str, Any]:
        """Benchmark the model performance."""
        if not self.model:
            self.load_model()
        
        # Warmup runs
        for _ in range(warmup_runs):
            self.model.encode(test_texts[:min(5, len(test_texts))], show_progress_bar=False)
        
        # Benchmark runs
        batch_sizes = [1, 4, 8, 16, 32]
        results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_texts):
                continue
                
            # Test with different batch sizes
            start_time = time.time()
            
            # Process in batches
            for i in range(0, len(test_texts), batch_size):
                batch = test_texts[i:i + batch_size]
                self.model.encode(batch, show_progress_bar=False)
            
            total_time = time.time() - start_time
            throughput = len(test_texts) / total_time
            
            results[f"batch_size_{batch_size}"] = {
                "total_time": total_time,
                "throughput": throughput,
                "avg_time_per_text": total_time / len(test_texts)
            }
        
        # Memory usage (if GPU)
        memory_info = {}
        if torch.cuda.is_available():
            memory_info = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        
        return {
            "model_name": self.config.model_name,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "embedding_dimension": self._embedding_dim,
            "test_texts_count": len(test_texts),
            "performance": results,
            "memory_info": memory_info
        }
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {}
        
        return {
            "model_name": self.config.model_name,
            "embedding_dimension": self._embedding_dim,
            "max_seq_length": self.config.max_seq_length,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "model_config": self.model._modules if hasattr(self.model, '_modules') else {}
        }