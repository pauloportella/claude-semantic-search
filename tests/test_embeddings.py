"""
Tests for the embeddings module.
"""

import os
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.embeddings import EmbeddingGenerator, EmbeddingConfig, EmbeddingStats
from src.chunker import Chunk


class TestEmbeddingConfig:
    """Test suite for EmbeddingConfig."""
    
    def test_config_defaults(self):
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.model_name == "all-mpnet-base-v2"
        assert config.batch_size == 16
        assert config.max_seq_length == 384
        assert config.device == "auto"
        assert config.normalize_embeddings is True
        assert config.show_progress is True
        assert config.cache_dir is None
    
    def test_config_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            max_seq_length=512,
            device="cpu",
            normalize_embeddings=False,
            show_progress=False,
            cache_dir="/tmp/models"
        )
        
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.max_seq_length == 512
        assert config.device == "cpu"
        assert config.normalize_embeddings is False
        assert config.show_progress is False
        assert config.cache_dir == "/tmp/models"


class TestEmbeddingStats:
    """Test suite for EmbeddingStats."""
    
    def test_stats_defaults(self):
        """Test EmbeddingStats default values."""
        stats = EmbeddingStats()
        assert stats.total_chunks == 0
        assert stats.total_tokens == 0
        assert stats.generation_time == 0.0
        assert stats.average_chunk_length == 0.0
        assert stats.throughput_chunks_per_second == 0.0
        assert stats.model_info == {}


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",  # Smaller model for faster tests
            batch_size=4,
            show_progress=False
        )
        self.generator = EmbeddingGenerator(self.config)
        
        # Create test chunks
        self.test_chunks = [
            Chunk(
                id="chunk_001",
                text="This is a test chunk about machine learning and AI.",
                metadata={"type": "test", "index": 0}
            ),
            Chunk(
                id="chunk_002", 
                text="Another chunk discussing natural language processing techniques.",
                metadata={"type": "test", "index": 1}
            ),
            Chunk(
                id="chunk_003",
                text="A third chunk containing information about deep learning models.",
                metadata={"type": "test", "index": 2}
            )
        ]
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        assert self.generator.config.model_name == "all-MiniLM-L6-v2"
        assert self.generator.model is None
        assert self.generator._embedding_dim is None
        assert not self.generator.is_model_loaded
    
    def test_generator_default_config(self):
        """Test generator with default config."""
        generator = EmbeddingGenerator()
        assert generator.config.model_name == "all-mpnet-base-v2"
        assert generator.config.batch_size == 16
    
    @patch('src.embeddings.SentenceTransformer')
    def test_load_model_success(self, mock_st):
        """Test successful model loading."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        self.generator.load_model()
        
        # Check model was loaded
        mock_st.assert_called_once_with(
            "all-MiniLM-L6-v2",
            cache_folder=None
        )
        assert self.generator.model is not None
        assert self.generator._embedding_dim == 384
        assert self.generator.is_model_loaded
    
    @patch('src.embeddings.SentenceTransformer')
    def test_load_model_with_cache_dir(self, mock_st):
        """Test model loading with cache directory."""
        config = EmbeddingConfig(cache_dir="/tmp/cache")
        generator = EmbeddingGenerator(config)
        
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        generator.load_model()
        
        mock_st.assert_called_once_with(
            "all-mpnet-base-v2",
            cache_folder="/tmp/cache"
        )
        assert os.environ.get("SENTENCE_TRANSFORMERS_HOME") == "/tmp/cache"
    
    @patch('src.embeddings.SentenceTransformer')
    def test_load_model_failure(self, mock_st):
        """Test model loading failure."""
        mock_st.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            self.generator.load_model()
    
    @patch('src.embeddings.SentenceTransformer')
    def test_generate_single_embedding(self, mock_st):
        """Test generating a single embedding."""
        # Mock the model
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model
        
        # Generate embedding
        embedding = self.generator.generate_single_embedding("test text")
        
        # Check result
        np.testing.assert_array_equal(embedding, mock_embedding)
        mock_model.encode.assert_called_once_with(
            "test text",
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    @patch('src.embeddings.SentenceTransformer')
    def test_generate_embeddings_batch(self, mock_st):
        """Test generating embeddings for multiple chunks."""
        # Mock the model
        mock_model = Mock()
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model
        
        # Generate embeddings
        embeddings = self.generator.generate_embeddings(self.test_chunks)
        
        # Check results
        assert len(embeddings) == 3
        np.testing.assert_array_equal(embeddings[0], mock_embeddings[0])
        np.testing.assert_array_equal(embeddings[1], mock_embeddings[1])
        np.testing.assert_array_equal(embeddings[2], mock_embeddings[2])
        
        # Check chunks were updated
        for i, chunk in enumerate(self.test_chunks):
            assert chunk.embedding == mock_embeddings[i].tolist()
        
        # Check model was called correctly
        expected_texts = [chunk.text for chunk in self.test_chunks]
        mock_model.encode.assert_called_once_with(
            expected_texts,
            batch_size=4,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    def test_generate_embeddings_empty_list(self):
        """Test generating embeddings for empty list."""
        embeddings = self.generator.generate_embeddings([])
        assert embeddings == []
    
    def test_compute_similarity(self):
        """Test cosine similarity computation."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        emb3 = np.array([1.0, 0.0, 0.0])
        
        # Test orthogonal vectors
        similarity = self.generator.compute_similarity(emb1, emb2)
        assert abs(similarity) < 1e-10  # Should be 0
        
        # Test identical vectors
        similarity = self.generator.compute_similarity(emb1, emb3)
        assert abs(similarity - 1.0) < 1e-10  # Should be 1
    
    def test_compute_similarity_matrix(self):
        """Test similarity matrix computation."""
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        ]
        
        matrix = self.generator.compute_similarity_matrix(embeddings)
        
        assert matrix.shape == (3, 3)
        assert abs(matrix[0, 0] - 1.0) < 1e-10  # Self-similarity
        assert abs(matrix[1, 1] - 1.0) < 1e-10  # Self-similarity
        assert abs(matrix[2, 2] - 1.0) < 1e-10  # Self-similarity
        assert abs(matrix[0, 2] - 1.0) < 1e-10  # Identical vectors
        assert abs(matrix[0, 1]) < 1e-10  # Orthogonal vectors
    
    def test_find_similar_chunks(self):
        """Test finding similar chunks."""
        query_embedding = np.array([1.0, 0.0, 0.0])
        chunk_embeddings = [
            np.array([1.0, 0.0, 0.0]),  # Identical
            np.array([0.0, 1.0, 0.0]),  # Orthogonal
            np.array([0.7, 0.7, 0.0])   # Partially similar
        ]
        
        similar_chunks = self.generator.find_similar_chunks(
            query_embedding, 
            chunk_embeddings, 
            top_k=2
        )
        
        assert len(similar_chunks) == 2
        assert similar_chunks[0][0] == 0  # Most similar is index 0
        assert similar_chunks[0][1] == 1.0  # Perfect similarity
        assert similar_chunks[1][0] == 2  # Second most similar is index 2
        assert similar_chunks[1][1] > 0.5  # Positive similarity
    
    def test_get_embedding_stats(self):
        """Test embedding statistics generation."""
        # Set up chunks with embeddings
        for i, chunk in enumerate(self.test_chunks):
            chunk.embedding = [0.1 * i, 0.2 * i, 0.3 * i]
        
        stats = self.generator.get_embedding_stats(self.test_chunks)
        
        assert stats.total_chunks == 3
        assert stats.total_tokens > 0  # Should count words
        assert stats.average_chunk_length > 0
        assert stats.model_info == {}  # Model not loaded yet
    
    @patch('src.embeddings.SentenceTransformer')
    def test_get_embedding_stats_with_model(self, mock_st):
        """Test embedding statistics with loaded model."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st.return_value = mock_model
        
        self.generator.load_model()
        stats = self.generator.get_embedding_stats(self.test_chunks)
        
        assert stats.model_info["model_name"] == "all-MiniLM-L6-v2"
        assert stats.model_info["embedding_dimension"] == 384
        assert stats.model_info["device"] == "cpu"
    
    def test_get_embedding_stats_empty(self):
        """Test embedding statistics for empty list."""
        stats = self.generator.get_embedding_stats([])
        assert stats.total_chunks == 0
        assert stats.total_tokens == 0
        assert stats.average_chunk_length == 0
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        # Set up chunks with embeddings
        for i, chunk in enumerate(self.test_chunks):
            chunk.embedding = [0.1 * i, 0.2 * i, 0.3 * i]
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save embeddings
            self.generator.save_embeddings(self.test_chunks, tmp_path)
            
            # Load embeddings
            loaded_chunks = self.generator.load_embeddings(tmp_path)
            
            # Check results
            assert len(loaded_chunks) == len(self.test_chunks)
            for original, loaded in zip(self.test_chunks, loaded_chunks):
                assert loaded.id == original.id
                assert loaded.text == original.text
                assert loaded.metadata == original.metadata
                assert loaded.embedding == original.embedding
        
        finally:
            os.unlink(tmp_path)
    
    def test_validate_embeddings_success(self):
        """Test embedding validation with valid embeddings."""
        # Set up chunks with valid embeddings
        for i, chunk in enumerate(self.test_chunks):
            chunk.embedding = [0.1 * i, 0.2 * i, 0.3 * i]
        
        results = self.generator.validate_embeddings(self.test_chunks)
        
        assert results["total_chunks"] == 3
        assert results["chunks_with_embeddings"] == 3
        assert results["embedding_dimension"] == 3
        assert len(results["issues"]) == 0
        assert "embedding_stats" in results
    
    def test_validate_embeddings_issues(self):
        """Test embedding validation with issues."""
        # Set up chunks with missing and inconsistent embeddings
        self.test_chunks[0].embedding = [0.1, 0.2, 0.3]
        self.test_chunks[1].embedding = None  # Missing
        self.test_chunks[2].embedding = [0.1, 0.2]  # Different dimension
        
        results = self.generator.validate_embeddings(self.test_chunks)
        
        assert results["total_chunks"] == 3
        assert results["chunks_with_embeddings"] == 2
        assert results["embedding_dimension"] == 3
        assert len(results["issues"]) == 2
        assert "Missing embedding" in results["issues"][0]
        assert "Inconsistent embedding dimension" in results["issues"][1]
        
        # Check that stats are computed for different dimensions
        assert "embedding_stats" in results
        assert "note" in results["embedding_stats"]
        assert "norm_mean" in results["embedding_stats"]
    
    @patch('src.embeddings.SentenceTransformer')
    def test_benchmark_model(self, mock_st):
        """Test model benchmarking."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model
        
        test_texts = ["text1", "text2", "text3", "text4", "text5"]
        
        results = self.generator.benchmark_model(test_texts, warmup_runs=1)
        
        assert results["model_name"] == "all-MiniLM-L6-v2"
        assert results["embedding_dimension"] == 3
        assert results["test_texts_count"] == 5
        assert "performance" in results
        assert "batch_size_1" in results["performance"]
        assert "batch_size_4" in results["performance"]
        
        # Check performance metrics
        perf = results["performance"]["batch_size_1"]
        assert "total_time" in perf
        assert "throughput" in perf
        assert "avg_time_per_text" in perf
    
    @patch('src.embeddings.SentenceTransformer')
    def test_get_model_info(self, mock_st):
        """Test getting model information."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.device = "cpu"
        mock_st.return_value = mock_model
        
        # Test without loaded model
        info = self.generator.get_model_info()
        assert info == {}
        
        # Test with loaded model
        self.generator.load_model()
        info = self.generator.get_model_info()
        
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] == 384
        assert info["max_seq_length"] == 384
        assert info["device"] == "cpu"
    
    @patch('src.embeddings.SentenceTransformer')
    def test_embedding_dimension_property(self, mock_st):
        """Test embedding dimension property."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        # Test before model load
        assert self.generator.embedding_dimension is None
        
        # Test after model load
        self.generator.load_model()
        assert self.generator.embedding_dimension == 768
    
    def test_is_model_loaded_property(self):
        """Test is_model_loaded property."""
        assert not self.generator.is_model_loaded
        
        # Mock model loading
        self.generator.model = Mock()
        assert self.generator.is_model_loaded