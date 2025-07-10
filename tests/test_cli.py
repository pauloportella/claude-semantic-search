"""
Tests for the CLI interface.

This module tests the command-line interface functionality including
indexing, searching, and stats commands.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import cli, SemanticSearchCLI
from src.chunker import Chunk
from src.storage import SearchResult


class TestSemanticSearchCLI:
    """Test the SemanticSearchCLI class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cli_instance = SemanticSearchCLI(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test CLI initialization."""
        assert self.cli_instance.data_dir == Path(self.temp_dir)
        assert self.cli_instance.parser is not None
        assert self.cli_instance.chunker is not None
        assert self.cli_instance.embedder is not None
        assert self.cli_instance.storage is not None
    
    def test_scan_claude_projects_not_found(self):
        """Test scanning when Claude projects directory doesn't exist."""
        with pytest.raises(SystemExit):
            self.cli_instance.scan_claude_projects("/nonexistent/path")
    
    def test_scan_claude_projects_empty(self):
        """Test scanning when Claude projects directory is empty."""
        empty_dir = Path(self.temp_dir) / "empty_claude"
        empty_dir.mkdir()
        
        with pytest.raises(SystemExit):
            self.cli_instance.scan_claude_projects(str(empty_dir))
    
    def test_scan_claude_projects_success(self):
        """Test successful scanning of Claude projects."""
        # Create test JSONL files
        claude_dir = Path(self.temp_dir) / "claude_projects"
        claude_dir.mkdir()
        
        test_file1 = claude_dir / "test1.jsonl"
        test_file2 = claude_dir / "test2.jsonl"
        test_file1.write_text('{"test": "data"}')
        test_file2.write_text('{"test": "data"}')
        
        files = self.cli_instance.scan_claude_projects(str(claude_dir))
        
        assert len(files) == 2
        assert all(f.suffix == '.jsonl' for f in files)
    
    @patch('src.cli.SemanticSearchCLI.scan_claude_projects')
    @patch('src.parser.JSONLParser.parse_file')
    @patch('src.chunker.ConversationChunker.chunk_conversation')
    @patch('src.embeddings.EmbeddingGenerator.generate_embeddings')
    @patch('src.storage.HybridStorage.add_chunks')
    def test_index_conversations_success(self, mock_add_chunks, mock_generate_embeddings, 
                                       mock_chunk_conversation, mock_parse_file, 
                                       mock_scan_claude_projects):
        """Test successful conversation indexing."""
        # Mock dependencies
        mock_scan_claude_projects.return_value = [Path("test1.jsonl"), Path("test2.jsonl")]
        
        mock_conversation = Mock()
        mock_conversation.messages = [Mock(), Mock()]
        mock_parse_file.return_value = mock_conversation
        
        mock_chunk = Chunk(
            id="test_chunk",
            text="Test chunk text",
            metadata={"test": "metadata"}
        )
        mock_chunk_conversation.return_value = [mock_chunk]
        
        mock_generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock storage and embedder
        self.cli_instance.storage = Mock()
        self.cli_instance.storage.initialize = Mock()
        self.cli_instance.embedder = Mock()
        self.cli_instance.embedder.is_model_loaded = True
        
        # Test indexing
        files = [Path("test1.jsonl"), Path("test2.jsonl")]
        stats = self.cli_instance.index_conversations(files)
        
        assert stats["files_processed"] == 2
        assert stats["chunks_created"] == 2
        assert stats["chunks_indexed"] == 2
        assert len(stats["errors"]) == 0
    
    @patch('src.cli.SemanticSearchCLI.scan_claude_projects')
    @patch('src.parser.JSONLParser.parse_file')
    def test_index_conversations_parse_error(self, mock_parse_file, mock_scan_claude_projects):
        """Test indexing with parsing errors."""
        mock_scan_claude_projects.return_value = [Path("test1.jsonl")]
        mock_parse_file.side_effect = Exception("Parse error")
        
        # Mock storage
        self.cli_instance.storage = Mock()
        self.cli_instance.storage.initialize = Mock()
        self.cli_instance.embedder = Mock()
        self.cli_instance.embedder.is_model_loaded = True
        
        files = [Path("test1.jsonl")]
        stats = self.cli_instance.index_conversations(files)
        
        assert stats["files_processed"] == 0
        assert stats["files_skipped"] == 0
        assert len(stats["errors"]) == 1
        assert "Parse error" in stats["errors"][0]
    
    @patch('src.embeddings.EmbeddingGenerator.generate_single_embedding')
    @patch('src.storage.HybridStorage.search')
    def test_search_conversations_success(self, mock_search, mock_generate_single_embedding):
        """Test successful conversation search."""
        # Mock dependencies
        mock_generate_single_embedding.return_value = [0.1, 0.2, 0.3]
        
        mock_result = SearchResult(
            chunk_id="test_chunk",
            similarity=0.95,
            text="Test result text",
            metadata={"project_name": "test_project", "session_id": "test_session"}
        )
        mock_search.return_value = [mock_result]
        
        # Mock storage and embedder
        self.cli_instance.storage = Mock()
        self.cli_instance.storage.initialize = Mock()
        self.cli_instance.storage.search = mock_search
        self.cli_instance.embedder = Mock()
        self.cli_instance.embedder.is_model_loaded = True
        self.cli_instance.embedder.generate_single_embedding = mock_generate_single_embedding
        
        # Test search
        results = self.cli_instance.search_conversations("test query")
        
        assert len(results) == 1
        assert results[0]["chunk_id"] == "test_chunk"
        assert results[0]["similarity"] == 0.95
        assert results[0]["text"] == "Test result text"
        assert results[0]["project"] == "test_project"
        assert results[0]["session"] == "test_session"
    
    @patch('src.storage.HybridStorage.get_stats')
    def test_get_index_stats(self, mock_get_stats):
        """Test getting index statistics."""
        mock_stats = {
            "total_chunks": 100,
            "total_sessions": 10,
            "total_projects": 5,
            "faiss_index_size": 1024,
            "database_size": 2048,
            "embedding_dimension": 768
        }
        mock_get_stats.return_value = mock_stats
        
        # Mock storage
        self.cli_instance.storage = Mock()
        self.cli_instance.storage.initialize = Mock()
        self.cli_instance.storage.get_stats = mock_get_stats
        
        stats = self.cli_instance.get_index_stats()
        
        assert stats["total_chunks"] == 100
        assert stats["total_sessions"] == 10
        assert stats["total_projects"] == 5
        assert stats["embedding_dimension"] == 768


class TestCLICommands:
    """Test the CLI commands using Click's testing framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Claude Semantic Search CLI" in result.output
    
    def test_index_command_help(self):
        """Test index command help."""
        result = self.runner.invoke(cli, ['index', '--help'])
        assert result.exit_code == 0
        assert "Index Claude conversations" in result.output
    
    def test_search_command_help(self):
        """Test search command help."""
        result = self.runner.invoke(cli, ['search', '--help'])
        assert result.exit_code == 0
        assert "Search through indexed conversations" in result.output
    
    def test_stats_command_help(self):
        """Test stats command help."""
        result = self.runner.invoke(cli, ['stats', '--help'])
        assert result.exit_code == 0
        assert "Show statistics about the current index" in result.output
    
    @patch('src.cli.SemanticSearchCLI.scan_claude_projects')
    @patch('src.cli.SemanticSearchCLI.index_conversations')
    def test_index_command_success(self, mock_index_conversations, mock_scan_claude_projects):
        """Test successful index command."""
        mock_scan_claude_projects.return_value = [Path("test1.jsonl"), Path("test2.jsonl")]
        mock_index_conversations.return_value = {
            "files_processed": 2,
            "files_skipped": 0,
            "chunks_created": 10,
            "chunks_indexed": 10,
            "errors": [],
            "duration": 5.0
        }
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'index'])
        
        assert result.exit_code == 0
        assert "Indexing complete" in result.output
        assert "Files processed: 2" in result.output
        assert "Chunks created: 10" in result.output
    
    @patch('src.cli.SemanticSearchCLI.scan_claude_projects')
    @patch('src.cli.SemanticSearchCLI.index_conversations')
    def test_index_command_with_errors(self, mock_index_conversations, mock_scan_claude_projects):
        """Test index command with errors."""
        mock_scan_claude_projects.return_value = [Path("test1.jsonl")]
        mock_index_conversations.return_value = {
            "files_processed": 0,
            "files_skipped": 1,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "errors": ["Error processing file"],
            "duration": 1.0
        }
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'index'])
        
        assert result.exit_code == 0
        assert "Indexing complete" in result.output
        assert "Errors: 1" in result.output
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_success(self, mock_search_conversations):
        """Test successful search command."""
        mock_search_conversations.return_value = [
            {
                "chunk_id": "test_chunk",
                "similarity": 0.95,
                "text": "Test result text",
                "project": "test_project",
                "session": "test_session",
                "timestamp": "2023-01-01T00:00:00Z",
                "has_code": True,
                "has_tools": False
            }
        ]
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'search', 'test query'])
        
        assert result.exit_code == 0
        assert "Found 1 results" in result.output
        assert "test_project" in result.output
        assert "Contains code" in result.output
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_json_output(self, mock_search_conversations):
        """Test search command with JSON output."""
        mock_search_conversations.return_value = [
            {
                "chunk_id": "test_chunk",
                "similarity": 0.95,
                "text": "Test result text",
                "project": "test_project",
                "session": "test_session",
                "timestamp": "2023-01-01T00:00:00Z",
                "has_code": False,
                "has_tools": False
            }
        ]
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'search', 'test query', '--json'])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output_data = json.loads(result.output)
        assert "items" in output_data
        assert len(output_data["items"]) == 1
        assert output_data["items"][0]["uid"] == "test_chunk"
        assert output_data["items"][0]["variables"]["similarity"] == 0.95
        assert output_data["items"][0]["variables"]["project"] == "test_project"
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_with_filters(self, mock_search_conversations):
        """Test search command with filters."""
        mock_search_conversations.return_value = []
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir, 
            'search', 'test query',
            '--project', 'test_project',
            '--has-code',
            '--has-tools',
            '--top-k', '5'
        ])
        
        assert result.exit_code == 0
        assert "Found 0 results" in result.output
        
        # Verify the search was called with correct filters
        mock_search_conversations.assert_called_once()
        args, kwargs = mock_search_conversations.call_args
        assert args[0] == "test query"
        assert args[1]['project_name'] == 'test_project'  # filters is second positional arg
        assert args[1]['has_code'] is True
        assert args[1]['has_tools'] is True
        assert args[2] == 5  # top_k is third positional arg
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_error(self, mock_search_conversations):
        """Test search command with error."""
        mock_search_conversations.side_effect = Exception("Search failed")
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'search', 'test query'])
        
        assert result.exit_code == 1
        assert "Search failed" in result.output
    
    @patch('src.cli.SemanticSearchCLI.get_index_stats')
    def test_stats_command_success(self, mock_get_index_stats):
        """Test successful stats command."""
        mock_get_index_stats.return_value = {
            "total_chunks": 100,
            "total_sessions": 10,
            "total_projects": 5,
            "faiss_index_size": 1024,
            "database_size": 2048,
            "total_storage_size": 3072,
            "embedding_dimension": 768,
            "index_type": "flat",
            "chunk_types": {"qa_pair": 50, "code_block": 30, "context_segment": 20}
        }
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'stats'])
        
        assert result.exit_code == 0
        assert "Index Statistics" in result.output
        assert "Total chunks: 100" in result.output
        assert "Total sessions: 10" in result.output
        assert "Total projects: 5" in result.output
        assert "Embedding dimension: 768" in result.output
        assert "qa_pair: 50" in result.output
    
    @patch('src.cli.SemanticSearchCLI.get_index_stats')
    def test_stats_command_error(self, mock_get_index_stats):
        """Test stats command with error."""
        mock_get_index_stats.side_effect = Exception("Stats failed")
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'stats'])
        
        assert result.exit_code == 1
        assert "Failed to get stats" in result.output
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_date_filtering(self, mock_search_conversations):
        """Test search command with date filtering."""
        mock_search_conversations.return_value = [
            {
                "chunk_id": "test_chunk",
                "similarity": 0.95,
                "text": "Test result within date range",
                "project": "test_project",
                "session": "test_session",
                "timestamp": "2025-06-15T12:00:00Z",
                "has_code": False,
            }
        ]
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir, 
            'search', 'test query',
            '--after', '2025-06-01',
            '--before', '2025-06-30'
        ])
        
        assert result.exit_code == 0
        assert "Found 1 results" in result.output
        
        # Verify the search_conversations was called with date filters
        mock_search_conversations.assert_called_once()
        call_args = mock_search_conversations.call_args
        filters = call_args[0][1]  # Second argument is filters
        
        assert 'timestamp' in filters
        assert 'gte' in filters['timestamp']
        assert 'lte' in filters['timestamp']
        assert filters['timestamp']['gte'] == '2025-06-01T00:00:00+00:00'
        assert filters['timestamp']['lte'] == '2025-06-30T23:59:59+00:00'
    
    def test_search_command_invalid_after_date(self):
        """Test search command with invalid after date."""
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'search', 'test query',
            '--after', 'invalid-date'
        ])
        
        assert result.exit_code == 1
        assert "Invalid date format for --after" in result.output
        assert "Use YYYY-MM-DD format" in result.output
    
    def test_search_command_invalid_before_date(self):
        """Test search command with invalid before date."""
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'search', 'test query',
            '--before', 'not-a-date'
        ])
        
        assert result.exit_code == 1
        assert "Invalid date format for --before" in result.output
        assert "Use YYYY-MM-DD format" in result.output
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_only_after_date(self, mock_search_conversations):
        """Test search command with only after date."""
        mock_search_conversations.return_value = []
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'search', 'test query',
            '--after', '2025-01-01'
        ])
        
        assert result.exit_code == 0
        
        # Verify the search_conversations was called with after filter only
        call_args = mock_search_conversations.call_args
        filters = call_args[0][1]
        
        assert 'timestamp' in filters
        assert 'gte' in filters['timestamp']
        assert 'lte' not in filters['timestamp']
        assert filters['timestamp']['gte'] == '2025-01-01T00:00:00+00:00'
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_search_command_only_before_date(self, mock_search_conversations):
        """Test search command with only before date."""
        mock_search_conversations.return_value = []
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'search', 'test query',
            '--before', '2025-12-31'
        ])
        
        assert result.exit_code == 0
        
        # Verify the search_conversations was called with before filter only
        call_args = mock_search_conversations.call_args
        filters = call_args[0][1]
        
        assert 'timestamp' in filters
        assert 'lte' in filters['timestamp']
        assert 'gte' not in filters['timestamp']
        assert filters['timestamp']['lte'] == '2025-12-31T23:59:59+00:00'


class TestCLIIntegration:
    """Integration tests for the CLI."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_data_dir_creation(self):
        """Test that CLI creates data directory if it doesn't exist."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        # This should create the directory
        cli_instance = SemanticSearchCLI(str(nonexistent_dir))
        
        assert nonexistent_dir.exists()
        assert cli_instance.data_dir == nonexistent_dir
    
    def test_cli_workflow_empty_index(self):
        """Test CLI workflow with empty index."""
        # Test stats on empty index
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'stats'])
        
        assert result.exit_code == 0
        assert "Total chunks: 0" in result.output
    
    @patch('src.cli.SemanticSearchCLI.search_conversations')
    def test_cli_search_no_results(self, mock_search_conversations):
        """Test CLI search with no results."""
        mock_search_conversations.return_value = []
        
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, 'search', 'nonexistent'])
        
        assert result.exit_code == 0
        assert "Found 0 results" in result.output


if __name__ == "__main__":
    pytest.main([__file__])