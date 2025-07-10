"""
Tests for the file watcher module.
"""

import os
import time
import tempfile
import shutil
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.watcher import ConversationFileHandler, ConversationWatcher
from src.cli import SemanticSearchCLI


class TestConversationFileHandler:
    """Test suite for ConversationFileHandler."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_cli = Mock(spec=SemanticSearchCLI)
        self.handler = ConversationFileHandler(self.mock_cli, debounce_seconds=0.1)
    
    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, 'handler') and self.handler.timer and self.handler.timer.is_alive():
            self.handler.timer.cancel()
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_handler_initialization(self):
        """Test handler initialization."""
        assert self.handler.cli_instance == self.mock_cli
        assert self.handler.debounce_seconds == 0.1
        assert len(self.handler.pending_files) == 0
        assert self.handler.last_trigger_time is None
        assert self.handler.timer is None
    
    def test_is_conversation_file(self):
        """Test conversation file detection."""
        assert self.handler._is_conversation_file("test.jsonl")
        assert self.handler._is_conversation_file("/path/to/conversation.jsonl")
        assert not self.handler._is_conversation_file("test.txt")
        assert not self.handler._is_conversation_file("test.json")
        assert not self.handler._is_conversation_file("test")
    
    def test_on_created_event(self):
        """Test file creation event handling."""
        # Create mock event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/conversation.jsonl"
        
        # Process event
        self.handler.on_created(mock_event)
        
        # Check that file was scheduled
        assert "/test/conversation.jsonl" in self.handler.pending_files
        assert self.handler.timer is not None
    
    def test_on_modified_event(self):
        """Test file modification event handling."""
        # Create mock event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/conversation.jsonl"
        
        # Process event
        self.handler.on_modified(mock_event)
        
        # Check that file was scheduled
        assert "/test/conversation.jsonl" in self.handler.pending_files
        assert self.handler.timer is not None
    
    def test_ignore_non_conversation_files(self):
        """Test that non-conversation files are ignored."""
        # Create mock event for non-JSONL file
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/other_file.txt"
        
        # Process event
        self.handler.on_created(mock_event)
        
        # Check that file was not scheduled
        assert len(self.handler.pending_files) == 0
        assert self.handler.timer is None
    
    def test_ignore_directory_events(self):
        """Test that directory events are ignored."""
        # Create mock event for directory
        mock_event = Mock()
        mock_event.is_directory = True
        mock_event.src_path = "/test/directory"
        
        # Process event
        self.handler.on_created(mock_event)
        
        # Check that event was ignored
        assert len(self.handler.pending_files) == 0
        assert self.handler.timer is None
    
    def test_debouncing(self):
        """Test that multiple events are debounced."""
        # Create multiple mock events
        events = []
        for i in range(3):
            event = Mock()
            event.is_directory = False
            event.src_path = f"/test/conversation_{i}.jsonl"
            events.append(event)
        
        # Process events quickly
        for event in events:
            self.handler.on_created(event)
        
        # Check that all files are pending
        assert len(self.handler.pending_files) == 3
        for i in range(3):
            assert f"/test/conversation_{i}.jsonl" in self.handler.pending_files
        
        # Should have only one timer
        assert self.handler.timer is not None
    
    @patch('src.watcher.Path')
    def test_trigger_indexing(self, mock_path_class):
        """Test indexing trigger mechanism."""
        # Setup mock Path behavior
        mock_path = Mock()
        mock_path.parent = "/test"
        mock_path.glob.return_value = [Path("/test/conv1.jsonl"), Path("/test/conv2.jsonl")]
        mock_path_class.return_value = mock_path
        
        # Setup mock CLI behavior
        self.mock_cli.index_conversations.return_value = {
            'files_processed': 2,
            'files_skipped': 0,
            'chunks_indexed': 10,
            'duration': 1.5,
            'errors': []
        }
        
        # Add files to pending
        self.handler.pending_files.add("/test/conv1.jsonl")
        self.handler.pending_files.add("/test/conv2.jsonl")
        
        # Trigger indexing
        self.handler._trigger_indexing()
        
        # Check that indexing was called
        self.mock_cli.index_conversations.assert_called_once()
        
        # Check that pending files were cleared
        assert len(self.handler.pending_files) == 0
        
        # Check that last trigger time was set
        assert self.handler.last_trigger_time is not None
    
    def test_trigger_indexing_with_errors(self):
        """Test indexing trigger with errors."""
        # Setup mock CLI to raise exception
        self.mock_cli.index_conversations.side_effect = Exception("Test error")
        
        # Add file to pending
        self.handler.pending_files.add("/test/conv1.jsonl")
        
        # Trigger indexing (should not raise exception)
        self.handler._trigger_indexing()
        
        # Check that pending files were cleared despite error
        assert len(self.handler.pending_files) == 0


class TestConversationWatcher:
    """Test suite for ConversationWatcher."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.claude_dir = os.path.join(self.temp_dir, "claude_projects")
        os.makedirs(self.claude_dir)
        
        # Create test JSONL file
        test_file = os.path.join(self.claude_dir, "test.jsonl")
        with open(test_file, 'w') as f:
            f.write('{"test": "data"}\n')
    
    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_watcher_initialization(self):
        """Test watcher initialization."""
        watcher = ConversationWatcher(self.temp_dir, debounce_seconds=1)
        
        assert watcher.data_dir == self.temp_dir
        assert watcher.debounce_seconds == 1
        assert isinstance(watcher.cli_instance, SemanticSearchCLI)
        assert not watcher.is_running
    
    def test_watcher_with_nonexistent_directory(self):
        """Test watcher with non-existent directory."""
        watcher = ConversationWatcher(self.temp_dir)
        
        with pytest.raises(FileNotFoundError):
            watcher.start_watching("/nonexistent/path")
    
    @patch('src.watcher.SemanticSearchCLI')
    def test_start_stop_watcher(self, mock_cli_class):
        """Test starting and stopping watcher."""
        # Mock CLI instance
        mock_cli = Mock()
        mock_cli.storage = Mock()
        mock_cli.embedder = Mock()
        mock_cli.embedder.is_model_loaded = True
        mock_cli_class.return_value = mock_cli
        
        watcher = ConversationWatcher(self.temp_dir)
        
        # Start watcher in thread
        def start_watcher():
            try:
                watcher.start_watching(self.claude_dir)
            except Exception:
                pass  # Expected when we stop it
        
        watcher_thread = threading.Thread(target=start_watcher)
        watcher_thread.start()
        
        # Wait a bit for watcher to start
        time.sleep(0.1)
        
        # Check that watcher is running
        assert watcher.is_running
        
        # Stop watcher
        watcher.stop_watching()
        
        # Wait for thread to finish
        watcher_thread.join(timeout=1)
        
        # Check that watcher stopped
        assert not watcher.is_running
    
    @patch('src.watcher.SemanticSearchCLI')
    def test_get_status(self, mock_cli_class):
        """Test getting watcher status."""
        # Mock CLI instance
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        watcher = ConversationWatcher(self.temp_dir, debounce_seconds=3)
        
        status = watcher.get_status()
        
        assert status["is_running"] is False
        assert status["watching_path"] is None
        assert status["pending_files"] == 0
        assert status["last_trigger_time"] is None
        assert status["debounce_seconds"] == 3
    
    def test_run_watcher_function(self):
        """Test run_watcher function."""
        from src.watcher import run_watcher
        
        # This should raise FileNotFoundError for non-existent directory
        with pytest.raises(FileNotFoundError):
            run_watcher(self.temp_dir, "/nonexistent/path", 1)


class TestWatcherIntegration:
    """Integration tests for the watcher system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.claude_dir = os.path.join(self.temp_dir, "claude_projects")
        os.makedirs(self.claude_dir)
    
    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('src.watcher.SemanticSearchCLI')
    def test_file_creation_triggers_indexing(self, mock_cli_class):
        """Test that creating a file triggers indexing."""
        # Mock CLI instance
        mock_cli = Mock()
        mock_cli.storage = Mock()
        mock_cli.embedder = Mock()
        mock_cli.embedder.is_model_loaded = True
        mock_cli.index_conversations.return_value = {
            'files_processed': 1,
            'files_skipped': 0,
            'chunks_indexed': 5,
            'duration': 0.5,
            'errors': []
        }
        mock_cli_class.return_value = mock_cli
        
        # Create watcher with very short debounce
        watcher = ConversationWatcher(self.temp_dir, debounce_seconds=0.1)
        
        # Start watcher in thread
        def start_watcher():
            try:
                watcher.start_watching(self.claude_dir)
            except Exception:
                pass  # Expected when we stop it
        
        watcher_thread = threading.Thread(target=start_watcher)
        watcher_thread.start()
        
        # Wait for watcher to start
        time.sleep(0.2)
        
        # Create a new JSONL file
        test_file = os.path.join(self.claude_dir, "new_conversation.jsonl")
        with open(test_file, 'w') as f:
            f.write('{"message": "test conversation"}\n')
        
        # Wait for debounce and processing
        time.sleep(0.5)
        
        # Stop watcher
        watcher.stop_watching()
        watcher_thread.join(timeout=1)
        
        # Check that indexing was triggered
        assert mock_cli.index_conversations.called