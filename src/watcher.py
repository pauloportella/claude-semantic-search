#!/usr/bin/env python3
"""
File watcher for automatic index updates.

This module provides a file watcher that monitors Claude conversation
directories for changes and automatically triggers incremental indexing.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Set

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .cli import SemanticSearchCLI

logger = logging.getLogger(__name__)


class ConversationFileHandler(FileSystemEventHandler):
    """Handler for conversation file changes."""

    def __init__(self, cli_instance: SemanticSearchCLI, debounce_seconds: int = 5):
        """Initialize handler with CLI instance and debounce settings."""
        self.cli_instance = cli_instance
        self.debounce_seconds = debounce_seconds
        self.pending_files: Set[str] = set()
        self.last_trigger_time: Optional[datetime] = None
        self.timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._is_conversation_file(event.src_path):
            logger.info(f"New conversation file detected: {event.src_path}")
            self._schedule_indexing(event.src_path)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self._is_conversation_file(event.src_path):
            logger.info(f"Modified conversation file detected: {event.src_path}")
            self._schedule_indexing(event.src_path)

    def _is_conversation_file(self, file_path: str) -> bool:
        """Check if file is a conversation file."""
        return file_path.endswith(".jsonl")

    def _schedule_indexing(self, file_path: str):
        """Schedule indexing with debounce."""
        with self._lock:
            self.pending_files.add(file_path)

            # Cancel existing timer
            if self.timer and self.timer.is_alive():
                self.timer.cancel()

            # Schedule new indexing
            self.timer = threading.Timer(self.debounce_seconds, self._trigger_indexing)
            self.timer.start()

    def _trigger_indexing(self):
        """Trigger incremental indexing for pending files."""
        with self._lock:
            if not self.pending_files:
                return

            files_to_index = list(self.pending_files)
            self.pending_files.clear()

        logger.info(f"Triggering incremental indexing for {len(files_to_index)} files")

        try:
            # Get unique parent directories to scan
            dirs_to_scan = set()
            for file_path in files_to_index:
                dirs_to_scan.add(str(Path(file_path).parent))

            # For each directory, run incremental indexing
            for dir_path in dirs_to_scan:
                logger.info(f"Scanning directory for changes: {dir_path}")

                # Find all JSONL files in the directory
                dir_files = list(Path(dir_path).glob("*.jsonl"))

                if dir_files:
                    # Run indexing on the files
                    stats = self.cli_instance.index_conversations(
                        dir_files, force=False
                    )

                    logger.info(f"Incremental indexing complete:")
                    logger.info(f"  Files processed: {stats['files_processed']}")
                    logger.info(f"  Files unchanged: {stats.get('files_unchanged', 0)}")
                    logger.info(f"  Files skipped: {stats['files_skipped']}")
                    logger.info(f"  Chunks indexed: {stats['chunks_indexed']}")
                    if stats.get("chunks_removed", 0) > 0:
                        logger.info(f"  Chunks removed: {stats['chunks_removed']}")
                    logger.info(f"  Duration: {stats['duration']:.1f}s")

                    if stats["errors"]:
                        logger.warning(f"  Errors: {len(stats['errors'])}")
                        for error in stats["errors"][:3]:
                            logger.warning(f"    - {error}")

            self.last_trigger_time = datetime.now()

        except Exception as e:
            logger.error(f"Error during automatic indexing: {str(e)}")


class ConversationWatcher:
    """File watcher for Claude conversations."""

    def __init__(
        self, data_dir: str = None, debounce_seconds: int = 5, use_gpu: bool = False
    ):
        """Initialize watcher."""
        self.data_dir = data_dir or os.environ.get("CLAUDE_SEARCH_DATA_DIR", "~/.claude-semantic-search/data")
        # Expand user path to handle ~ properly
        self.data_dir = str(Path(self.data_dir).expanduser())
        self.debounce_seconds = debounce_seconds
        self.use_gpu = use_gpu
        self.cli_instance = SemanticSearchCLI(self.data_dir, use_gpu)
        self.observer = Observer()
        self.handler = ConversationFileHandler(self.cli_instance, debounce_seconds)
        self.is_running = False
        self.pid_file = Path(self.data_dir) / "watcher.pid"
        self.log_file = Path(self.data_dir) / "watcher.log"

    def start_watching(self, claude_dir: str = "~/.claude/projects"):
        """Start watching for file changes."""
        claude_path = Path(claude_dir).expanduser()

        if not claude_path.exists():
            raise FileNotFoundError(
                f"Claude projects directory not found: {claude_path}"
            )

        logger.info(f"Starting file watcher for: {claude_path}")
        logger.info(f"Debounce interval: {self.debounce_seconds} seconds")

        # Initialize storage and models
        self.cli_instance.storage.initialize()
        if not self.cli_instance.embedder.is_model_loaded:
            logger.info("Loading embedding model...")
            self.cli_instance.embedder.load_model()
            logger.info("Model loaded successfully")

        # Start watching recursively
        self.observer.schedule(self.handler, str(claude_path), recursive=True)
        self.observer.start()
        self.is_running = True

        logger.info("File watcher started. Monitoring for changes...")

        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping watcher...")
        finally:
            self.stop_watching()

    def stop_watching(self):
        """Stop watching for file changes."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        # Cancel any pending indexing
        if self.handler.timer and self.handler.timer.is_alive():
            self.handler.timer.cancel()

        self.is_running = False
        logger.info("File watcher stopped")

    def get_status(self) -> dict:
        """Get watcher status."""
        return {
            "is_running": self.is_running,
            "watching_path": (
                self.observer.emitters[0].watch.path if self.observer.emitters else None
            ),
            "pending_files": len(self.handler.pending_files),
            "last_trigger_time": (
                self.handler.last_trigger_time.isoformat()
                if self.handler.last_trigger_time
                else None
            ),
            "debounce_seconds": self.debounce_seconds,
        }

    def setup_daemon_logging(self):
        """Setup logging for daemon mode."""
        print(f"setup_daemon_logging: self.log_file = {self.log_file!r}")
        print(f"setup_daemon_logging: current directory = {os.getcwd()!r}")
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Create file handler for daemon logging
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Also add to root logger to catch all messages
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

    def write_pid_file(self):
        """Write PID file."""
        # Ensure directory exists
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pid_file, "w") as f:
            f.write(str(os.getpid()))
        logger.info(f"PID file written: {self.pid_file}")

    def remove_pid_file(self):
        """Remove PID file."""
        if self.pid_file.exists():
            self.pid_file.unlink()
            logger.info(f"PID file removed: {self.pid_file}")

    def is_daemon_running(self) -> bool:
        """Check if daemon is already running."""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process is still running
            os.kill(pid, 0)  # This will raise OSError if process doesn't exist
            return True
        except (OSError, ValueError):
            # Process doesn't exist or PID file is corrupted
            self.remove_pid_file()
            return False

    def get_daemon_pid(self) -> Optional[int]:
        """Get daemon PID if running."""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, "r") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def start_daemon(self, claude_dir: str = "~/.claude/projects"):
        """Start watcher as daemon."""
        if self.is_daemon_running():
            raise RuntimeError("Watcher daemon is already running")

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, stopping daemon...")
            self.stop_watching()
            self.remove_pid_file()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Setup daemon logging
        self.setup_daemon_logging()

        # Write PID file
        self.write_pid_file()

        logger.info("Starting watcher daemon...")
        logger.info(f"Watching directory: {claude_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Debounce interval: {self.debounce_seconds} seconds")

        try:
            self.start_watching(claude_dir)
        except Exception as e:
            logger.error(f"Daemon failed: {str(e)}")
            self.remove_pid_file()
            raise

    def stop_daemon(self):
        """Stop the watcher daemon."""
        if not self.is_daemon_running():
            raise RuntimeError("Watcher daemon is not running")

        pid = self.get_daemon_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait a bit for graceful shutdown
                time.sleep(2)

                # Check if still running, force kill if necessary
                if self.is_daemon_running():
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(1)

                logger.info(f"Daemon stopped (PID: {pid})")
            except OSError as e:
                logger.error(f"Failed to stop daemon: {str(e)}")
                raise

        self.remove_pid_file()


def run_watcher(
    data_dir: str = None,
    claude_dir: str = "~/.claude/projects",
    debounce_seconds: int = 5,
    use_gpu: bool = False,
):
    """Run the file watcher in interactive mode."""
    data_dir = data_dir or os.environ.get("CLAUDE_SEARCH_DATA_DIR", "~/.claude-semantic-search/data")
    data_dir = str(Path(data_dir).expanduser())  # Expand ~ to full path
    watcher = ConversationWatcher(data_dir, debounce_seconds, use_gpu)

    try:
        watcher.start_watching(claude_dir)
    except Exception as e:
        logger.error(f"Failed to start watcher: {str(e)}")
        raise


def start_daemon(
    data_dir: str = None,
    claude_dir: str = "~/.claude/projects",
    debounce_seconds: int = 5,
    use_gpu: bool = False,
):
    """Start the file watcher as a daemon."""
    print(f"start_daemon called with data_dir: {data_dir!r}")
    data_dir = data_dir or os.environ.get("CLAUDE_SEARCH_DATA_DIR", "~/.claude-semantic-search/data")
    print(f"After default: {data_dir!r}")
    data_dir = str(Path(data_dir).expanduser())  # Expand ~ to full path
    print(f"After expand: {data_dir!r}")
    watcher = ConversationWatcher(data_dir, debounce_seconds, use_gpu)

    # Fork process to run in background
    try:
        print(f"Parent process CWD: {os.getcwd()}")
        print(f"About to fork...")
        pid = os.fork()
        print(f"Fork returned pid: {pid}")
        if pid > 0:
            # Parent process - exit
            print(f"✅ Watcher daemon started with PID: {pid}")
            print(f"📁 Watching: {claude_dir}")
            print(f"💾 Data directory: {data_dir}")
            print(f"📝 Log file: {watcher.log_file}")
            return
        else:
            # Child process
            print(f"In child process, pid=0")
    except OSError:
        # Fork not supported (Windows), run directly
        print("Fork failed, running directly")
        pass

    # Child process or non-Unix system
    # Debug immediately 
    with open("/tmp/claude_debug.txt", "w") as f:
        f.write("Child process started\n")
        f.write(f"Child process CWD: {os.getcwd()}\n")
        f.write(f"Child process - watcher.data_dir: {watcher.data_dir!r}\n")
        f.write(f"Child process - watcher.log_file: {watcher.log_file!r}\n")
    
    try:
        watcher.start_daemon(claude_dir)
    except Exception as e:
        with open("/tmp/claude_debug.txt", "a") as f:
            f.write(f"Exception: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        print(f"❌ Failed to start daemon: {str(e)}")
        sys.exit(1)


def stop_daemon(data_dir: str = None):
    """Stop the file watcher daemon."""
    data_dir = data_dir or os.environ.get("CLAUDE_SEARCH_DATA_DIR", "~/.claude-semantic-search/data")
    data_dir = str(Path(data_dir).expanduser())  # Expand ~ to full path
    watcher = ConversationWatcher(data_dir)

    try:
        watcher.stop_daemon()
        print("✅ Watcher daemon stopped")
    except RuntimeError as e:
        print(f"❌ {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to stop daemon: {str(e)}")
        sys.exit(1)


def daemon_status(data_dir: str = None):
    """Check daemon status."""
    data_dir = data_dir or os.environ.get("CLAUDE_SEARCH_DATA_DIR", "~/.claude-semantic-search/data")
    data_dir = str(Path(data_dir).expanduser())  # Expand ~ to full path
    watcher = ConversationWatcher(data_dir)

    if watcher.is_daemon_running():
        pid = watcher.get_daemon_pid()
        print(f"✅ Watcher daemon is running (PID: {pid})")
        print(f"📝 Log file: {watcher.log_file}")
        print(f"🔧 PID file: {watcher.pid_file}")

        # Show recent log entries if available
        if watcher.log_file.exists():
            print("\n📋 Recent log entries:")
            try:
                with open(watcher.log_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-5:]:  # Show last 5 lines
                        print(f"   {line.rstrip()}")
            except Exception:
                print("   (Could not read log file)")
    else:
        print("❌ Watcher daemon is not running")

        # Check if log file exists
        if watcher.log_file.exists():
            print(f"📝 Log file available: {watcher.log_file}")

    return watcher.is_daemon_running()
