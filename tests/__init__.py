"""
Tests for Claude Semantic Search.

This module contains comprehensive tests for all components of the
semantic search system.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test configuration
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test_fixtures"
TEST_MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_MODELS_DIR.mkdir(parents=True, exist_ok=True)