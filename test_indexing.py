#!/usr/bin/env python3
"""Test indexing script - processes only 2 files to verify everything works."""

import sys
from pathlib import Path
from src.cli import SemanticSearchCLI
from src.storage import StorageConfig
import logging

logging.basicConfig(level=logging.INFO)

def test_indexing():
    """Test indexing with just 2 files."""
    
    # Initialize CLI with new data directory
    data_dir = Path.home() / ".claude-semantic-search" / "data-test"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using test data directory: {data_dir}")
    
    cli = SemanticSearchCLI(str(data_dir), use_gpu=True)
    
    # Test with specific files that were failing
    claude_dir = Path.home() / ".claude" / "projects" / "-Users-jrbaron-dev-pauloportella-dotfiles"
    
    # These are the files that were failing with readonly database error
    test_files = [
        "434f1ce1-dd24-47c8-b922-7adc058d4318.jsonl",
        "a0e5a5b4-f38c-444f-ba56-7c6c34a060b4.jsonl",
        "f031a754-1ac7-4d47-9db4-b4bcde6d7790.jsonl",
        "f40db0f8-a01b-4e55-82e6-723a3c9ee7af.jsonl",
        "c44552a4-2881-4607-ab7e-35b5a8be5619.jsonl"
    ]
    
    files = []
    for fname in test_files:
        fpath = claude_dir / fname
        if fpath.exists():
            files.append(fpath)
        if len(files) >= 5:  # Test with 5 files
            break
    
    if not files:
        print("No conversation files found!")
        return
    
    print(f"\nTesting with {len(files)} files from dotfiles project:")
    for f in files:
        print(f"  - {f.name}")
    
    # Check database permissions
    print(f"\nChecking database permissions...")
    db_path = data_dir / "metadata.db"
    if db_path.exists():
        import os
        stat = os.stat(db_path)
        print(f"  Database exists: {db_path}")
        print(f"  Permissions: {oct(stat.st_mode)}")
        print(f"  Writable: {os.access(db_path, os.W_OK)}")
    
    # Initialize storage
    print("\nInitializing storage...")
    cli.storage.initialize()
    
    # Process files
    print("\nIndexing files...")
    try:
        stats = cli.index_conversations(files, force=True)
        
        print(f"\nIndexing complete:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Chunks indexed: {stats['chunks_indexed']}")
        print(f"  Duration: {stats['duration']:.1f}s")
        
        if stats["errors"]:
            print(f"\nErrors encountered:")
            for error in stats["errors"]:
                print(f"  - {error}")
        
        # Test search
        print("\nTesting search...")
        results = cli.search_conversations("test query", top_k=5)
        print(f"  Found {len(results)} results")
        
        # Get stats
        storage_stats = cli.get_index_stats()
        print(f"\nStorage stats:")
        print(f"  Total chunks: {storage_stats['total_chunks']}")
        print(f"  Total sessions: {storage_stats['total_sessions']}")
        
        print("\n✅ Test successful! You can now run full indexing.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_indexing()
    sys.exit(0 if success else 1)