#!/usr/bin/env python3
"""
Debug batch processing errors that occur during full indexing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import JSONLParser
from src.chunker import ConversationChunker, ChunkingConfig
from src.embeddings import EmbeddingGenerator, EmbeddingConfig

# Files that failed
problem_files = [
    "/Users/jrbaron/.claude/projects/-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine/4c1d6a71-d7cc-42c3-a0eb-9f77f125a485.jsonl",
    "/Users/jrbaron/.claude/projects/-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine/3f3e8d5a-1fb3-42a7-b7d1-bbf3c95b4938.jsonl",
    "/Users/jrbaron/.claude/projects/-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine/b9efbf00-a9d4-4dcc-8778-027abeb11060.jsonl"
]

def test_batch_processing(file_path: str, use_gpu: bool = False):
    """Test batch processing to reproduce the error."""
    print(f"\n{'='*60}")
    print(f"Testing batch processing ({'GPU' if use_gpu else 'CPU'}): {Path(file_path).name}")
    print(f"{'='*60}")
    
    # 1. Parse the file
    parser = JSONLParser()
    conversation = parser.parse_file(file_path)
    
    if not conversation:
        print("❌ Failed to parse conversation")
        return
    
    # 2. Create all chunks
    chunker = ConversationChunker(ChunkingConfig())
    chunks = chunker.chunk_conversation(conversation)
    print(f"✅ Created {len(chunks)} chunks")
    
    # 3. Check each chunk for potential issues
    print("\n🔍 Checking chunks for issues...")
    issues_found = False
    
    for i, chunk in enumerate(chunks):
        issues = []
        
        # Check for None text
        if chunk.text is None:
            issues.append("Text is None")
        # Check for non-string text
        elif not isinstance(chunk.text, str):
            issues.append(f"Text is not string: {type(chunk.text)}")
        # Check for empty text
        elif not chunk.text:
            issues.append("Text is empty string")
        # Check for whitespace-only text
        elif not chunk.text.strip():
            issues.append("Text is whitespace only")
        
        if issues:
            issues_found = True
            print(f"\n❌ Chunk {i} has issues: {', '.join(issues)}")
            print(f"   Type: {chunk.metadata.get('chunk_type')}")
            print(f"   Text repr: {repr(chunk.text)[:200]}")
            print(f"   Messages: {chunk.metadata.get('message_uuids', [])}")
    
    if not issues_found:
        print("✅ No text issues found in chunks")
    
    # 4. Test batch embedding generation
    print("\n🚀 Testing batch embedding generation...")
    
    config = EmbeddingConfig(
        batch_size=64 if use_gpu else 16,
        use_gpu=use_gpu,
        show_progress=True
    )
    embedder = EmbeddingGenerator(config)
    embedder.load_model()
    
    try:
        # Filter out problematic chunks before embedding
        valid_chunks = []
        skipped_chunks = []
        
        for chunk in chunks:
            if chunk.text and isinstance(chunk.text, str) and chunk.text.strip():
                valid_chunks.append(chunk)
            else:
                skipped_chunks.append(chunk)
        
        print(f"\n📊 Chunk statistics:")
        print(f"   Valid chunks: {len(valid_chunks)}")
        print(f"   Skipped chunks: {len(skipped_chunks)}")
        
        if valid_chunks:
            embeddings = embedder.generate_embeddings(valid_chunks)
            print(f"✅ Successfully generated {len(embeddings)} embeddings")
        else:
            print("⚠️  No valid chunks to process")
            
    except Exception as e:
        print(f"\n❌ Error during batch embedding: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to identify which chunk caused the error
        print("\n🔍 Attempting to identify problematic chunk...")
        for i, chunk in enumerate(chunks[:50]):  # Test first 50
            try:
                if chunk.text:
                    _ = embedder.generate_single_embedding(chunk.text)
            except Exception as chunk_error:
                print(f"   ❌ Chunk {i} fails: {str(chunk_error)}")
                print(f"      Text type: {type(chunk.text)}")
                print(f"      Text repr: {repr(chunk.text)[:100]}")

def main():
    """Test all problem files."""
    for file_path in problem_files:
        if Path(file_path).exists():
            # Test CPU mode
            test_batch_processing(file_path, use_gpu=False)
            
            # Test GPU mode
            print("\n" + "="*60)
            print("Testing with GPU mode...")
            test_batch_processing(file_path, use_gpu=True)
        else:
            print(f"\n❌ File not found: {file_path}")

if __name__ == "__main__":
    main()