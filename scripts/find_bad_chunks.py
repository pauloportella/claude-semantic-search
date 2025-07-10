#!/usr/bin/env python3
"""
Find the exact chunks causing the TextEncodeInput error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import JSONLParser
from src.chunker import ConversationChunker, ChunkingConfig
from src.embeddings import EmbeddingGenerator, EmbeddingConfig

problem_files = [
    "/Users/jrbaron/.claude/projects/-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine/4c1d6a71-d7cc-42c3-a0eb-9f77f125a485.jsonl",
    "/Users/jrbaron/.claude/projects/-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine/3f3e8d5a-1fb3-42a7-b7d1-bbf3c95b4938.jsonl"
]

def find_bad_chunks(file_path: str):
    """Find chunks that cause the TextEncodeInput error."""
    print(f"\n{'='*60}")
    print(f"Finding bad chunks in: {Path(file_path).name}")
    print(f"{'='*60}")
    
    # Parse and chunk
    parser = JSONLParser()
    conversation = parser.parse_file(file_path)
    
    if not conversation:
        print("❌ Failed to parse conversation")
        return
    
    chunker = ConversationChunker(ChunkingConfig())
    chunks = chunker.chunk_conversation(conversation)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Test each chunk individually
    config = EmbeddingConfig(show_progress=False)
    embedder = EmbeddingGenerator(config)
    embedder.load_model()
    
    bad_chunks = []
    
    print("\n🔍 Testing chunks individually...")
    for i, chunk in enumerate(chunks):
        try:
            # Check text type and content
            if chunk.text is None:
                bad_chunks.append((i, chunk, "Text is None"))
                continue
            elif not isinstance(chunk.text, str):
                bad_chunks.append((i, chunk, f"Text is not string: {type(chunk.text)}"))
                continue
            elif isinstance(chunk.text, str):
                # Additional checks for string content
                if '\x00' in chunk.text:
                    bad_chunks.append((i, chunk, "Contains null character"))
                    continue
                
            # Try to encode
            _ = embedder.generate_single_embedding(chunk.text)
            
        except Exception as e:
            bad_chunks.append((i, chunk, str(e)))
    
    if bad_chunks:
        print(f"\n❌ Found {len(bad_chunks)} bad chunks:")
        for idx, chunk, error in bad_chunks:
            print(f"\n--- Bad Chunk {idx} ---")
            print(f"Error: {error}")
            print(f"Type: {chunk.metadata.get('chunk_type')}")
            print(f"Text type: {type(chunk.text)}")
            print(f"Text length: {len(chunk.text) if chunk.text else 0}")
            print(f"Text repr: {repr(chunk.text)[:500] if chunk.text else 'None'}")
            print(f"Message UUIDs: {chunk.metadata.get('message_uuids', [])}")
            
            # Debug the text content
            if chunk.text and isinstance(chunk.text, str):
                print(f"\nText analysis:")
                print(f"- Contains null char: {'\\x00' in chunk.text}")
                print(f"- Is ASCII: {chunk.text.isascii()}")
                print(f"- First 100 chars: {repr(chunk.text[:100])}")
                
                # Check for unusual characters
                non_printable = [c for c in chunk.text if ord(c) < 32 and c not in '\n\r\t']
                if non_printable:
                    print(f"- Non-printable chars: {[hex(ord(c)) for c in non_printable[:10]]}")
    else:
        print("✅ All chunks are valid")
    
    # Now test batch processing to find exact batch issue
    print("\n🚀 Testing batch processing...")
    batch_size = 16
    
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_num = batch_start // batch_size
        
        try:
            # Extract texts
            texts = [chunk.text for chunk in batch_chunks]
            
            # Check for None or non-string texts
            for i, text in enumerate(texts):
                if text is None or not isinstance(text, str):
                    print(f"\n❌ Batch {batch_num} has invalid text at position {i}:")
                    print(f"   Chunk index: {batch_start + i}")
                    print(f"   Text type: {type(text)}")
                    print(f"   Text value: {repr(text)}")
            
            # Try batch encoding
            _ = embedder.model.encode(texts, show_progress_bar=False)
            
        except Exception as e:
            print(f"\n❌ Batch {batch_num} (chunks {batch_start}-{batch_end-1}) failed: {str(e)}")
            
            # Find exact problematic chunk in batch
            for i, chunk in enumerate(batch_chunks):
                try:
                    _ = embedder.generate_single_embedding(chunk.text)
                except:
                    print(f"   Bad chunk in batch: index {batch_start + i}")

def main():
    for file_path in problem_files:
        if Path(file_path).exists():
            find_bad_chunks(file_path)

if __name__ == "__main__":
    main()