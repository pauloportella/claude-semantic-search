#!/usr/bin/env python3
"""
Debug the parsing errors from specific JSONL files.
"""

import sys
import json
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

def debug_file(file_path: str):
    """Debug a single file to find parsing issues."""
    print(f"\n{'='*60}")
    print(f"Debugging: {Path(file_path).name}")
    print(f"{'='*60}")
    
    # 1. Parse the file
    parser = JSONLParser()
    conversation = parser.parse_file(file_path)
    
    if not conversation:
        print("❌ Failed to parse conversation")
        return
    
    print(f"✅ Parsed conversation: {conversation.total_messages} messages")
    
    # 2. Create chunks
    chunker = ConversationChunker(ChunkingConfig())
    chunks = chunker.chunk_conversation(conversation)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # 3. Check for empty/invalid chunks
    problematic_chunks = []
    for i, chunk in enumerate(chunks):
        if not chunk.text or chunk.text.strip() == "":
            problematic_chunks.append((i, chunk, "Empty text"))
        elif chunk.text is None:
            problematic_chunks.append((i, chunk, "None text"))
        elif not isinstance(chunk.text, str):
            problematic_chunks.append((i, chunk, f"Non-string text: {type(chunk.text)}"))
    
    if problematic_chunks:
        print(f"\n❌ Found {len(problematic_chunks)} problematic chunks:")
        for idx, chunk, issue in problematic_chunks[:5]:  # Show first 5
            print(f"  Chunk {idx}: {issue}")
            print(f"    Type: {chunk.metadata.get('chunk_type')}")
            print(f"    Text repr: {repr(chunk.text)[:100]}")
    
    # 4. Try to generate embeddings to reproduce the error
    print("\n🔍 Testing embedding generation...")
    
    config = EmbeddingConfig(show_progress=False)
    embedder = EmbeddingGenerator(config)
    embedder.load_model()
    
    for i, chunk in enumerate(chunks[:10]):  # Test first 10 chunks
        try:
            if chunk.text and isinstance(chunk.text, str) and chunk.text.strip():
                embedding = embedder.generate_single_embedding(chunk.text)
                print(f"  ✅ Chunk {i}: OK (embedding dim: {len(embedding)})")
            else:
                print(f"  ⚠️  Chunk {i}: Skipped (invalid text)")
        except Exception as e:
            print(f"  ❌ Chunk {i}: Error - {str(e)}")
            print(f"     Text repr: {repr(chunk.text)[:100]}")
            print(f"     Text type: {type(chunk.text)}")
            
    # 5. Check raw JSONL data for patterns
    print("\n📄 Checking raw JSONL data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"  Total lines: {len(lines)}")
        
        # Check for empty/malformed lines
        empty_lines = 0
        malformed_lines = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                empty_lines += 1
            else:
                try:
                    data = json.loads(line)
                    # Check for messages with empty content
                    if 'message' in data and isinstance(data['message'], dict):
                        content = data['message'].get('content', [])
                        if not content or (isinstance(content, list) and len(content) == 0):
                            print(f"  ⚠️  Line {i}: Empty content in message")
                except json.JSONDecodeError:
                    malformed_lines += 1
        
        print(f"  Empty lines: {empty_lines}")
        print(f"  Malformed lines: {malformed_lines}")

def main():
    """Debug all problem files."""
    for file_path in problem_files:
        if Path(file_path).exists():
            try:
                debug_file(file_path)
            except Exception as e:
                print(f"\n❌ Fatal error debugging {Path(file_path).name}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n❌ File not found: {file_path}")

if __name__ == "__main__":
    main()