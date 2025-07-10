# Claude Search MCP Server Implementation Plan

## Overview

Add an MCP (Model Context Protocol) server interface to the `claude-search` semantic search system, enabling seamless integration with Claude Desktop, Cursor, and other AI agents. This MCP interface will complement existing interfaces (CLI, future Alfred workflow, etc.) without replacing them.

## Architecture

```
                    Interfaces Layer
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│     CLI     │  │  MCP Server  │  │    Alfred    │  │   Future     │
│  (existing) │  │    (new)     │  │  (planned)   │  │  Interfaces  │
└──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                │                  │                  │
       └────────────────┴──────────────────┴──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Core Search API      │
                    │   (SemanticSearchCLI)   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐           ┌──────▼──────┐
              │  Search   │           │   Storage   │
              │  Engine   │           │  (FAISS +   │
              │           │           │   SQLite)   │
              └───────────┘           └─────────────┘

MCP Server Detail:
┌─────────────────┐     JSON-RPC 2.0      ┌──────────────────┐
│   Claude/AI     │◄──────over stdio─────►│   MCP Server     │
│   Agent         │                        │   Interface      │
└─────────────────┘                        └──────────────────┘
```

## Implementation Phases

### Phase 1: MCP Server Foundation

**Objective**: Create basic MCP server structure with JSON-RPC communication

**Implementation**:
```python
# src/mcp_server.py
import asyncio
import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class MCPServer:
    """MCP server for Claude semantic search"""
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC requests"""
        
    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize server with capabilities"""
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return available tools"""
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool"""
        
    async def run(self):
        """Main server loop reading from stdin"""
```

**Key Features**:
- JSON-RPC 2.0 protocol over stdio
- Async/await for non-blocking operations
- Error handling and validation
- Logging and debugging support

### Phase 2: Search Tools Implementation

**Objective**: Expose search functionality as MCP tools

**Tools to Implement**:

1. **semantic_search**
   ```python
   {
       "name": "semantic_search",
       "description": "Search Claude conversations using semantic similarity",
       "inputSchema": {
           "type": "object",
           "properties": {
               "query": {
                   "type": "string",
                   "description": "Search query"
               },
               "top_k": {
                   "type": "integer",
                   "description": "Number of results to return",
                   "default": 20
               },
               "project": {
                   "type": "string",
                   "description": "Filter by project name"
               },
               "has_code": {
                   "type": "boolean",
                   "description": "Filter for chunks with code"
               },
               "after": {
                   "type": "string",
                   "description": "Filter after date (YYYY-MM-DD)"
               },
               "before": {
                   "type": "string",
                   "description": "Filter before date (YYYY-MM-DD)"
               },
               "session": {
                   "type": "string",
                   "description": "Filter by session ID"
               },
               "related_to": {
                   "type": "string",
                   "description": "Find chunks related to given chunk ID"
               },
               "same_session": {
                   "type": "boolean",
                   "description": "Include chunks from same session as related_to"
               },
               "full_content": {
                   "type": "boolean",
                   "description": "Show full content instead of truncated",
                   "default": false
               },
               "use_gpu": {
                   "type": "boolean",
                   "description": "Use GPU acceleration for faster search",
                   "default": false
               }
           },
           "required": ["query"]
       }
   }
   ```

2. **get_chunk_by_id**
   ```python
   {
       "name": "get_chunk_by_id",
       "description": "Retrieve a specific conversation chunk by ID",
       "inputSchema": {
           "type": "object",
           "properties": {
               "chunk_id": {
                   "type": "string",
                   "description": "Unique chunk identifier"
               },
               "full_content": {
                   "type": "boolean",
                   "description": "Return full content instead of truncated",
                   "default": true
               }
           },
           "required": ["chunk_id"]
       }
   }
   ```


3. **list_projects**
   ```python
   {
       "name": "list_projects",
       "description": "List all indexed Claude projects",
       "inputSchema": {
           "type": "object",
           "properties": {}
       }
   }
   ```

4. **get_index_stats**
   ```python
   {
       "name": "get_index_stats",
       "description": "Get statistics about the search index",
       "inputSchema": {
           "type": "object",
           "properties": {}
       }
   }
   ```

5. **get_status**
   ```python
   {
       "name": "get_status",
       "description": "Get the status of the indexing daemon and last index update",
       "inputSchema": {
           "type": "object",
           "properties": {}
       }
   }
   ```

### Phase 3: Resource Providers

**Objective**: Expose conversation data as MCP resources

**Resources**:
1. **Recent Conversations**: Last N conversations
2. **Project Conversations**: All conversations for a project
3. **Code Snippets**: Extracted code blocks

```python
class ResourceProvider:
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read resource content"""
```

### Phase 4: Configuration & Setup

**Files to Create**:

1. **pyproject.toml** update:
   ```toml
   [project.scripts]
   claude-search-mcp = "src.mcp_server:main"
   
   [project.optional-dependencies]
   mcp = [
       "mcp>=0.1.0",  # or appropriate version
       "jsonrpc>=1.0.0",
   ]
   ```

2. **MCP manifest** (`mcp.json`):
   ```json
   {
       "name": "claude-search",
       "version": "1.0.0",
       "description": "Semantic search for Claude conversations",
       "author": "Your Name",
       "license": "MIT",
       "runtime": "python",
       "main": "uv run claude-search-mcp",
       "tools": [
           "semantic_search",
           "get_chunk_by_id",
           "list_projects",
           "get_index_stats",
           "get_status"
       ],
       "resources": [
           "recent_conversations",
           "project_conversations",
           "code_snippets"
       ]
   }
   ```

3. **Claude Desktop Config**:
   ```json
   {
       "mcpServers": {
           "claude-search": {
               "command": "uv",
               "args": ["run", "claude-search-mcp"],
               "cwd": "/path/to/semantic-search"
           }
       }
   }
   ```

### Phase 5: Testing & Integration

**Test Implementation**:
```python
# tests/test_mcp_server.py
import pytest
import asyncio
from src.mcp_server import MCPServer

async def test_server_initialization():
    """Test server initializes correctly"""
    
async def test_semantic_search_tool():
    """Test semantic search tool"""
    
async def test_error_handling():
    """Test error responses"""
```

**Integration Tests**:
1. Manual testing with Claude Desktop
2. Automated tests using MCP test client
3. Performance testing with concurrent requests
4. Error scenario testing

### Phase 6: Documentation & Deployment

**Documentation**:
1. Installation guide
2. Configuration instructions
3. Tool usage examples
4. Troubleshooting guide

**Example Usage in Claude**:
```
User: Search for conversations about GPU performance