"""Tests for MCP server interface."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import McpError

from src.mcp_server import call_tool, list_tools
from src.storage import Chunk


class TestMCPServer:
    """Test MCP server functionality."""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that all tools are listed correctly."""
        tools = await list_tools()
        tool_names = [tool.name for tool in tools]

        assert len(tools) == 5
        assert "semantic_search" in tool_names
        assert "get_chunk_by_id" in tool_names
        assert "list_projects" in tool_names
        assert "get_stats" in tool_names
        assert "get_status" in tool_names

        # Check semantic_search tool schema
        search_tool = next(t for t in tools if t.name == "semantic_search")
        assert search_tool.inputSchema["required"] == ["query"]
        assert "query" in search_tool.inputSchema["properties"]
        assert "top_k" in search_tool.inputSchema["properties"]
        assert "project" in search_tool.inputSchema["properties"]

    @pytest.mark.asyncio
    async def test_semantic_search_basic(self):
        """Test basic semantic search functionality."""
        mock_results = [
            MagicMock(
                chunk_id="chunk_001",
                text="This is a test result about Python programming.",
                similarity=0.95,
                metadata={
                    "project": "test-project",
                    "timestamp": "2025-07-10T10:00:00Z",
                    "session_id": "session_123",
                    "has_code": True,
                },
            )
        ]

        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=mock_results)
            mock_get_cli.return_value = mock_cli

            results = await call_tool("semantic_search", {"query": "Python programming"})

            assert len(results) == 1
            result_text = results[0].text
            assert "Found 1 results for: 'Python programming'" in result_text
            assert "chunk_001" in result_text
            assert "test-project" in result_text
            assert "🔧 Contains code" in result_text

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self):
        """Test semantic search with various filters."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=[])
            mock_get_cli.return_value = mock_cli

            # Test with all filters
            await call_tool(
                "semantic_search",
                {
                    "query": "test query",
                    "top_k": 5,
                    "project": "my-project",
                    "has_code": True,
                    "after": "2025-07-01",
                    "before": "2025-07-10",
                    "session": "session_456",
                    "use_gpu": True,
                    "full_content": True,
                },
            )

            # Verify filters were passed correctly
            call_args = mock_cli.search_conversations.call_args
            filters = call_args[0][1]  # Second positional argument
            config = call_args[0][2]  # Third positional argument

            assert filters["project"] == "my-project"
            assert filters["has_code"] is True
            assert filters["after_date"] == "2025-07-01"
            assert filters["before_date"] == "2025-07-10"
            assert filters["session_id"] == "session_456"
            assert config.top_k == 5
            
            # Verify GPU was requested
            mock_get_cli.assert_called_with(True)

    @pytest.mark.asyncio
    async def test_semantic_search_related_to(self):
        """Test semantic search with related_to functionality."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=[])
            mock_get_cli.return_value = mock_cli

            await call_tool(
                "semantic_search",
                {
                    "query": "",
                    "related_to": "chunk_123",
                    "same_session": True,
                },
            )

            # Verify related_to filters
            call_args = mock_cli.search_conversations.call_args
            filters = call_args[0][1]

            assert filters["related_to"] == "chunk_123"
            assert filters["same_session"] is True

    @pytest.mark.asyncio
    async def test_get_chunk_by_id_success(self):
        """Test getting a chunk by ID."""
        mock_chunk = MagicMock()
        mock_chunk.text = "This is the chunk content."
        mock_chunk.metadata = {
            "project": "test-project",
            "timestamp": "2025-07-10T10:00:00Z",
        }

        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.storage.get_chunk_by_id = MagicMock(return_value=mock_chunk)
            mock_get_cli.return_value = mock_cli

            results = await call_tool("get_chunk_by_id", {"chunk_id": "chunk_123"})

            assert len(results) == 1
            result_text = results[0].text
            assert "chunk_123" in result_text
            assert "This is the chunk content." in result_text
            assert "test-project" in result_text

    @pytest.mark.asyncio
    async def test_get_chunk_by_id_not_found(self):
        """Test getting a non-existent chunk."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.storage.get_chunk_by_id = MagicMock(return_value=None)
            mock_get_cli.return_value = mock_cli

            with pytest.raises(McpError) as exc_info:
                await call_tool("get_chunk_by_id", {"chunk_id": "nonexistent"})

            assert "Chunk not found: nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_projects(self):
        """Test listing all projects."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.get_index_stats = MagicMock(
                return_value={
                    "projects": ["project-a", "project-b", "project-c"],
                }
            )
            mock_get_cli.return_value = mock_cli

            results = await call_tool("list_projects", {})

            assert len(results) == 1
            result_text = results[0].text
            assert "Indexed Projects (3)" in result_text
            assert "- project-a" in result_text
            assert "- project-b" in result_text
            assert "- project-c" in result_text

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting index statistics."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.get_index_stats = MagicMock(
                return_value={
                    "total_chunks": 1000,
                    "total_sessions": 50,
                    "total_projects": 5,
                    "faiss_size_mb": 100.5,
                    "db_size_mb": 50.2,
                    "total_size_mb": 150.7,
                    "chunk_types": {
                        "qa_pair": 500,
                        "code_block": 300,
                        "context_segment": 200,
                    },
                }
            )
            mock_get_cli.return_value = mock_cli

            results = await call_tool("get_stats", {})

            assert len(results) == 1
            result_text = results[0].text
            assert "Total chunks: 1,000" in result_text
            assert "Total sessions: 50" in result_text
            assert "qa_pair: 500" in result_text
            assert "code_block: 300" in result_text

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting daemon status."""
        with patch("src.mcp_server.Path") as mock_path:
            # Mock PID file exists
            pid_file_mock = MagicMock()
            pid_file_mock.exists.return_value = True
            pid_file_mock.read_text.return_value = "12345"

            # Mock DB path exists
            db_path_mock = MagicMock()
            db_path_mock.exists.return_value = True

            mock_path.home.return_value = MagicMock()
            mock_path.home.return_value.__truediv__.side_effect = lambda x: (
                pid_file_mock if "daemon.pid" in x else db_path_mock
            )

            with patch("src.mcp_server.psutil.pid_exists", return_value=True):
                with patch("src.mcp_server.sqlite3.connect") as mock_connect:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = ("2025-07-10T12:00:00Z",)
                    mock_connect.return_value.__enter__.return_value.cursor.return_value = (
                        mock_cursor
                    )

                    results = await call_tool("get_status", {})

                    assert len(results) == 1
                    result_text = results[0].text
                    assert "Daemon running: ✅ Yes" in result_text
                    assert "Last index update: 2025-07-10T12:00:00Z" in result_text

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """Test calling an unknown tool."""
        with pytest.raises(McpError) as exc_info:
            await call_tool("unknown_tool", {})

        assert "Unknown tool: unknown_tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_full_content_truncation(self):
        """Test content truncation behavior."""
        long_text = "x" * 1000  # 1000 character text
        mock_results = [
            MagicMock(
                chunk_id="chunk_001",
                text=long_text,
                similarity=0.95,
                metadata={},
            )
        ]

        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=mock_results)
            mock_get_cli.return_value = mock_cli

            # Test with truncation (default)
            results = await call_tool("semantic_search", {"query": "test"})
            result_text = results[0].text
            assert "..." in result_text
            # The truncated content should be around 500 chars
            truncated_content = result_text.split("\n\n")[1].split("..")[0]
            assert len(truncated_content) <= 510  # Allow small buffer for edge cases

            # Test without truncation
            results = await call_tool(
                "semantic_search", {"query": "test", "full_content": True}
            )
            result_text = results[0].text
            assert "xxx" in result_text  # Full content should be there
            assert "..." not in result_text.split("---")[0]  # No truncation in content