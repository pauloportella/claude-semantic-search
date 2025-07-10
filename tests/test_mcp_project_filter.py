"""Test MCP server project filtering with partial matching."""

import pytest
from unittest.mock import MagicMock, patch
from src.mcp_server import call_tool


class TestMCPProjectFilter:
    """Test MCP server handles project filters correctly."""

    @pytest.mark.asyncio
    async def test_mcp_partial_project_filter(self):
        """Test that MCP server correctly passes project filter for partial matching."""
        mock_results = [
            {
                "chunk_id": "chunk_001",
                "text": "Persistence implementation in daisy-hft-engine",
                "similarity": 0.95,
                "metadata": {
                    "project": "-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine",
                    "timestamp": "2025-07-07T20:00:00Z",
                    "session_id": "session_123",
                    "has_code": True,
                },
                "project": "-Users-jrbaron-dev-pauloportella-2025-trading-daisy-hft-engine",
                "session": "session_123",
                "timestamp": "2025-07-07T20:00:00Z",
                "has_code": True,
            }
        ]

        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=mock_results)
            mock_get_cli.return_value = mock_cli

            # Test partial project name through MCP
            results = await call_tool(
                "claude_semantic_search", 
                {
                    "query": "persistence",
                    "project": "daisy-hft"  # Partial name
                }
            )

            # Verify the filter was passed correctly
            mock_cli.search_conversations.assert_called_once()
            call_args = mock_cli.search_conversations.call_args
            
            # Check that project_name filter was set
            filters = call_args[0][1]  # Second positional argument
            assert "project_name" in filters
            assert filters["project_name"] == "daisy-hft"
            
            # Verify results are returned
            assert len(results) == 1
            assert "daisy-hft-engine" in results[0].text

    @pytest.mark.asyncio
    async def test_mcp_date_filters(self):
        """Test that MCP server correctly formats date filters."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=[])
            mock_get_cli.return_value = mock_cli

            # Test date filters
            await call_tool(
                "claude_semantic_search",
                {
                    "query": "test",
                    "after": "2025-01-01",
                    "before": "2025-12-31"
                }
            )

            # Verify date filters were formatted correctly
            call_args = mock_cli.search_conversations.call_args
            filters = call_args[0][1]
            
            assert "timestamp" in filters
            assert "gte" in filters["timestamp"]
            assert "lte" in filters["timestamp"]
            assert filters["timestamp"]["gte"] == "2025-01-01T00:00:00+00:00"
            assert filters["timestamp"]["lte"] == "2025-12-31T23:59:59+00:00"

    @pytest.mark.asyncio  
    async def test_mcp_combined_filters(self):
        """Test MCP server with multiple filters including partial project match."""
        with patch("src.mcp_server.get_search_cli") as mock_get_cli:
            mock_cli = MagicMock()
            mock_cli.search_conversations = MagicMock(return_value=[])
            mock_get_cli.return_value = mock_cli

            # Test combined filters
            await call_tool(
                "claude_semantic_search",
                {
                    "query": "optimization",
                    "project": "semantic",  # Partial match
                    "has_code": True,
                    "session": "abc123"
                }
            )

            # Verify all filters were set correctly
            call_args = mock_cli.search_conversations.call_args
            filters = call_args[0][1]
            
            assert filters["project_name"] == "semantic"
            assert filters["has_code"] is True
            assert filters["session_id"] == "abc123"