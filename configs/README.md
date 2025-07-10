# Configuration Examples

This directory contains example configuration files for various integrations.

## Files

### claude_desktop_config.example.json
Example configuration for Claude Desktop. Copy this to:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/claude/claude_desktop_config.json`

Then edit the path to point to your semantic-search installation.

### mcp_manifest.json
MCP server manifest describing available tools and capabilities. This file is used by the MCP server itself and doesn't need to be copied elsewhere.

## Usage

```bash
# Copy Claude Desktop config (macOS example)
cp claude_desktop_config.example.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Edit the config to set your correct path
# Then restart Claude Desktop
```