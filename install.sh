#!/bin/bash
set -e

echo "Installing Claude Semantic Search MCP Server..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install the tool
echo "Installing claude-semantic-search..."
uv tool install .

# Download the embedding model
echo ""
echo "Downloading embedding model (this may take a minute)..."
setup-models

# Ask about data directory
echo ""
echo "Data Directory Configuration"
echo "============================"
echo "By default, semantic search data will be stored in: ~/.claude-semantic-search/data"
echo ""
read -p "Use the default location? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please enter your preferred data directory path:"
    read -r custom_data_dir
    export CLAUDE_SEARCH_DATA_DIR="$custom_data_dir"
    echo ""
    echo "✅ Data directory set to: $custom_data_dir"
    echo ""
    echo "To make this permanent, add to your shell profile:"
    echo "  export CLAUDE_SEARCH_DATA_DIR=\"$custom_data_dir\""
else
    echo "✅ Using default data directory: ~/.claude-semantic-search/data"
fi

# Configure MCP for various tools
echo ""
echo "Configuring MCP servers..."

# Function to add MCP server to a JSON config file
add_mcp_to_config() {
    local config_file="$1"
    local config_name="$2"
    local command_path="$3"
    
    if [ -f "$config_file" ]; then
        # Backup existing config
        cp "$config_file" "${config_file}.backup"
        echo "Backed up existing $config_name config to ${config_file}.backup"
        
        # Update or add the MCP server configuration
        if command -v jq &> /dev/null; then
            jq --arg cmd "$command_path" '.mcpServers."claude-semantic-search" = {"command": $cmd, "args": []}' "$config_file" > "${config_file}.tmp" && mv "${config_file}.tmp" "$config_file"
            echo "✅ Updated $config_name with MCP server configuration"
        else
            echo "⚠️  Warning: jq not found. Please manually add this to your $config_file:"
            echo '  "mcpServers": {'
            echo '    "claude-semantic-search": {'
            echo "      \"command\": \"$command_path\","
            echo '      "args": []'
            echo '    }'
            echo '  }'
        fi
    else
        # Create new config
        echo "Creating $config_name configuration..."
        mkdir -p "$(dirname "$config_file")"
        cat > "$config_file" << EOF
{
  "mcpServers": {
    "claude-semantic-search": {
      "command": "$command_path",
      "args": []
    }
  }
}
EOF
        echo "✅ Created $config_name configuration"
    fi
}

# Get the full path to claude-search-mcp
mcp_command=$(which claude-search-mcp)

# Configure Claude Code
add_mcp_to_config ~/.claude.json "Claude Code" "claude-search-mcp"

# Configure Claude Desktop (macOS)
claude_desktop_config="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Claude Desktop needs full path since it doesn't have ~/.local/bin in PATH
    add_mcp_to_config "$claude_desktop_config" "Claude Desktop" "$mcp_command"
fi

# Configure Cursor
cursor_config="$HOME/.cursor/mcp.json"
if [ -d "$HOME/.cursor" ] || [ -f "$cursor_config" ]; then
    add_mcp_to_config "$cursor_config" "Cursor" "claude-search-mcp"
else
    echo "ℹ️  Cursor not detected. To configure Cursor later, add the MCP server to ~/.cursor/mcp.json"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "The following commands are now available:"
echo "  - claude-search: Search your Claude conversations"
echo "  - claude-index: Index your Claude conversations"
echo "  - claude-stats: View search statistics"
echo "  - claude-start: Start the indexing daemon"
echo "  - claude-stop: Stop the indexing daemon"
echo "  - claude-status: Check daemon status"
echo ""
echo "The MCP server has been configured for:"
echo "  ✅ Claude Code"
if [[ "$OSTYPE" == "darwin"* ]] && [ -f "$claude_desktop_config" ]; then
    echo "  ✅ Claude Desktop"
fi
if [ -f "$cursor_config" ]; then
    echo "  ✅ Cursor"
fi
echo ""
echo "Please restart these applications to use the semantic search features."
echo ""

# Prompt for indexing
read -p "Would you like to index your Claude conversations now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting indexing..."
    claude-index
    echo ""
    echo "✅ Initial indexing complete!"
    echo ""
    echo "You can now search your conversations using:"
    echo "  - In Claude Code/Desktop/Cursor: Ask to search your conversations"
    echo "  - In terminal: claude-search \"your query\""
else
    echo ""
    echo "You can index your conversations later by running: claude-index"
fi