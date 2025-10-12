#!/bin/bash
set -e

echo "Installing Claude Semantic Search MCP Server..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Detect GPU type
detect_gpu() {
    # Check for AMD ROCm
    if command -v rocminfo &> /dev/null || [ -d /sys/class/kfd ]; then
        echo "amd"
        return
    fi

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia"
        return
    fi

    # Check for Apple Silicon
    if [[ "$OSTYPE" == "darwin"* ]] && [ "$(uname -m)" == "arm64" ]; then
        echo "apple"
        return
    fi

    echo "none"
}

gpu_type=$(detect_gpu)
install_gpu_support=false

# Check if user wants GPU support
if [ "$gpu_type" != "none" ]; then
    echo ""
    echo "GPU Detection"
    echo "============="
    case "$gpu_type" in
        amd)
            echo "✅ AMD GPU detected (ROCm)"
            echo ""
            read -p "Install PyTorch with ROCm for GPU-accelerated embeddings? (Y/n) " -r
            echo ""
            # Default to yes if empty
            if [[ -z "$REPLY" ]] || [[ $REPLY =~ ^[Yy]$ ]]; then
                install_gpu_support=true
            fi
            ;;
        nvidia)
            echo "✅ NVIDIA GPU detected"
            echo ""
            echo "For GPU support, install with conda first:"
            echo "  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
            echo "  conda install faiss-gpu -c conda-forge"
            echo ""
            read -p "Have you already installed CUDA PyTorch via conda? (y/N) " -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                install_gpu_support=true
            fi
            ;;
        apple)
            echo "✅ Apple Silicon detected"
            echo "PyTorch with MPS support is included in base installation."
            install_gpu_support=true
            ;;
    esac
fi

# Clean up Python bytecode cache to ensure fresh install
echo ""
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Install the tool
echo ""
echo "Installing claude-semantic-search..."
uv tool install . --force --reinstall

# Install GPU-specific packages after main installation
if [ "$install_gpu_support" = true ] && [ "$gpu_type" = "amd" ]; then
    echo ""
    echo "Replacing PyTorch with ROCm version in tool environment..."

    # Get the tool's python executable path
    tool_python=$(uv tool run --from claude-semantic-search -- python -c "import sys; print(sys.executable)")

    # Remove CUDA versions of torch and triton
    echo "Removing CUDA PyTorch..."
    uv pip uninstall torch triton --python "$tool_python" --quiet 2>/dev/null || true

    # Install ROCm versions
    echo "Installing ROCm PyTorch..."
    uv pip install --pre torch pytorch-triton-rocm --index-url https://download.pytorch.org/whl/nightly/rocm7.0 --python "$tool_python" --quiet

    echo "✅ PyTorch with ROCm installed in tool environment"
    echo "Note: Embeddings will use GPU. FAISS search uses CPU (no ROCm support)."
fi

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
read -p "Use the default location? (Y/n) " -r
echo ""

if [[ -n "$REPLY" ]] && [[ ! $REPLY =~ ^[Yy]$ ]]; then
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
read -p "Would you like to index your Claude conversations now? (Y/n) " -r
echo ""
if [[ -z "$REPLY" ]] || [[ $REPLY =~ ^[Yy]$ ]]; then
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