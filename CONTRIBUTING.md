# Contributing to Claude Semantic Search

We welcome contributions to make Claude conversation search even better!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/claude-semantic-search`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment: `uv sync --dev`

## Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src

# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/
```

## Guidelines

### Code Style
- Use Black for formatting (line length: 88)
- Use isort for import sorting
- Follow PEP 8 conventions
- Add type hints to all functions

### Testing
- Write tests for new features
- Maintain or improve code coverage
- Run the full test suite before submitting

### Commits
- Use clear, descriptive commit messages
- Reference issues in commit messages when applicable
- Keep commits focused and atomic

## Submitting Changes

1. Ensure all tests pass: `uv run pytest`
2. Update documentation if needed
3. Push to your fork
4. Create a Pull Request with:
   - Clear description of changes
   - Any related issue numbers
   - Screenshots/examples if applicable

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bugs
- Provide system information (OS, Python version)
- Check existing issues before creating new ones

## Feature Ideas

We're especially interested in:
- Performance optimizations
- New search capabilities
- Integration improvements (Alfred, other editors)
- Better chunking strategies
- UI/UX enhancements

Thank you for contributing!