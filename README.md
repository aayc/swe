# SWE - Software Engineering Agent Engineer

A software engineering agent engineer designed to assist with development tasks, code analysis, and project management.

## Usage

```
swe "there's a problem where ... can you fix it?"
```

## Features

- Intelligent code analysis and generation
- Project structure management
- Development workflow automation
- Code quality assurance

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

### Prerequisites

- Python 3.10+
- uv package manager

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd swe
   ```

2. Install dependencies:
   ```bash
   uv sync --dev
   ```

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

### Development Commands

- **Run tests**: `uv run pytest`
- **Format code**: `uv run black src/ tests/`
- **Lint code**: `uv run ruff check src/ tests/`
- **Type check**: `uv run mypy src/`
- **Run pre-commit**: `uv run pre-commit run --all-files`
