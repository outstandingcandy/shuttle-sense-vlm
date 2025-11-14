# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Shuttle-Sense-VLM** is a badminton video analysis tool that uses Vision-Language Models (VLMs) to detect serves and analyze rallies in badminton videos. The system supports both Qwen-VL series models via DashScope API and local vLLM deployments.

## Core Architecture

### Key Design Patterns

1. **Two-stage Detection Pipeline**
   - Stage 1: Serve detection identifies serve moments in video segments
   - Stage 2: Rally analysis segments video based on detected serves

2. **Flexible Inference Backends**
   - DashScope SDK: Native Alibaba Cloud API integration
   - OpenAI-compatible API: Supports local vLLM deployments
   - Backend selection configured via YAML

3. **Few-shot Learning Support**
   - `MessageManager` (src/core/few_shot_manager.py) creates messages with optional example demonstrations
   - Example frames stored in `few_shot_examples/` directory structure: `{category}/{label}/frame_*.png`
   - Zero-shot mode: Direct query without examples
   - Few-shot mode: Includes positive/negative examples before query

### Module Responsibilities

- **ServeDetector** (src/core/serve_detector.py): Detects badminton serves in video segments using VLM inference
- **MessageManager** (src/core/few_shot_manager.py): Constructs multi-modal messages with images and text, manages example caching
- **ResponseParser** (src/utils/response_parser.py): Parses model outputs to extract structured information (serve detection, rally detection)
- **BadmintonAnalyzer** (src/badminton_analyzer.py): Main orchestrator for video analysis workflow

### Configuration System

Configuration is split across two files:

1. **serve_detector_config.yaml** - Infrastructure and model settings
   - `api_endpoints`: Define API endpoints (DashScope, local vLLM, OpenAI)
   - `models`: Map model names to API endpoints
   - `active_model`: Select which model to use
   - `video_frames`: Frame extraction parameters
   - `inference`: Batch size, tokens, temperature
   - `few_shot.enable`: Toggle few-shot learning mode

2. **prompts_config.py** - Prompts and few-shot configuration
   - `SERVE_PROMPTS`, `RALLY_PROMPTS`: Task-specific prompts in Chinese
   - `FEW_SHOT_SYSTEM_PROMPTS`: System-level instructions for tasks
   - `FEW_SHOT_EXAMPLES`: Maps tasks to example categories/labels
   - `FEW_SHOT_RESPONSES`: Expected assistant responses for examples
   - `FEW_SHOT_CONFIG`: Few-shot parameters (num examples, examples directory)

### Critical Implementation Details

**API Selection Logic** (serve_detector.py:46-141):
- Config loads `active_model` which references a model in `models` dict
- Model specifies `api` field which references key in `api_endpoints`
- API endpoint determines `type` (dashscope/openai), `base_url`, and `api_key_env`
- `_generate_batch()` routes to either `_generate_with_dashscope()` or `_generate_with_openai_api()`

**Message Construction** (few_shot_manager.py:212-376):
- `create_messages()` supports both zero-shot and few-shot modes via parameters
- Few-shot: Loads examples from disk, creates user/assistant message pairs, appends actual query
- Zero-shot: Only includes optional system prompt + user query
- All images stored as `file://` paths, converted to base64 for OpenAI API
- Session directories created under `temp_frames/{session_id}/` for query frames

**Frame Extraction**:
- Video segments processed with overlap to avoid missing boundary events
- Frames extracted uniformly across segment duration (np.linspace)
- Frames resized to config-specified dimensions before inference

## Common Development Commands

### Running Serve Detection
```bash
# Basic serve detection
python src/badminton_analyzer.py --video path/to/video.mp4

# With custom segment parameters
python src/badminton_analyzer.py --video video.mp4 \
    --segment-duration 3.0 --overlap 1.0 --max-segments 10

# Enable verbose logging
python src/badminton_analyzer.py --video video.mp4 --verbose --debug
```

### Preparing Few-shot Examples
```bash
# Extract positive examples (videos with serves)
python tools/prepare_few_shot_examples.py \
    --video reference_serve.mp4 \
    --category serve \
    --label has_serve \
    --start-time 10.5 \
    --duration 2.0 \
    --num-frames 8

# Extract negative examples (videos without serves)
python tools/prepare_few_shot_examples.py \
    --video reference_no_serve.mp4 \
    --category serve \
    --label no_serve \
    --start-time 5.0 \
    --duration 2.0 \
    --num-frames 8
```

### Configuration Testing
When modifying config files, verify loading:
```python
from src.core.serve_detector import ServeDetector
detector = ServeDetector()
print(detector.config)  # Verify active_model, API settings
```

## Environment Setup

### Required Environment Variables
- `DASHSCOPE_API_KEY`: For DashScope API access (Qwen-VL models)
- `OPENAI_API_KEY`: For OpenAI-compatible APIs (local vLLM)

### Typical Development Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install vLLM for local high-performance inference
pip install vllm>=0.11.0 ray>=2.0.0
```

### Starting Local vLLM Server (if using local models)
```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 \
    --max-model-len 8192
```

## Important Notes for Code Changes

### When Adding New Detection Tasks
1. Add prompts to `prompts_config.py` (SERVE_PROMPTS/RALLY_PROMPTS dictionaries)
2. If using few-shot: Add system prompt to `FEW_SHOT_SYSTEM_PROMPTS`
3. Define example mappings in `FEW_SHOT_EXAMPLES`
4. Define expected responses in `FEW_SHOT_RESPONSES`
5. Add parsing logic to `ResponseParser` for extracting structured results

### When Adding New Models
1. Define API endpoint in `serve_detector_config.yaml` under `api_endpoints` if new
2. Add model entry under `models` with `name` and `api` fields
3. Update `active_model` to select the new model
4. Ensure model name matches what the API expects

### When Modifying Video Processing
- Frame extraction logic in `video_processor.py` and `serve_detector.py:171-221`
- Segment generation in `serve_detector.py:223-291`
- Batch processing with ThreadPoolExecutor in `serve_detector.py:270-289`

### Response Parsing Logic
- `ResponseParser.contains_serve()` uses keyword matching on Chinese/English indicators
- Both positive and negative indicators checked; negative indicators have priority
- When adding new detection types, follow same pattern: define positive/negative indicators

## Testing Strategy

### Quick Integration Test
```bash
# Test with limited segments to verify pipeline
python src/badminton_analyzer.py \
    --video test_video.mp4 \
    --max-segments 5 \
    --verbose
```

### Testing Few-shot vs Zero-shot
1. Set `few_shot.enable: false` in config for zero-shot
2. Set `few_shot.enable: true` for few-shot (requires prepared examples)
3. Compare results to evaluate few-shot effectiveness

### Verifying New Model Integration
1. Add model to config, set as active
2. Run with `--debug` flag to save selected frames
3. Check `selected_frames/` directory for debugging output
4. Verify API responses in logs

## Git Workflow

### Current Branch State
- Working on `main` branch
- Staged changes include few-shot manager, serve detector, response parser
- Untracked directories: data/, docs/, results/, few_shot_examples/

### Making Changes
- Configuration changes (YAML/prompts): Test with small video segments first
- API changes: Verify both DashScope and OpenAI paths still work
- Frame extraction changes: Check memory usage with long videos

## Build & Test Commands

### Using uv (recommended)
- Install dependencies: `uv pip install --system -e .`
- Install dev dependencies: `uv pip install --system -e ".[dev]"`
- Update lock file: `uv pip compile --system pyproject.toml -o uv.lock`
- Install from lock file: `uv pip sync --system uv.lock`

### Using pip (alternative)
- Install dependencies: `pip install -e .`
- Install dev dependencies: `pip install -e ".[dev]"`

### Testing and linting
- Run tests: `pytest`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name -v`
- Run tests with coverage: `python -m pytest --cov=src/aws_mcp_server tests/`
- Run linter: `ruff check src/ tests/`
- Format code: `ruff format src/ tests/`

## Technical Stack

- **Python version**: Python 3.13+
- **Project config**: `pyproject.toml` for configuration and dependency management
- **Environment**: Use virtual environment in `.venv` for dependency isolation
- **Package management**: Use `uv` for faster, more reliable dependency management with lock file
- **Dependencies**: Separate production and dev dependencies in `pyproject.toml`
- **Version management**: Use `setuptools_scm` for automatic versioning from Git tags
- **Linting**: `ruff` for style and error checking
- **Type checking**: Use VS Code with Pylance for static type checking
- **Project layout**: Organize code with `src/` layout

## Code Style Guidelines

- **Formatting**: Black-compatible formatting via `ruff format`
- **Imports**: Sort imports with `ruff` (stdlib, third-party, local)
- **Type hints**: Use native Python type hints (e.g., `list[str]` not `List[str]`)
- **Documentation**: Google-style docstrings for all modules, classes, functions
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Function length**: Keep functions short (< 30 lines) and single-purpose
- **PEP 8**: Follow PEP 8 style guide (enforced via `ruff`)
- **Comment Language**: English

## Python Best Practices

- **File handling**: Prefer `pathlib.Path` over `os.path`
- **Debugging**: Use `logging` module instead of `print`
- **Error handling**: Use specific exceptions with context messages and proper logging
- **Data structures**: Use list/dict comprehensions for concise, readable code
- **Function arguments**: Avoid mutable default arguments
- **Data containers**: Leverage `dataclasses` to reduce boilerplate
- **Configuration**: Use environment variables (via `python-dotenv`) for configuration
- **AWS CLI**: Validate all commands before execution (must start with "aws")
- **Security**: Never store/log AWS credentials, set command timeouts

## Development Patterns & Best Practices

- **Favor simplicity**: Choose the simplest solution that meets requirements
- **DRY principle**: Avoid code duplication; reuse existing functionality
- **Configuration management**: Use environment variables for different environments
- **Focused changes**: Only implement explicitly requested or fully understood changes
- **Preserve patterns**: Follow existing code patterns when fixing bugs
- **File size**: Keep files under 300 lines; refactor when exceeding this limit
- **Modular design**: Create reusable, modular components
- **Logging**: Implement appropriate logging levels (debug, info, error)
- **Error handling**: Implement robust error handling for production reliability