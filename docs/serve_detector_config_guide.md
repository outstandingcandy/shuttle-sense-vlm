# Serve Detector Configuration Guide

This guide explains how to configure the ServeDetector using YAML configuration files with flexible model-API mapping.

## Overview

The new configuration structure separates API endpoints from models, allowing:
- **Multiple models to share the same API endpoint** (e.g., different Qwen models via DashScope)
- **Same model accessible through different APIs** (e.g., cloud vs. local deployment)
- **Easy model switching** by changing the `active_model` setting

## Default Configuration

By default, ServeDetector loads configuration from `src/config/serve_detector_config.yaml`.

```python
from core.serve_detector import ServeDetector

# Uses default config file
detector = ServeDetector()
```

## Custom Configuration

You can provide a custom configuration file path:

```python
from core.serve_detector import ServeDetector

# Use custom config file
detector = ServeDetector(config_path="path/to/your/config.yaml")
```

## Configuration Structure

```yaml
# API Endpoints Configuration
# Define available API endpoints
api_endpoints:
  dashscope:
    type: "dashscope"  # SDK type: "dashscope" or "openai"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"  # Environment variable name

  local_vllm:
    type: "openai"  # OpenAI-compatible API
    base_url: "http://localhost:8000/v1"
    api_key_env: "OPENAI_API_KEY"

  openai:
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"

# Models Configuration
# Define available models and map them to API endpoints
models:
  qwen-vl-max:
    name: "qwen-vl-max-latest"  # Model name as recognized by API
    api: "dashscope"  # References api_endpoints key
    description: "Qwen VL Max model via DashScope"

  qwen-vl-plus:
    name: "qwen3-vl-plus"
    api: "dashscope"  # Same API, different model
    description: "Qwen VL Plus model via DashScope"

  qwen-vl-local:
    name: "Qwen/Qwen3-VL-8B-Instruct"
    api: "local_vllm"  # Different API
    description: "Local Qwen VL model via vLLM"

  gpt4-vision:
    name: "gpt-4-vision-preview"
    api: "openai"
    description: "GPT-4 Vision via OpenAI API"

# Active Model Selection
active_model: "qwen-vl-max"  # Which model to use

# Video Frame Extraction
video_frames:
  max_frames: 8
  frame_size: [1024, 1024]

# Inference Parameters
inference:
  batch_size: 4
  max_new_tokens: 50
  temperature: 0.1
  top_p: 0.8
  do_sample: false

# Few-shot Learning
few_shot:
  enable: false
```

## Environment Variables

API keys must be set as environment variables:

```bash
# For DashScope API
export DASHSCOPE_API_KEY="your-api-key"

# For OpenAI API
export OPENAI_API_KEY="your-api-key"
```

## Common Use Cases

### 1. Switch Between Different Models (Same API)

Multiple Qwen models via DashScope:

```yaml
api_endpoints:
  dashscope:
    type: "dashscope"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"

models:
  qwen-vl-max:
    name: "qwen-vl-max-latest"
    api: "dashscope"

  qwen-vl-plus:
    name: "qwen3-vl-plus"
    api: "dashscope"

# Switch between models by changing this
active_model: "qwen-vl-max"  # or "qwen-vl-plus"
```

### 2. Switch Between Cloud and Local Deployment

Same model, different APIs:

```yaml
api_endpoints:
  cloud:
    type: "dashscope"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"

  local:
    type: "openai"
    base_url: "http://localhost:8000/v1"
    api_key_env: "OPENAI_API_KEY"

models:
  qwen-cloud:
    name: "qwen-vl-max-latest"
    api: "cloud"

  qwen-local:
    name: "qwen-vl-max-latest"  # Same model name
    api: "local"  # Different API endpoint

# Easy cloud/local switching
active_model: "qwen-cloud"  # or "qwen-local"
```

### 3. High Accuracy Configuration

```yaml
active_model: "qwen-vl-max"

inference:
  temperature: 0.0  # Deterministic
  batch_size: 2  # Smaller batches

video_frames:
  max_frames: 16  # More context

few_shot:
  enable: true  # Use examples
```

### 4. Fast Processing Configuration

```yaml
active_model: "qwen-vl-plus"  # Faster model

inference:
  temperature: 0.1
  batch_size: 8  # Parallel processing

video_frames:
  max_frames: 4  # Fewer frames

few_shot:
  enable: false  # No examples
```

### 5. Multi-Provider Setup

```yaml
api_endpoints:
  dashscope:
    type: "dashscope"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"

  openai:
    type: "openai"
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"

  local:
    type: "openai"
    base_url: "http://localhost:8000/v1"
    api_key_env: "OPENAI_API_KEY"

models:
  qwen-cloud:
    name: "qwen-vl-max-latest"
    api: "dashscope"

  gpt4:
    name: "gpt-4-vision-preview"
    api: "openai"

  qwen-local:
    name: "Qwen/Qwen3-VL-8B-Instruct"
    api: "local"

# Switch provider and model easily
active_model: "qwen-cloud"
```

## Adding New Models

To add a new model:

1. **Define the API endpoint** (if not already defined):
   ```yaml
   api_endpoints:
     my_api:
       type: "openai"  # or "dashscope"
       base_url: "http://my-server:8000/v1"
       api_key_env: "MY_API_KEY"
   ```

2. **Define the model**:
   ```yaml
   models:
     my_model:
       name: "model-name-on-api"
       api: "my_api"
       description: "My custom model"
   ```

3. **Activate the model**:
   ```yaml
   active_model: "my_model"
   ```

## Fallback Behavior

If the configuration file is not found or contains errors, ServeDetector falls back to default values:
- Model: `qwen-vl-max-latest`
- API: DashScope
- Logs warnings for troubleshooting

## Benefits

✅ **Flexible**: Easy switching between models and APIs
✅ **Scalable**: Add new models and APIs without code changes
✅ **Clean**: Separation of concerns (models vs. endpoints)
✅ **Reusable**: Same API can serve multiple models
✅ **Testable**: Easy to test different configurations

