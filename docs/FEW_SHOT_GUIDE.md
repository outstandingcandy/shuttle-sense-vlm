# Few-shot Example Preparation Guide

This guide explains how to prepare few-shot examples for the Shuttle-Sense-VLM system.

## Overview

The `prepare_few_shot_examples.py` tool extracts frames from videos to create few-shot learning examples. It processes multiple videos using annotation files in JSON or CSV format.

## Usage

Extract examples from multiple videos using an annotation file:

```bash
# Using JSON annotation file
python tools/prepare_few_shot_examples.py --annotation-file annotations.json

# Using CSV annotation file
python tools/prepare_few_shot_examples.py --annotation-file annotations.csv
```

### Annotation File Formats

#### JSON Format

```json
{
  "examples": [
    {
      "video": "data/videos/match1.mp4",
      "category": "serve",
      "label": "has_serve",
      "start_time": 10.5,
      "duration": 2.0,
      "num_frames": 8
    },
    {
      "video": "data/videos/match2.mp4",
      "category": "serve",
      "label": "no_serve",
      "start_time": 5.0,
      "duration": 2.0,
      "num_frames": 8
    }
  ]
}
```

**Required fields:**
- `video`: Path to video file
- `category`: Example category
- `label`: Example label

**Optional fields:**
- `start_time`: Start time in seconds (default: 0)
- `duration`: Duration in seconds (default: entire video from start_time)
- `num_frames`: Number of frames to extract (default: 8)

#### CSV Format

```csv
video,category,label,start_time,duration,num_frames
data/videos/match1.mp4,serve,has_serve,10.5,2.0,8
data/videos/match2.mp4,serve,no_serve,5.0,2.0,8
```

**Required columns:**
- `video`: Path to video file
- `category`: Example category
- `label`: Example label

**Optional columns:**
- `start_time`: Start time in seconds (default: 0)
- `duration`: Duration in seconds (default: entire video from start_time)
- `num_frames`: Number of frames to extract (default: 8)

## Example Workflow

### Step 1: Prepare Your Annotation File

Create an annotation file (JSON or CSV) listing all the video segments you want to use as examples:

```json
{
  "examples": [
    {"video": "serve_videos/game1.mp4", "category": "serve", "label": "has_serve", "start_time": 12.0, "duration": 2.0},
    {"video": "serve_videos/game1.mp4", "category": "serve", "label": "has_serve", "start_time": 45.0, "duration": 2.0},
    {"video": "no_serve_videos/warmup.mp4", "category": "serve", "label": "no_serve", "start_time": 0, "duration": 2.0},
    {"video": "no_serve_videos/warmup.mp4", "category": "serve", "label": "no_serve", "start_time": 10.0, "duration": 2.0}
  ]
}
```

### Step 2: Run Batch Processing

```bash
python tools/prepare_few_shot_examples.py --annotation-file my_annotations.json
```

### Step 3: Verify Results

The tool will output a summary showing:
- Total examples processed
- Successful extractions
- Failed extractions
- All available examples in the database

Example output:
```
============================================================
Processing Summary:
  Total: 4 examples
  Success: 4
  Failed: 0
============================================================

📋 Current Available Examples:
  serve:
    - has_serve: 16 frames (source: game1.mp4)
    - no_serve: 16 frames (source: warmup.mp4)
```

### Step 4: Enable Few-shot Mode

Update `src/config/serve_detector_config.yaml`:

```yaml
few_shot:
  enable: true
```

## Directory Structure

Examples are stored in the following structure:

```
few_shot_examples/
├── serve/
│   ├── has_serve/
│   │   ├── frame_000.png
│   │   ├── frame_001.png
│   │   ├── ...
│   │   └── metadata.json
│   └── no_serve/
│       ├── frame_000.png
│       ├── ...
│       └── metadata.json
└── rally/
    ├── has_rally/
    │   └── ...
    └── no_rally/
        └── ...
```

## Best Practices

### Selecting Example Segments

1. **Positive Examples (has_serve, has_rally)**
   - Choose clear, unambiguous examples
   - Include variety (different angles, lighting, players)
   - Ensure the action is centered in the time segment

2. **Negative Examples (no_serve, no_rally)**
   - Include similar-looking but different actions
   - Add challenging cases (preparation poses, similar movements)
   - Balance difficulty with clarity

### Number of Examples

- Start with 2-3 positive and 2-3 negative examples per category
- Test performance and add more if needed
- Too many examples can slow inference; find the right balance

### Frame Count

- For short actions (serves): 6-8 frames usually sufficient
- For longer sequences (rallies): 10-12 frames may be better
- Ensure frames capture the complete action

## Troubleshooting

### Video Not Found

**Error:** `❌ 视频文件不存在: path/to/video.mp4`

**Solution:**
- Check that video paths in annotation file are correct
- Use absolute paths or paths relative to where you run the script
- Verify files exist: `ls path/to/video.mp4`

### Extraction Failed

**Error:** `❌ 提取失败`

**Common causes:**
- Video file is corrupted
- Insufficient permissions to read video
- Invalid time range (start_time + duration exceeds video length)
- Missing video codecs

### Invalid Annotation Format

**Error:** `JSON解析错误` or `CSV文件缺少必需列`

**Solution:**
- Verify JSON syntax is valid (use a JSON validator)
- Ensure CSV has required columns: `video,category,label`
- Check for missing commas, quotes, or brackets

## Advanced Usage

### Custom Examples Directory

Store examples in a different location:

```bash
python tools/prepare_few_shot_examples.py \
    --annotation-file annotations.json \
    --examples-dir /path/to/custom/examples
```

Remember to update the `examples_dir` in `prompts_config.py` to match.

### Adding Examples Incrementally

You can run the tool multiple times to add more examples:

```bash
# First batch of examples
python tools/prepare_few_shot_examples.py --annotation-file batch1.json

# Later, add more examples from a different annotation file
python tools/prepare_few_shot_examples.py --annotation-file batch2.json
```

All examples will be accumulated in the same `few_shot_examples` directory (or your custom directory).

## See Also

- Example annotation files: `docs/annotations_example.json`, `docs/annotations_example.csv`
- Configuration: `src/config/serve_detector_config.yaml`
- Few-shot configuration: `src/config/prompts_config.py`
- MessageManager documentation: `src/core/few_shot_manager.py`
