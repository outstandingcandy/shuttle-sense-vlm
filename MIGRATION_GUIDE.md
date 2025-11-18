# Migration Guide: Category/Label to ID-Based Structure

## Migration Complete! ✅

The few-shot example system has been successfully migrated from category/label structure to ID-based structure.

## What Changed

### Before Migration
```
few_shot_examples/
└── serve/
    └── {long_descriptive_label}/
        └── {video}_{seq}/
            ├── frame_*.png
            └── metadata.json

data/annotations_example.json  # Old format (no IDs)
```

### After Migration
```
few_shot_examples/
├── 1/  # Simple numeric IDs
├── 2/
├── 3/
├── 4/
└── 5/

data/annotations.json  # New format (with IDs)
data/annotations_example.json  # Old format (kept for reference)
```

## Files Created/Updated

1. **New annotation file**: `data/annotations.json` - USE THIS FILE going forward
2. **Backup**: `few_shot_examples_backup/` - Original examples (just in case)
3. **Migration script**: `tools/migrate_to_id_structure.py`
4. **Test script**: `test_annotation_config.py`

## How to Use the New System

### 1. Use the New Annotation File

**IMPORTANT**: Always use `data/annotations.json` (with IDs), not `data/annotations_example.json`

```bash
# ✅ CORRECT: Use the new annotation file
python tools/prepare_few_shot_examples.py --annotation-file data/annotations.json

# ❌ WRONG: Don't use the old file (no IDs)
python tools/prepare_few_shot_examples.py --annotation-file data/annotations_example.json
```

### 2. Current Example IDs

Your migrated examples:
- **ID 1**: Positive example (有发球动作)
- **ID 2-5**: Negative examples (没有发球动作)

### 3. Adding New Examples

When adding new examples, assign new sequential IDs:

```json
{
  "examples": [
    {
      "id": 6,  // New ID
      "video": "path/to/video.mp4",
      "start_time": 0,
      "duration": 2.0,
      "num_frames": 8,
      "expected_response": "是的，这段视频中有发球动作。",
      "query": "这段羽毛球视频中是否包含发球动作？..."
    }
  ]
}
```

Then extract frames:
```bash
python tools/prepare_few_shot_examples.py --annotation-file data/annotations.json
```

### 4. Updating Existing Examples

To modify an example:
1. Edit `data/annotations.json` 
2. Re-run the preparation tool to regenerate frames:
```bash
python tools/prepare_few_shot_examples.py --annotation-file data/annotations.json
```

### 5. Testing Your Changes

Run the test script to verify everything works:
```bash
python test_annotation_config.py
```

## Code Changes for Your Application

### Old Way (Deprecated)
```python
# Don't use this anymore
from config.prompts_config import FEW_SHOT_EXAMPLES

example_config = FEW_SHOT_EXAMPLES.get("serve_detection", {})
messages = message_manager.create_messages(
    frames=frames,
    text=query,
    example_category=example_config.get("category"),
    positive_label=example_config.get("positive_label"),
    negative_label=example_config.get("negative_label"),
    num_positive_examples=1,
    num_negative_examples=4
)
```

### New Way (Recommended)
```python
# Use this approach
from config.annotations_loader import AnnotationConfigLoader

annotation_config = AnnotationConfigLoader("data/annotations.json")
positive_ids, negative_ids = annotation_config.get_example_ids_for_task(
    task_name="serve_detection",
    num_positive=1,
    num_negative=4
)

messages = message_manager.create_messages(
    frames=frames,
    text=query,
    positive_example_ids=positive_ids,  # e.g., [1]
    negative_example_ids=negative_ids   # e.g., [2, 3, 4, 5]
)
```

## Benefits of ID-Based Structure

1. **Simpler directory structure** - No more long Chinese label names
2. **Easier management** - Just use numbers to reference examples
3. **Flexible labeling** - Labels can be edited without breaking references
4. **Automatic selection** - System auto-detects positive/negative based on `expected_response`
5. **Better metadata** - Each example stores its own query and expected response

## Troubleshooting

### Error: "Missing required field: id"
**Cause**: You're using the old `annotations_example.json` file  
**Fix**: Use `data/annotations.json` instead

### Error: "Example ID X not found"
**Cause**: The example directory `few_shot_examples/X/` doesn't exist  
**Fix**: Run the preparation tool to extract frames for that example

### Need to revert the migration?
```bash
# Remove new structure
rm -rf few_shot_examples/

# Restore backup
mv few_shot_examples_backup/ few_shot_examples/

# Use old annotation file
# (but you'll need to update code to use old API)
```

## Next Steps

1. Update your application code to use the new ID-based API
2. Test with `test_annotation_config.py`
3. Add new examples using sequential IDs (6, 7, 8...)
4. Optionally delete `data/annotations_example.json` if you don't need it

## Questions?

The system maintains backward compatibility, so old code will still work but will show deprecation warnings. Migrate to the new API when you can!
