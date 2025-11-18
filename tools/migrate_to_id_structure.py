#!/usr/bin/env python3
"""Migration script to convert few-shot examples from category/label structure to ID-based structure.

This script:
1. Reads existing annotations from data/annotations_example.json
2. Scans few_shot_examples directory for existing examples
3. Assigns sequential numeric IDs to each example
4. Reorganizes directory structure from {category}/{label}/{video}_{seq}/ to {id}/
5. Updates metadata.json files with IDs
6. Generates new data/annotations.json with IDs and expected_response fields
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def infer_expected_response(label: str) -> str:
    """Infer expected response based on label content.

    Args:
        label: Descriptive label (e.g., "这段视频展示了标准的羽毛球发球动作...")

    Returns:
        Expected assistant response
    """
    # Check if this is a positive example (has serve)
    has_serve_keywords = ["发球动作", "发出", "发球"]
    no_serve_keywords = ["没有", "不是"]

    label_lower = label.lower()

    # Check for negative indicators first (higher priority)
    for keyword in no_serve_keywords:
        if keyword in label:
            return "不，这段视频中没有发球动作。"

    # Then check for positive indicators
    for keyword in has_serve_keywords:
        if keyword in label:
            return "是的，这段视频中有发球动作。"

    # Default to negative if unclear
    return "不，这段视频中没有发球动作。"


def sanitize_label(label: str) -> str:
    """Sanitize label to create filesystem-safe directory name.

    This mirrors the logic in MessageManager.sanitize_label()
    """
    # Replace filesystem-unsafe characters
    safe_label = label.replace("/", "_").replace("\\", "_")
    safe_label = safe_label.replace(":", "_").replace("*", "_")
    safe_label = safe_label.replace("?", "_").replace("\"", "_")
    safe_label = safe_label.replace("<", "_").replace(">", "_")
    safe_label = safe_label.replace("|", "_")

    # Replace multiple underscores with single underscore
    while "__" in safe_label:
        safe_label = safe_label.replace("__", "_")

    # Truncate to reasonable length
    if len(safe_label) > 100:
        safe_label = safe_label[:100]

    return safe_label.strip("_")


def find_existing_examples(examples_dir: Path) -> List[Dict[str, Any]]:
    """Find all existing example directories in few_shot_examples.

    Args:
        examples_dir: Path to few_shot_examples directory

    Returns:
        List of dicts with 'path', 'category', 'label', 'metadata'
    """
    found_examples = []

    if not examples_dir.exists():
        logger.warning(f"Examples directory not found: {examples_dir}")
        return found_examples

    # Iterate through category directories
    for category_dir in examples_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        # Iterate through label directories
        for label_dir in category_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label_sanitized = label_dir.name

            # Iterate through video sequence directories
            for video_dir in label_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                # Check if metadata.json exists
                metadata_path = video_dir / "metadata.json"
                metadata = {}

                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                found_examples.append({
                    "old_path": video_dir,
                    "category": category,
                    "label_sanitized": label_sanitized,
                    "label_original": metadata.get("label", ""),
                    "metadata": metadata
                })

                logger.info(f"Found example: {video_dir.relative_to(examples_dir)}")

    return found_examples


def create_backup(examples_dir: Path, backup_dir: Path):
    """Create backup of examples directory.

    Args:
        examples_dir: Source directory to backup
        backup_dir: Destination backup directory
    """
    if backup_dir.exists():
        logger.info(f"Backup already exists: {backup_dir}")
        response = input("Overwrite existing backup? (y/n): ")
        if response.lower() != "y":
            logger.info("Skipping backup creation")
            return
        shutil.rmtree(backup_dir)

    logger.info(f"Creating backup: {examples_dir} -> {backup_dir}")
    shutil.copytree(examples_dir, backup_dir)
    logger.info("Backup created successfully")


def migrate_examples(
    annotations_path: Path,
    examples_dir: Path,
    output_annotations_path: Path,
    backup_dir: Path,
    dry_run: bool = False
):
    """Migrate examples from category/label structure to ID-based structure.

    Args:
        annotations_path: Path to input annotations_example.json
        examples_dir: Path to few_shot_examples directory
        output_annotations_path: Path to output annotations.json
        backup_dir: Path to backup directory
        dry_run: If True, only simulate migration without making changes
    """
    logger.info("=" * 80)
    logger.info("Starting migration to ID-based structure")
    logger.info("=" * 80)

    # Load existing annotations
    logger.info(f"Loading annotations from: {annotations_path}")
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    tasks = annotations_data.get("tasks", {})
    examples = annotations_data.get("examples", [])

    logger.info(f"Found {len(examples)} examples in annotations file")

    # Find existing example directories
    logger.info(f"Scanning examples directory: {examples_dir}")
    existing_examples = find_existing_examples(examples_dir)
    logger.info(f"Found {len(existing_examples)} example directories on disk")

    # Create backup if not dry run
    if not dry_run and existing_examples:
        create_backup(examples_dir, backup_dir)

    # Assign sequential IDs to examples
    updated_examples = []
    id_counter = 1

    for example in examples:
        # Infer expected response from label
        label = example.get("label", "")
        expected_response = infer_expected_response(label)

        # Create updated example with ID
        updated_example = {
            "id": id_counter,
            "video": example.get("video", ""),
            "start_time": example.get("start_time", 0),
            "duration": example.get("duration", 2.0),
            "num_frames": example.get("num_frames", 8),
            "expected_response": expected_response,
            "label": label,  # Keep for reference
            "query": example.get("query", "")
        }

        updated_examples.append(updated_example)

        logger.info(f"ID {id_counter}: {label[:50]}...")
        logger.info(f"  Expected response: {expected_response}")

        # Find matching directory on disk
        category = example.get("category", "serve")
        label_sanitized = sanitize_label(label)

        matching_dirs = [
            ex for ex in existing_examples
            if ex["category"] == category and ex["label_sanitized"] == label_sanitized
        ]

        if matching_dirs:
            old_dir = matching_dirs[0]["old_path"]
            new_dir = examples_dir / str(id_counter)

            logger.info(f"  Will migrate: {old_dir.relative_to(examples_dir)} -> {id_counter}/")

            if not dry_run:
                # Move directory to new ID-based location
                if new_dir.exists():
                    logger.warning(f"  Target directory already exists: {new_dir}")
                else:
                    shutil.move(str(old_dir), str(new_dir))
                    logger.info(f"  Moved to: {new_dir}")

                # Update metadata.json with ID
                metadata_path = new_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    metadata["id"] = id_counter
                    metadata["expected_response"] = expected_response

                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)

                    logger.info(f"  Updated metadata.json with ID")

            # Remove from existing_examples to avoid reprocessing
            existing_examples.remove(matching_dirs[0])
        else:
            logger.warning(f"  No matching directory found on disk for ID {id_counter}")

        id_counter += 1

    # Create output annotations with IDs
    output_data = {
        "tasks": tasks,
        "examples": updated_examples
    }

    if not dry_run:
        logger.info(f"Writing updated annotations to: {output_annotations_path}")
        with open(output_annotations_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info("Updated annotations file created")

    # Clean up empty category/label directories
    if not dry_run and examples_dir.exists():
        logger.info("Cleaning up empty category/label directories...")
        for category_dir in examples_dir.iterdir():
            if category_dir.is_dir() and category_dir.name.isdigit():
                continue  # Skip ID directories

            if category_dir.is_dir():
                # Check if empty
                try:
                    if not any(category_dir.iterdir()):
                        category_dir.rmdir()
                        logger.info(f"  Removed empty directory: {category_dir.name}")
                    else:
                        # Check label subdirectories
                        for label_dir in category_dir.iterdir():
                            if label_dir.is_dir() and not any(label_dir.iterdir()):
                                label_dir.rmdir()
                                logger.info(f"  Removed empty directory: {category_dir.name}/{label_dir.name}")

                        # Check category again
                        if not any(category_dir.iterdir()):
                            category_dir.rmdir()
                            logger.info(f"  Removed empty directory: {category_dir.name}")
                except Exception as e:
                    logger.warning(f"  Could not clean up {category_dir}: {e}")

    # Summary
    logger.info("=" * 80)
    logger.info("Migration complete!")
    logger.info(f"Total examples migrated: {len(updated_examples)}")
    logger.info(f"Output annotations: {output_annotations_path}")
    if not dry_run and backup_dir.exists():
        logger.info(f"Backup created at: {backup_dir}")
    logger.info("=" * 80)


def main():
    """Main entry point for migration script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate few-shot examples from category/label structure to ID-based structure"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations_example.json"),
        help="Path to input annotations file (default: data/annotations_example.json)"
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path("few_shot_examples"),
        help="Path to few_shot_examples directory (default: few_shot_examples)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations.json"),
        help="Path to output annotations file (default: data/annotations.json)"
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("few_shot_examples_backup"),
        help="Path to backup directory (default: few_shot_examples_backup)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making changes"
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    migrate_examples(
        annotations_path=args.annotations,
        examples_dir=args.examples_dir,
        output_annotations_path=args.output,
        backup_dir=args.backup_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
