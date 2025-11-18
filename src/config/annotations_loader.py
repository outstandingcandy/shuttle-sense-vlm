#!/usr/bin/env python3
"""
Annotation Config Loader - Loads configuration from annotation JSON files
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class AnnotationConfigLoader:
    """
    Loads and manages configuration from annotation JSON files.

    Replaces hardcoded prompts_config.py by providing dynamic configuration
    loading from annotation files with sentence-based labels and per-example queries.
    """

    def __init__(self, annotation_path: str = "docs/annotations_example.json"):
        """
        Initialize annotation config loader.

        Args:
            annotation_path: Path to annotation JSON file
        """
        self.annotation_path = Path(annotation_path)
        self.data = self._load_json()
        logger.info(f"Loaded annotation config from: {annotation_path}")

    def _load_json(self) -> Dict[str, Any]:
        """Load and parse annotation JSON file."""
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")

        try:
            with open(self.annotation_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate required structure
            if "examples" not in data:
                raise ValueError("Annotation file must contain 'examples' field")

            if "tasks" not in data:
                logger.warning("No 'tasks' configuration found in annotation file")
                data["tasks"] = {}

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in annotation file: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load annotation file: {str(e)}")

    def get_task_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get task-level configuration.

        Args:
            task_name: Name of the task (e.g., "serve_detection")

        Returns:
            Task configuration dictionary or None if not found

        Example:
            {
                "category": "serve",
                "system_prompt": "你是一名专业的羽毛球动作识别专家...",
            }
        """
        return self.data.get("tasks", {}).get(task_name)

    def get_system_prompt(self, task_name: str) -> Optional[str]:
        """
        Get system prompt for a task.

        Args:
            task_name: Name of the task

        Returns:
            System prompt string or None if not found
        """
        task_config = self.get_task_config(task_name)
        return task_config.get("system_prompt") if task_config else None

    def get_task_category(self, task_name: str) -> Optional[str]:
        """
        Get category for a task.

        Args:
            task_name: Name of the task

        Returns:
            Category string or None if not found
        """
        task_config = self.get_task_config(task_name)
        return task_config.get("category") if task_config else None

    def get_all_examples(self) -> List[Dict[str, Any]]:
        """
        Get all examples from annotation file.

        Returns:
            List of example dictionaries
        """
        return self.data.get("examples", [])

    def get_examples_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get examples filtered by category.

        Args:
            category: Category to filter by (e.g., "serve", "rally")

        Returns:
            List of examples matching the category
        """
        return [
            example for example in self.get_all_examples()
            if example.get("category") == category
        ]

    def get_example_by_video_and_time(
        self,
        video_path: str,
        start_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific example by video path and start time.

        Args:
            video_path: Video file path
            start_time: Start time in seconds

        Returns:
            Example dictionary or None if not found
        """
        for example in self.get_all_examples():
            if (example.get("video") == video_path and
                example.get("start_time") == start_time):
                return example
        return None

    def get_example_by_id(self, example_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific example by its ID.

        Args:
            example_id: Unique example ID

        Returns:
            Example dictionary or None if not found
        """
        for example in self.get_all_examples():
            if example.get("id") == example_id:
                return example
        return None

    def get_examples_by_ids(self, example_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Get multiple examples by their IDs.

        Args:
            example_ids: List of example IDs

        Returns:
            List of example dictionaries (may be shorter if some IDs not found)
        """
        examples = []
        for example_id in example_ids:
            example = self.get_example_by_id(example_id)
            if example:
                examples.append(example)
            else:
                logger.warning(f"Example ID {example_id} not found in annotations")
        return examples

    def get_positive_examples(self, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get positive examples (those with positive expected responses).

        Identifies positive examples by checking if "是" or "有" appears in expected_response
        and negative indicators like "没有" or "不" do not appear.

        Args:
            task_name: Optional task name to filter by (uses task's category if available)

        Returns:
            List of positive example dictionaries with their IDs
        """
        # For ID-based structure, we don't filter by category anymore
        # All examples are evaluated based on expected_response field
        positive_examples = []

        for example in self.get_all_examples():
            expected_response = example.get("expected_response", "")

            # Check for positive indicators
            has_positive = ("是" in expected_response or "有" in expected_response)
            has_negative = ("没有" in expected_response or "不" in expected_response)

            if has_positive and not has_negative:
                positive_examples.append(example)

        logger.debug(f"Found {len(positive_examples)} positive examples")
        return positive_examples

    def get_negative_examples(self, task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get negative examples (those with negative expected responses).

        Identifies negative examples by checking if "没有" or "不" appears in expected_response.

        Args:
            task_name: Optional task name to filter by (uses task's category if available)

        Returns:
            List of negative example dictionaries with their IDs
        """
        # For ID-based structure, we don't filter by category anymore
        # All examples are evaluated based on expected_response field
        negative_examples = []

        for example in self.get_all_examples():
            expected_response = example.get("expected_response", "")

            # Check for negative indicators
            has_negative = ("没有" in expected_response or "不" in expected_response)

            if has_negative:
                negative_examples.append(example)

        logger.debug(f"Found {len(negative_examples)} negative examples")
        return negative_examples

    def get_example_ids_for_task(
        self,
        task_name: str,
        num_examples: int = 5
    ) -> List[int]:
        """
        Get example IDs for a task.

        Args:
            task_name: Name of the task
            num_examples: Number of examples to select

        Returns:
            List of example IDs
        """
        all_examples = self.get_all_examples()

        # Extract IDs
        example_ids = [ex["id"] for ex in all_examples if "id" in ex][:num_examples]

        logger.info(
            f"Selected {len(example_ids)} example IDs for task {task_name}: {example_ids}"
        )

        return example_ids

    def list_available_tasks(self) -> List[str]:
        """
        List all available task names.

        Returns:
            List of task names
        """
        return list(self.data.get("tasks", {}).keys())

    def list_available_categories(self) -> List[str]:
        """
        List all unique categories from examples.

        Returns:
            List of unique category names
        """
        categories = set()
        for example in self.get_all_examples():
            if "category" in example:
                categories.add(example["category"])
        return sorted(list(categories))

    def get_example_count(self, category: Optional[str] = None) -> int:
        """
        Get count of examples, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            Number of examples
        """
        if category:
            return len(self.get_examples_by_category(category))
        return len(self.get_all_examples())

    def validate_example(self, example: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate an example has required fields.

        Args:
            example: Example dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        # Required fields for ID-based structure
        required_fields = ["id", "video", "start_time", "duration", "num_frames", "expected_response", "query"]
        errors = []

        for field in required_fields:
            if field not in example:
                errors.append(f"Missing required field: {field}")

        # Type validation
        if "id" in example and not isinstance(example["id"], int):
            errors.append("id must be an integer")

        if "start_time" in example and not isinstance(example["start_time"], (int, float)):
            errors.append("start_time must be a number")

        if "duration" in example and not isinstance(example["duration"], (int, float)):
            errors.append("duration must be a number")

        if "num_frames" in example and not isinstance(example["num_frames"], int):
            errors.append("num_frames must be an integer")

        return (len(errors) == 0, errors)

    def validate_all_examples(self) -> Dict[str, List[str]]:
        """
        Validate all examples in the annotation file.

        Also checks for ID uniqueness across all examples.

        Returns:
            Dictionary mapping example indices/IDs to error lists
        """
        all_errors = {}
        seen_ids = set()

        for i, example in enumerate(self.get_all_examples()):
            is_valid, errors = self.validate_example(example)

            # Check for duplicate IDs
            example_id = example.get("id")
            if example_id is not None:
                if example_id in seen_ids:
                    errors.append(f"Duplicate ID {example_id}")
                else:
                    seen_ids.add(example_id)

            if not is_valid or errors:
                identifier = f"example_{example_id}" if example_id is not None else f"example_index_{i}"
                all_errors[identifier] = errors

        return all_errors

    def reload(self) -> None:
        """Reload annotation file from disk."""
        self.data = self._load_json()
        logger.info(f"Reloaded annotation config from: {self.annotation_path}")
