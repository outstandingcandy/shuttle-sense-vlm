#!/usr/bin/env python3
"""
Message Manager - Manages message creation for VLM models, supporting both few-shot and zero-shot scenarios
"""

import logging
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
from utils.video_processor import extract_frames_from_video


logger = logging.getLogger(__name__)

class MessageManager:
    """
    Message Manager for Vision Language Models.

    Supports both few-shot learning (with example images) and zero-shot inference.
    Manages frame storage, message formatting, and example caching.
    """

    def __init__(self, examples_dir: str = "few_shot_examples", temp_dir: str = "temp_frames"):
        """
        Initialize Message Manager.

        Args:
            examples_dir: Directory for storing few-shot example frames
            temp_dir: Directory for storing temporary query frames
        """
        self.examples_dir = examples_dir
        os.makedirs(examples_dir, exist_ok=True)

        # Cache for loaded examples
        self.examples_cache = {}

        # Temporary directory for storing query frames
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(f"MessageManager initialized - Examples dir: {examples_dir}, Temp dir: {temp_dir}")
    
    def _save_frames_to_temp(self, frames: List[Image.Image], session_dir: str) -> str:
        """
        Save image frames to specified session directory and return file:// format paths.

        Args:
            frames: List of PIL Image objects
            session_dir: Session directory path

        Yields:
            File paths in file:// format
        """
        # Ensure session directory exists
        os.makedirs(session_dir, exist_ok=True)

        # Save images
        for i, frame in enumerate(frames):
            filename = f"{i:03d}.png"
            filepath = os.path.join(session_dir, filename)
            frame.save(filepath)
            abs_path = os.path.abspath(filepath)
            yield f"file://{abs_path}"
    
    def extract_example_frames(self,
                              video_path: str,
                              category: str,
                              label: str,
                              num_frames: int = 4,
                              start_time: float = 0,
                              duration: Optional[float] = None,
                              frame_size: tuple = (1024, 1024)) -> List[Image.Image]:
        """
        Extract example frames from reference video.

        Args:
            video_path: Path to reference video
            category: Example category (e.g., "serve", "rally", "action")
            label: Example label (e.g., "has_serve", "no_serve")
            num_frames: Number of frames to extract
            start_time: Start time in seconds
            duration: Duration in seconds
            frame_size: Frame size

        Returns:
            List of extracted frames
        """
        frames = extract_frames_from_video(video_path, num_frames, start_time, duration, frame_size)
        self._save_example_frames(frames, category, label, video_path)
        logger.info(f"Extracted {len(frames)} frames from {video_path} as {category}/{label} examples")
        return frames
    
    def _save_example_frames(self,
                            frames: List[Image.Image],
                            category: str,
                            label: str,
                            source_video: str):
        """Save example frames to disk in unique subdirectories."""
        try:
            # Extract video filename (without extension)
            video_name = os.path.splitext(os.path.basename(source_video))[0]
            # Clean filename to avoid filesystem issues
            video_name = video_name.replace(' ', '_').replace('/', '_').replace('\\', '_')

            # Base directory for this category/label
            base_dir = os.path.join(self.examples_dir, category, label)
            os.makedirs(base_dir, exist_ok=True)

            # Find next available sequence number for this video
            existing_dirs = []
            if os.path.exists(base_dir):
                existing_dirs = [
                    d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"{video_name}_")
                ]

            # Extract sequence numbers and find max
            max_seq = 0
            for dir_name in existing_dirs:
                try:
                    # Extract number from pattern: video_name_NNN
                    # Use removeprefix to handle video names with underscores
                    prefix = f"{video_name}_"
                    if dir_name.startswith(prefix):
                        seq_str = dir_name[len(prefix):]  # Remove prefix
                        seq_num = int(seq_str)
                        max_seq = max(max_seq, seq_num)
                except (ValueError, IndexError):
                    continue

            # Create new subdirectory with next sequence number
            next_seq = max_seq + 1
            example_dir = os.path.join(base_dir, f"{video_name}_{next_seq:03d}")
            os.makedirs(example_dir, exist_ok=True)

            # Save frames
            for i, frame in enumerate(frames):
                frame_path = os.path.join(example_dir, f"frame_{i:03d}.png")
                frame.save(frame_path)

            # Save metadata
            metadata = {
                "category": category,
                "label": label,
                "source_video": source_video,
                "video_name": video_name,
                "sequence_number": next_seq,
                "num_frames": len(frames),
                "timestamp": datetime.now().isoformat()
            }

            metadata_path = os.path.join(example_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Example frames saved to: {example_dir}")

        except Exception as e:
            logger.error(f"Failed to save example frames: {str(e)}")
    
    def load_examples(self, category: str, label: str) -> List[List[str]]:
        """
        Load saved example frames from all subdirectories.

        Each subdirectory represents a separate example, and frames are kept separate.

        Args:
            category: Example category
            label: Example label

        Returns:
            List of examples, where each example is a list of frame paths from one subdirectory.
            e.g., [[frames from example_001], [frames from example_002], ...]
        """
        cache_key = f"{category}/{label}"

        # Check cache
        if cache_key in self.examples_cache:
            num_examples = len(self.examples_cache[cache_key])
            total_frames = sum(len(ex) for ex in self.examples_cache[cache_key])
            logger.info(f"Loaded {num_examples} {category}/{label} examples ({total_frames} total frames) from cache")
            return self.examples_cache[cache_key]

        try:
            base_dir = os.path.join(self.examples_dir, category, label)

            if not os.path.exists(base_dir):
                logger.warning(f"Example directory does not exist: {base_dir}")
                return []

            # Load frames from each subdirectory as a separate example
            examples = []
            subdirs = sorted([
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ])

            for subdir in subdirs:
                subdir_path = os.path.join(base_dir, subdir)
                frame_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.png')])

                # Collect all frames for this example
                example_frames = []
                for frame_file in frame_files:
                    frame_path = os.path.join(subdir_path, frame_file)
                    example_frames.append(frame_path)

                if example_frames:
                    examples.append(example_frames)

            # Cache the results
            self.examples_cache[cache_key] = examples

            total_frames = sum(len(ex) for ex in examples)
            logger.info(f"Loaded {len(examples)} {category}/{label} examples ({total_frames} total frames) from {len(subdirs)} subdirectories")

            return examples

        except Exception as e:
            logger.error(f"Failed to load example frames: {str(e)}")
            return []

    def _generate_session_id(self,
                            video_path: Optional[str] = None,
                            start_time: Optional[float] = None,
                            end_time: Optional[float] = None) -> str:
        """
        Generate a unique session ID for message creation.

        Args:
            video_path: Video path for contextual naming
            start_time: Video segment start time
            end_time: Video segment end time

        Returns:
            Unique session ID string
        """
        parts = []

        # Add video filename (without extension and path)
        if video_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            # Clean filename, remove or replace special characters
            video_name = video_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            parts.append(video_name)

        # Add time segment information
        if start_time is not None and end_time is not None:
            parts.append(f"{start_time:.2f}_{end_time:.2f}")
        elif start_time is not None:
            parts.append(f"{start_time:.2f}")

        # Add UUID for uniqueness
        parts.append(uuid.uuid4().hex[:8])

        # If no information provided, use timestamp
        if not parts or len(parts) == 1:  # Only UUID
            parts.insert(0, datetime.now().strftime('%Y%m%d_%H%M%S'))

        return '_'.join(parts)

    def create_messages(self,
                       frames: List[Image.Image],
                       text: str,
                       system_prompt: Optional[str] = None,
                       session_id: Optional[str] = None,
                       video_path: Optional[str] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       example_category: Optional[str] = None,
                       positive_label: Optional[str] = None,
                       positive_response: str = "Yes, there is the action in this video.",
                       negative_label: Optional[str] = None,
                       negative_response: str = "No, there is no action in this video.",
                       num_positive_examples: int = 0,
                       num_negative_examples: int = 0) -> List[Dict[str, Any]]:
        """
        Create messages for VLM models, supporting both zero-shot and few-shot scenarios.

        When example_category and positive_label are provided with num_positive_examples > 0,
        this creates few-shot messages with example demonstrations. Otherwise, it creates
        simple zero-shot messages with just the query.

        Args:
            frames: Query frames to analyze
            text: Query text/question
            system_prompt: Optional system prompt
            session_id: Optional session ID (auto-generated if not provided)
            video_path: Video path for session ID generation
            start_time: Video segment start time (seconds)
            end_time: Video segment end time (seconds)
            example_category: Example category for few-shot learning (optional)
            positive_label: Positive example label for few-shot learning (optional)
            positive_response: Assistant's reply for positive examples (few-shot mode)
            negative_label: Negative example label for few-shot learning (optional)
            negative_response: Assistant's reply for negative examples (few-shot mode)
            num_positive_examples: Number of positive examples to include (0 for zero-shot)
            num_negative_examples: Number of negative examples to include (0 for zero-shot)

        Returns:
            Message list in multi-turn conversation format:
            - Zero-shot: [{"role": "system", ...}, {"role": "user", "content": [...]}]
            - Few-shot: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ..., {"role": "user", "content": [...]}]

        Examples:
            # Zero-shot usage
            messages = manager.create_messages(
                frames=video_frames,
                text="Is there a serve in this video?"
            )

            # Few-shot usage
            messages = manager.create_messages(
                frames=video_frames,
                text="Is there a serve in this video?",
                example_category="serve",
                positive_label="has_serve",
                num_positive_examples=2,
                negative_label="no_serve",
                num_negative_examples=1
            )
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = self._generate_session_id(video_path, start_time, end_time)

        session_dir = os.path.join(self.temp_dir, session_id)
        logger.info(f"Creating session directory: {session_dir}")

        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Determine if this is few-shot mode
        is_few_shot = (example_category and
                      (num_positive_examples > 0 or num_negative_examples > 0))

        if is_few_shot:
            # Few-shot mode: Add example demonstrations
            logger.info(f"Creating few-shot messages with category '{example_category}'")

            # Add positive examples
            if num_positive_examples > 0:
                positive_examples = self.load_examples(example_category, positive_label)
                if positive_examples:
                    # Select the requested number of examples (each example is already a complete set of frames)
                    num_to_use = min(num_positive_examples, len(positive_examples))

                    for i in range(num_to_use):
                        example_frames = positive_examples[i]

                        if example_frames:
                            # User message: example frames + question
                            user_content = []
                            for frame in example_frames:
                                image_url = f"file://{os.path.abspath(frame)}"
                                user_content.append({"type": "image", "image": image_url})
                            user_content.append({"type": "text", "text": text})

                            messages.append({
                                "role": "user",
                                "content": user_content
                            })

                            # Assistant reply: positive answer
                            messages.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": positive_response}]
                            })

                    logger.info(f"Added {num_to_use} positive examples")

            # Add negative examples
            if negative_label and num_negative_examples > 0:
                negative_examples = self.load_examples(example_category, negative_label)
                if negative_examples:
                    # Select the requested number of examples (each example is already a complete set of frames)
                    num_to_use = min(num_negative_examples, len(negative_examples))

                    for i in range(num_to_use):
                        example_frames = negative_examples[i]

                        if example_frames:
                            # User message: example frames + question
                            user_content = []
                            for frame in example_frames:
                                image_url = f"file://{os.path.abspath(frame)}"
                                user_content.append({"type": "image", "image": image_url})
                            user_content.append({"type": "text", "text": text})

                            messages.append({
                                "role": "user",
                                "content": user_content
                            })

                            # Assistant reply: negative answer
                            messages.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": negative_response}]
                            })

                    logger.info(f"Added {num_to_use} negative examples")

        # Add actual query (user's final question)
        query_content = []
        for image_url in self._save_frames_to_temp(frames, session_dir):
            query_content.append({"type": "image", "image": image_url})
        query_content.append({"type": "text", "text": text})

        messages.append({
            "role": "user",
            "content": query_content
        })

        mode = "few-shot" if is_few_shot else "zero-shot"
        logger.info(f"Created {mode} message with {len(frames)} query frames")

        return messages
    
    def list_available_examples(self) -> Dict[str, List[str]]:
        """List all available examples."""
        examples = {}

        if not os.path.exists(self.examples_dir):
            return examples

        for category in os.listdir(self.examples_dir):
            category_path = os.path.join(self.examples_dir, category)
            if os.path.isdir(category_path):
                labels = [label for label in os.listdir(category_path)
                         if os.path.isdir(os.path.join(category_path, label))]
                examples[category] = labels

        return examples

    def get_example_metadata(self, category: str, label: str) -> Optional[Dict[str, Any]]:
        """
        Get aggregated metadata for all examples under a category/label.

        Returns a summary including all subdirectories (examples) and their metadata.
        """
        base_dir = os.path.join(self.examples_dir, category, label)

        if not os.path.exists(base_dir):
            return None

        try:
            subdirs = sorted([
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
            ])

            if not subdirs:
                return None

            # Collect metadata from all subdirectories
            all_metadata = []
            total_frames = 0
            source_videos = []

            for subdir in subdirs:
                metadata_path = os.path.join(base_dir, subdir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        all_metadata.append(metadata)
                        total_frames += metadata.get("num_frames", 0)
                        source_videos.append(metadata.get("source_video", "unknown"))

            if not all_metadata:
                return None

            # Return aggregated metadata
            return {
                "category": category,
                "label": label,
                "num_examples": len(subdirs),
                "num_frames": total_frames,
                "source_videos": source_videos,
                "examples": all_metadata
            }

        except Exception as e:
            logger.error(f"Failed to read metadata: {str(e)}")
            return None

