#!/usr/bin/env python3
"""
Message Manager - Manages message creation for VLM models, supporting both few-shot and zero-shot scenarios
"""

import logging
import os
import json
import uuid
import re
import shutil
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
                              example_id: int,
                              query: Optional[str] = None,
                              expected_response: Optional[str] = None,
                              num_frames: int = 4,
                              start_time: float = 0,
                              duration: Optional[float] = None,
                              frame_size: tuple = (1024, 1024)) -> List[Image.Image]:
        """
        Extract example frames from reference video.

        Args:
            video_path: Path to reference video
            example_id: Unique example ID
            query: Query text associated with this example (optional)
            expected_response: Expected assistant response for this example (optional)
            num_frames: Number of frames to extract
            start_time: Start time in seconds
            duration: Duration in seconds
            frame_size: Frame size

        Returns:
            List of extracted frames
        """
        frames = extract_frames_from_video(video_path, num_frames, start_time, duration, frame_size)
        self._save_example_frames(frames, example_id, video_path, query, expected_response)
        logger.info(f"Extracted {len(frames)} frames from {video_path} as example ID {example_id}")
        return frames
    
    def _save_example_frames(self,
                            frames: List[Image.Image],
                            example_id: int,
                            source_video: str,
                            query: Optional[str] = None,
                            expected_response: Optional[str] = None):
        """
        Save example frames to disk using ID-based directory structure.

        Args:
            frames: List of PIL Image objects
            example_id: Unique example ID
            source_video: Source video path
            query: Optional query text for this example
            expected_response: Optional expected assistant response
        """
        try:
            # Create directory with ID
            example_dir = os.path.join(self.examples_dir, str(example_id))

            # Check if directory already exists
            if os.path.exists(example_dir):
                logger.warning(f"Example ID {example_id} already exists, overwriting")
                shutil.rmtree(example_dir)

            os.makedirs(example_dir, exist_ok=True)

            # Save frames
            for i, frame in enumerate(frames):
                frame_path = os.path.join(example_dir, f"frame_{i:03d}.png")
                frame.save(frame_path)

            # Save metadata with ID
            metadata = {
                "id": example_id,
                "query": query,
                "expected_response": expected_response,
                "source_video": source_video,
                "num_frames": len(frames),
                "timestamp": datetime.now().isoformat()
            }

            metadata_path = os.path.join(example_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Example frames saved to: {example_dir}")

        except Exception as e:
            logger.error(f"Failed to save example frames: {str(e)}")
    

    def load_example_by_id(self, example_id: int) -> Optional[Dict[str, Any]]:
        """
        Load a single example by its ID.

        Args:
            example_id: Unique example ID

        Returns:
            Dictionary with 'frames' and 'metadata', or None if not found
        """
        cache_key = f"id_{example_id}"

        # Check cache
        if cache_key in self.examples_cache:
            logger.debug(f"Loaded example ID {example_id} from cache")
            return self.examples_cache[cache_key]

        try:
            example_dir = os.path.join(self.examples_dir, str(example_id))

            if not os.path.exists(example_dir):
                logger.warning(f"Example ID {example_id} not found: {example_dir}")
                return None

            # Load frames
            frame_files = sorted([
                f for f in os.listdir(example_dir)
                if f.endswith('.png') and f.startswith('frame_')
            ])

            example_frames = []
            for frame_file in frame_files:
                frame_path = os.path.join(example_dir, frame_file)
                example_frames.append(frame_path)

            # Load metadata from centralized annotations.json
            annotations_path = "data/annotations.json"
            metadata = {}
            if os.path.exists(annotations_path):
                try:
                    with open(annotations_path, "r", encoding="utf-8") as f:
                        annotations_data = json.load(f)
                        # Find the example with matching ID
                        for example in annotations_data.get("examples", []):
                            if example.get("id") == example_id:
                                metadata = example
                                break
                        if not metadata:
                            logger.warning(f"Example ID {example_id} not found in {annotations_path}")
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {annotations_path}: {str(e)}")
            else:
                logger.warning(f"Annotations file not found: {annotations_path}")

            if not example_frames:
                logger.warning(f"No frames found for example ID {example_id}")
                return None

            example_data = {
                'frames': example_frames,
                'metadata': metadata
            }

            # Cache the result
            self.examples_cache[cache_key] = example_data

            logger.debug(f"Loaded example ID {example_id} with {len(example_frames)} frames")
            return example_data

        except Exception as e:
            logger.error(f"Failed to load example ID {example_id}: {str(e)}")
            return None

    def load_examples_by_ids(self, example_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Load multiple examples by their IDs.

        Args:
            example_ids: List of example IDs to load

        Returns:
            List of dictionaries, each containing:
            - 'frames': List of frame file paths
            - 'metadata': Metadata dictionary with ID, query, expected_response, etc.
        """
        examples = []

        for example_id in example_ids:
            example_data = self.load_example_by_id(example_id)
            if example_data:
                examples.append(example_data)
            else:
                logger.warning(f"Skipping missing example ID {example_id}")

        total_frames = sum(len(ex['frames']) for ex in examples)
        logger.info(f"Loaded {len(examples)}/{len(example_ids)} examples with {total_frames} total frames")

        return examples

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
                       example_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Create messages for VLM models, supporting both zero-shot and few-shot scenarios.

        When no examples are provided, creates zero-shot messages.

        Args:
            frames: Query frames to analyze
            text: Query text/question
            system_prompt: Optional system prompt
            session_id: Optional session ID (auto-generated if not provided)
            video_path: Video path for session ID generation
            start_time: Video segment start time (seconds)
            end_time: Video segment end time (seconds)
            example_ids: List of example IDs to use for few-shot learning

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
                example_ids=[1, 2, 3, 4, 5]
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

        # Add few-shot examples if provided
        if example_ids:
            logger.info(f"Creating few-shot messages with {len(example_ids)} examples")

            examples = self.load_examples_by_ids(example_ids)

            for example_data in examples:
                example_frames = example_data['frames']
                example_metadata = example_data['metadata']

                if example_frames:
                    # Use query from metadata or default text
                    example_query = example_metadata.get('query', text)
                    expected_response = example_metadata.get(
                        'expected_response',
                        "Yes, there is the action in this video."
                    )

                    # User message: example frames + question
                    user_content = []
                    for frame in example_frames:
                        image_url = f"file://{os.path.abspath(frame)}"
                        user_content.append({"type": "image", "image": image_url})
                    user_content.append({"type": "text", "text": example_query})

                    messages.append({
                        "role": "user",
                        "content": user_content
                    })

                    # Assistant reply: expected response from metadata
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": expected_response}]
                    })

            logger.info(f"Added {len(examples)} examples")

        # Add actual query (user's final question)
        query_content = []
        for image_url in self._save_frames_to_temp(frames, session_dir):
            query_content.append({"type": "image", "image": image_url})
        query_content.append({"type": "text", "text": text})

        messages.append({
            "role": "user",
            "content": query_content
        })

        mode = "few-shot" if example_ids else "zero-shot"
        logger.info(f"Created {mode} message with {len(frames)} query frames")

        return messages

    def save_all_images_from_messages(self, messages: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Extract and save all images from a messages list to a specified directory.

        Args:
            messages: List of message dictionaries with content
            output_dir: Directory to save images to

        Returns:
            List of saved image file paths
        """
        import shutil

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        image_counter = 0

        for msg_idx, message in enumerate(messages):
            if message.get("role") not in ["user", "assistant"]:
                continue

            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for content_item in content:
                if content_item.get("type") == "image":
                    image_url = content_item.get("image", "")

                    # Handle file:// URLs
                    if image_url.startswith("file://"):
                        source_path = image_url[7:]  # Remove "file://" prefix

                        if os.path.exists(source_path):
                            # Create descriptive filename
                            role = message.get("role", "unknown")
                            ext = os.path.splitext(source_path)[1] or ".png"
                            dest_filename = f"{role}_{msg_idx:02d}_{image_counter:03d}{ext}"
                            dest_path = os.path.join(output_dir, dest_filename)

                            # Copy the image
                            shutil.copy2(source_path, dest_path)
                            saved_paths.append(dest_path)
                            image_counter += 1
                            logger.debug(f"Saved image: {dest_filename}")
                        else:
                            logger.warning(f"Image file not found: {source_path}")

        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths

