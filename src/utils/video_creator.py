#!/usr/bin/env python3
"""
Video Creator Module - Combines image frames into video files
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoCreator:
    """Creates video files from image frames"""

    def __init__(self, fps: int = 8, codec: str = "mp4v"):
        """
        Initialize VideoCreator.

        Args:
            fps: Frames per second for output video (default: 8)
            codec: Video codec to use (default: "mp4v" for MP4)
        """
        self.fps = fps
        self.codec = codec

    def create_video_from_frames(
        self,
        frames: List[Union[Image.Image, str, Path]],
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
        frame_size: Optional[tuple] = None
    ) -> str:
        """
        Create a video file from a list of image frames.

        Args:
            frames: List of PIL Images or paths to image files
            output_path: Path for output video file (defaults to temp file)
            fps: Override default FPS (optional)
            frame_size: Resize all frames to (width, height). If None, uses first frame size

        Returns:
            Path to created video file

        Raises:
            ValueError: If frames list is empty or frames have inconsistent sizes
        """
        if not frames:
            raise ValueError("Frames list cannot be empty")

        # Use custom FPS if provided
        fps = fps or self.fps

        # Create temporary file if output path not specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = temp_file.name
            temp_file.close()
            logger.info(f"Using temporary video file: {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Load first frame to determine video dimensions
        first_frame = self._load_frame(frames[0])
        if frame_size is None:
            frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)

        logger.info(f"Creating video: {output_path} ({frame_size[0]}x{frame_size[1]} @ {fps} fps)")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )

        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        try:
            # Write all frames to video
            for idx, frame_input in enumerate(frames):
                frame = self._load_frame(frame_input)

                # Resize frame if needed
                if (frame.shape[1], frame.shape[0]) != frame_size:
                    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LANCZOS4)

                video_writer.write(frame)

            logger.info(f"Successfully created video with {len(frames)} frames: {output_path}")
            return output_path

        finally:
            video_writer.release()

    def _load_frame(self, frame_input: Union[Image.Image, str, Path]) -> np.ndarray:
        """
        Load a frame from various input types.

        Args:
            frame_input: PIL Image, file path string, or Path object

        Returns:
            Frame as numpy array in BGR format (OpenCV format)
        """
        if isinstance(frame_input, Image.Image):
            # Convert PIL Image to numpy array (RGB -> BGR)
            frame_rgb = np.array(frame_input)
            if len(frame_rgb.shape) == 2:  # Grayscale
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
            elif frame_rgb.shape[2] == 4:  # RGBA
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return frame_bgr

        elif isinstance(frame_input, (str, Path)):
            # Load from file
            frame_path = str(frame_input)
            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"Frame file not found: {frame_path}")
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load frame from: {frame_path}")
            return frame

        else:
            raise TypeError(f"Unsupported frame input type: {type(frame_input)}")

    def create_video_from_directory(
        self,
        frames_dir: Union[str, Path],
        output_path: Optional[str] = None,
        pattern: str = "*.png",
        fps: Optional[int] = None
    ) -> str:
        """
        Create a video from all frames in a directory.

        Args:
            frames_dir: Directory containing frame images
            output_path: Path for output video file (defaults to temp file)
            pattern: Glob pattern for matching frame files (default: "*.png")
            fps: Override default FPS (optional)

        Returns:
            Path to created video file
        """
        frames_dir = Path(frames_dir)

        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Get all frame files matching pattern, sorted by name
        frame_files = sorted(frames_dir.glob(pattern))

        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir} matching pattern '{pattern}'")

        logger.info(f"Found {len(frame_files)} frames in {frames_dir}")

        return self.create_video_from_frames(
            frames=frame_files,
            output_path=output_path,
            fps=fps
        )
