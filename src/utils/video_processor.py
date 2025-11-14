#!/usr/bin/env python3
"""
Video processing utilities
"""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image to target size while preserving aspect ratio by adding padding.

    Args:
        image: Input image as numpy array (H, W, C)
        target_size: Target size as (width, height)
        pad_color: Color for padding in RGB format, default is black (0, 0, 0)

    Returns:
        Resized image with padding as numpy array
    """
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    # Calculate aspect ratios
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    # Calculate scaling factor to fit image within target size
    if original_aspect > target_aspect:
        # Image is wider than target, fit by width
        scale = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale)
    else:
        # Image is taller than target, fit by height
        scale = target_height / original_height
        new_height = target_height
        new_width = int(original_width * scale)

    # Resize image maintaining aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create canvas with target size filled with pad_color
    padded_image = np.full((target_height, target_width, 3), pad_color, dtype=np.uint8)

    # Calculate position to paste resized image (center alignment)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Paste resized image onto canvas
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image


def extract_frames_from_video(
    video_path: str,
    num_frames: int,
    start_time: float = 0,
    duration: Optional[float] = None,
    frame_size: Union[Tuple[int, int], List[int]] = (1024, 1024)
) -> List[Image.Image]:
    """
    Extract specified number of frames from video.

    The frames are resized to the target size while preserving the original aspect ratio.
    Black padding is added as needed to fill the target dimensions.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        start_time: Start time in seconds, default is 0
        duration: Duration in seconds, None means until end of video
        frame_size: Frame size in format (width, height), default is (1024, 1024).
                   Images will be resized to fit within these dimensions while preserving
                   aspect ratio, with black padding added to fill remaining space.

    Returns:
        List of PIL Image objects with preserved aspect ratio and padding

    Raises:
        ValueError: If video cannot be opened or parameters are invalid
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        # Get video basic information
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            logger.error(f"Invalid video FPS: {fps}")
            raise ValueError(f"Invalid video FPS: {fps}")

        # Calculate frame range
        start_frame = int(start_time * fps)
        if duration is not None:
            end_frame = min(start_frame + int(duration * fps), total_frames)
        else:
            end_frame = total_frames

        # Validate frame range
        if start_frame >= total_frames:
            logger.warning(f"Start frame {start_frame} exceeds total frames {total_frames}")
            return []

        if start_frame >= end_frame:
            logger.warning(f"Start frame {start_frame} >= end frame {end_frame}")
            return []

        # Calculate frame indices to extract
        segment_frames = end_frame - start_frame

        if segment_frames <= num_frames:
            # If segment has fewer frames than needed, extract all frames
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Otherwise, uniformly sample frames
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)

        logger.info(f"Extracting {len(frame_indices)} frames from video {video_path} "
                   f"(start frame: {start_frame}, end frame: {end_frame})")

        # Extract frames
        frames = []
        frame_size_tuple = tuple(frame_size) if isinstance(frame_size, list) else frame_size

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame with padding to preserve aspect ratio
                frame = resize_with_padding(frame, frame_size_tuple)
                # Convert to PIL Image
                frames.append(Image.fromarray(frame))
            else:
                logger.warning(f"Cannot read frame {idx}")

        logger.info(f"Successfully extracted {len(frames)} frames")

        return frames

    finally:
        cap.release()


def get_video_info(video_path: str) -> dict:
    """
    Get basic video information.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary containing video information:
        - fps: Frame rate
        - total_frames: Total number of frames
        - duration: Video duration in seconds
        - width: Video width
        - height: Video height

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height
        }
    finally:
        cap.release()
