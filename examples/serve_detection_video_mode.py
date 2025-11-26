#!/usr/bin/env python3
"""
Example: Serve Detection with Video Mode and S3 Upload

This example demonstrates how to use the video mode feature where frames are
combined into videos and optionally uploaded to S3 before being sent to the VLM.

Features:
- Extract frames from video segments
- Combine frames into short video clips
- Upload videos to S3 (optional)
- Use video URLs in VLM messages instead of individual frames
- Reduce token usage and improve inference efficiency
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from badminton_analyzer import BadmintonAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_video_mode_local():
    """
    Example 1: Video mode with local video files (no S3)

    This mode:
    - Extracts frames from each segment
    - Creates a local video file from the frames
    - Uses the local video file path in messages
    """
    print("\n" + "="*80)
    print("Example 1: Video Mode with Local Files")
    print("="*80 + "\n")

    # Create analyzer with video mode enabled in config
    # Config should have: video_frames.mode = "video"
    analyzer = BadmintonAnalyzer(
        config_path="src/config/config.yaml",
        debug=True
    )

    # Run serve detection
    video_path = "path/to/your/badminton_video.mp4"

    if os.path.exists(video_path):
        result = analyzer.detect_serves(
            video_path=video_path,
            workers=1
        )

        print("\n" + "-"*80)
        print("Results:")
        print("-"*80)
        for idx, serve in enumerate(result.get("serves", []), 1):
            print(f"\nServe #{idx}:")
            print(f"  Time: {serve['start_time']:.1f}s - {serve['end_time']:.1f}s")
            print(f"  Mode: {serve.get('mode', 'unknown')}")
            if serve.get('video_url'):
                print(f"  Video: {serve['video_url']}")
            print(f"  Response: {serve.get('serve_response', 'N/A')}")
    else:
        print(f"Video not found: {video_path}")
        print("Please update the video_path variable with a valid video file")


def example_video_mode_with_s3():
    """
    Example 2: Video mode with S3 upload

    This mode:
    - Extracts frames from each segment
    - Creates a local video file from the frames
    - Uploads the video to S3
    - Uses the S3 presigned URL in messages
    - Optionally cleans up local video files

    Prerequisites:
    - AWS credentials configured (via environment variables or AWS CLI)
    - S3 bucket created
    - Config file updated with S3 settings
    """
    print("\n" + "="*80)
    print("Example 2: Video Mode with S3 Upload")
    print("="*80 + "\n")

    # Check if AWS credentials are available
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("WARNING: AWS credentials not found in environment variables")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("\nSkipping this example...")
        return

    if not os.getenv("S3_BUCKET_NAME"):
        print("WARNING: S3_BUCKET_NAME not set in environment variables")
        print("Please set S3_BUCKET_NAME to your S3 bucket name")
        print("\nSkipping this example...")
        return

    # Create analyzer with S3 and video mode enabled in config
    # Config should have:
    #   - video_frames.mode = "video"
    #   - s3.enabled = true
    #   - s3.bucket_name = "your-bucket-name" (or set S3_BUCKET_NAME env var)
    analyzer = BadmintonAnalyzer(
        config_path="src/config/config.yaml",
        debug=True
    )

    # Run serve detection with S3 upload
    video_path = "path/to/your/badminton_video.mp4"

    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        print(f"S3 Bucket: {os.getenv('S3_BUCKET_NAME')}")
        print(f"AWS Region: {os.getenv('AWS_REGION', 'us-west-2')}\n")

        result = analyzer.detect_serves(
            video_path=video_path,
            workers=1
        )

        print("\n" + "-"*80)
        print("Results:")
        print("-"*80)
        for idx, serve in enumerate(result.get("serves", []), 1):
            print(f"\nServe #{idx}:")
            print(f"  Time: {serve['start_time']:.1f}s - {serve['end_time']:.1f}s")
            print(f"  Mode: {serve.get('mode', 'unknown')}")
            if serve.get('video_url'):
                print(f"  S3 URL: {serve['video_url'][:80]}...")
            print(f"  Response: {serve.get('serve_response', 'N/A')}")
    else:
        print(f"Video not found: {video_path}")
        print("Please update the video_path variable with a valid video file")


def example_configuration_guide():
    """
    Example 3: Configuration guide for video mode and S3
    """
    print("\n" + "="*80)
    print("Example 3: Configuration Guide")
    print("="*80 + "\n")

    config_example = """
# config.yaml configuration for video mode with S3

# Enable video mode (combine frames into videos)
video_frames:
  mode: "video"  # Options: "frame" or "video"
  video_fps: 8   # FPS for created videos
  video_temp_dir: "temp_videos"  # Directory for temporary videos
  max_frames: 8
  frame_size: [1536, 1536]

# S3 configuration (optional)
s3:
  enabled: true  # Set to false to disable S3 upload
  bucket_name: "your-bucket-name"  # Or set S3_BUCKET_NAME env var
  region: "us-west-2"  # Or set AWS_REGION env var
  prefix: "videos/"  # Prefix for S3 object keys
  auto_upload: true  # Not currently used
  cleanup_local: false  # Delete local video after upload
  expiration: 3600  # Presigned URL expiration (seconds)

# Model configuration
active_model: "qwen3-vl-32b-local"  # Or any other supported model
"""

    print("Configuration Example:")
    print(config_example)

    print("\n" + "-"*80)
    print("Environment Variables:")
    print("-"*80)
    print("""
Required for S3:
  - AWS_ACCESS_KEY_ID: Your AWS access key
  - AWS_SECRET_ACCESS_KEY: Your AWS secret key
  - AWS_REGION (optional): AWS region (default: us-west-2)
  - S3_BUCKET_NAME (optional): S3 bucket name (can also set in config.yaml)

Required for VLM:
  - DASHSCOPE_API_KEY: For Qwen models via DashScope
  - OPENAI_API_KEY: For OpenAI-compatible APIs
""")

    print("\n" + "-"*80)
    print("Benefits of Video Mode:")
    print("-"*80)
    print("""
1. Reduced token usage: 1 video URL vs 8 image tokens
2. Better temporal context: Models can see motion between frames
3. Faster inference: Less data to process
4. Cost savings: Fewer tokens = lower API costs
5. Presigned URLs: Secure, time-limited access to videos
""")

    print("\n" + "-"*80)
    print("When to Use Video Mode:")
    print("-"*80)
    print("""
Use video mode when:
  ✓ Your VLM supports video input (e.g., Qwen-VL models)
  ✓ You need to analyze temporal sequences
  ✓ You want to reduce token costs
  ✓ You have S3 or similar storage available

Use frame mode when:
  ✓ Your model only supports images
  ✓ You need to analyze individual frames independently
  ✓ You don't have external storage available
""")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("Serve Detection with Video Mode and S3 - Examples")
    print("="*80)

    # Example 1: Local video mode
    example_video_mode_local()

    # Example 2: Video mode with S3
    example_video_mode_with_s3()

    # Example 3: Configuration guide
    example_configuration_guide()

    print("\n" + "="*80)
    print("Examples Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
