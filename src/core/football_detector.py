#!/usr/bin/env python3
"""
Football Action Detection Module - Specialized for football action detection and analysis
"""

import logging
import json
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import os
import uuid
import base64
import yaml
from dashscope import MultiModalConversation

from core.few_shot_manager import MessageManager
from utils.response_parser import ResponseParser
from utils.video_processor import extract_frames_from_video
from utils.video_creator import VideoCreator
from utils.s3_uploader import S3VideoUploader

logger = logging.getLogger(__name__)

class FootballDetector:
    """Football Action Detector"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Football Detector.

        Args:
            config_path: Path to YAML configuration file. If not provided, uses default config file
                        in src/config/config.yaml
        """
        # Load configuration from YAML file
        self.config = self._load_config(config_path)

        # Load football detection prompts from config file
        self.football_questions = self.config.get("prompts", {}).get("football", {})

        # Initialize message manager with examples directory from config
        self.message_manager = MessageManager(
            self.config.get("few_shot", {}).get("examples_dir", "few_shot_examples")
        )

        # Initialize video creator if video mode is enabled
        video_config = self.config.get("video_frames", {})
        self.video_mode = video_config.get("mode", "frame") == "video"
        if self.video_mode:
            self.video_creator = VideoCreator(
                fps=video_config.get("video_fps", 8),
                codec="mp4v"
            )
            logger.info("Video mode enabled: will create videos from frames")
        else:
            self.video_creator = None
            logger.info("Frame mode enabled: will send individual frames")

        # Initialize S3 uploader if S3 is enabled
        s3_config = self.config.get("s3", {})
        if s3_config.get("enabled", False):
            try:
                self.s3_uploader = S3VideoUploader(
                    bucket_name=s3_config.get("bucket_name") or os.getenv("S3_BUCKET_NAME"),
                    aws_region=s3_config.get("region") or os.getenv("AWS_REGION", "us-west-2"),
                    s3_prefix=s3_config.get("prefix", "videos/")
                )
                self.s3_enabled = True
                logger.info(f"S3 uploader initialized for bucket: {self.s3_uploader.bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize S3 uploader: {e}. S3 features disabled.")
                self.s3_uploader = None
                self.s3_enabled = False
        else:
            self.s3_uploader = None
            self.s3_enabled = False
            logger.info("S3 upload disabled")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configuration dictionary
        """
        # Default config path
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "config",
                "config.yaml"
            )

        # Try to load from YAML file
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from: {config_path}")

                # Get active model and resolve API endpoint
                active_model_key = yaml_config.get("active_model", "qwen-vl-max")
                models = yaml_config.get("models", {})

                if active_model_key not in models:
                    logger.warning(f"Active model '{active_model_key}' not found in models config")
                    logger.info("Using default configuration")
                    return self._get_default_config()

                # Get model configuration
                model_config = models[active_model_key]
                model_name = model_config.get("name")

                # Get API endpoint configuration
                api_type = model_config.get("api_type", "openai")
                api_base_url = model_config.get("base_url")
                api_key_env = model_config.get("api_key_env", "OPENAI_API_KEY")

                # Start with the full YAML config
                config = yaml_config.copy()

                # Add computed model and API fields
                config.update({
                    "active_model_key": active_model_key,
                    "model_name": model_name,
                    "use_dashscope_sdk": (api_type == "dashscope"),
                    "api_base_url": api_base_url,
                    "api_key": os.getenv(api_key_env),
                    "api_type": api_type,
                })

                logger.info(f"Active model: {active_model_key} ({model_name})")
                return config

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
                logger.info("Using default configuration")

        else:
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Using default configuration")

        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration as fallback.

        Returns:
            Default configuration dictionary
        """
        return {
            # Model and API configuration
            "active_model_key": "qwen-vl-max",
            "model_name": "qwen-vl-max-latest",
            "use_dashscope_sdk": True,
            "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "api_type": "dashscope",

            # Video frames configuration
            "video_frames": {
                "max_frames": 8,
                "frame_size": [1024, 1024],
                "mode": "frame",
                "video_fps": 8,
                "video_temp_dir": "temp_videos",
            },

            # Inference parameters
            "inference": {
                "batch_size": 4,
                "max_new_tokens": 50,
                "temperature": 0.1,
                "top_p": 0.8,
                "do_sample": False,
            },

            # Few-shot configuration
            "few_shot": {
                "enabled": False,
                "examples_dir": "few_shot_examples",
                "example_ids": [],
                "example_frames_per_video": 8,
                "save_message_images": False,
                "saved_images_dir": "debug_message_images",
            },

            # Video segmentation configuration
            "video_segmentation": {
                "segment_duration": 3.0,
                "overlap_duration": 1.0,
                "max_segments": None,
            },

            # Prompts and system prompts
            "prompts": {},
            "system_prompts": {},

            # Annotations path
            "annotations_path": "data/annotations_exp.json",
        }

    def detect_all_actions(self,
                          video_path: str,
                          action_type: str = "goal",
                          segment_duration: float = None,
                          overlap_duration: float = None,
                          max_segments: int = None,
                          workers: int = None) -> List[Dict[str, Any]]:
        """
        检测视频中所有的足球动作

        Args:
            video_path: 视频文件路径
            action_type: 动作类型 (goal, pass, tackle, shot, etc.)
            segment_duration: 每段时长（秒），None时使用config中的值
            overlap_duration: 重叠时长（秒），None时使用config中的值
            max_segments: 最大处理片段数（用于测试），None时使用config中的值
            workers: 并行处理的最大工作线程数

        Returns:
            包含动作检测结果的列表
        """
        # Use config values if not provided
        if segment_duration is None:
            segment_duration = self.config.get("video_segmentation", {}).get("segment_duration", 3.0)
        if overlap_duration is None:
            overlap_duration = self.config.get("video_segmentation", {}).get("overlap_duration", 1.0)
        if max_segments is None:
            max_segments = self.config.get("video_segmentation", {}).get("max_segments", None)

        # 获取视频信息
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # 生成时间片段
        segments = []
        current_start = 0.0
        segment_index = 0

        while current_start < total_duration:
            current_end = min(current_start + segment_duration, total_duration)

            segments.append({
                "segment_index": segment_index,
                "start_time": current_start,
                "end_time": current_end,
                "duration": current_end - current_start
            })

            segment_index += 1
            current_start += (segment_duration - overlap_duration)

            if overlap_duration >= segment_duration:
                current_start += 0.1

            if max_segments and len(segments) >= max_segments:
                break

        logger.info(f"生成了 {len(segments)} 个动作检测片段")

        # 批量检测动作
        action_segments = []
        batch_size = min(self.config.get("inference", {}).get("batch_size", 4), len(segments))
        segment_batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(self._detect_actions_in_batch, batch, video_path, action_type): batch
                for batch in segment_batches
            }

            for future in tqdm(as_completed(future_to_batch),
                             desc=f"检测{action_type}动作",
                             total=len(segment_batches)):
                try:
                    batch_results = future.result()
                    for segment_result in batch_results:
                        if segment_result.get("has_action", False):
                            action_segments.append(segment_result)
                            logger.info(f"检测到{action_type}: {segment_result.get('start_time', 0):.1f}s - {segment_result.get('end_time', 0):.1f}s")

                except Exception as e:
                    logger.error(f"动作检测批处理失败: {str(e)}")
                    raise e

        return action_segments

    def _detect_actions_in_batch(self, segments_batch: List[Dict[str, Any]], video_path: str, action_type: str) -> List[Dict[str, Any]]:
        """批量检测足球动作"""
        try:
            # Generate unique request ID for this batch
            request_id = str(uuid.uuid4())

            # 批量提取帧并处理（frame模式或video模式）
            frame_batches = []
            video_url_batches = []
            valid_segments = []

            # Create temporary directory for videos if in video mode
            video_temp_dir = None
            if self.video_mode:
                video_config = self.config.get("video_frames", {})
                video_temp_dir = video_config.get("video_temp_dir", "temp_videos")
                os.makedirs(video_temp_dir, exist_ok=True)

            for segment_idx, segment in enumerate(segments_batch):
                # Extract frames from video file
                frames_with_timestamps = extract_frames_from_video(
                    video_path,
                    num_frames=self.config.get("video_frames", {}).get("max_frames", 8),
                    start_time=segment["start_time"],
                    duration=segment["duration"],
                    frame_size=self.config.get("video_frames", {}).get("frame_size", [1024, 1024])
                )

                if not frames_with_timestamps:
                    continue

                # Unpack frames and timestamps
                frames = [frame for frame, _ in frames_with_timestamps]
                timestamps = [timestamp for _, timestamp in frames_with_timestamps]

                # Save message images if configured
                if self.config.get("few_shot", {}).get("save_message_images", False):
                    saved_images_dir = self.config.get("few_shot", {}).get("saved_images_dir", "debug_message_images")
                    saved_images_sub_dir = os.path.join(
                        saved_images_dir,
                        request_id,
                        f"segment_{segment_idx:03d}_{segment['start_time']:.1f}s-{segment['start_time'] + segment['duration']:.1f}s"
                    )
                    os.makedirs(saved_images_sub_dir, exist_ok=True)
                    for i, frame in enumerate(frames):
                        frame.save(os.path.join(saved_images_sub_dir, f"frame_{i:03d}.png"))

                # If video mode is enabled, create video from frames
                if self.video_mode and self.video_creator:
                    try:
                        # Create video from frames
                        video_filename = f"segment_{segment_idx:03d}_{request_id[:8]}_{segment['start_time']:.1f}s.mp4"
                        local_video_path = os.path.join(video_temp_dir, video_filename)

                        self.video_creator.create_video_from_frames(
                            frames=frames,
                            output_path=local_video_path,
                            frame_size=tuple(self.config.get("video_frames", {}).get("frame_size", [1024, 1024]))
                        )
                        logger.info(f"Created video for segment {segment_idx}: {local_video_path}")

                        # Upload to S3 if enabled
                        video_url = None
                        if self.s3_enabled and self.s3_uploader:
                            try:
                                s3_config = self.config.get("s3", {})
                                expiration = s3_config.get("expiration", 3600)
                                cleanup_local = s3_config.get("cleanup_local", False)

                                video_url = self.s3_uploader.upload_video(
                                    local_path=local_video_path,
                                    expiration=expiration
                                )
                                logger.info(f"Uploaded video to S3: {video_url}")

                                # Clean up local video if configured
                                if cleanup_local:
                                    try:
                                        os.remove(local_video_path)
                                        logger.info(f"Deleted local video: {local_video_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to delete local video: {e}")
                            except Exception as e:
                                logger.error(f"Failed to upload video to S3: {e}")
                                logger.info("Falling back to local video path")
                                video_url = local_video_path
                        else:
                            video_url = local_video_path

                        # Store video URL
                        video_url_batches.append(video_url)
                        frame_batches.append(None)  # Placeholder
                        valid_segments.append(segment)

                    except Exception as e:
                        logger.error(f"Failed to create video for segment {segment_idx}: {e}")
                        logger.info("Falling back to frame mode for this segment")
                        frame_batches.append(frames_with_timestamps)
                        video_url_batches.append(None)
                        valid_segments.append(segment)
                else:
                    # Frame mode: store frames directly
                    frame_batches.append(frames_with_timestamps)
                    video_url_batches.append(None)
                    valid_segments.append(segment)

            if not valid_segments:
                return [{
                    **segment,
                    "has_action": False,
                    "response": "无法提取帧",
                    "success": False
                } for segment in segments_batch]

            # Get appropriate question based on action type
            query_text = self.football_questions.get(f"has_{action_type}",
                                                     self.football_questions.get("has_action", ""))
            system_prompt = self.config.get("system_prompts", {}).get("football_detection")

            # Build messages based on few-shot configuration and mode (frame or video)
            messages_list = []
            if self.config.get("few_shot", {}).get("enabled", False):
                # Get example IDs from config
                example_ids = self.config.get("few_shot", {}).get("example_ids", [])

                # If example_ids is empty or None, load all available examples from annotations
                if not example_ids:
                    logger.info("No example_ids specified, loading all available examples from annotations")
                    try:
                        annotations_path = self.config.get("annotations_path", "data/annotations_exp.json")
                        with open(annotations_path, 'r', encoding='utf-8') as f:
                            annotations_data = json.load(f)
                            examples = annotations_data.get('examples', [])
                            example_ids = [ex['id'] for ex in examples]
                            logger.info(f"Found {len(example_ids)} examples in annotations: {example_ids}")
                    except Exception as e:
                        logger.warning(f"Failed to load all examples from annotations: {e}")
                        logger.warning("Falling back to empty example list")
                        example_ids = []

                logger.info(f"Using few-shot examples: {example_ids}")

                # Get annotations path from config for MessageManager
                annotations_path = self.config.get("annotations_path", "data/annotations_exp.json")

                for idx in range(len(valid_segments)):
                    video_url = video_url_batches[idx]
                    frames_with_timestamps = frame_batches[idx]

                    # Check if we're using video URL mode or frame mode
                    if video_url is not None:
                        # Video URL mode - use MessageManager with video URL support
                        messages = self.message_manager.create_messages(
                            video_url=video_url,
                            text=query_text,
                            system_prompt=system_prompt,
                            example_ids=example_ids,
                            annotations_path=annotations_path
                        )
                    else:
                        # Frame mode - use MessageManager with frames
                        messages = self.message_manager.create_messages(
                            frames_with_timestamps=frames_with_timestamps,
                            text=query_text,
                            system_prompt=system_prompt,
                            example_ids=example_ids,
                            annotations_path=annotations_path
                        )
                    messages_list.append(messages)
            else:
                # Zero-shot mode: no examples
                for idx in range(len(valid_segments)):
                    video_url = video_url_batches[idx]
                    frames_with_timestamps = frame_batches[idx]

                    # Check if we're using video URL mode or frame mode
                    if video_url is not None:
                        # Video URL mode - use MessageManager with video URL support
                        messages = self.message_manager.create_messages(
                            video_url=video_url,
                            text=query_text,
                            system_prompt=system_prompt
                        )
                    else:
                        # Frame mode - use MessageManager with frames
                        messages = self.message_manager.create_messages(
                            frames_with_timestamps=frames_with_timestamps,
                            text=query_text,
                            system_prompt=system_prompt
                        )
                    messages_list.append(messages)

            mode = "few-shot" if self.config.get("few_shot", {}).get("enabled", False) else "zero-shot"
            logger.info(f"Using {mode} mode to analyze {len(messages_list)} segments")
            batch_analyses = self._generate_batch(messages_list, request_id=request_id)

            # 处理结果
            batch_results = []
            for i, (segment, analysis) in enumerate(zip(valid_segments, batch_analyses)):
                # Parse action detection based on action_type
                has_action = ResponseParser.contains_football_action(analysis, action_type)

                # Determine the mode used and add appropriate metadata
                video_url = video_url_batches[i]
                frames_with_timestamps = frame_batches[i]

                segment_result = {
                    **segment,
                    "action_type": action_type,
                    "has_action": has_action,
                    "action_response": analysis,
                    "success": True,
                }

                # Add mode-specific metadata
                if video_url is not None:
                    segment_result["mode"] = "video"
                    segment_result["video_url"] = video_url
                else:
                    segment_result["mode"] = "frame"
                    segment_result["num_frames"] = len(frames_with_timestamps) if frames_with_timestamps else 0

                batch_results.append(segment_result)

            # 处理无法提取帧的片段
            for segment in segments_batch:
                if segment not in valid_segments:
                    batch_results.append({
                        **segment,
                        "action_type": action_type,
                        "has_action": False,
                        "action_response": "无法提取帧",
                        "success": False
                    })

            return batch_results

        except Exception as e:
            logger.error(f"批量动作检测失败: {str(e)}")
            raise e

    def _generate_batch(self, messages_batch: List[List[Dict]], request_id: str = None, debug_mode: bool = False) -> List[str]:
        """
        批量生成响应（统一支持单轮和多轮对话）

        Args:
            messages_batch: 消息批次
            request_id: 请求ID，用于组织保存的文件
            debug_mode: 是否保存调试信息

        Returns:
            生成的响应列表
        """
        # Generate request_id if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())

        # 如果使用DashScope SDK，调用专用方法
        if self.config["use_dashscope_sdk"]:
            logger.info("使用DashScope原生SDK进行推理")
            return self._generate_with_dashscope(messages_batch, request_id, debug_mode)
        else:
            logger.info("使用OpenAI API进行推理")
            return self._generate_with_openai_api(messages_batch, request_id, debug_mode)

    def _generate_with_dashscope(self, messages_batch: List[List[Dict]], request_id: str, debug_mode: bool = False) -> List[str]:
        """
        使用DashScope原生SDK进行批量生成

        Args:
            messages_batch: 消息批次
            request_id: 请求ID，用于组织保存的文件
            debug_mode: 是否保存调试信息
        """
        try:
            responses = []
            for messages in messages_batch:
                response = MultiModalConversation.call(
                    api_key=os.getenv('DASHSCOPE_API_KEY'),
                    model='qwen3-vl-flash',
                    messages=messages)
                responses.append(response.output.choices[0].message.content[0]['text'].strip())
                logger.info(f"DashScope原生SDK批量生成响应: {response}")
            return responses
        except Exception as e:
            logger.error(f"使用DashScope原生SDK批量生成失败: {str(e)}")
            raise e

    def _encode_image(self, image: str) -> str:
        """文件路径图像编码为base64"""
        with open(image.split("file://")[1], "rb") as f:
            img_str = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{img_str}"

    def _generate_with_openai_api(self, messages_batch: List[List[Dict]], request_id: str, debug_mode: bool = False) -> List[str]:
        """
        使用OpenAI API进行批量生成

        Args:
            messages_batch: 消息批次
            request_id: 请求ID，用于组织保存的文件
            debug_mode: 是否保存调试信息
        """
        try:
            # Create request-specific debug directory if needed
            debug_dir = None
            if debug_mode:
                debug_dir = os.path.join("selected_frames", request_id)
                os.makedirs(debug_dir, exist_ok=True)

            responses = []

            for batch_index, conversation in enumerate(messages_batch):
                if not conversation:
                    responses.append("输入为空")
                    continue

                transformed_conversation = []
                image_index = 0

                for msg in conversation:
                    transformed_msg = {"role": msg["role"], "content": []}
                    frame_list = []
                    video_url = None
                    for item in msg.get("content", []):
                        if item["type"] == "text":
                            transformed_msg["content"].append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item["type"] == "image":
                            image_local_path = item["image"].split("file://")[1]
                            image_url = self.s3_uploader.upload_video(image_local_path, expiration=3600)
                            frame_list.append({"image": image_url, "timestamp": item["timestamp"]})
                            image_index += 1
                            if debug_mode:
                                debug_sub_dir = os.path.join(debug_dir, f"batch_{batch_index:03d}")
                                os.makedirs(debug_sub_dir, exist_ok=True)
                                item["image"].save(os.path.join(debug_sub_dir, f"image_{image_index:03d}.png"))
                        elif item["type"] == "video":
                            video_url = item["video"]
                    if frame_list:
                        for frame in frame_list:
                            transformed_msg["content"].append({
                                "type": "text",
                                "text": f"<{frame['timestamp']:.1f} seconds>"
                            })
                            transformed_msg["content"].append({
                                "type": "image_url",
                                "image_url": {"url": frame["image"]}
                            })
                    if video_url:
                        transformed_msg["content"].append({
                            "type": "video_url",
                            "video_url": {"url": video_url}
                        })
                    transformed_conversation.append(transformed_msg)

                # 调用OpenAI API
                payload = {
                    "model": self.config["model_name"],
                    "messages": transformed_conversation,
                    "max_tokens": self.config.get("inference", {}).get("max_new_tokens", 50),
                    "temperature": self.config.get("inference", {}).get("temperature", 0.1),
                    "top_p": self.config.get("inference", {}).get("top_p", 0.8),
                    "max_completion_tokens": 16000,
                }

                try:
                    response = requests.post(
                        f"{self.config['api_base_url']}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config['api_key']}",
                            "Content-Type": "application/json"
                        },
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get('choices') and len(result['choices']) > 0:
                            text = result['choices'][0]['message']['content'].strip()
                            responses.append(text)
                            logger.info(f"request_id: {request_id}, batch_index: {batch_index}, text: {text}")
                            logger.info(f"openai api response: {result}")
                        else:
                            responses.append("生成失败：无响应内容")
                    else:
                        logger.error(f"API调用失败: {response.status_code} - {response.text}")
                        responses.append(f"API调用失败: {response.status_code}")

                except requests.exceptions.Timeout:
                    logger.error("API调用超时")
                    responses.append("请求超时")
                except Exception as api_e:
                    logger.error(f"API调用异常: {str(api_e)}")
                    responses.append(f"API调用异常: {str(api_e)}")

            return responses

        except Exception as e:
            logger.error(f"批量生成失败: {str(e)}")
            raise e
