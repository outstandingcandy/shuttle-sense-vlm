#!/usr/bin/env python3
"""
Serve Detection Module - Specialized for badminton serve action detection and analysis
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

class ServeDetector:
    """Serve Detector"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Serve Detector.

        Args:
            config_path: Path to YAML configuration file. If not provided, uses default config file
                        in src/config/config.yaml
        """
        # Load configuration from YAML file
        self.config = self._load_config(config_path)

        # Load serve detection prompts from config file
        self.serve_questions = self.config.get("prompts", {}).get("serve", {})

        # Initialize message manager with examples directory from config
        self.message_manager = MessageManager(self.config.get("few_shot_examples_dir", "few_shot_examples"))

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

                # Build unified configuration
                config = {
                    # Video frame extraction
                    "max_frames": yaml_config.get("video_frames", {}).get("max_frames", 8),
                    "frame_size": yaml_config.get("video_frames", {}).get("frame_size", [1024, 1024]),

                    # Inference parameters
                    "batch_size": yaml_config.get("inference", {}).get("batch_size", 4),
                    "max_new_tokens": yaml_config.get("inference", {}).get("max_new_tokens", 50),
                    "temperature": yaml_config.get("inference", {}).get("temperature", 0.1),
                    "top_p": yaml_config.get("inference", {}).get("top_p", 0.8),
                    "do_sample": yaml_config.get("inference", {}).get("do_sample", False),

                    # Model and API (unified from new structure)
                    "active_model_key": active_model_key,
                    "model_name": model_name,
                    "use_dashscope_sdk": (api_type == "dashscope"),
                    "api_base_url": api_base_url,
                    "api_key": os.getenv(api_key_env),
                    "api_type": api_type,

                    # Backward compatibility fields
                    "dashscope_api_base": api_base_url if api_type == "dashscope" else "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "openai_api_base": api_base_url if api_type == "openai" else "http://localhost:8000/v1",
                    "dashscope_api_key": os.getenv(api_key_env) if api_type == "dashscope" else os.getenv("DASHSCOPE_API_KEY"),

                    # Few-shot configuration
                    "enable_few_shot": yaml_config.get("few_shot", {}).get("enabled", False),
                    "few_shot_examples_dir": yaml_config.get("few_shot", {}).get("examples_dir", "few_shot_examples"),
                    "few_shot_example_ids": yaml_config.get("few_shot", {}).get("example_ids", []),
                    "few_shot_example_frames_per_video": yaml_config.get("few_shot", {}).get("example_frames_per_video", 8),
                    "save_message_images": yaml_config.get("few_shot", {}).get("save_message_images", False),
                    "saved_images_dir": yaml_config.get("few_shot", {}).get("saved_images_dir", "debug_message_images"),

                    # Video segmentation configuration
                    "segment_duration": yaml_config.get("video_segmentation", {}).get("segment_duration", 3.0),
                    "overlap_duration": yaml_config.get("video_segmentation", {}).get("overlap_duration", 1.0),
                    "max_segments": yaml_config.get("video_segmentation", {}).get("max_segments", None),

                    # Prompts configuration
                    "prompts": yaml_config.get("prompts", {}),
                    "system_prompts": yaml_config.get("system_prompts", {}),

                    # Annotations path
                    "annotations_path": yaml_config.get("annotations_path", "data/annotations_exp.json"),
                }

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
            "max_frames": 8,
            "frame_size": [1024, 1024],
            "batch_size": 4,
            "max_new_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.8,
            "do_sample": False,
            "active_model_key": "qwen-vl-max",
            "model_name": "qwen-vl-max-latest",
            "use_dashscope_sdk": True,
            "api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("DASHSCOPE_API_KEY"),
            "api_type": "dashscope",
            "dashscope_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dashscope_api_key": os.getenv("DASHSCOPE_API_KEY"),
            "openai_api_base": "http://localhost:8000/v1",
            "enable_few_shot": False,
            "segment_duration": 3.0,
            "overlap_duration": 1.0,
            "max_segments": None,
        }

    def detect_all_serves(self,
                         video_path: str,
                         segment_duration: float = None,
                         overlap_duration: float = None,
                         max_segments: int = None,
                         workers: int = None) -> List[Dict[str, Any]]:
        """
        检测视频中所有的发球时刻

        Args:
            video_path: 视频文件路径
            segment_duration: 每段时长（秒），None时使用config中的值
            overlap_duration: 重叠时长（秒），None时使用config中的值
            max_segments: 最大处理片段数（用于测试），None时使用config中的值
            workers: 并行处理的最大工作线程数

        Returns:
            包含发球检测结果的列表
        """
        # Use config values if not provided
        if segment_duration is None:
            segment_duration = self.config.get("segment_duration", 3.0)
        if overlap_duration is None:
            overlap_duration = self.config.get("overlap_duration", 1.0)
        if max_segments is None:
            max_segments = self.config.get("max_segments", None)
        
        # 获取视频信息
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # 生成时间片段（用于发球检测）
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
        
        logger.info(f"生成了 {len(segments)} 个发球检测片段")
        
        # 批量检测发球
        serve_segments = []
        batch_size = min(self.config["batch_size"], len(segments))
        segment_batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(self._detect_serves_in_batch, batch, video_path): batch
                for batch in segment_batches
            }
            
            for future in tqdm(as_completed(future_to_batch), 
                             desc="检测发球动作", 
                             total=len(segment_batches)):
                try:
                    batch_results = future.result()
                    for segment_result in batch_results:
                        if segment_result.get("has_serve", False):
                            serve_segments.append(segment_result)
                            logger.info(f"检测到发球: {segment_result.get('start_time', 0):.1f}s - {segment_result.get('end_time', 0):.1f}s")
                        
                except Exception as e:
                    logger.error(f"发球检测批处理失败: {str(e)}")
                    raise e
        
        return serve_segments
    
    def _detect_serves_in_batch(self, segments_batch: List[Dict[str, Any]], video_path: str) -> List[Dict[str, Any]]:
        """批量检测发球动作"""
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
                frames = extract_frames_from_video(
                    video_path,
                    num_frames=self.config["max_frames"],
                    start_time=segment["start_time"],
                    duration=segment["duration"],
                    frame_size=self.config["frame_size"]
                )

                if not frames:
                    continue

                # Save message images if configured
                if self.config.get("save_message_images", False):
                    # Create request-specific directory structure: saved_images_dir/request_id/segment_idx/
                    saved_images_sub_dir = os.path.join(
                        self.config["saved_images_dir"],
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
                            frame_size=tuple(self.config["frame_size"])
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
                        frame_batches.append(frames)
                        video_url_batches.append(None)
                        valid_segments.append(segment)
                else:
                    # Frame mode: store frames directly
                    frame_batches.append(frames)
                    video_url_batches.append(None)
                    valid_segments.append(segment)

            if not valid_segments:
                return [{
                    **segment,
                    "has_serve": False,
                    "response": "无法提取帧",
                    "success": False
                } for segment in segments_batch]
            query_text = self.serve_questions["has_serve"]
            system_prompt = self.config.get("system_prompts", {}).get("serve_detection")

            # Build messages based on few-shot configuration and mode (frame or video)
            messages_list = []
            if self.config["enable_few_shot"]:
                # Get example IDs from config
                example_ids = self.config.get("few_shot_example_ids", [])

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
                    frames = frame_batches[idx]

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
                            frames=frames,
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
                    frames = frame_batches[idx]

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
                            frames=frames,
                            text=query_text,
                            system_prompt=system_prompt
                        )
                    messages_list.append(messages)

            mode = "few-shot" if self.config["enable_few_shot"] else "zero-shot"
            logger.info(f"Using {mode} mode to analyze {len(messages_list)} segments")
            batch_analyses = self._generate_batch(messages_list, request_id=request_id)
        
            # 处理结果
            batch_results = []
            for i, (segment, analysis) in enumerate(zip(valid_segments, batch_analyses)):
                has_serve = ResponseParser.contains_serve(analysis)

                # Determine the mode used and add appropriate metadata
                video_url = video_url_batches[i]
                frames = frame_batches[i]

                segment_result = {
                    **segment,
                    "has_serve": has_serve,
                    "serve_response": analysis,
                    "success": True,
                }

                # Add mode-specific metadata
                if video_url is not None:
                    segment_result["mode"] = "video"
                    segment_result["video_url"] = video_url
                else:
                    segment_result["mode"] = "frame"
                    segment_result["num_frames"] = len(frames) if frames else 0

                batch_results.append(segment_result)
            
            # 处理无法提取帧的片段
            for segment in segments_batch:
                if segment not in valid_segments:
                    batch_results.append({
                        **segment,
                        "has_serve": False,
                        "serve_response": "无法提取帧",
                        "success": False
                    })
            
            return batch_results

        except Exception as e:
            logger.error(f"批量发球检测失败: {str(e)}")
            raise e

    def _create_messages_with_video_url(
        self,
        video_url: str,
        text: str,
        system_prompt: Optional[str] = None,
        segment: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create messages with video URL for DashScope API.

        Args:
            video_url: URL of the video (S3 or other public URL)
            text: Query text
            system_prompt: Optional system prompt
            segment: Optional segment information (for time-based queries)

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })

        # Build user message with video URL
        user_content = [
            {
                "type": "video_url",
                "video_url": {"url": video_url}
            }
        ]

        # Add time range information if segment is provided
        if segment:
            time_info = f" (时间段: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s)"
            user_content.append({
                "type": "text",
                "text": text + time_info
            })
        else:
            user_content.append({
                "type": "text",
                "text": text
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        logger.info(f"Created messages with video URL: {video_url}")
        return messages

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
                    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
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
                    for item in msg.get("content", []):
                        if item["type"] == "text":
                            transformed_msg["content"].append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item["type"] == "image":
                            image_b64 = self._encode_image(item["image"])
                            frame_list.append(image_b64)
                            image_index += 1
                            if debug_mode:
                                # Create subdirectory for each batch: request_id/batch_xxx/
                                debug_sub_dir = os.path.join(debug_dir, f"batch_{batch_index:03d}")
                                os.makedirs(debug_sub_dir, exist_ok=True)
                                item["image"].save(os.path.join(debug_sub_dir, f"image_{image_index:03d}.png"))
                    if frame_list:
                        transformed_msg["content"].append({
                            "type": "video",
                            "video": frame_list
                        })
                    transformed_conversation.append(transformed_msg)

                # 调用OpenAI API
                payload = {
                    "model": self.config["model_name"],
                    "messages": transformed_conversation,
                    "max_tokens": self.config["max_new_tokens"],
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "max_completion_tokens": 16000,
                }

                try:
                    response = requests.post(
                        f"{self.config['openai_api_base']}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
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

