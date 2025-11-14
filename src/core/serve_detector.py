#!/usr/bin/env python3
"""
Serve Detection Module - Specialized for badminton serve action detection and analysis
"""

import logging
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
from config.prompts_config import SERVE_PROMPTS, FEW_SHOT_CONFIG, FEW_SHOT_EXAMPLES, FEW_SHOT_SYSTEM_PROMPTS, FEW_SHOT_RESPONSES

logger = logging.getLogger(__name__)

class ServeDetector:
    """Serve Detector"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Serve Detector.

        Args:
            config_path: Path to YAML configuration file. If not provided, uses default config file
                        in src/config/serve_detector_config.yaml
        """
        # Load serve detection prompts from config file
        self.serve_questions = SERVE_PROMPTS

        # Load configuration from YAML file
        self.config = self._load_config(config_path)

        self.message_manager = MessageManager(FEW_SHOT_CONFIG["examples_dir"])

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
                "serve_detector_config.yaml"
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
                api_endpoints = yaml_config.get("api_endpoints", {})

                if active_model_key not in models:
                    logger.warning(f"Active model '{active_model_key}' not found in models config")
                    logger.info("Using default configuration")
                    return self._get_default_config()

                # Get model configuration
                model_config = models[active_model_key]
                model_name = model_config.get("name")
                api_key = model_config.get("api")

                if api_key not in api_endpoints:
                    logger.warning(f"API endpoint '{api_key}' not found for model '{active_model_key}'")
                    logger.info("Using default configuration")
                    return self._get_default_config()

                # Get API endpoint configuration
                api_config = api_endpoints[api_key]
                api_type = api_config.get("type", "openai")
                api_base_url = api_config.get("base_url")
                api_key_env = api_config.get("api_key_env", "OPENAI_API_KEY")

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
                    "enable_few_shot": yaml_config.get("few_shot", {}).get("enable", False),

                    # Video segmentation configuration
                    "segment_duration": yaml_config.get("video_segmentation", {}).get("segment_duration", 3.0),
                    "overlap_duration": yaml_config.get("video_segmentation", {}).get("overlap_duration", 1.0),
                    "max_segments": yaml_config.get("video_segmentation", {}).get("max_segments", None),
                }

                logger.info(f"Active model: {active_model_key} ({model_name})")
                logger.info(f"Using API: {api_key} ({api_type}) at {api_base_url}")

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
            # 批量提取帧
            frame_batches = []
            valid_segments = []
            
            for segment in segments_batch:
                frames = extract_frames_from_video(
                    video_path, 
                    num_frames=self.config["max_frames"],
                    start_time=segment["start_time"],
                    duration=segment["duration"],
                    frame_size=self.config["frame_size"]
                )
                
                if frames:
                    frame_batches.append(frames)
                    valid_segments.append(segment)
            
            if not frame_batches:
                return [{
                    **segment,
                    "has_serve": False,
                    "response": "无法提取帧",
                    "success": False
                } for segment in segments_batch]

            # Get query text and system prompt
            query_text = self.serve_questions["has_serve"]
            system_prompt = FEW_SHOT_SYSTEM_PROMPTS.get("serve_detection")

            # Build messages based on few-shot configuration
            messages_list = []
            if self.config["enable_few_shot"]:
                # Few-shot mode: include examples
                example_config = FEW_SHOT_EXAMPLES.get("serve_detection", {})
                responses_config = FEW_SHOT_RESPONSES.get("serve_detection", {})

                for frames in frame_batches:
                    messages = self.message_manager.create_messages(
                        frames=frames,
                        text=query_text,
                        system_prompt=system_prompt,
                        example_category=example_config.get("category"),
                        positive_label=example_config.get("positive_label"),
                        positive_response=responses_config.get("positive"),
                        negative_label=example_config.get("negative_label"),
                        negative_response=responses_config.get("negative"),
                        num_positive_examples=FEW_SHOT_CONFIG["num_positive_examples"],
                        num_negative_examples=FEW_SHOT_CONFIG["num_negative_examples"],
                    )
                    messages_list.append(messages)
            else:
                # Zero-shot mode: no examples
                for frames in frame_batches:
                    messages = self.message_manager.create_messages(
                        frames=frames,
                        text=query_text,
                        system_prompt=system_prompt
                    )
                    messages_list.append(messages)

            mode = "few-shot" if self.config["enable_few_shot"] else "zero-shot"
            logger.info(f"Using {mode} mode to analyze {len(messages_list)} segments")
            batch_analyses = self._generate_batch(messages_list)
        
            # 处理结果
            batch_results = []
            for i, (segment, analysis) in enumerate(zip(valid_segments, batch_analyses)):
                has_serve = ResponseParser.contains_serve(analysis)
                
                segment_result = {
                    **segment,
                    "has_serve": has_serve,
                    "serve_response": analysis,
                    "success": True,
                    "num_frames": len(frame_batches[i])
                }
                
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

    def _generate_batch(self, messages_batch: List[List[Dict]], debug_mode: bool = False) -> List[str]:
        """
        批量生成响应（统一支持单轮和多轮对话）
        
        Args:
            messages_batch: 消息批次
            debug_mode: 是否保存调试信息
            
        Returns:
            生成的响应列表
        """
        # 如果使用DashScope SDK，调用专用方法
        if self.config["use_dashscope_sdk"]:
            logger.info("使用DashScope原生SDK进行推理")
            return self._generate_with_dashscope(messages_batch, debug_mode)
        else:
            logger.info("使用OpenAI API进行推理")
            return self._generate_with_openai_api(messages_batch, debug_mode)

    def _generate_with_dashscope(self, messages_batch: List[List[Dict]], debug_mode: bool = False) -> List[str]:
        """
        使用DashScope原生SDK进行批量生成
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

    def _generate_with_openai_api(self, messages_batch: List[List[Dict]], debug_mode: bool = False) -> List[str]:
        """
        使用OpenAI API进行批量生成
        """
        try:
            debug_dir = f"selected_frames/{uuid.uuid4()}"
            if debug_mode:
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
                    for item in msg.get("content", []):
                        if item["type"] == "text":
                            transformed_msg["content"].append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item["type"] == "image":
                            image_b64 = self._encode_image(item["image"])
                            image_index += 1
                            if debug_mode:
                                debug_sub_dir = f"{debug_dir}/{batch_index}"
                                os.makedirs(debug_sub_dir, exist_ok=True)
                                item["image"].save(f"{debug_sub_dir}/{image_index}.png")
                            transformed_msg["content"].append({
                                "type": "image_url",
                                "image_url": {"url": image_b64}
                            })
                    transformed_conversation.append(transformed_msg)

                # 调用OpenAI API
                payload = {
                    "model": self.config["model_name"],
                    "messages": transformed_conversation,
                    "max_tokens": self.config["max_new_tokens"],
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                }
                
                try:
                    response = requests.post(
                        f"{self.config["openai_api_base"]}/chat/completions",
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

