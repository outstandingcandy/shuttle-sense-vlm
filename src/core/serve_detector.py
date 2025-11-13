#!/usr/bin/env python3
"""
发球检测模块 - 专门处理发球动作的检测和分析
"""

import logging
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.response_parser import ResponseParser
from config.prompts_config import SERVE_PROMPTS, FEW_SHOT_CONFIG, FEW_SHOT_EXAMPLES, FEW_SHOT_SYSTEM_PROMPTS, FEW_SHOT_RESPONSES
from core.few_shot_manager import FewShotManager
import requests
import os
import uuid
import io
import base64
from dashscope import MultiModalConversation

logger = logging.getLogger(__name__)

class ServeDetector:
    """发球检测器"""
    
    def __init__(self):
        """
        初始化发球检测器
        """
        # 发球检测相关的提示词（从配置文件加载）
        self.serve_questions = SERVE_PROMPTS

        self.config = {
            "max_frames": 8,
            "frame_size": [1024, 1024],
            "batch_size": 4,
            "max_new_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.8,
            "do_sample": False,
            "use_dashscope_sdk": True,
            "model_name": "qwen-vl-max-latest",
            "dashscope_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dashscope_api_key": os.getenv("DASHSCOPE_API_KEY"),
            # "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            # "openai_api_base": "http://localhost:8000/v1",
        }
        
        self.few_shot_manager = FewShotManager(FEW_SHOT_CONFIG["examples_dir"])

    def extract_video_frames(self, video_path: str, start_time: float = 0, duration: float = None) -> List[Image.Image]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            duration: 持续时间（秒），None表示到视频结束
            
        Returns:
            PIL图像列表
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算帧范围
        start_frame = int(start_time * fps)
        if duration is not None:
            end_frame = min(start_frame + int(duration * fps), total_frames)
        else:
            end_frame = total_frames
        
        # 计算要提取的帧索引
        segment_frames = end_frame - start_frame
        max_frames = self.config["max_frames"]
        
        if segment_frames <= max_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # 转换BGR到RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 调整大小
                frame = cv2.resize(frame, tuple(self.config["frame_size"]))
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames
    
    def detect_all_serves(self, 
                         video_path: str,
                         segment_duration: float,
                         overlap_duration: float,
                         max_segments: int,
                         workers: int) -> List[Dict[str, Any]]:
        """检测视频中所有的发球时刻"""
        
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
                frames = self.extract_video_frames(
                    video_path, 
                    segment["start_time"], 
                    segment["duration"]
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
            
            # 批量分析是否有发球
            # 如果启用few-shot，使用多轮对话模式
            example_config = FEW_SHOT_EXAMPLES.get("serve_detection", {})
            query_text = self.serve_questions["has_serve"]
            
            # 获取 system prompt 和响应配置
            system_prompt = FEW_SHOT_SYSTEM_PROMPTS.get("serve_detection")
            responses_config = FEW_SHOT_RESPONSES.get("serve_detection", {})
            
            # 构建多轮对话消息
            messages_list = []
            for frames in frame_batches:
                messages = self.few_shot_manager.create_few_shot_messages(
                    query_frames=frames,
                    query_text=query_text,
                    example_category=example_config.get("category", "serve"),
                    positive_label=example_config.get("positive_label", "has_serve"),
                    positive_response=responses_config.get("positive", "是的，这段视频中有发球动作。"),
                    negative_label=example_config.get("negative_label") if FEW_SHOT_CONFIG["num_negative_examples"] > 0 else None,
                    negative_response=responses_config.get("negative", "不，这段视频中没有发球动作。"),
                    num_positive_examples=FEW_SHOT_CONFIG["num_positive_examples"],
                    num_negative_examples=FEW_SHOT_CONFIG["num_negative_examples"],
                    system_prompt=system_prompt
                )
                messages_list.append(messages)
            
            logger.info(f"使用 Few-shot 多轮对话模式分析 {len(messages_list)} 个片段")
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
                    model='qwen3-vl-plus',
                    messages=messages)
                responses.append(response.output.choices[0].message.content[0]['text'].strip())
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
                    content = []
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

