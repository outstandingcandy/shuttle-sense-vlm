#!/usr/bin/env python3
"""
羽毛球分析器
"""

import logging
import argparse
import cv2
from typing import Dict, Any, Optional

from core.serve_detector import ServeDetector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BadmintonAnalyzer:
    """羽毛球分析器，支持Qwen3-VL"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen3-VL-3B-Instruct", max_workers: int = None, use_vllm: bool = True, api_base: str = None, api_key: str = None, debug: bool = False, config_path: str = None):
        """
        初始化检测器

        Args:
            base_model_name: 模型名称
            max_workers: 并行处理的最大工作线程数，None表示自动选择
            use_vllm: 是否使用vLLM加速（推荐）
            api_base: API服务器地址 (例如: http://localhost:8000/v1)
            api_key: API密钥
            debug: 是否启用调试模式
        """
        self.debug = debug
        self.serve_detector = ServeDetector(config_path=config_path)
 
    def detect_serves(self, 
                      video_path: str, workers: int = 1) -> Dict[str, Any]:
        """
        基于发球检测来分割和检测羽毛球回合
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含发球和回合检测结果的字典
        """
        logger.info(f"开始基于发球的回合分析: {video_path}")

        # 获取视频信息
        video_info = self._get_video_info(video_path)
        if "error" in video_info:
            return video_info
        
        # 第一步：检测所有发球时刻
        logger.info("第一步：检测发球时刻...")
        serve_segments = self.serve_detector.detect_all_serves(video_path=video_path, workers=workers)
        logger.info(f"检测到 {len(serve_segments)} 个发球片段")
        
        # 构造结果
        result = {
            "video_path": video_path,
            "processing_mode": "serve_based_segmentation",
            "video_info": video_info,
            "detected_serves": serve_segments,
            "serve_count": len(serve_segments),
            "processing_summary": {
                "method": "serve_detection",
                "successful_serves": len([s for s in serve_segments if s.get("success", False)]),
            }
        }
        
        logger.info(f"基于发球的检测结果: 找到 {len(serve_segments)} 个发球")
        return result
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "error": f"Cannot open video: {video_path}",
                "video_path": video_path
            }
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"视频信息: {total_frames}帧, {fps:.2f}FPS, {total_duration:.2f}秒")
        
        return {
            "duration": total_duration,
            "fps": fps,
            "total_frames": total_frames
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用羽毛球回合检测器 - 支持发球检测和回合分割")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--output-format", type=str, choices=["json", "csv", "txt"], default="json", help="输出格式")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--config-path", type=str, help="配置文件路径")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        analyzer = BadmintonAnalyzer(
            debug=args.debug
        )

        result = analyzer.detect_serves(video_path=args.video)
        print(result)

    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        raise e

if __name__ == "__main__":
    main()