#!/usr/bin/env python3
"""
羽毛球分析器
"""

import logging
import argparse
import cv2
from typing import Dict, Any

from core.serve_detector import ServeDetector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BadmintonAnalyzer:
    """羽毛球分析器，支持Qwen3-VL"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen3-VL-3B-Instruct", max_workers: int = None, use_vllm: bool = True, api_base: str = None, api_key: str = None, debug: bool = False):
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
        self.serve_detector = ServeDetector()
    
    def detect_serves(self, 
                                         video_path: str,
                                         segment_duration: float = 3.0,
                                         overlap_duration: float = 1.0,
                                         max_segments: int = None,
                                         max_workers: int = None) -> Dict[str, Any]:
        """
        基于发球检测来分割和检测羽毛球回合
        
        Args:
            video_path: 视频文件路径
            segment_duration: 每段时长（秒）
            overlap_duration: 重叠时长（秒）
            max_segments: 最大处理片段数（用于测试）
            max_workers: 并行处理的最大工作线程数
            
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
        serve_segments = self.serve_detector.detect_all_serves(
            video_path, segment_duration, overlap_duration, max_segments, max_workers
        )
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                       help="模型名称 (默认: Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--segment-duration", type=float, default=3.0, help="分析片段时长（秒），默认3.0")
    parser.add_argument("--overlap", type=float, default=1.0, help="片段重叠时长（秒），默认1.0")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--output-format", type=str, choices=["json", "csv", "txt"], default="json", help="输出格式")
    parser.add_argument("--max-segments", type=int, help="最大处理片段数（用于测试）")
    parser.add_argument("--method", type=str, choices=["serve_based", "time_based"], default="serve_based", 
                       help="检测方法：serve_based（基于发球） 或 time_based（基于时间片段）")
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers", "auto"], default="auto",
                       help="推理后端：vllm（高速） 或 transformers（标准） 或 auto（自动选择）")
    parser.add_argument("--api-base", type=str, default=None,
                       help="API服务器地址 (例如: http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API密钥")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--max-workers", type=int, default=1, help="最大工作线程数")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 确定后端配置
        use_vllm = args.backend in ["vllm", "auto"]
        api_base = args.api_base
        api_key = args.api_key
        
        # 创建检测器
        analyzer = BadmintonAnalyzer(
            debug=args.debug
        )
        
        result = analyzer.detect_serves(
            video_path=args.video,
            segment_duration=args.segment_duration,
            overlap_duration=args.overlap,
            max_segments=args.max_segments,
            max_workers=args.max_workers
        )
        
        if "error" in result:
            logger.error(f"分析失败: {result['error']}")
            return 1
        
        logger.info(f"检测完成: 找到 {result['serve_count']} 个发球")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise e

if __name__ == "__main__":
    main()