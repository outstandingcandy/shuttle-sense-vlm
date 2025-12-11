#!/usr/bin/env python3
"""
足球分析器 (Football Analyzer)
"""

import logging
import argparse
import cv2
from typing import Dict, Any, Optional

from core.football_detector import FootballDetector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FootballAnalyzer:
    """足球分析器，支持Qwen3-VL"""

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
            config_path: 配置文件路径
        """
        self.debug = debug
        self.football_detector = FootballDetector(config_path=config_path)

    def detect_actions(self,
                      video_path: str,
                      action_type: str = "goal",
                      workers: int = 1) -> Dict[str, Any]:
        """
        基于动作检测来分割和检测足球比赛片段

        Args:
            video_path: 视频文件路径
            action_type: 动作类型 (goal, pass, tackle, shot, dribble, etc.)
            workers: 并行处理的工作线程数

        Returns:
            包含动作检测结果的字典
        """
        logger.info(f"开始基于{action_type}的足球分析: {video_path}")

        # 获取视频信息
        video_info = self._get_video_info(video_path)
        if "error" in video_info:
            return video_info

        # 检测所有指定类型的动作
        logger.info(f"检测{action_type}动作...")
        action_segments = self.football_detector.detect_all_actions(
            video_path=video_path,
            action_type=action_type,
            workers=workers
        )
        logger.info(f"检测到 {len(action_segments)} 个{action_type}片段")

        # 构造结果
        result = {
            "video_path": video_path,
            "processing_mode": "football_action_detection",
            "action_type": action_type,
            "video_info": video_info,
            "detected_actions": action_segments,
            "action_count": len(action_segments),
            "processing_summary": {
                "method": f"{action_type}_detection",
                "successful_actions": len([s for s in action_segments if s.get("success", False)]),
            }
        }

        logger.info(f"基于{action_type}的检测结果: 找到 {len(action_segments)} 个动作")
        return result

    def detect_multiple_actions(self,
                               video_path: str,
                               action_types: list = None,
                               workers: int = 1) -> Dict[str, Any]:
        """
        检测视频中的多种动作类型

        Args:
            video_path: 视频文件路径
            action_types: 动作类型列表，默认检测进球、射门、传球
            workers: 并行处理的工作线程数

        Returns:
            包含所有动作检测结果的字典
        """
        if action_types is None:
            action_types = ["goal", "shot", "pass"]

        logger.info(f"开始多动作类型分析: {video_path}")
        logger.info(f"检测的动作类型: {', '.join(action_types)}")

        # 获取视频信息
        video_info = self._get_video_info(video_path)
        if "error" in video_info:
            return video_info

        # 检测所有动作类型
        all_actions = {}
        total_actions = 0

        for action_type in action_types:
            logger.info(f"检测{action_type}动作...")
            action_segments = self.football_detector.detect_all_actions(
                video_path=video_path,
                action_type=action_type,
                workers=workers
            )
            all_actions[action_type] = action_segments
            total_actions += len(action_segments)
            logger.info(f"检测到 {len(action_segments)} 个{action_type}片段")

        # 构造结果
        result = {
            "video_path": video_path,
            "processing_mode": "football_multiple_action_detection",
            "action_types": action_types,
            "video_info": video_info,
            "detected_actions_by_type": all_actions,
            "total_action_count": total_actions,
            "processing_summary": {
                "method": "multiple_action_detection",
                "actions_per_type": {
                    action_type: len(segments)
                    for action_type, segments in all_actions.items()
                }
            }
        }

        logger.info(f"多动作检测完成: 总共找到 {total_actions} 个动作")
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(f"视频信息: {total_frames}帧, {fps:.2f}FPS, {total_duration:.2f}秒, {width}x{height}")

        return {
            "duration": total_duration,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="通用足球动作检测器 - 支持进球、射门、传球等动作检测")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--action-type", type=str, default="goal",
                       help="动作类型: goal (进球), shot (射门), pass (传球), tackle (铲球), dribble (带球)")
    parser.add_argument("--multiple-actions", action="store_true",
                       help="检测多种动作类型 (进球、射门、传球)")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--output-format", type=str, choices=["json", "csv", "txt"], default="json", help="输出格式")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--config-path", type=str, help="配置文件路径")
    parser.add_argument("--workers", type=int, default=1, help="并行处理的工作线程数")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        analyzer = FootballAnalyzer(
            debug=args.debug,
            config_path=args.config_path
        )

        if args.multiple_actions:
            result = analyzer.detect_multiple_actions(
                video_path=args.video,
                workers=args.workers
            )
        else:
            result = analyzer.detect_actions(
                video_path=args.video,
                action_type=args.action_type,
                workers=args.workers
            )

        # 输出结果
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                if args.output_format == "json":
                    json.dump(result, f, ensure_ascii=False, indent=2)
                else:
                    print("Only JSON format is currently supported")
            logger.info(f"结果已保存到: {args.output}")
        else:
            print(result)

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
