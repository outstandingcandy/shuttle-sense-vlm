#!/usr/bin/env python3
"""
Few-shot 示例准备工具
用于从参考视频中提取示例帧
"""

import sys
import os
import argparse
import logging

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.few_shot_manager import FewShotManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='从参考视频中提取Few-shot示例帧')
    
    parser.add_argument('--video', required=True, help='参考视频路径')
    parser.add_argument('--category', required=True, help='示例类别（如：serve, rally）')
    parser.add_argument('--label', required=True, help='示例标签（如：has_serve, no_serve）')
    parser.add_argument('--start-time', type=float, default=0, help='开始时间（秒）')
    parser.add_argument('--duration', type=float, default=None, help='持续时间（秒）')
    parser.add_argument('--num-frames', type=int, default=4, help='提取的帧数')
    parser.add_argument('--examples-dir', default='few_shot_examples', help='示例存储目录')
    
    args = parser.parse_args()
    
    # 初始化管理器
    manager = FewShotManager(args.examples_dir)
    
    logger.info(f"开始从视频提取示例: {args.video}")
    logger.info(f"类别: {args.category}, 标签: {args.label}")
    
    # 提取示例帧
    frames = manager.extract_example_frames(
        video_path=args.video,
        category=args.category,
        label=args.label,
        num_frames=args.num_frames,
        start_time=args.start_time,
        duration=args.duration
    )
    
    if frames:
        logger.info(f"✅ 成功提取 {len(frames)} 帧")
        logger.info(f"📁 保存位置: {os.path.join(args.examples_dir, args.category, args.label)}")
    else:
        logger.error("❌ 提取失败")
        return 1
    
    # 显示所有可用示例
    logger.info("\n📋 当前所有可用示例:")
    available = manager.list_available_examples()
    for category, labels in available.items():
        logger.info(f"  {category}:")
        for label in labels:
            metadata = manager.get_example_metadata(category, label)
            if metadata:
                logger.info(f"    - {label} ({metadata.get('num_frames', 0)} 帧)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

