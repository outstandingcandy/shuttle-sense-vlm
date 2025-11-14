#!/usr/bin/env python3
"""
Few-shot 示例准备工具
用于从注释文件批量提取视频示例帧

注释文件格式：
- JSON: {"examples": [{"video": "...", "category": "...", "label": "...", "start_time": 0, "duration": 2.0, "num_frames": 8}, ...]}
- CSV: video,category,label,start_time,duration,num_frames
"""

import sys
import os
import argparse
import logging
import json
import csv
from typing import List, Dict, Any
from pathlib import Path

# 添加项目路径
project_root = os.path.join(os.path.dirname(__file__), '..')
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from core.few_shot_manager import MessageManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_annotation_file(annotation_path: str) -> List[Dict[str, Any]]:
    """
    解析注释文件，支持JSON和CSV格式

    Args:
        annotation_path: 注释文件路径

    Returns:
        示例列表，每个示例包含: video, category, label, start_time, duration, num_frames
    """
    annotation_path = Path(annotation_path)

    if not annotation_path.exists():
        raise FileNotFoundError(f"注释文件不存在: {annotation_path}")

    suffix = annotation_path.suffix.lower()

    if suffix == '.json':
        return parse_json_annotation(annotation_path)
    elif suffix == '.csv':
        return parse_csv_annotation(annotation_path)
    else:
        raise ValueError(f"不支持的注释文件格式: {suffix}. 仅支持 .json 和 .csv")


def parse_json_annotation(json_path: Path) -> List[Dict[str, Any]]:
    """解析JSON格式的注释文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'examples' not in data:
            raise ValueError("JSON文件必须包含 'examples' 键")

        examples = []
        for idx, example in enumerate(data['examples']):
            # 验证必需字段
            required_fields = ['video', 'category', 'label']
            for field in required_fields:
                if field not in example:
                    raise ValueError(f"示例 {idx} 缺少必需字段: {field}")

            # 设置默认值
            example.setdefault('start_time', 0)
            example.setdefault('duration', None)
            example.setdefault('num_frames', 8)

            examples.append(example)

        logger.info(f"从 {json_path} 加载了 {len(examples)} 个示例")
        return examples

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {str(e)}")


def parse_csv_annotation(csv_path: Path) -> List[Dict[str, Any]]:
    """解析CSV格式的注释文件"""
    try:
        examples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # 验证必需列
            required_columns = {'video', 'category', 'label'}
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(f"CSV文件缺少必需列: {missing}")

            for row_idx, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    example = {
                        'video': row['video'],
                        'category': row['category'],
                        'label': row['label'],
                        'start_time': float(row.get('start_time', 0)),
                        'duration': float(row['duration']) if row.get('duration') else None,
                        'num_frames': int(row.get('num_frames', 8))
                    }
                    examples.append(example)
                except (ValueError, KeyError) as e:
                    logger.warning(f"跳过CSV第{row_idx}行，解析错误: {str(e)}")
                    continue

        logger.info(f"从 {csv_path} 加载了 {len(examples)} 个示例")
        return examples

    except Exception as e:
        raise ValueError(f"CSV解析错误: {str(e)}")


def process_single_example(
    manager: MessageManager,
    video: str,
    category: str,
    label: str,
    start_time: float = 0,
    duration: float = None,
    num_frames: int = 8
) -> bool:
    """
    处理单个示例提取

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"处理: {video} -> {category}/{label} (时间: {start_time:.1f}s, 帧数: {num_frames})")

    # 检查视频文件是否存在
    if not os.path.exists(video):
        logger.error(f"  ❌ 视频文件不存在: {video}")
        return False

    try:
        frames = manager.extract_example_frames(
            video_path=video,
            category=category,
            label=label,
            num_frames=num_frames,
            start_time=start_time,
            duration=duration
        )

        if frames:
            logger.info(f"  ✅ 成功提取 {len(frames)} 帧")
            return True
        else:
            logger.error(f"  ❌ 提取失败")
            return False

    except Exception as e:
        logger.error(f"  ❌ 提取失败: {str(e)}")
        return False


def process_batch(manager: MessageManager, examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    批量处理多个示例

    Returns:
        处理统计信息 {"total": N, "success": M, "failed": K}
    """
    stats = {"total": len(examples), "success": 0, "failed": 0}

    logger.info(f"\n开始批量处理 {len(examples)} 个示例...\n")

    for idx, example in enumerate(examples, start=1):
        logger.info(f"[{idx}/{len(examples)}]")

        success = process_single_example(
            manager=manager,
            video=example['video'],
            category=example['category'],
            label=example['label'],
            start_time=example.get('start_time', 0),
            duration=example.get('duration'),
            num_frames=example.get('num_frames', 8)
        )

        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        logger.info("")  # Empty line for readability

    return stats


def print_summary(manager: MessageManager, stats: Dict[str, int] = None):
    """打印处理摘要"""
    if stats:
        logger.info("=" * 60)
        logger.info("处理摘要:")
        logger.info(f"  总计: {stats['total']} 个示例")
        logger.info(f"  成功: {stats['success']} 个")
        logger.info(f"  失败: {stats['failed']} 个")
        logger.info("=" * 60)

    logger.info("\n📋 当前所有可用示例:")
    available = manager.list_available_examples()

    if not available:
        logger.info("  (无)")
        return

    for category, labels in available.items():
        logger.info(f"  {category}:")
        for label in labels:
            metadata = manager.get_example_metadata(category, label)
            if metadata:
                num_examples = metadata.get('num_examples', 0)
                num_frames = metadata.get('num_frames', 0)
                source_videos = metadata.get('source_videos', [])
                # Create a summary of source videos
                unique_sources = list(set(Path(v).name for v in source_videos))
                sources_str = ', '.join(unique_sources[:3])  # Show first 3
                if len(unique_sources) > 3:
                    sources_str += f', ... (+{len(unique_sources) - 3} more)'
                logger.info(f"    - {label}: {num_examples} examples, {num_frames} frames (sources: {sources_str})")


def main():
    parser = argparse.ArgumentParser(
        description='从注释文件批量提取Few-shot示例帧',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 使用 JSON 注释文件
  python prepare_few_shot_examples.py --annotation-file annotations.json

  # 使用 CSV 注释文件
  python prepare_few_shot_examples.py --annotation-file annotations.csv

  # 指定自定义存储目录
  python prepare_few_shot_examples.py \\
      --annotation-file annotations.json \\
      --examples-dir custom_examples

注释文件格式:
  JSON: {"examples": [{"video": "...", "category": "...", "label": "...", "start_time": 0, "duration": 2.0, "num_frames": 8}]}
  CSV:  video,category,label,start_time,duration,num_frames

详细文档: docs/FEW_SHOT_GUIDE.md
示例文件: docs/annotations_example.json, docs/annotations_example.csv
        """
    )

    # Required arguments
    parser.add_argument('--annotation-file', type=str, required=True,
                       help='注释文件路径 (JSON或CSV格式)')

    # Optional arguments
    parser.add_argument('--examples-dir', default='few_shot_examples',
                       help='示例存储目录 (默认: few_shot_examples)')

    args = parser.parse_args()

    # 初始化管理器
    manager = MessageManager(args.examples_dir)

    try:
        # 批处理模式
        logger.info("=" * 60)
        logger.info("批量提取Few-shot示例")
        logger.info(f"注释文件: {args.annotation_file}")
        logger.info("=" * 60)

        examples = parse_annotation_file(args.annotation_file)

        if not examples:
            logger.warning("注释文件中没有找到任何示例")
            return 1

        stats = process_batch(manager, examples)
        print_summary(manager, stats)

        return 0 if stats['failed'] == 0 else 1

    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

