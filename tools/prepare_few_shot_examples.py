#!/usr/bin/env python3
"""
Few-shot 示例准备工具
用于从注释文件批量提取视频示例帧

注释文件格式（ID-based structure）:
- JSON: {"examples": [{"id": 1, "video": "...", "start_time": 0, "duration": 2.0, "num_frames": 8, "expected_response": "...", "query": "..."}, ...]}
- CSV: id,video,start_time,duration,num_frames,expected_response,query

⚠️  MIGRATION NOTE:
The old category/label-based structure is deprecated. Use IDs for all new examples.
Legacy fields (category, label) are kept for backward compatibility but are no longer required.
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
        示例列表，每个示例包含: id, video, start_time, duration, num_frames, expected_response, query
        Legacy fields (category, label) are optional for backward compatibility
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
    """解析JSON格式的注释文件（ID-based structure）"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'examples' not in data:
            raise ValueError("JSON文件必须包含 'examples' 键")

        examples = []
        seen_ids = set()

        for idx, example in enumerate(data['examples']):
            # Validate required fields for ID-based structure
            required_fields = ['id', 'video']
            for field in required_fields:
                if field not in example:
                    raise ValueError(f"示例 {idx} 缺少必需字段: {field}")

            # Check for duplicate IDs
            example_id = example['id']
            if example_id in seen_ids:
                raise ValueError(f"重复的示例 ID: {example_id}")
            seen_ids.add(example_id)

            # Validate ID type
            if not isinstance(example_id, int):
                raise ValueError(f"示例 {idx}: ID 必须是整数，当前类型: {type(example_id)}")

            # Set default values
            example.setdefault('start_time', 0)
            example.setdefault('duration', None)
            example.setdefault('num_frames', 8)
            example.setdefault('query', None)
            example.setdefault('expected_response', None)

            examples.append(example)

        logger.info(f"从 {json_path} 加载了 {len(examples)} 个示例")
        return examples

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {str(e)}")


def parse_csv_annotation(csv_path: Path) -> List[Dict[str, Any]]:
    """解析CSV格式的注释文件（ID-based structure）"""
    try:
        examples = []
        seen_ids = set()

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required columns for ID-based structure
            required_columns = {'id', 'video'}
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                raise ValueError(f"CSV文件缺少必需列: {missing}")

            for row_idx, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    example_id = int(row['id'])

                    # Check for duplicate IDs
                    if example_id in seen_ids:
                        logger.warning(f"跳过CSV第{row_idx}行，重复的 ID: {example_id}")
                        continue
                    seen_ids.add(example_id)

                    example = {
                        'id': example_id,
                        'video': row['video'],
                        'start_time': float(row.get('start_time', 0)),
                        'duration': float(row['duration']) if row.get('duration') else None,
                        'num_frames': int(row.get('num_frames', 8)),
                        'query': row.get('query'),
                        'expected_response': row.get('expected_response')
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
    example_id: int,
    video: str,
    query: str = None,
    expected_response: str = None,
    start_time: float = 0,
    duration: float = None,
    num_frames: int = 8
) -> bool:
    """
    处理单个示例提取（ID-based structure）

    Args:
        manager: MessageManager instance
        example_id: Unique example ID
        video: Video path
        query: Query text for this example (optional)
        expected_response: Expected assistant response (optional)
        start_time: Start time in seconds
        duration: Duration in seconds
        num_frames: Number of frames to extract

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"处理: ID {example_id} - {video} (时间: {start_time:.1f}s, 帧数: {num_frames})")

    # 检查视频文件是否存在
    if not os.path.exists(video):
        logger.error(f"  ❌ 视频文件不存在: {video}")
        return False

    try:
        frames = manager.extract_example_frames(
            video_path=video,
            example_id=example_id,
            query=query,
            expected_response=expected_response,
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
    批量处理多个示例（ID-based structure）

    Returns:
        处理统计信息 {"total": N, "success": M, "failed": K}
    """
    stats = {"total": len(examples), "success": 0, "failed": 0}

    logger.info(f"\n开始批量处理 {len(examples)} 个示例...\n")

    for idx, example in enumerate(examples, start=1):
        logger.info(f"[{idx}/{len(examples)}]")

        success = process_single_example(
            manager=manager,
            example_id=example['id'],
            video=example['video'],
            query=example.get('query'),
            expected_response=example.get('expected_response'),
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

