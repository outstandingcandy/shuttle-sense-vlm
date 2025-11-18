#!/usr/bin/env python3
"""
提示词配置文件 - 统一管理所有模块的提示词

⚠️  DEPRECATION NOTICE:
This module is being replaced by the annotation-based configuration system.
New code should use AnnotationConfigLoader (src/config/annotations_loader.py) to load
prompts and configuration from annotation JSON files (e.g., docs/annotations_example.json).

This file is kept for backward compatibility with existing code.
"""

# 发球检测相关提示词
SERVE_PROMPTS = {
    "has_serve": """这段羽毛球视频中是否包含发球动作？如果包含请给出发球动作所在的时间区间，如果不包含请回答没有发球动作""",
    "serve_type": "描述这段羽毛球视频中的发球类型：高远球发球、平快球发球、网前球发球、反手发球或其他",
}

# 回合分析相关提示词
RALLY_PROMPTS = {
    "has_rally": "这段羽毛球视频中是否有正在进行的回合？请回答：有回合 或 无回合",
    "action_type": "描述这段羽毛球视频中的主要动作类型：发球、接发球、对拉、杀球、吊球、网前球或其他",
    "rally_intensity": "评估这段羽毛球回合的激烈程度：低、中、高",
    "player_activity": "这段视频中羽毛球选手的活动状态：活跃比赛、准备阶段、休息或无活动"
}

# 可以添加更多模块的提示词
# 例如：动作识别、比分检测等

# 通用提示词
GENERAL_PROMPTS = {
    "describe_content": "详细描述这段羽毛球视频中的内容",
    "count_players": "这段视频中有多少名羽毛球选手？",
    "court_position": "描述羽毛球选手在场地上的位置：前场、中场、后场"
}

# 提示词模板（可用于动态生成提示词）
PROMPT_TEMPLATES = {
    "yes_no_question": "这段羽毛球视频中是否{action}？请回答：{yes_answer} 或 {no_answer}",
    "describe_action": "描述这段羽毛球视频中的{action_category}：{action_options}",
    "evaluate_level": "评估这段羽毛球{aspect}的{metric}：{levels}"
}

def get_serve_prompt(key: str) -> str:
    """获取发球检测提示词"""
    return SERVE_PROMPTS.get(key, "")

def get_rally_prompt(key: str) -> str:
    """获取回合分析提示词"""
    return RALLY_PROMPTS.get(key, "")

def get_general_prompt(key: str) -> str:
    """获取通用提示词"""
    return GENERAL_PROMPTS.get(key, "")

def get_all_prompts() -> dict:
    """获取所有提示词"""
    return {
        "serve": SERVE_PROMPTS,
        "rally": RALLY_PROMPTS,
        "general": GENERAL_PROMPTS,
        "templates": PROMPT_TEMPLATES
    }

# Few-shot 配置
FEW_SHOT_CONFIG = {
    "enabled": True,  # 是否启用few-shot
    "examples_dir": "few_shot_examples",  # 示例存储目录
    "example_ids": [1, 2, 3, 4, 5],  # 使用的示例ID列表
    "example_frames_per_video": 8,  # 每个示例视频提取的帧数
}

# Few-shot System Prompts（系统提示词）
FEW_SHOT_SYSTEM_PROMPTS = {
    "serve_detection": """你是一名专业的羽毛球动作识别专家。
你的任务是分析羽毛球视频片段，判断其中是否包含发球动作。

【发球定义】
羽毛球发球是指发球员在发球区内，以一个连续向前的挥拍动作，用球拍击打羽毛球底部，使球飞过球网进入对方发球区的动作。

【关键特征】
1. 发球员在发球区内、身体相对静止
2. 挥拍时球拍杆朝下、球在腰部以下
3. 击球后羽毛球飞过球网，进入对方场地
4. 发球时不能把球抛起

【分析要求】
- 仔细观察视频中的每一帧
- 识别发球的准备、挥拍和击球动作
- 基于典型特征做出准确判断""",

}
