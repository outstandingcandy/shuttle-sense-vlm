"""配置模块"""
from .prompts_config import (
    SERVE_PROMPTS,
    RALLY_PROMPTS,
    GENERAL_PROMPTS,
    PROMPT_TEMPLATES,
    FEW_SHOT_CONFIG,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_SYSTEM_PROMPTS,
    FEW_SHOT_RESPONSES,
    get_serve_prompt,
    get_rally_prompt,
    get_general_prompt,
    get_all_prompts
)

__all__ = [
    "SERVE_PROMPTS",
    "RALLY_PROMPTS",
    "GENERAL_PROMPTS",
    "PROMPT_TEMPLATES",
    "FEW_SHOT_CONFIG",
    "FEW_SHOT_EXAMPLES",
    "FEW_SHOT_SYSTEM_PROMPTS",
    "FEW_SHOT_RESPONSES",
    "get_serve_prompt",
    "get_rally_prompt",
    "get_general_prompt",
    "get_all_prompts"
]

