#!/usr/bin/env python3
"""
响应解析工具 - 解析模型响应并提取关键信息
"""

import json

class ResponseParser:
    """模型响应解析器"""
    
    @staticmethod
    def contains_serve(response: str) -> bool:
        """判断模型响应是否表示包含发球动作"""
        response = response.lower().strip()
        try:
            result = json.loads(response)
            if result["action_type"] == "serve":
                return True
            else:
                return False
        except:
            return False

    @staticmethod
    def contains_rally(response: str) -> bool:
        """判断模型响应是否表示包含回合"""
        response = response.lower().strip()

        # 积极指标
        positive_indicators = [
            "有回合", "有羽毛球回合", "包含回合", "存在回合", "正在进行",
            "活跃比赛", "比赛中", "对打", "击球", "发球", "接发球",
            "对拉", "杀球", "吊球", "网前", "羽毛球动作"
        ]

        # 消极指标
        negative_indicators = [
            "无回合", "没有回合", "无羽毛球回合", "不包含回合", "无活动",
            "无动作", "静止", "暂停", "休息", "准备阶段", "准备"
        ]

        # 检查消极指标
        for neg in negative_indicators:
            if neg in response:
                return False

        # 检查积极指标
        for pos in positive_indicators:
            if pos in response:
                return True

        return False

    @staticmethod
    def contains_football_action(response: str, action_type: str = "goal") -> bool:
        """
        判断模型响应是否表示包含足球动作

        Args:
            response: 模型响应文本
            action_type: 动作类型 (goal, shot, pass, tackle, dribble, etc.)

        Returns:
            是否包含指定动作
        """
        response = response.lower().strip()

        # Try to parse JSON response first
        try:
            result = json.loads(response)
            detected_action = result.get("action_type", "").lower()
            if detected_action == action_type:
                return True
            elif detected_action == f"no_{action_type}":
                return False
        except:
            pass

        # Define indicators for different action types
        action_indicators = {
            "goal": {
                "positive": [
                    "进球", "得分", "破门", "入网", "球进了", "goal", "score",
                    "球过球门线", "皮球入网", "攻入一球"
                ],
                "negative": [
                    "无进球", "没有进球", "未进球", "no goal", "没得分",
                    "射偏", "被扑出", "击中门柱", "越位"
                ]
            },
            "shot": {
                "positive": [
                    "射门", "起脚", "打门", "抽射", "劲射", "shot", "shoot",
                    "远射", "近射", "头球攻门", "凌空抽射"
                ],
                "negative": [
                    "无射门", "没有射门", "未射门", "no shot", "传球", "控球"
                ]
            },
            "pass": {
                "positive": [
                    "传球", "传递", "传中", "直塞", "分球", "pass", "传出",
                    "横传", "长传", "短传", "回传", "助攻"
                ],
                "negative": [
                    "无传球", "没有传球", "未传球", "no pass", "射门", "带球"
                ]
            },
            "tackle": {
                "positive": [
                    "铲球", "抢断", "拦截", "tackle", "防守", "断球",
                    "铲倒", "滑铲", "封堵", "破坏"
                ],
                "negative": [
                    "无铲球", "没有铲球", "未铲球", "no tackle", "进攻"
                ]
            },
            "dribble": {
                "positive": [
                    "带球", "运球", "盘带", "过人", "突破", "dribble",
                    "控球", "摆脱", "变向", "过掉防守"
                ],
                "negative": [
                    "无带球", "没有带球", "未带球", "no dribble", "失球", "被断"
                ]
            },
            "action": {
                "positive": [
                    "动作", "活动", "比赛", "进行", "踢球", "action",
                    "足球", "球员", "运动"
                ],
                "negative": [
                    "无动作", "没有动作", "无活动", "no action", "静止",
                    "暂停", "休息", "中场休息"
                ]
            }
        }

        # Get indicators for specified action type, default to "action" if not found
        indicators = action_indicators.get(action_type, action_indicators["action"])

        # Check negative indicators first
        for neg in indicators["negative"]:
            if neg in response:
                return False

        # Check positive indicators
        for pos in indicators["positive"]:
            if pos in response:
                return True

        return False