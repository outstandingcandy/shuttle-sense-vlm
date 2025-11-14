#!/usr/bin/env python3
"""
响应解析工具 - 解析模型响应并提取关键信息
"""

class ResponseParser:
    """模型响应解析器"""
    
    @staticmethod
    def contains_serve(response: str) -> bool:
        """判断模型响应是否表示包含发球动作"""
        response = response.lower().strip()
        
        # 发球积极指标
        serve_indicators = [
            "有发球", "发球动作", "发球", "serve", "serving",
            "高远球发球", "平快球发球", "网前球发球", "反手发球",
            "准备发球", "发球姿势", "发球瞬间", "开球"
        ]
        
        # 发球消极指标
        no_serve_indicators = [
            "无发球", "没有发球", "不是发球", "非发球",
            "接发球", "回球", "对拉", "杀球", "吊球"
        ]
        
        # 检查消极指标
        for neg in no_serve_indicators:
            if neg in response:
                return False
        
        # 检查积极指标
        for pos in serve_indicators:
            if pos in response:
                return True
        
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