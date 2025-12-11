import requests
import os
import logging
logger = logging.getLogger(__name__)

from utils.s3_uploader import S3VideoUploader

api_base = "http://ec2-44-249-79-188.us-west-2.compute.amazonaws.com/v1"
model_name = "Qwen3-VL-30B-A3B-Thinking"
# api_base = "http://localhost:8000/v1"
# model_name = "Qwen/Qwen3-VL-8B-Instruct"

# video_path = "/home/ubuntu/shuttle-sense-vlm/data/videos/badminton_2_segment_000.mp4"
video_path = "/home/ubuntu/shuttle-sense-vlm/data/videos/test.mp4"

s3_uploader = S3VideoUploader(
    bucket_name="gmis-test-1",
    aws_region="us-west-2",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
video_url = s3_uploader.upload_video(video_path, s3_key="data/videos/test.mp4")

print(video_url)
messages = [
    {
        "role": "system",
        "content": """你是一名专业的羽毛球动作识别专家。
    你的任务是分析羽毛球视频片段，判断其中是否包含发球动作。

    【发球定义】
    羽毛球发球是指发球员在发球区内，以一个连续向前的挥拍动作，用球拍击打羽毛球底部，使球飞过球网进入对方发球区的动作。

    【关键特征】
    - 站位：靠近中线，身体正对球网，重心略低。
    - 握拍：采用反手握拍，拇指靠在拍柄宽面上。
    - 托球：非持拍手将球托在拍面前方，球大约与腰部同高。
    - 引拍：持拍手肘略抬起，拍面略向下倾。
    - 击球：轻轻用手指和手腕向前推球，动作短、小、快。
    - 路线：球过网后落点尽可能贴近前发球线，压低球网。
    - 收拍与准备：球出手后立即回到中后场准备位置。
    - 对手臂的要求
      - 手臂不能把击球点抬到腰以上：
        - 因为球被击中时必须在腰部以下。
      - 手臂和手腕的姿势必须保证 “拍头低于持拍手”：
        - 任何导致“拍头高于持拍手”的手臂抬高与甩动，都是潜在违例风险。
      - 手臂挥动方向必须包含明显的“由下向上成分”：
        - 不能做近似水平或向下砍的击球动作。
      - 手臂的挥动必须一气呵成，不得中途停顿或假动作：
        - 手臂前送一旦开始，中间不能再停顿再发力。
      - 托球那只手的手臂要把球放在身体前方，并自然放球，不得向上抛球：
        - 手臂动作太大、明显抛球，会被判违例。

    【分析要求】
    - 仔细观察视频中的每一帧
    - 识别发球的准备、挥拍和击球动作
    - 基于典型特征做出准确判断
    - 请严格按照发球定义和关键特征进行分析，不要进行任何主观判断
    - 输入的图像为视频帧，每秒4帧，请根据视频帧分析发球动作
    - 请仔细思考后给出答案，不要轻易给出答案"""
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video_url", "video_url": {
                    "url": video_url
                }
            },
            {
                "type": "text",
                "text": "请找出视频中发球所处的时间段"
            }
        ]
    }
]
 # 调用OpenAI API
payload = {
    "model": model_name,
    "messages": messages,
    "mm_processor_kwargs": {
        "fps": 2,
        "do_sample_frames": True
    }
}

response = requests.post(f"{api_base}/chat/completions", headers={"Content-Type": "application/json"}, json=payload)
result = response.json()
print(result)
print(result["choices"][0]["message"]["content"])