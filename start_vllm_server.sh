#!/bin/bash
# vLLM服务器启动脚本 - 为羽毛球检测器提供API服务

# 默认配置
MODEL=${1:-"Qwen/Qwen3-VL-8B-Instruct"}
HOST=${2:-"0.0.0.0"}
PORT=${3:-8000}
GPU_MEMORY=${4:-0.8}

echo "🚀 启动vLLM服务器..."
echo "模型: $MODEL"
echo "地址: $HOST:$PORT"
echo "GPU内存使用率: $GPU_MEMORY"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 使用虚拟环境中的vllm直接执行
"$SCRIPT_DIR/.venv/bin/vllm" serve $MODEL \
    --host $HOST \
    --port $PORT \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.96

echo "✅ vLLM服务器已启动在 http://$HOST:$PORT"
echo "💡 测试命令:"
echo "   python src/universal_rally_detector.py --video test.mp4 --vllm-api-base http://localhost:8000/v1"