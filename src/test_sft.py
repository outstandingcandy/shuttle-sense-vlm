import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# ==========================================
# 1. 核心配置 (Configuration)
# ==========================================
max_seq_length = 4096   # Agent 轨迹通常较长，建议设为 4096 或 8192
dtype = None            # None = 自动检测 (Bfloat16 for Ampere+)
load_in_4bit = True     # 开启 4bit 量化，显存占用降低 4 倍

# 模型选择：可以直接换成 "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
model_name = "unsloth/Qwen2.5-7B-Instruct"

# ==========================================
# 2. 加载模型与 Tokenizer (Model Loading)
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# ==========================================
# 3. LoRA 适配器配置 (LoRA Config)
# 我们使用了 Rank=64, Alpha=128 的"黄金配比"
# ==========================================
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,                  # Rank: 越高参数量越大，Agent 任务建议 32-128
    target_modules = [       # 关键：对所有线性层进行微调，增强逻辑推理能力
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 128,        # 通常是 Rank 的 2 倍
    lora_dropout = 0,        # Unsloth 建议为 0 以加速
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 显存优化技术
    random_state = 3407,
    use_rslora = False,      # 如果要用 Rank Stabilization LoRA 可开启
    loftq_config = None,     # LoftQ 初始化
)

# ==========================================
# 4. 数据处理 (Data Formatting)
# 假设你的数据是 JSON 格式的 ShareGPT 风格 (system, user, assistant)
# ==========================================
from unsloth.chat_templates import get_chat_template

# 自动适配 Llama-3 的 Chat Template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

# 加载你的 Agent 数据集 (假设是本地 json)
# dataset = load_dataset("json", data_files="my_agent_data.json", split="train")
# 这里为了演示，我们模拟一条数据
# 注意：使用 "role"/"content" 格式（标准 OpenAI 格式），而非 "from"/"value" (ShareGPT 格式)
dummy_data = [
    {
        "conversations": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Draw a cat."},
            {"role": "assistant", "content": "Thought: The user wants a cat.\nAction: ```json\n{\"tool\": \"draw\", \"obj\": \"cat\"}\n```"}
        ]
    }
] * 100 # 复制100次

from datasets import Dataset
dataset = Dataset.from_list(dummy_data)
dataset = dataset.map(formatting_prompts_func, batched = True)

# ==========================================
# 5. 训练器配置 (Trainer Config)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    
    # === 关键技术点: Sequence Packing ===
    # 将多条短数据拼接到 max_seq_length，大幅提升训练效率
    packing = True, 

    args = TrainingArguments(
        per_device_train_batch_size = 2,  # 根据显存调整
        gradient_accumulation_steps = 4,  # 模拟大 Batch Size (2*4=8)
        warmup_steps = 5,
        max_steps = 60,                   # 测试用，实际训练建议用 num_train_epochs = 3
        learning_rate = 2e-4,             # QLoRA 标准学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",             # 8bit 优化器，省显存
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# ==========================================
# 6. 开始训练与保存 (Train & Save)
# ==========================================
print("开始训练 Agent 模型...")
trainer_stats = trainer.train()

# 保存 LoRA Adapter
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# (可选) 合并保存为 GGUF 或 16bit 模型
# model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit")