#!/usr/bin/env python3
"""
GPT-OSS-20B LoRA Fine-tuning Script
====================================

使用 Unsloth 在您的資料集上進行 LoRA fine-tuning

使用方法:
    python train_gpt_oss_20b.py

在 tmux 背景執行（建議）:
    tmux new -s finetune
    python train_gpt_oss_20b.py
    # 按 Ctrl+B 然後按 D 來 detach

    # 重新連接:
    tmux attach -t finetune

注意事項:
    - 請將訓練資料放在 ./data/YourDataset.csv
    - 資料格式：CSV with 'input' and 'output' columns
    - 訓練時間視資料量和 GPU 而定，建議使用 tmux 背景執行
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTConfig, SFTTrainer

print("=" * 80)
print("🦥 GPT-OSS-20B LoRA Fine-tuning Script")
print("=" * 80)

# ============================================================================
# 模型設定
# ============================================================================
MODEL_NAME = "unsloth/gpt-oss-20b-BF16"
MAX_SEQ_LENGTH = 1024
DTYPE = None  # None for auto detection
LOAD_IN_4BIT = False
FULL_FINETUNING = False

# ============================================================================
# LoRA 設定
# ============================================================================
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_BIAS = "none"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
USE_GRADIENT_CHECKPOINTING = "unsloth"
LORA_RANDOM_STATE = 3407
USE_RSLORA = False
LOFTQ_CONFIG = None

# ============================================================================
# System Prompt
# ============================================================================
SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in technical support.\n"
    "Knowledge cutoff: 2024-06"
)

# ============================================================================
# 訓練設定
# ============================================================================
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
MAX_STEPS = -1  # -1 表示由 num_train_epochs 控制
NUM_TRAIN_EPOCHS = 300
LEARNING_RATE = 2e-4
OPTIM = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
TRAINING_SEED = 3407
OUTPUT_DIR = "outputs"
REPORT_TO = "none"

# 評估設定
EVAL_STRATEGY = "epoch"

# ============================================================================
# 資料集設定
# ============================================================================
CSV_FILE = "./data/YourDataset.csv"  # 請將您的資料集命名為 YourDataset.csv
TEST_SIZE = 110  # Test 和 Validation 各 110 筆（可依實際資料量調整）
RANDOM_SEED = 42

# ============================================================================
# 1. 載入基礎模型
# ============================================================================
print("\n" + "=" * 80)
print("📦 載入基礎模型...")
print("=" * 80)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    dtype=DTYPE,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
    full_finetuning=FULL_FINETUNING,
)

print(f"✅ 成功載入模型: {MODEL_NAME}")

# ============================================================================
# 2. 配置 LoRA
# ============================================================================
print("\n" + "=" * 80)
print("🔧 配置 LoRA...")
print("=" * 80)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias=LORA_BIAS,
    use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    random_state=LORA_RANDOM_STATE,
    use_rslora=USE_RSLORA,
    loftq_config=LOFTQ_CONFIG,
)

print(f"✅ LoRA 配置完成 (r={LORA_R}, alpha={LORA_ALPHA})")

# ============================================================================
# 3. 資料處理函數
# ============================================================================

def convert_to_sharegpt(example):
    """將 CSV 的 input/output 轉換為 ShareGPT 格式"""
    conversations = [
        {"from": "human", "value": example["input"]},
        {"from": "gpt", "value": example["output"]}
    ]
    return {"conversations": conversations}


def formatting_prompts_func(examples):
    """應用 chat template"""
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}


def prepare_dataset(csv_file):
    """準備資料集"""
    # 步驟 1: Read CSV
    print(f"📂 讀取 CSV: {csv_file}...")
    dataset = load_dataset("csv", data_files=csv_file, split="train")
    print(f"✅ 載入了 {len(dataset)} 筆資料")
    print(f"第一筆資料: {dataset[0]}")

    # 步驟 2: 轉換為 ShareGPT 格式
    print("🔄 轉換為 ShareGPT 格式...")
    dataset = dataset.map(convert_to_sharegpt, remove_columns=dataset.column_names)
    print(f"第一筆資料: {dataset[0]}")

    # 步驟 3: 標準化 ShareGPT 格式
    print("🔧 標準化 ShareGPT...")
    dataset = standardize_sharegpt(dataset)
    print(f"第一筆資料: {dataset[0]}")

    # 步驟 4: 應用 chat template 轉為訓練文字
    print("📝 應用 Chat Template...")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 預覽第一筆資料
    print("\n📋 預覽第一筆處理後的資料:")
    print("=" * 80)
    print(dataset[0]["text"][:500])  # 只顯示前 500 字元
    print("..." if len(dataset[0]["text"]) > 500 else "")
    print("=" * 80)

    return dataset


# ============================================================================
# 4. 載入並分割資料集
# ============================================================================
print("\n" + "=" * 80)
print("📊 載入並分割資料集...")
print("=" * 80)

# 處理資料集
processed_dataset = prepare_dataset(CSV_FILE)
print(f"\n📊 原始資料集大小: {len(processed_dataset)} 筆")

# 資料集分割：Train / Validation / Test
print("\n" + "=" * 80)
print("✂️  開始分割資料集...")
print("=" * 80)

# 先切出 test set (110 筆)
split_1 = processed_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
train_val_dataset = split_1['train']
test_dataset = split_1['test']

# 再從剩下的資料中切出 validation set (110 筆)
split_2 = train_val_dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
train_dataset = split_2['train']
val_dataset = split_2['test']

print(f"✅ Train 資料集: {len(train_dataset)} 筆")
print(f"✅ Validation 資料集: {len(val_dataset)} 筆")
print(f"✅ Test 資料集: {len(test_dataset)} 筆")
print(f"📊 總計: {len(train_dataset) + len(val_dataset) + len(test_dataset)} 筆")

# 儲存分割後的資料集
output_path_train = "processed_dataset_gpt_oss/train"
output_path_val = "processed_dataset_gpt_oss/validation"
output_path_test = "processed_dataset_gpt_oss/test"

train_dataset.save_to_disk(output_path_train)
val_dataset.save_to_disk(output_path_val)
test_dataset.save_to_disk(output_path_test)

print(f"\n💾 Train 資料集已儲存至: {output_path_train}")
print(f"💾 Validation 資料集已儲存至: {output_path_val}")
print(f"💾 Test 資料集已儲存至: {output_path_test}")

# ============================================================================
# 5. 初始化訓練器
# ============================================================================
print("\n" + "=" * 80)
print("🏋️  初始化訓練器...")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=TRAINING_SEED,
        output_dir=OUTPUT_DIR,
        report_to=REPORT_TO,
        # 評估設定
        eval_strategy=EVAL_STRATEGY,
    ),
)

print("✅ 訓練器初始化完成")

# ============================================================================
# 6. 開始訓練
# ============================================================================
print("\n" + "=" * 80)
print("🚀 開始訓練...")
print(f"📊 Train: {len(processed_dataset)} 筆 | Validation: {len(val_dataset)} 筆")
print("=" * 80)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("✅ 訓練完成!")
print("=" * 80)

# ============================================================================
# 7. 儲存模型
# ============================================================================
print("\n" + "=" * 80)
print("💾 儲存模型...")
print("=" * 80)

model.save_pretrained("RANK32_gpt_oss_finetuned")
tokenizer.save_pretrained("RANK32_gpt_oss_finetuned")

print("✅ 模型已儲存至: ./RANK32_gpt_oss_finetuned")

# ============================================================================
# 8. 測試推論
# ============================================================================
print("\n" + "=" * 80)
print("🧪 測試推論...")
print("=" * 80)

# 準備測試訊息
messages = [
    {"role": "user", "content": "What is the purpose of SEMI E88?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",
).to(model.device)

from transformers import TextStreamer

print("\n📝 生成回答:")
print("-" * 80)
_ = model.generate(**inputs, max_new_tokens=512, streamer=TextStreamer(tokenizer))
print("-" * 80)

print("\n" + "=" * 80)
print("✅ 全部完成!")
print("=" * 80)
print(f"📁 模型位置: ./RANK32_gpt_oss_finetuned")
print(f"📁 輸出位置: ./{OUTPUT_DIR}")
print(f"📁 資料集位置: ./processed_dataset_gpt_oss/")
