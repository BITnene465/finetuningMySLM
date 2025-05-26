#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用unsloth微调qwen3-1.7b模型
基于凉宫春日ShareGPT格式对话数据
该版本代码没办法使用 token级掩码，会计算所有文本的loss
"""

import unsloth
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免警告

# 加载ShareGPT格式数据
def load_sharegpt_data(data_path):
    dataset = load_dataset("json", data_files=data_path)
    return dataset

# 数据预处理：将ShareGPT格式转换为 qwen 的chat格式
def preprocess_data(dataset, tokenizer): # Added tokenizer argument
    def process_conversations(example):
        messages = []
        role_mapping = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
        }
        
        if "conversations" in example and isinstance(example["conversations"], list):
            for turn in example["conversations"]:
                if isinstance(turn, dict) and "from" in turn and "value" in turn:
                    messages.append({
                        "role": role_mapping.get(turn["from"], "user"), # Default to "user" if role is unknown
                        "content": turn["value"]
                    })
                else:
                    # Optionally, handle or log malformed turns
                    print(f"Skipping malformed turn in example: {example.get('id', 'Unknown ID')}")
        
        if not messages:
            # Optionally, handle or log examples with no valid messages
            print(f"No valid messages found in example: {example.get('id', 'Unknown ID')}")
            return {"conversation_text": ""}

        try:
            # add_generation_prompt=False because the assistant's response is already in messages.
            # SFTTrainer handles training to predict only the assistant's parts.
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            print(f"Error applying chat template for example {example.get('id', 'Unknown ID')}: {e}")
            print(f"Messages that caused error: {messages}")
            return {"conversation_text": ""} # Return empty string on error

        return {"conversation_text": formatted_text}
    
    processed_dataset = dataset.map(
        process_conversations, 
    )
    processed_dataset = processed_dataset.filter(lambda x: x["conversation_text"] != "")
    return processed_dataset

# 监控训练进度的回调函数
class TrainingMonitor(TrainerCallback):
    def __init__(self):
        self.training_loss = []
        self.eval_loss = []
        self.learning_rates = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            self.steps.append(step)
            # 记录训练损失
            if "loss" in logs:
                self.training_loss.append((step, logs["loss"]))
            # 记录评估损失
            if "eval_loss" in logs:
                self.eval_loss.append((step, logs["eval_loss"]))
            # 记录学习率
            if "learning_rate" in logs:
                self.learning_rates.append((step, logs["learning_rate"]))
    
    def plot_metrics(self, save_dir):
        """绘制训练指标图表"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 绘制训练损失
        if self.training_loss:
            steps, losses = zip(*self.training_loss)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.title("Training Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "training_loss.png"))
            plt.close()
        
        # 绘制评估损失
        if self.eval_loss:
            steps, losses = zip(*self.eval_loss)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.title("Evaluation Loss")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "eval_loss.png"))
            plt.close()
        
        # 绘制学习率
        if self.learning_rates:
            steps, lrs = zip(*self.learning_rates)
            plt.figure(figsize=(10, 6))
            plt.plot(steps, lrs)
            plt.title("Learning Rate Schedule")
            plt.xlabel("Steps")
            plt.ylabel("Learning Rate")
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, "learning_rate.png"))
            plt.close()
        
        # 保存指标数据为CSV
        metrics_df = pd.DataFrame({
            "step": self.steps,
            "training_loss": [loss for _, loss in self.training_loss] if self.training_loss else None,
            "eval_loss": [loss for _, loss in self.eval_loss] if self.eval_loss else None,
            "learning_rate": [lr for _, lr in self.learning_rates] if self.learning_rates else None
        })
        metrics_df.to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)

def main():
    # 设置参数
    model_name = "./Qwen3-1.7B"  # 使用本地模型路径
    output_dir = "./haruhi_qwen_model"
    data_path_dict = {
        "train": "./sharegpt_data/haruhi_train.jsonl",
        "dev": "./sharegpt_data/haruhi_dev.jsonl",
    }
    
    # Unsloth parameters
    max_seq_length = 4096  # Max sequence length
    dtype = None  # Autodetect, or torch.float16, torch.bfloat16
    load_in_4bit = False 
    
    print("开始微调 Qwen3-1.7B 模型")
    
    # 设置随机种子
    set_seed(42)
    print("随机种子设置完成")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录创建完成: {output_dir}")
    # 检查数据文件是否存在
    for key, data_path in data_path_dict.items():
        if not os.path.exists(data_path):
            print(f"错误：数据文件不存在 {data_path}")
            return
        else:
            print(f"{key} 数据文件存在: {data_path}")
    # 检查模型文件是否存在
    if not os.path.exists(model_name):
        print(f"错误：模型目录不存在 {model_name}")
        return
        
    # 加载数据集
    print("正在加载数据集...")
    try:
        dataset = load_sharegpt_data(data_path_dict)
        print(f"数据集加载完成")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return
    
    # 加载模型 
    print(f"正在加载模型 {model_name}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            local_files_only=True,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 应用LoRA配置 - 使用unsloth的方式
    print("正在配置 LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                # LoRA注意力维度
        lora_alpha=32,       # LoRA alpha参数
        lora_dropout=0.0,    # 修正：Dropout = 0 is supported for fast patching.
        target_modules=[     # 目标模块
            "q_proj",
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 使用unsloth的梯度检查点
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    print("LoRA 配置完成")
    
    # 创建训练监控器
    monitor = TrainingMonitor()
    
    # 设置训练参数
    print("正在设置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-5,
        weight_decay=0.001,
        warmup_steps=100,
        max_steps=1000,
        logging_steps=10,                
        report_to="tensorboard",
        save_total_limit=3,
        remove_unused_columns=True,              # (NOTE: Changed to True to fix DataLoader error)
        fp16=False,             
        bf16=True,               # 使用bfloat16精度
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,     # 添加：避免内存问题
        disable_tqdm=False,              # 启用进度条
    )
    print("训练参数设置完成")
    
    # 预处理数据集
    print("正在预处理数据集...")
    try:
        train_dataset = preprocess_data(dataset["train"], tokenizer)
        eval_dataset = preprocess_data(dataset["dev"], tokenizer)
        print("数据集预处理完成")
    except Exception as e:
        print(f"数据集预处理失败: {e}")
        return
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    # 创建SFT训练器
    print("正在初始化训练器...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="conversation_text",  # 使用的训练字段
        max_seq_length=max_seq_length,
        packing=False,   # 关闭packing，使用自定义的 labels
        callbacks=[monitor],
    )
    print("训练器初始化完成")
    
    # 训练模型
    print("\n开始训练模型...")
    print("训练进度将在下方显示...")
    
    try:
        trainer.train()
        print("\n训练完成!")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        return
    
    # 保存训练指标
    print("\n正在保存训练指标...")
    try:
        monitor.plot_metrics(os.path.join(output_dir, "metrics"))
        print("训练指标已保存")
    except Exception as e:
        print(f"保存训练指标失败: {e}")
    
    # 保存最终模型
    print("\n正在保存模型...")
    try:
        trainer.save_model(os.path.join(output_dir, "final_model"))
        print("模型已保存")
    except Exception as e:
        print(f"保存模型失败: {e}")
    
    # 保存训练参数
    print("\n正在保存配置文件...")
    try:
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            config_dict = {
                "model_name": model_name,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"
                ],
                "train_dataset_size": len(train_dataset),
                "eval_dataset_size": len(eval_dataset),
            }
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        print("配置文件已保存")
    except Exception as e:
        print(f"保存配置文件失败: {e}")
    
    print("\n所有任务完成!")
    print(f"输出目录: {output_dir}")

if __name__ == "__main__":
    main()