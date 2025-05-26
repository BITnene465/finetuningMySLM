#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将LoRA适配器权重合并到基础模型中，并保存合并后的模型。
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_to_base_model(base_model_path: str, lora_adapter_path: str, output_path: str):
    """
    加载本地的基础模型和LoRA适配器，合并它们，并将合并后的模型保存到指定路径。
    """
    print(f"开始合并过程...")
    print(f"  基础模型: {base_model_path}")
    print(f"  LoRA适配器: {lora_adapter_path}")
    print(f"  输出目录: {output_path}")

    # 1. 加载基础模型 (强制本地)
    print(f"正在加载基础模型: {base_model_path}...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True, # 强制本地
        )
    except Exception as e:
        print(f"加载基础模型失败: {e}")
        return

    # 2. 加载Tokenizer (强制本地)
    # 优先从LoRA适配器路径加载，然后回退到基础模型路径
    tokenizer_path_to_try = lora_adapter_path
    print(f"正在加载Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_to_try, 
            trust_remote_code=True, 
            local_files_only=True # 强制本地
        )
        print(f"从 {tokenizer_path_to_try} 加载Tokenizer成功。")
    except Exception:
        print(f"无法从 {tokenizer_path_to_try} 加载Tokenizer，尝试从基础模型路径 {base_model_path} 加载。")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, 
                trust_remote_code=True, 
                local_files_only=True # 强制本地
            )
            print(f"从 {base_model_path} 加载Tokenizer成功。")
        except Exception as e_base_tok:
            print(f"无法从基础模型路径加载Tokenizer: {e_base_tok}")
            return
            
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 调整词嵌入（如果需要）
    if len(tokenizer) > base_model.config.vocab_size:
        print(f"调整Token Embeddings大小...")
        base_model.resize_token_embeddings(len(tokenizer))

    # 4. 加载LoRA适配器
    print(f"正在加载LoRA适配器: {lora_adapter_path}...")
    try:
        model = PeftModel.from_pretrained(base_model, lora_adapter_path, local_files_only=True)
    except Exception as e:
        print(f"加载LoRA适配器失败: {e}")
        return

    # 5. 合并权重
    print("正在合并LoRA权重...")
    try:
        model = model.merge_and_unload()
    except Exception as e:
        print(f"合并LoRA权重失败: {e}")
        return

    # 6. 保存模型和Tokenizer
    print(f"正在保存合并后的模型到: {output_path}...")
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"模型和Tokenizer已保存到 {output_path}")
    except Exception as e:
        print(f"保存失败: {e}")
        return
    
    print("合并过程完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将本地LoRA适配器权重合并到本地基础模型中并保存。")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True,
        help="本地基础模型的路径 (例如 ./Qwen3-1.7B)。"
    )
    parser.add_argument(
        "--lora_adapter_path", 
        type=str, 
        required=True,
        help="本地训练好的LoRA适配器模型的路径 (例如 ./qwen3_haruhi_lora/final_lora_adapter)。"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="保存合并后模型的目录路径 (例如 ./merged_qwen3_haruhi)。"
    )
    args = parser.parse_args()

    if not os.path.exists(args.base_model_path):
        print(f"错误: 基础模型路径 '{args.base_model_path}' 不存在。")
        exit(1)
    if not os.path.exists(args.lora_adapter_path):
        print(f"错误: LoRA适配器路径 '{args.lora_adapter_path}' 不存在。")
        exit(1)
        
    merge_lora_to_base_model(
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        output_path=args.output_path
    )
