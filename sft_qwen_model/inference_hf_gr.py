#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用微调后的Qwen LoRA模型 (Hugging Face PEFT) 进行推理
"""

import os
import json
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# 加载模型
def load_hf_lora_model(base_model_path, lora_adapter_path=None, load_lora=True):
    """加载微调后的LoRA模型或仅基础模型"""
    print(f"正在加载基础模型: {base_model_path}...")
    # For Qwen models, trust_remote_code=True is often needed.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"正在加载Tokenizer...")
    # Load tokenizer from the LoRA adapter path if available, as it would contain any modifications
    # (e.g., added tokens) made during the finetuning setup.
    # Fallback to base model path if not present in adapter directory.
    try:
        print(f"尝试从LoRA适配器路径加载Tokenizer: {lora_adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True, local_files_only=True)
    except Exception as e: 
        print(f"无法从LoRA适配器路径加载Tokenizer ({e}), 尝试从基础模型路径加载: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)

    # Ensure pad_token is set in the tokenizer.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            print(f"Tokenizer的pad_token未设置，将其设置为eos_token: {tokenizer.eos_token}")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("警告: Tokenizer既没有pad_token也没有eos_token。推理时可能出现问题。")
            # For Qwen, eos_token is typically <|endoftext|> (ID 151643).

    # Resize token embeddings if the tokenizer's vocabulary size is larger than the model's.
    if len(tokenizer) > base_model.config.vocab_size:
        print(f"基础模型的词汇表大小 ({base_model.config.vocab_size}) 小于Tokenizer的词汇表大小 ({len(tokenizer)})。")
        print("正在调整基础模型的token embeddings大小...")
        base_model.resize_token_embeddings(len(tokenizer))

    if load_lora:
        if not lora_adapter_path:
            raise ValueError("load_lora为True时，必须提供lora_adapter_path。")
        print(f"正在加载LoRA适配器: {lora_adapter_path}...")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        print("LoRA适配器加载完成。")
    else:
        print("选择不加载LoRA适配器，使用基础模型。")
        model = base_model
    
    # Optional: Merge LoRA weights for potentially faster inference if LoRA was loaded.
    # if load_lora:
    #     print("Merging LoRA adapter weights into the base model...")
    #     model = model.merge_and_unload()

    model.eval()  # Set the model to evaluation mode
    print("模型和Tokenizer准备完成。")
    return model, tokenizer

# 生成回复
def generate_response(model, tokenizer, user_input, chat_history=None, system_prompt=None):
    """生成模型回复"""
    if chat_history is None:
        chat_history = []
    
    messages = []
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}]
    
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": user_input})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True, 
        enable_thinking=False,  # important!! disable thinking mode(no train for this)
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.8,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            generation_config=generation_config,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # Parsing thinking content (specific to Qwen's <think>...</think> format)
    try:
        think_token_sequence = tokenizer.encode("</think>", add_special_tokens=False)
        think_token_id_value = think_token_sequence[-1] if think_token_sequence else 151668 # Fallback to known ID

        if think_token_id_value in output_ids:
            # Find the last occurrence of '</think>'
            index = len(output_ids) - output_ids[::-1].index(think_token_id_value)
        else:
            index = 0 
            
    except ValueError: 
        index = 0
        
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    return content

# 创建Gradio界面
def create_gradio_demo(base_model_path, lora_adapter_path=None, load_lora=True):
    """创建Gradio演示界面"""
    model, tokenizer = load_hf_lora_model(base_model_path, lora_adapter_path, load_lora)
    system_prompt = ""
    
    def respond(message, chat_history_list):
        # chat_history_list is passed by gr.ChatInterface, format: [["user1", "bot1"], ["user2", "bot2"]]
        bot_message = generate_response(model, tokenizer, message, chat_history_list, system_prompt)
        return bot_message
    
    demo = gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=600, label="聊天框"),
        title=" AI - HF LoRA (Qwen 微调)",
        description="",
        examples=[
            "你是谁？", 
            "你喜欢什么？",
            "你能告诉我关于SOS团的事吗？",
            "你能帮我解决一个难题吗？",
        ],
        cache_examples=False,
        theme="soft",
        submit_btn="发送",
    )
    
    return demo

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用微调后的Qwen LoRA模型 (Hugging Face PEFT) 进行推理")
    parser.add_argument("--base_model_path", type=str, default="./Qwen3-1.7B", 
                        help="基础模型的路径或Hugging Face Hub ID (例如 ./Qwen3-1.7B 或 Qwen/Qwen1.5-1.8B-Chat)")
    parser.add_argument("--lora_adapter_path", type=str, default=None, 
                        help="LoRA适配器模型的路径 (例如 ./qwen3_haruhi_lora/final_lora_adapter)。如果提供了此参数且 --no_lora 未设置，则加载LoRA。")
    parser.add_argument("--no_lora", action="store_true", help="如果设置此项，则不加载LoRA适配器，即使提供了lora_adapter_path。")
    parser.add_argument("--share", action="store_true", help="是否共享Gradio链接")
    
    args = parser.parse_args()
    
    load_lora_flag = True
    if args.no_lora:
        load_lora_flag = False
        print("参数 --no_lora 已设置，将不会加载LoRA适配器。")
    elif not args.lora_adapter_path:
        load_lora_flag = False
        print("未提供 --lora_adapter_path，将不会加载LoRA适配器。")

    # Check if local base_model_path exists if it looks like a path
    is_local_path = any(s in args.base_model_path for s in ['/', '.'])
    if is_local_path and not os.path.exists(args.base_model_path):
        print(f"错误: 基础模型路径 {args.base_model_path} 不存在。请确保路径正确。")
        exit(1)
    elif not is_local_path:
        print(f"提示: 基础模型路径 '{args.base_model_path}' 看起来不像本地路径，将尝试从Hugging Face Hub加载。")

    if load_lora_flag and not os.path.exists(args.lora_adapter_path):
        print(f"错误: LoRA适配器路径 {args.lora_adapter_path} 不存在。请确保路径正确，并且LoRA模型已训练并保存。")
        exit(1)
    
    print("正在启动Gradio界面...")
    demo = create_gradio_demo(args.base_model_path, args.lora_adapter_path, load_lora_flag)
    demo.launch(share=args.share)
