#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用微调后的Qwen模型进行推理
"""

import os
import json
import torch
import gradio as gr
from unsloth import FastLanguageModel
from transformers import GenerationConfig

# 加载模型
def load_model(model_path):
    """加载微调后的模型"""
    print(f"正在加载模型: {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        device_map="auto",
        # load_in_4bit=True  # 4-bit量化加载，节省内存
    )
    
    return model, tokenizer

# 生成回复
def generate_response(model, tokenizer, user_input, chat_history=None, system_prompt=None):
    """生成模型回复"""
    if chat_history is None:
        chat_history = []
    
    # 添加系统提示
    messages = []
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史对话
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # 添加当前用户输入
    messages.append({"role": "user", "content": user_input})
    
    # 使用聊天模板格式化输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking = False,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 设置生成配置
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        temperature=0.8,
        top_p=0.8,
        do_sample=True,
    )
    
    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            generation_config=generation_config,
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content

# 创建Gradio界面
def create_gradio_demo(model_path):
    """创建Gradio演示界面"""
    model, tokenizer = load_model(model_path)
    system_prompt = "你是来自《凉宫春日的忧郁》的凉宫春日。作为SOS团团长，你性格强势、自信、充满活力，对普通的事物毫无兴趣，只对外星人、未来人、异世界人和超能力者感兴趣。你坚信这些超自然现象的存在，并执着于寻找它们。你说话直接、有时傲慢，但也显示出你的魅力和领导能力。你总是充满想法和计划，喜欢指挥他人，特别是阿虚。"
    
    # 定义回调函数
    def respond(message, history):
        bot_message = generate_response(model, tokenizer, message, history, system_prompt)
        return bot_message
    
    # 创建Gradio界面
    demo = gr.ChatInterface(
        fn=respond,
        title="凉宫春日 AI - 基于Qwen3-1.7B微调模型",
        description="与《凉宫春日的忧郁》中的凉宫春日进行对话。她是SOS团团长，性格强势、自信、充满活力，只对超自然现象感兴趣。",
        examples=[
            "你是谁？", 
            "阿虚: 「春日，今天我们要做什么活动？」", 
            "阿虚: 「你真的相信有外星人吗？」",
            "朝比奈: 「春日，我觉得这个活动有点危险...」",
            "古泉: 「我们今天要去哪里探险？」"
        ],
        cache_examples=False,
        theme="soft"
    )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用微调后的Qwen模型进行推理")
    parser.add_argument("--model_path", type=str, default="./Qwen3-1.7B", 
                        help="微调后模型的路径")
    parser.add_argument("--share", action="store_true", help="是否共享Gradio链接")
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"警告: 模型路径 {args.model_path} 不存在")
        print("请确保模型路径正确，或者已完成模型微调")
        exit(1)
    
    demo = create_gradio_demo(args.model_path)
    demo.launch(share=args.share)
