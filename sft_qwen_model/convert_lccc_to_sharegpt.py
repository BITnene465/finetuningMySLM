'''
将LCCC数据集从Hugging Face下载并转换为ShareGPT格式 (JSONL) 的脚本。
'''
import json
import argparse # 导入 argparse
from datasets import load_dataset, Value, Sequence # 导入 Sequence

def format_conversation(idx, dialogue_turns):
    """将单个LCCC对话格式化为ShareGPT结构，不含system prompt，并去除多余空格。"""
    conversation = []
    for i, turn in enumerate(dialogue_turns):
        role = "human" if i % 2 == 0 else "gpt"
        # 去除value中的所有空格
        processed_turn = turn.replace(" ", "")
        conversation.append({"from": role, "value": processed_turn})
    
    return {"id": f"lccc_data_{idx}", "conversations": conversation} # idx 从0开始

def convert_lccc_to_jsonl(dataset_name="lccc-base", split="train", output_file="lccc_sharegpt.jsonl", 
                          max_samples=None, seed=42, shuffle_all_if_no_sampling=True):
    """
    下载/加载LCCC数据集，将其转换为ShareGPT JSONL格式并保存。
    """
    print(f"正在加载数据集 {dataset_name}, 切分 {split}...")
    try:
        hf_config_name = dataset_name
        if dataset_name == "lccc-base":
            hf_config_name = "base"
        elif dataset_name == "lccc-large":
            hf_config_name = "large"
            
        dataset = load_dataset("silver/lccc", hf_config_name, split=split)
        
        num_total_samples = len(dataset)
        processed_dataset = dataset 

        if max_samples is not None and max_samples > 0 and max_samples < num_total_samples:
            print(f"数据集包含 {num_total_samples} 个样本，将从中随机采样 {max_samples} 个样本 (seed={seed})。")
            processed_dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        else: 
            if max_samples is not None and max_samples > 0:
                print(f"请求的样本数 ({max_samples}) 大于或等于数据集中的总样本数 ({num_total_samples})。")
            print(f"将处理所有 {num_total_samples} 个样本。")
            if shuffle_all_if_no_sampling:
                 print(f"将打乱所有样本的顺序 (seed={seed})。")
                 processed_dataset = dataset.shuffle(seed=seed)
            else:
                 print("将按原始顺序处理所有样本。")
        
        print(f"最终将处理 {len(processed_dataset)} 个样本。")

        dialogue_column = None
        if "conversation" in dataset.features:
            dialogue_column = "conversation"
        elif "dialog" in dataset.features:
            dialogue_column = "dialog"
        elif "text" in dataset.features: 
            dialogue_column = "text"
        else:
            for col_name, feature_type in dataset.features.items():
                if isinstance(feature_type, Sequence) and hasattr(feature_type, 'feature') and isinstance(feature_type.feature, Value) and feature_type.feature.dtype == 'string':
                    if len(dataset) > 0 and isinstance(dataset[0][col_name], list):
                        dialogue_column = col_name
                        break
            if not dialogue_column:
                 print(f"无法自动确定对话列。数据集特征: {dataset.features}")
                 print("请检查数据集结构并相应地修改脚本。")
                 return

        print(f"使用 '{dialogue_column}' 作为对话轮次的来源。")

    except Exception as e:
        print(f"加载或预处理数据集 {dataset_name} 时出错: {e}")
        return

    print(f"正在转换并写入到 {output_file}...")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(processed_dataset):
            dialogue_turns = example[dialogue_column]
            if not isinstance(dialogue_turns, list) or not all(isinstance(turn, str) for turn in dialogue_turns):
                print(f"由于 '{dialogue_column}' 中的格式非预期，跳过条目 {idx}。预期为字符串列表。")
                print(f"内容: {dialogue_turns}")
                continue

            if not dialogue_turns: 
                continue
                
            formatted_entry = format_conversation(idx, dialogue_turns)
            f.write(json.dumps(formatted_entry, ensure_ascii=False) + '\n')
            count += 1
            if count % 1000 == 0:
                print(f"已处理 {count} 个样本...")
    
    print(f"成功将 {count} 个样本转换为 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将LCCC数据集从Hugging Face下载并转换为ShareGPT JSONL格式。")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lccc-base",
        choices=["lccc-base", "lccc-large"],
        help="要使用的LCCC数据集版本 (默认: lccc-base)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="要使用的数据集切分 (默认: train)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出JSONL文件的路径。如果未提供，将根据dataset_name自动生成。"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="要随机采样的最大样本数。如果未提供、为0或负数，则处理所有样本。(默认: None，即全部样本)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="用于随机采样和打乱的随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--no_shuffle_when_all",
        action="store_true",
        help="如果处理所有样本（即未通过max_samples进行子采样），则不打乱数据集顺序。"
    )

    args = parser.parse_args()

    output_filename = args.output_file
    if output_filename is None:
        output_filename = f"{args.dataset_name.replace('-', '_')}_sharegpt.jsonl"

    actual_max_samples = args.max_samples
    if args.max_samples is not None and args.max_samples <= 0:
        print("max_samples被设置为0或负数，将处理所有样本。")
        actual_max_samples = None
    
    shuffle_all_if_no_sampling_param = not args.no_shuffle_when_all

    print(f"命令参数:")
    print(f"  数据集名称: {args.dataset_name}")
    print(f"  数据集切分: {args.split}")
    print(f"  输出文件: {output_filename}")
    print(f"  最大样本数: {'所有' if actual_max_samples is None else actual_max_samples}")
    print(f"  随机种子: {args.seed}")
    print(f"  处理所有样本时是否打乱: {not args.no_shuffle_when_all}")
    print("-" * 30)

    convert_lccc_to_jsonl(
        dataset_name=args.dataset_name,
        split=args.split,
        output_file=output_filename,
        max_samples=actual_max_samples,
        seed=args.seed,
        shuffle_all_if_no_sampling=shuffle_all_if_no_sampling_param
    )
    print("转换过程完成。")


