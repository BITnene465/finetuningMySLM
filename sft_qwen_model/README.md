# Qwen 模型微调 实现角色扮演

通过微调 Qwen3 系列模型，实现富有个性且有固定风格的 chatbot

## 主要流程

###  下载 Qwen 模型
    首先，克隆 Qwen 模型仓库到本地。 （保证 git lfs 已经下载）
    ```bash
    git clone https://www.modelscope.cn/Qwen/Qwen3-1.7B.git
    ```
    根据需求可以选择更大的模型，qwen3系列均可。

### 在 lccc 上训练
用于提升模型的多轮对话能力和拟人化能力。
首先将 lccc 数据集转换为脚本能够处理的 sharegpt 格式，jsonl文件
运行以下脚本（会自动下载 lccc-base 数据集（需要梯子））
```bash
# 提取 训练集样本
python convert_lccc_to_sharegpt.py --dataset_name lccc-base --split train --seed 114514 --max_samples 40000 --output_file lccc_base_train_sample_40k.jsonl

# 提取验证集
python convert_lccc_to_sharegpt.py --dataset_name lccc-base --split validation --output_file lccc_base_validation.jsonl
```

###  **数据预处理:**
    使用 `extract_dailogue.ipynb` 脚本从原始数据集中提取对话。打开并运行此 Jupyter Notebook 中的所有单元格以生成处理后的数据。

    `split_tr_dev.py`: 用于切分训练集和开发集。
    在脚本里面更改参数

    使用 `lora_torch.py` 脚本对 Qwen 模型进行 LoRA 微调。
    ```bash
    python lora_torch.py 
    ```
    请根据 `lora_torch.py` 脚本的实际需求**填写必要的参数**。
    如果要微调其他模型可以更改脚本的中代码，Qwen3 系列均可行。

###  **(可选) 合并 LoRA权重:**
    如果需要将 LoRA 权重合并到基础模型中，可以使用 `merge_lora_weights.py` 脚本。
    ```bash
    python merge_lora_weights.py --base_model_path ./Qwen3-1.7B --lora_model_path  ./runs/qwen3_haruhi_lora/final_lora_adapter --output_path merged_qwen3_haruhi
    ```

### **(可选) 推理:**
    使用 `inference_hf_gr.py` 脚本进行推理，该脚本可能带有一个 Gradio 界面。
    ```bash
    python inference_hf_gr.py --base_model_path ./Qwen3-1.7B --lora_model_path runs/qwen3_haruhi_lora/final_lora_adapter
    ```

## 其他脚本

-   `old/` 文件夹: 包含一些旧版本的脚本，例如 `inference_with_us_gr.py` 和 `lora_us.py`。这些脚本可能用于对比或参考，但不是当前推荐的主要流程。

## 目录结构说明

-   `ChatHaruhi-54K-Role-Playing-Dialogue/`: 存放原始 ChatHaruhi 数据集。
-   `extract_data/`: 存放从原始数据集中提取处理后的数据。
-   `Qwen3-1.7B/`: 存放下载的 Qwen 模型文件。
-   `runs/`: 存放微调过程中产生的日志和模型检查点。
-   `sharegpt_data/`: 存放 ShareGPT 格式的数据。

## 注意事项

-   请确保已安装所有必要的 Python 依赖库。(大多数常见库，pip 即可)
-   根据你的硬件配置调整微调脚本中的参数，例如批处理大小、学习率等。