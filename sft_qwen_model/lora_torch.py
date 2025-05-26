
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import os

# Configuration
MODEL_NAME = "Qwen3-1.7B" 
BASE_DIR = "."
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "sharegpt_data/haruhi_train.jsonl")
DEV_DATA_PATH = os.path.join(BASE_DIR, "sharegpt_data/haruhi_dev.jsonl") 
OUTPUT_DIR = os.path.join(BASE_DIR, "qwen3_haruhi_lora")
SYSTEM_PROMPT = None  # dataset has its own system prompt !!!

# dataset class -- sharegpt 2 qwen format
class ShareGPTDataset(Dataset):
    def __init__(self, data_path, tokenizer, system_prompt=None, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.data = []
        
        print(f"Loading data from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_num+1} in {data_path}: {e}")
                    continue
        print(f"Loaded {len(self.data)} examples.")

        # IMPORTANT: The apply_chat_template might add a generation prompt or other tokens
        # if not handled carefully. For masking, we are interested in the raw tokens
        # Let's get the tokens for "<|im_start|>assistant\n"
        # This is the sequence that *precedes* the actual assistant content we want to train on.
        assistant_role_prompt_str = "<|im_start|>assistant\n"
        self.assistant_role_prompt_tokens = tuple(self.tokenizer.encode(assistant_role_prompt_str, add_special_tokens=False))
        
        self.im_end_tokens = tuple(self.tokenizer.encode("<|im_end|>", add_special_tokens=False))

        if not self.assistant_role_prompt_tokens:
            print("Warning: assistant_role_prompt_tokens is empty. Masking logic might fail.")
        else:
            print(f"assistant_role_prompt_tokens: {self.assistant_role_prompt_tokens} -> {self.tokenizer.convert_ids_to_tokens(list(self.assistant_role_prompt_tokens))}")
        if not self.im_end_tokens:
            print("Warning: im_end_tokens is empty. Masking logic might fail.")
        else:
            print(f"im_end_tokens: {self.im_end_tokens} -> {self.tokenizer.convert_ids_to_tokens(list(self.im_end_tokens))}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversations = item['conversations']
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        for turn in conversations:
            if turn['from'] == 'system':
                role = "system"
            elif turn['from'] == 'human':
                role = "user"
            elif turn['from'] == 'gpt':
                role = "assistant"
            else:
                raise ValueError(f"Unknown role in conversation: {turn['from']}")
            messages.append({"role": role, "content": turn['value']})

        # Tokenize the entire conversation using the chat template
        tokenized_output = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False, # Important for training data
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False 
        )
        
        input_ids = tokenized_output.squeeze(0) 
        input_ids_list = input_ids.cpu().tolist()
        labels = [-100] * len(input_ids_list)

        i = 0
        while i < len(input_ids_list):
            # Check for the start of an assistant's turn (i.e., "<|im_start|>assistant\n")
            if tuple(input_ids_list[i : i + len(self.assistant_role_prompt_tokens)]) == self.assistant_role_prompt_tokens:
                # We found the assistant's role prompt. Skip these tokens for loss calculation.
                i += len(self.assistant_role_prompt_tokens)
                # Now, unmask tokens until <|im_end|> is found
                while i < len(input_ids_list):
                    # Check if the current position starts with <|im_end|>
                    is_im_end_match = True
                    if i + len(self.im_end_tokens) <= len(input_ids_list): # Ensure we don't go out of bounds
                        for k in range(len(self.im_end_tokens)):
                            if input_ids_list[i+k] != self.im_end_tokens[k]:
                                is_im_end_match = False
                                break
                    else:
                        is_im_end_match = False # Not enough tokens left for a full match

                    if is_im_end_match:
                        # We found <|im_end|>. The tokens of <|im_end|> itself are part of the assistant's turn
                        # and should be predicted. So, unmask them.
                        for k in range(len(self.im_end_tokens)):
                            labels[i+k] = input_ids_list[i+k]
                        i += len(self.im_end_tokens) # Move past <|im_end|>
                        break # Exit inner loop (processing this assistant turn)
                    else:
                        # This is an assistant content token, unmask it.
                        labels[i] = input_ids_list[i]
                        i += 1
                continue # Continue outer loop to find next assistant turn or end of sequence
            i += 1 # Move to the next token if not an assistant role prompt
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids), # Collator will handle padding mask
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created TensorBoard log directory: {log_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True) 
    if tokenizer.pad_token is None:
        # Qwen3 typically uses <|endoftext|> as eos_token. Some models might not have pad_token set.
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"tokenizer.pad_token was None, set to tokenizer.eos_token ({tokenizer.eos_token})")
        else:
            raise ValueError("Tokenizer does not have a pad_token or eos_token set. Please check the tokenizer configuration.")

    # create Datasets
    print("Initializing training dataset...")
    train_dataset = ShareGPTDataset(TRAIN_DATA_PATH, tokenizer, SYSTEM_PROMPT)
    dev_dataset = ShareGPTDataset(DEV_DATA_PATH, tokenizer, SYSTEM_PROMPT) # Uncomment if you need evaluation

    if len(train_dataset) == 0:
        print("Training dataset is empty. Please check the data path and format.")
        return
    
    # Verify a sample from the dataset to check masking
    if len(train_dataset) > 0:
        print("\n--- Sample 1 from Dataset ---")
        sample = train_dataset[1]
        print(f"Input IDs: {sample['input_ids'].tolist()}")
        print(f"Labels:    {sample['labels'].tolist()}")
        print("Decoded tokens with labels (masked tokens shown as -100):")
        for token_id, label_id in zip(sample['input_ids'].tolist(), sample['labels'].tolist()):
            token_str = tokenizer.decode([token_id])
            if label_id != -100:
                print(f"  '{token_str}' (ID: {token_id}, Label: {label_id}) -> UNMASKED")
            else:
                print(f"  '{token_str}' (ID: {token_id}, Label: {label_id}) -> MASKED")
        print("--- End Sample ---")

    # Data Collator
    # padding="longest" is generally fine. If you have very disparate sequence lengths and are concerned
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, padding="longest")

    # Model
    print(f"Loading base model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", 
        local_files_only=True, 
        # trust_remote_code=True,  # 视情况使用
    )
    
    # If pad_token was added and tokenizer resized, model embeddings must be resized *before* LoRA
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
        print(f"Resizing token embeddings to include new pad_token. Old vocab size: {model.config.vocab_size}, new: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # After resizing, you might need to initialize the new embedding if it's not done automatically
        # with model.model.embed_tokens.weight.data[tokenizer.pad_token_id].normal_(mean=0.0, std=model.config.initializer_range)

    # LoRA Configuration
    # depending on the model architecture (maybe need to adjust)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32, 
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05, 
        bias="none", #  "none" or "all" or "lora_only".
        task_type=TaskType.CAUSAL_LM
    )
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training Arguments
    per_device_bs = 1
    grad_accum_steps = 4
    
    print(f"Using Per_device_batch_size: {per_device_bs}, Gradient_accumulation_steps: {grad_accum_steps}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03, # Warmup steps can stabilize training early on
        lr_scheduler_type="cosine", # Cosine learning rate schedule
        logging_dir=log_dir,
        logging_strategy="steps",
        logging_steps=5, 
        save_strategy="steps",
        save_steps=50, 
        eval_strategy="steps", # Uncomment if you have a dev_dataset 
        eval_steps=50,   # Uncomment if you have a dev_dataset 
        fp16=(model.dtype == torch.float16) and not torch.backends.mps.is_available(), # Use fp16 if available and not on MPS
        bf16=(model.dtype == torch.bfloat16),
        report_to="tensorboard",
        remove_unused_columns=False, # Important for custom __getitem__ with extra keys if any
        gradient_checkpointing= False # why? 用了就报错
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset, # Uncomment if using dev_dataset
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    try:
        trainer.train()
        print("Training finished successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        print(f"Model config: {model.config}")
        print(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
        print(f"Tokenizer eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
        raise

    # Save the LoRA adapter
    final_save_path = os.path.join(OUTPUT_DIR, "final_lora_adapter")
    print(f"Saving LoRA model to {final_save_path}")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path) 
    print("Model and tokenizer saved.")

if __name__ == "__main__":
    main()
