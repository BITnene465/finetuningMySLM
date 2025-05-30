CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ./Qwen3-1.7B \
    --train_type lora \
    # --dataset xxx
    --custom_train_dataset ./qwen_data/haruhi_train.json \
    --custom_eval_dataset ./qwen_data/haruhi_eval.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author Bitnene465 \
    --output_dir ms_runs/qwen3-1.7b-lora-haruhi \
    --model_name qwen3-1.7b-lora-haruhi \