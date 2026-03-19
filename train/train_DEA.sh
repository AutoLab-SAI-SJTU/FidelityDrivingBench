export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export DEEPSPEED_COMM_BACKEND=nccl
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

torchrun --nproc_per_node=2 \
  --master_addr 127.0.0.1 \
  --master_port 33229 \
  train_qwen_moe_lora_v4_flash.py \
  --model_name_or_path /home/ma-user/work/model/Qwen2.5-VL-3B-Instruct \
  --data_path /home/ma-user/modelarts/user-job-dir/A100/FidelityAD/train/train_annotations.jsonl \
  --image_root /home/ma-user/work/dataset/FidelityAD/train \
  --output_dir /home/ma-user/modelarts/user-job-dir/A100/FidelityAD/output2/ \
  --seq_len 4496 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 200 \
  --optim adamw_torch \
  --gradient_checkpointing \
  --deepspeed ds_config.json \
  --num_experts 8 \
  --top_k 1 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --prompt_mixture \
  --prompt_bank_size 4 \
  --prompt_len 8 \
  --prompt_gate_hidden 512 \
  --prompt_top_k 1 \
  # --gradient_accumulation_steps 3\
  # --use_qwen_image_processor \ # 开启后不进行resize