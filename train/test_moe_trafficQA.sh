export CUDA_VISIBLE_DEVICES=2
python infer_qwen_vl_lora_v4.py \
  --base_model_path ./data/model/Qwen2.5-VL-3B-Instruct \
  --lora_path ./test/checkpoint-1700 \
  --data_path ./data/test/test_trafficQA/trafficQA.jsonl \
  --image_root ./data/test/test_trafficQA  \
  --output_path ./checkpoint-1700_trafficQA.jsonl \
  --prompt_str "" \
  --max_new_tokens 12228 \
  --num_experts 8 --top_k 1 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --router_aux_weight 0.01
