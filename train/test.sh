export CUDA_VISIBLE_DEVICES=1
python infer_qwen_vl_lora_v4.py \
  --base_model_path /home/ma-user/work/model/Qwen2.5-VL-3B-Instruct \
  --lora_path /home/ma-user/modelarts/user-job-dir/A100/FidelityAD/output_lora_pa/checkpoint-14088 \
  --data_path /home/ma-user/work/dataset/FidelityAD/test_annotations_output.jsonl \
  --image_root /home/ma-user/work/dataset/FidelityAD/test  \
  --output_path /home/ma-user/modelarts/user-job-dir/A100/FidelityAD/output_pa_jsonl/qwen2.5-VL-3B_lora_pa_ckpt14088_scene_description.jsonl \
  --prompt_str "<image>
  Suppose you are driving, and I'm providing you with the image captured by the car's front camera, generate a description of the driving scene in detail." \
  --max_new_tokens 12228 \
  --num_experts 8 --top_k 1 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --router_aux_weight 0.01
