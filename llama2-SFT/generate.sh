CUDA_VISIBLE_DEVICES=0 python generate_b4_ckpt.py \
    --base_model #llama_path \
    --lora_weights #finetune_lora_weight \
    --load_8bit \
    --saveI 1 \
    --from_json '../DG_Final/GM/test_similar_users.json' \
    --save_path #output_path 