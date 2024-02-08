output_model= #finetune_lora_weight
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
deepspeed --include localhost:1 --master_port 29001 finetune-lora.py \
    --model_name_or_path #path of llama2 \
    --tokenizer_name #path of llama2 \
    --train_files #lora_training_jsons \
    --validation_files  #lora_validation_jsons \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 10 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --warmup_steps 400 \
    --load_in_bits 8 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --report_to tensorboard \
    --ignore_data_skip true \
    --gradient_checkpointing \
    --ddp_timeout 18000000
