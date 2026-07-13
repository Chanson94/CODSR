# Stage1: LQFM+PGPA
accelerate launch --config_file='scripts/default_config.yaml' --main_process_port 29900 train_codsr_stage1.py \
    --pretrained_model_name_or_path="preset/models/SD21Base" \
    --ram_path="preset/models/ram_swin_large_14m.pth" \
    --learning_rate=5e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps 2000 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --output_dir=experience/CODSR_Stage1 \
    --dataset_txt_paths_list_val 'scripts/valdata.txt','scripts/valdata.txt' \
    --dataset_prob_paths_list_val 1,1 \
    --lora_rank_vae=4 \
    --lora_rank_unet=16 \
    --deg_file_path="params_realesrgan.yml" \
    --tracker_project_name "train_osediff" \
    --warm_up_steps=3000 \
    --max_train_steps 14000