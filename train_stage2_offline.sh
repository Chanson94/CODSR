export TOKENIZERS_PARALLELISM=false
# Stage2 offline: LQFM+PGPA+TMG with precomputed SAM2 masks
accelerate launch --config_file='scripts/default_config.yaml' --main_process_port 29800 train_codsr_stage2_offline.py \
    --pretrained_model_name_or_path="preset/models/SD21Base" \
    --ram_path="preset/models/ram_swin_large_14m.pth" \
    --learning_rate=5e-5 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --enable_xformers_memory_efficient_attention --checkpointing_steps 1000 \
    --mixed_precision='fp16' \
    --report_to "tensorboard" \
    --seed 123 \
    --output_dir=experience/CODSR_Stage2_offline \
    --dataset_txt_paths_list_val 'scripts/valdata.txt','scripts/valdata.txt' \
    --dataset_prob_paths_list_val 1,1 \
    --deg_file_path="params_realesrgan.yml" \
    --tracker_project_name "train_osediff" \
    --resume_path="experience/CODSR_Stage1/checkpoints/model_14000.pkl" \
    --mask_root="/home/notebook/data/group/ch/Datasets/High_quality_training_data/stage2_offline_masks" \
    --max_train_steps 4000
