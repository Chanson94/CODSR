CUDA_VISIBLE_DEVICES=0 python test_inference_time.py \
    --warmup_iterations 5 \
    --inference_iterations 10 \
    --process_size 1024 \
    --pretrained_model_name_or_path /preset/models/SD21Base \
    --ram_ft_path preset/models/DAPE.pth  \
    --ram_path preset/models/ram_swin_large_14m.pth  \
    --codsr_path preset/models/codsr.pkl
