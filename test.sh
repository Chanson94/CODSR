### DRealSR
CUDA_VISIBLE_DEVICES=0 python test_codsr.py \
    --i preset/datasets/benchmark_drealsr/test_LR \
    --output_dir preset/datasets/benchmark_drealsr/result_codsr \
    --codsr_path preset/models/codsr.pkl \
    --pretrained_model_name_or_path preset/models/SD21Base \
    --ram_ft_path preset/models/DAPE.pth \
    --ram_path preset/models/ram_swin_large_14m.pth \
    --upscale=4

### RealSR
# CUDA_VISIBLE_DEVICES=1 python test_codsr.py \
#     --i preset/datasets/benchmark_realsr/test_LR \
#     --output_dir preset/datasets/benchmark_realsr/result_codsr \
#     --codsr_path preset/models/codsr.pkl \
#     --pretrained_model_name_or_path preset/models/SD21Base \
#     --ram_ft_path preset/models/DAPE.pth \
#     --ram_path preset/models/ram_swin_large_14m.pth \
#     --upscale=4

### RealPhoto60
# CUDA_VISIBLE_DEVICES=2 python test_codsr.py \
#     --i preset/datasets/benchmark_realphoto60/test_LR \
#     --output_dir preset/datasets/benchmark_realphoto60/result_codsr \
#     --codsr_path preset/models/codsr.pkl \
#     --pretrained_model_name_or_path preset/models/SD21Base \
#     --ram_ft_path preset/models/DAPE.pth \
#     --ram_path preset/models/ram_swin_large_14m.pth \
#     --upscale=2