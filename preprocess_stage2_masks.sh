# Offline mask preprocessing for stage2
python scripts/preprocess_stage2_masks.py \
    --ram_path="preset/models/ram_swin_large_14m.pth" \
    --sam2-checkpoint="seg_model/gsam2/checkpoints/sam2.1_hiera_large.pt" \
    --sam2-model-config="configs/sam2.1/sam2.1_hiera_l.yaml" \
    --mask_root="/home/notebook/data/group/ch/Datasets/High_quality_training_data/stage2_offline_masks" \
    --skip_existing
