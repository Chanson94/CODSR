
CUDA_VISIBLE_DEVICES=0 python test_metrics.py \
    --inp_imgs preset/datasets/benchmark_drealsr/result_codsr \
    --gt_imgs preset/datasets/benchmark_drealsr/test_HR \
    --log preset/datasets/benchmark_drealsr/result_codsr/metrics

# CUDA_VISIBLE_DEVICES=1 python test_metrics.py \
#     --inp_imgs preset/datasets/benchmark_realsr/result_codsr \
#     --gt_imgs preset/datasets/benchmark_realsr/test_HR \
#     --log preset/datasets/benchmark_realsr/result_codsr/metrics
