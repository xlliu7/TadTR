# Run on two GPUs in non-distributed mode (more convenient)
CUDA_VISIBLE_DEVICES=0,1 python -u main.py --cfg "CFG_PATH" --multi_gpu

# Run on two GPUs in distributed mode (more powerful)
MASTER_PORT=29510
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 --master_port ${MASTER_PORT} --use_env main.py --cfg "CFG_PATH"
