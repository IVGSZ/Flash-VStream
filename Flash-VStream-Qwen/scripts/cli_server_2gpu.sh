conda activate vstream

suffix=model_7b_test30
python cli_server_2gpu.py \
    --model-path output/best_ckpt \
    --video-file data/eval_video/videomme/frames/goyWFUzCqF4 \
    --log-file server_cli_${suffix}.log 

