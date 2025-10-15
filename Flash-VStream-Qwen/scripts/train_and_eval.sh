conda activate vstream

ngpus=8
pixel_expr=4*224*224
max_pixels=$((${pixel_expr}))
max_frames=240

DO_TRAIN=1
DO_EVAL=1

DISTRIBUTED_ARGS="
    --nproc_per_node ${ngpus} \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6002
"
DATE="$(date +%m%d)"

OUTPUT_BASE=output
OUTPUT_NAME=sft_lora_qwen2vl_7b
SAVE_PATH="${OUTPUT_BASE}/${OUTPUT_NAME}"
mkdir -p "$SAVE_PATH"

export NCCL_DEBUG=WARN
if [ $DO_TRAIN -eq 1 ]; then
    echo "start finetune, write to ${SAVE_PATH}"
    torchrun $DISTRIBUTED_ARGS finetune_flash.py \
        --model_name_or_path ./ckpt/Qwen2-VL-7B-Instruct \
        --data_path ./data/llava-video-178k/trainset_9k.json \
        --video_path ./data/llava-video-178k/frames \
        --use_flash_attn True \
        --bf16 True \
        --output_dir ${SAVE_PATH} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 10 \
        --save_total_limit 30 \
        --learning_rate 8e-4 \
        --weight_decay 0.1 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --model_max_length 14000 \
        --max_frames ${max_frames} \
        --max_pixels ${max_pixels} \
        --lazy_preprocess True \
        --use_lora \
        --lora_r 64 \
        --lora_alpha 32 \
        --gradient_checkpointing \
        --deepspeed deepspeed/zero2_config.json \
        >> "${SAVE_PATH}/lora_qwen2vl_7b_${suffix}.log" 2>&1
fi

if [ $DO_EVAL -eq 1 ]; then
    for dataset in egoschema mlvu lvbench mvbench videommewo videomme
    do
        echo start eval ${dataset}
        python3 eval_any_dataset.py \
            --model-path ${SAVE_PATH} \
            --dataset ${dataset} \
            --output_dir ${SAVE_PATH} \
            --evaluation_name evaluation \
            --num_chunks ${ngpus} \
            --max_frames ${max_frames} \
            --max_pixels ${max_pixels} \
            >> "${SAVE_PATH}/${DATE}_qwen2vl-7b-eval-${dataset}.log" 2>&1 
    done
fi
