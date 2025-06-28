#!/bin/bash

# set up python environment
conda activate vstream

# set important configurations
type=weighted_kmeans
suffix=STAR
cur_length=1
cur_size=8
long_length=25
long_size=4
Turing_length=25
Turing_size=1
ngpus=8
gputype=A100

# auto calculate configurations
gpus_list=$(seq -s, 0 $((ngpus - 1)))
date_device="$(date +%m%d)_${ngpus}${gputype}"

echo start pretrain
deepspeed --master_addr 127.0.0.1 --master_port 12345 --include localhost:${gpus_list} flash_vstream/train/train_mem.py \
    --deepspeed ./scripts/zero0.json \
    --model_name_or_path ./ckpt/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./data/pretrain/llava_558k_with_webvid.json \
    --image_folder ./data/pretrain/image_features \
    --video_folder ./data/pretrain/video_features \
    --vision_tower ./ckpt/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --compress_type mean \
    --compress_size ${cur_size} \
    --compress_long_memory_size ${long_size} \
    --compress_Turing_memory_size ${Turing_size} \
    --video_max_frames $((cur_length + long_length)) \
    --video_current_memory_length ${cur_length} \
    --video_long_memory_length ${long_length} \
    --video_Turing_memory_length ${Turing_length} \
    --video_sample_type ${type} \
    --group_by_modality_length \
    --bf16 \
    --output_dir ./checkpoints-pretrain/vstream-7b-pretrain-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    >> ${date_device}_vstream-7b-pretrain-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix}.log 2>&1 

echo start finetune
deepspeed --master_addr 127.0.2.1 --master_port 12345 --include localhost:${gpus_list} flash_vstream/train/train_mem.py \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path ./checkpoints-pretrain/vstream-7b-pretrain-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix}/checkpoint-3000 \
    --version v1 \
    --data_path ./data/finetune/llava_v1_5_mix665k_with_video_chatgpt.json \
    --image_folder ./data/finetune/image_features \
    --video_folder ./data/finetune/video_features \
    --vision_tower ./ckpt/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --video_fps 1 \
    --compress_type mean \
    --compress_size ${cur_size} \
    --compress_long_memory_size ${long_size} \
    --compress_Turing_memory_size ${Turing_size} \
    --video_max_frames $((cur_length + long_length)) \
    --video_current_memory_length ${cur_length} \
    --video_long_memory_length ${long_length} \
    --video_Turing_memory_length ${Turing_length} \
    --video_sample_type ${type} \
    --group_by_modality_length \
    --bf16 \
    --output_dir ./checkpoints-finetune/vstream-7b-finetune-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    >> ${date_device}_vstream-7b-finetune-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix}.log 2>&1 


echo start eval
# define your openai info here

for dataset in actnet nextoe msvd msrvtt vsmovienet vsego4d realtime_vsmovienet realtime_vsego4d
do
    echo start eval ${dataset}
    python -m flash_vstream.eval_video.eval_any_dataset_features \
        --model-path checkpoints-finetune/vstream-7b-finetune-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix}/checkpoint-5900 \
        --dataset ${dataset} \
        --num_chunks $ngpus \
        --api_key $OPENAIKEY \
        --api_base $OPENAIBASE \
        --api_type $OPENAITYPE \
        --api_version $OPENAIVERSION \
        >> ${date_device}_vstream-7b-eval-${dataset}-${type}${cur_length}*${cur_size}-${long_length}*${long_size}-${Turing_length}*${Turing_size}-${suffix}.log 2>&1 
done
