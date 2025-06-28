#!/bin/bash

# set up python environment
conda activate vstream

# set important configurations
ngpus=8
gputype=A100

# auto calculate configurations
gpus_list=$(seq -s, 0 $((ngpus - 1)))
date_device="$(date +%m%d)_${ngpus}${gputype}"

echo start eval
# define your openai info here

for dataset in actnet nextoe msvd msrvtt vsmovienet vsego4d realtime_vsmovienet realtime_vsego4d
do
    echo start eval ${dataset}
    python -m flash_vstream.eval_video.eval_any_dataset_features \
        --model-path your_model_checkpoint_path \
        --dataset ${dataset} \
        --num_chunks $ngpus \
        --api_key $OPENAIKEY \
        --api_base $OPENAIBASE \
        --api_type $OPENAITYPE \
        --api_version $OPENAIVERSION \
        --test \
        >> ${date_device}_vstream-7b-eval-${dataset}.log 2>&1 
done

