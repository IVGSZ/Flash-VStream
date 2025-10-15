# [ICCV 2025] Flash-VStream: Efficient Real-Time Understanding for Long Video Streams

<a href="https://zhang9302002.github.io/">Haoji Zhang</a><sup>\*</sup>,
<a href="https://github.com/InvincibleWyq/">Yiqin Wang</a><sup>\*</sup>,
<a href="https://andytang15.github.io/">Yansong Tang</a><sup>&#9993;</sup>,
<a href="https://yongliu20.github.io/">Yong Liu</a>,
<a href="https://sites.google.com/site/jshfeng/home">Jiashi Feng</a>,
<a href="https://scholar.google.com.sg/citations?user=OEZ816YAAAAJ&hl=en">Xiaojie Jin</a><sup>&#9993;&dagger;</sup>

<sup>\*</sup>Equally contributing first authors, 
<sup>&#9993;</sup>Correspondence, 
<sup>&dagger;</sup>Project Leader

**Work done when interning at Bytedance.**

<a href="https://zhang9302002.github.io/vstream-iccv-page/"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href="http://arxiv.org/abs/2506.23825"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href="https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>

We proposed Flash-VStream, an efficient VLM with a novel Flash Memory mechanism that enables real-time understanding and Q&A of extremely long video streams. Our model achieves outstanding efficiency on EgoSchema, MLVU, LVBench, MVBench and Video-MME Benchmarks.


## Contents
- [Install](#install)
- [Model](#model)
- [Preparation](#preparation)
- [Train](#train)
- [Evaluation](#evaluation)
- [Real-time CLI Inference](#Real-time-CLI-Inference)


## Install
Please follow the instructions below to install the required packages.

1. Clone this repository and enter it
```bash
git clone git@github.com:IVGSZ/Flash-VStream.git
cd Flash-VStream/Flash-VStream-Qwen
```

2. Install package following `setup.sh`


## Model
We provide the checkpoint of Flash-VStream model here:

| Model | Weight | Initialized from LLM |
| --- | --- | --- |
| Flash-VStream-Qwen-7b | [Flash-VStream-Qwen-7b](https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b) | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |


## Preparation

1. Please download videos from following datasets.
2. Extract frames with at least 1 FPS. 
You can use `scripts/extract_frames.py` directly or use your own script.
(Make sure the name order is correct, e.g., `000001.jpg` instead of `1.jpg`)
3. Put the extracted frames under the target folder in `data`.

### Evaluation Data

| Dataset | Source | Frame folder |
| --- | --- | --- |
| EgoSchema | https://egoschema.github.io/ | data/eval_video/EgoSchema/frames |
| LVBench | https://github.com/zai-org/LVBench | data/eval_video/lvbench/frames |
| MLVU | https://github.com/JUNJIE99/MLVU | data/eval_video/mlvu/frames |
| MVBench | https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md | data/eval_video/mvbench/frames |
| Video-MME | https://github.com/MME-Benchmarks/Video-MME | data/eval_video/videomme/frames |

### Training Data

| Dataset | Source | Frame folder |
| --- | --- | --- |
| LLaVA-Video-178K | https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K | data/llava-video-178k/frames |

## Train
Flash-VStream is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`. If your GPUs have less than 80GB memory, you may try ZeRO-2 and ZeRO-3 stages.

Please make sure you download and organize the data following [Preparation](#preparation) before training.
The Qwen2-VL-7b checkpoint should be saved in `ckpt/Qwen2-VL-7B-Instruct`. 

If you want to train Flash-VStream from pretrained LLM and evaluate it, please run the following command:

```bash
bash scripts/train_and_eval.sh
```


## Evaluation
Please make sure you download and organize the data following [Preparation](#preparation) before evaluation.

If you want to evaluate a Flash-VStream model, please run the following command:

```bash
bash scripts/eval.sh
```


## Real-time CLI Inference
We provide a real-time CLI inference script, which simulates video stream input by reading frames of a video file at a fixed frame speed. You can ask any question and get the answer at any timestamp of the video stream. Run the following command and have a try:

```bash
bash scripts/cli_server_2gpu.sh
```

## Citation
If you find this project useful in your research, please consider citing:

```
@article{zhang2025flashvstream,
    title={Flash-VStream: Efficient Real-Time Understanding for Long Video Streams}, 
    author={Haoji Zhang and Yiqin Wang and Yansong Tang and Yong Liu and Jiashi Feng and Xiaojie Jin},
    journal={arXiv preprint arXiv:2506.23825},
    year={2025},
}
```