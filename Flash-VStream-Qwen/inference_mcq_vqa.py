# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import re
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from models import (
    FlashVStreamQwen2VLModel,
    FlashVStreamQwen2VLConfig,
    FlashVStreamQwen2VLProcessor,
    DEFAULT_FLASH_MEMORY_CONFIG,
)
import warnings
from peft import AutoPeftModelForCausalLM  # depend on models.vstream_qwen2vl

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    # chunk_size = math.ceil(len(lst) / n)  # integer division
    # return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    res = [[] for i in range(n)]
    for i, x in enumerate(lst):
        res[i % n].append(x)
    return res

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def ranki_print(s):
    print(f'[cuda:{args.chunk_idx}] {s}')

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles

def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)

def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def get_subtitle(video_id, frame_num = 180):
    video_name = video_id
    SUBTITLE_PROMPT = "This video's subtitles are listed below: \n"
    QUESTION_PROMPT = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    data_path = os.path.join("./data/eval_video/videomme/Video-MME/data", video_name + ".mp4")
    subtitle_path = os.path.join("./data/eval_video/videomme/Video-MME/subtitle", video_name + ".srt")
    
    if os.path.exists(subtitle_path):  # Denote have subtitle
        print(f'try to open{subtitle_path}')
        subtitle = open(subtitle_path, encoding='utf-8').readlines()
    else:
        subtitle = ""
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        subtitle_by_frame, total_frame = extract_subtitles(data_path, subtitle_path)
        uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

        subtitle_by_frame_idx = []
        for frame_idx in uniform_sampled_frames:
            for idx, title in enumerate(subtitle_by_frame):
                if frame_idx < title[1] and frame_idx >= title[0]:
                    subtitle_by_frame_idx.append(idx)
        subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

        textlist = []
        for idx in subtitle_by_frame_idx:
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
            try:
                textlist.append(raw_text[0])
            except:
                continue
        subtitle = "\n".join(textlist)
    return subtitle

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    use_flash_attn = True
    qwen_path = 'ckpt/Qwen2-VL-7B-Instruct'
    if args.lora_path:
        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            qwen_path,
            trust_remote_code=True,
        )
        if args.flash_memory_dict is not None:
            model_config.vision_config.flash_memory_config = args.flash_memory_dict
            print(f'Override model config to {model_config}')
        if getattr(model_config.vision_config, 'flash_memory_config', None) is None:
            warnings.warn(f'Qwen2VLVisionConfig.flash_memory_config is not set. Set it to default, sample 10000')
            model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
        # use lora
        lora_path = args.lora_path
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path, 
            config=model_config,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        ).eval()
    else:
        # use full model
        model_path = args.model_path
        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if args.flash_memory_dict is not None:
            model_config.vision_config.flash_memory_config = args.flash_memory_dict
            print(f'Override model config to {model_config}')
        if getattr(model_config.vision_config, 'flash_memory_config', None) is None:
            warnings.warn(f'Qwen2VLVisionConfig.flash_memory_config is not set. Set it to default, sample 10000')
            model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
        model = FlashVStreamQwen2VLModel.from_pretrained(
            model_path, 
            config=model_config,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        ).eval()

    processor = FlashVStreamQwen2VLProcessor.from_pretrained(qwen_path)
    if args.flash_memory_dict is not None:
        flash_memory_config = args.flash_memory_dict
        ranki_print(f"Load processor success!, using new processor={flash_memory_config}, instead of training time={model.config.vision_config.flash_memory_config}")
    else:
        flash_memory_config = model.config.vision_config.flash_memory_config
        ranki_print(f"Load processor success!, processor with flash_memory_config={flash_memory_config}")
    for k, v in DEFAULT_FLASH_MEMORY_CONFIG.items():
        if k not in flash_memory_config:
            flash_memory_config[k] = v

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            ranki_print(f'mkdir Except: {e}')
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            id_set = [json.loads(row)['id'] for row in f.readlines()]
            id_set = set(id_set)
            gt_questions = [sample for sample in gt_questions if sample['id'] not in id_set]
    ans_file = open(answers_file, "a")

    for sample in tqdm(gt_questions, desc=f"cuda:{args.chunk_idx} "):
        try:
            if 'question' in sample:
                q_base_list = [sample['question']]
            else:
                q_base_list = [sample['question1'], sample['question2']]
            out_list = []
            question_list = []
            for q_base in q_base_list:
                QUESTION_PROMPT = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
                # QUESTION_PROMPT += "And explain your reason for the choice as detailed."
                video_name = sample['video_id']
                question = QUESTION_PROMPT + q_base
                is_mcq_flag = True
                if 'videommesub' in args.dataset:
                    subtitle = get_subtitle(video_name, args.subtitle_frames)
                    SUBTITLE_PROMPT = "This video's subtitles are listed below: \n"
                    QUESTION_PROMPT = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
                    question = SUBTITLE_PROMPT + subtitle + '\n' + QUESTION_PROMPT + q_base
                elif args.dataset in ['rvs_ego', 'rvs_movie', 'actnet', 'nextoe', 'videochatgpt']:
                    QUESTION_PROMPT = "Answer the following open-ended question based on the video. "
                    question = QUESTION_PROMPT + q_base
                    is_mcq_flag = False

                video_path = os.path.join(args.video_dir, video_name)
                # Check if the video exists
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file {video_path} does not exist.")
                    # print(f"Video file {video_path} does not exist.")
                    # continue
            
                frame_paths = os.listdir(video_path)
                frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                frame_paths = [os.path.join(video_path, frame_path) for frame_path in frame_paths]
                if args.reproduce:
                    frame_paths = frame_paths[::4]  # set to fps=2, only for egoschema
                    # frame_paths = frame_paths[::2]  # set to fps=2, for other
                    max_frames = None
                else:
                    max_frames = args.max_frames
                    if args.fps is None:
                        total_time = len(frame_paths)
                        ranki_print(f'Use max_frames mode, limit max_frames={args.max_frames}, real frames={total_time}')
                        if 'frames_fps4' in args.video_dir and total_time > max_frames:  # tight sample, (0, 0.5) (4, 4.5) (8, 8.5) (12, 12.5)
                            assert max_frames % 2 == 0, f"max_frames must be even, now is {max_frames}"
                            nframes = max_frames // 2
                            indices = torch.linspace(0, total_time - 1, nframes).round().long().tolist()
                            new_paths = []
                            for i in indices:  # tight sample
                                if i < total_time - 1:
                                    new_paths.append(frame_paths[i])
                                    new_paths.append(frame_paths[i + 1])
                                else:
                                    new_paths.append(frame_paths[i - 1])
                                    new_paths.append(frame_paths[i])
                            # for i in indices:  # 2 tight sample
                            #     if i < total_time - 2:
                            #         new_paths.append(frame_paths[i])
                            #         new_paths.append(frame_paths[i + 2])
                            #     else:
                            #         new_paths.append(frame_paths[i - 2])
                            #         new_paths.append(frame_paths[i])
                            assert len(new_paths) == max_frames
                            frame_paths = new_paths
                        elif 'rvs_movie' in args.dataset:
                            new_paths = []
                            nframes = min(total_time, max_frames // 2)
                            indices = torch.linspace(0, total_time - 1, nframes).round().long().tolist()
                            new_paths = []
                            for i in indices:  # twice sample
                                new_paths.append(frame_paths[i])
                                new_paths.append(frame_paths[i])
                            frame_paths = new_paths
                    else:
                        ranki_print(f'Use fps mode, limit fps={args.max_frames}')
                        total_time = len(frame_paths)
                        nframes = round(total_time * args.fps)
                        indices = torch.linspace(0, total_time - 1, nframes).round().long().tolist()
                        frame_paths = [frame_paths[i] for i in indices]
                        max_frames = 10000
                content_video = {
                    "type": "video",
                    "video": frame_paths,
                }
                if args.reproduce:
                    if args.reproduce_total_pixels is not None:
                        content_video['total_pixels'] = args.reproduce_total_pixels
                    # content_video['max_frames'] = 16  # only for 480 test
                    content_video['min_pixels'] = 32 * 28 * 28  # 1 / 8 224 * 224
                else:
                    if max_frames is not None:
                        content_video['max_frames'] = max_frames
                    if args.max_pixels is not None:
                        content_video['max_pixels'] = args.max_pixels
                    if args.resized_height is not None:
                        content_video['resized_height'] = args.resized_height
                    if args.resized_width is not None:
                        content_video['resized_width'] = args.resized_width
                messages = [
                    {
                        "role": "user",
                        "content": [
                            content_video,
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if is_mcq_flag == True:
                    text += 'Best option: ('
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    flash_memory_config=flash_memory_config,
                )
                input_ids = inputs.input_ids.cuda()
                attention_mask = inputs.attention_mask.cuda()
                pixel_values_videos = inputs.pixel_values_videos.cuda()
                video_grid_thw = inputs.video_grid_thw.cuda()
                visual_position_ids = inputs.visual_position_ids.cuda()
                
                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        max_new_tokens=128,
                        top_k=1,
                        do_sample=False,
                        visual_position_ids=visual_position_ids,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                outputs = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                output = outputs[0].strip()
                input_color = ""  # Blue
                output_color = ""  # Green
                reset_color = ""  # Reset to default color
                ranki_print(f'{input_color}input={text}{reset_color} {output_color}output={output}{reset_color}')
                out_list.append(output)
                question_list.append(question)
        except Exception as e:
            ranki_print(f'Except: {e}')
            continue

        if len(question_list) == 0:
            continue

        sample_set = {'id': sample['id'], 'question': question, 'answer': sample['answer']}
        if 'question' in sample:
            sample_set['question'] = question_list[0]
            sample_set['pred'] = out_list[0]
        else:
            sample_set['question1'] = question_list[0]
            sample_set['question2'] = question_list[1]
            sample_set['pred1'] = out_list[0]
            sample_set['pred2'] = out_list[1]
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=20)
    parser.add_argument("--subtitle_frames", type=int, default=1080)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    
    parser.add_argument("--flash_memory_dict", type=str, default=None)
    parser.add_argument("--reproduce", action="store_true", default=False) 
    parser.add_argument("--reproduce_total_pixels", type=int, default=None)

    global args
    args = parser.parse_args()
    if args.flash_memory_dict is not None:
        args.flash_memory_dict = json.loads(args.flash_memory_dict)

    run_inference(args)
