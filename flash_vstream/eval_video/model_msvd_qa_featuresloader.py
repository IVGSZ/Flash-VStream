# This file may have been modified by Flash-VStream Authors (Flash-VStream Modifications”). All Flash-VStream Modifications are Copyright 2024 Flash-VStream Authors. 
# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from llama_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llama_vstream.conversation import conv_templates, SeparatorStyle
from llama_vstream.model.builder import load_pretrained_model
from llama_vstream.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)
    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, questions, video_dir, tokenizer, image_processor, model_config):
        self.questions = questions
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        sample = self.questions[index]
        video_name = sample['video_id']
        try:
            video_path = os.path.join(self.video_dir, video_name + '.safetensors')
            video_tensor = load_file(video_path)['feature']
        except Exception as e:
            print(f'Dataset Exception: {e}, randomly choose one.')
            idx = random.randint(0, len(self.questions) - 1)
            return self.__getitem__(idx)
        qs = sample['question']
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[args.conv_mode].copy()
        if 'system' in sample:
            conv.system = conv.system + ' ' + sample['system']
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, video_tensor

    def __len__(self):
        return len(self.questions)
    

def create_data_loader(questions, video_dir, tokenizer, image_processor, model_config, batch_size=1, num_workers=2):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, video_dir, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)

    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            print(f'mkdir Except: {e}')

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    # resume from old exp
    exist_id_set = set()
    if os.path.exists(answers_file):
        with open(answers_file) as f:
            exist_pred_contents = [json.loads(line) for line in f]
        exist_id_set = set([x['id'] for x in exist_pred_contents])

    new_gt_questions = []
    for sample in tqdm(gt_questions):
        if not sample['id'] in exist_id_set:
            new_gt_questions.append(sample)
    gt_questions = new_gt_questions

    data_loader = create_data_loader(gt_questions, args.video_dir, tokenizer, image_processor, model.config)

    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    with open(answers_file, "a") as ans_file:
        for data, sample in tqdm(zip(data_loader, gt_questions), desc=f"cuda:{args.chunk_idx} ", total=len(gt_questions)):
            input_ids, video_tensors = data
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    features=video_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    do_sample=True,
                    temperature=0.002,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            sample_set = {
                'id': sample['id'], 
                'question': sample['question'], 
                'answer': sample['answer'],
                'answer_type': sample['answer_type'] if 'answer_type' in sample else None,
                'pred': outputs
            }
            ans_file.write(json.dumps(sample_set) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
