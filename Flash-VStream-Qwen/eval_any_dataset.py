#    Copyright 2024 Flash-VStream Authors 
#
#    Licensed under the Apache License, Version 2.0 (the "License"); 
#    you may not use this file except in compliance with the License. 
#    You may obtain a copy of the License at 
#
#        http://www.apache.org/licenses/LICENSE-2.0 
#
#    Unless required by applicable law or agreed to in writing, software 
#    distributed under the License is distributed on an "AS IS" BASIS, 
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#    See the License for the specific language governing permissions and 
#    limitations under the License. 

from collections import defaultdict
import csv
import json
import os
import argparse
import random
import re
import subprocess
import multiprocessing
import logging

def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(cmd, env=my_env)

def launch_multi_gpu_eval(args, dataset_name, frame_dir, data_file, evaluation_name='evaluation'):
    model_path = args.model_path
    num_chunks = args.num_chunks
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f'launch_multi_gpu_eval: args={args}')
    output_base = os.path.join(args.output_dir, evaluation_name, dataset_name)
    if not args.test:
        if 'videochatgpt' in dataset_name:
            split_list = ["generic", "temporal", "consistency"]
            data_file_list = data_file
        else:
            split_list = [""]
            data_file_list = [data_file]
        for data_file, split in zip(data_file_list, split_list):
            output_dir = output_base + split
            processes = []
            for idx in range(0, num_chunks):
                cmd = [ "python3", "inference_mcq_vqa.py",
                        "--dataset", dataset_name,
                        "--model-path", model_path,
                        "--video_dir", frame_dir,
                        "--gt_file", data_file,
                        "--output_dir", output_dir,
                        "--output_name", "pred",
                        "--num-chunks", str(num_chunks),
                        "--chunk-idx", str(idx),
                ]
                if args.reproduce:
                    cmd += ["--reproduce"]
                    if args.reproduce_total_pixels:
                        cmd += ["--reproduce_total_pixels", str(args.reproduce_total_pixels)]
                if args.fps:
                    cmd += ["--fps", str(args.fps)]
                if args.max_pixels:
                    cmd += ["--max_pixels", str(args.max_pixels)]
                if args.max_frames:
                    cmd += ["--max_frames", str(args.max_frames)]
                if args.resized_height:
                    cmd += ["--resized_height", str(args.resized_height)]
                if args.resized_width:
                    cmd += ["--resized_width", str(args.resized_width)]
                if args.lora_path:
                    cmd += ["--lora-path", args.lora_path]
                if args.flash_memory_dict:
                    cmd += ["--flash_memory_dict", args.flash_memory_dict]
                if dataset_name == 'videommesub':
                    cmd += ["--subtitle_frames", str(args.subtitle_frames)]
                logging.debug(f"Starting subprocess with command: {' '.join(cmd)}")
                # Start subprocess and capture output
                my_env = os.environ.copy()
                my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
                p = subprocess.Popen(cmd, env=my_env)
                processes.append(p)
            for idx, p in enumerate(processes):
                stdout, stderr = p.communicate()
                logging.debug(f"Subprocess {idx} stdout: {stdout}")
                if stderr:
                    logging.error(f"Subprocess {idx} stderr: {stderr}")
                if p.returncode != 0:
                    logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
                else:
                    logging.debug(f"Subprocess {idx} completed successfully")
    return output_base

def get_dataset_info(args):
    dataset = args.dataset
    dataset_list = [
        {'type': 'mc', 'dataset_name': 'egoschema', 'frame_dir': 'data/eval_video/EgoSchema/frames', 'data_file': 'data/eval_video/EgoSchema/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'egoschema_all', 'frame_dir': 'data/eval_video/EgoSchema/frames', 'data_file': 'data/eval_video/EgoSchema/all_qa.json'},
        {'type': 'mc', 'dataset_name': 'videommesub', 'frame_dir': 'data/eval_video/videomme/frames', 'data_file': 'data/eval_video/videomme/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'videommewo', 'frame_dir': 'data/eval_video/videomme/frames', 'data_file': 'data/eval_video/videomme/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'mvbench', 'frame_dir': 'data/eval_video/mvbench/frames', 'data_file': 'data/eval_video/mvbench/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'lvbench', 'frame_dir': 'data/eval_video/lvbench/frames', 'data_file': 'data/eval_video/lvbench/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'mlvu', 'frame_dir': 'data/eval_video/mlvu/frames', 'data_file': 'data/eval_video/mlvu/test_qa.json'},
        {'type': 'oe', 'dataset_name': 'rvs_ego', 'frame_dir': 'data/eval_video/vstream-realtime/ego4d_frames', 'data_file': 'data/eval_video/vstream-realtime/test_qa_ego4d.json'},
        {'type': 'oe', 'dataset_name': 'rvs_movie', 'frame_dir': 'data/eval_video/vstream-realtime/movienet_frames', 'data_file': 'data/eval_video/vstream-realtime/test_qa_movienet.json'},
        {'type': 'oe', 'dataset_name': 'actnet', 'frame_dir': 'data/eval_video/ActivityNet-QA/test_frames', 'data_file': 'data/eval_video/ActivityNet-QA/test_qa.json'},
        {'type': 'oe', 'dataset_name': 'nextoe', 'frame_dir': 'data/eval_video/nextoe/nextoe_frames', 'data_file': 'data/eval_video/nextoe/test_qa.json'},
    ]
    dataset_list.append({'type':'oe', 
        'dataset_name': f'videochatgpt', 
        'frame_dir': 'data/eval_video/VideoChatGPTBench/video_10000frames_high_fps1',
        'data_file': ['data/eval_video/VideoChatGPTBench/test_{split}_qa.json'.format(split=split) for split in ["generic", "temporal", "consistency"]]
    })
    for d in dataset_list:
        if d['dataset_name'] == dataset:
            if args.use_high_fps:
                d['frame_dir'] = d['frame_dir'].replace('frames', 'frames_fps4')
            return d
    return None

def extract_answer(llm_message):
    answer = re.findall(r'[A-E]', llm_message)
    if len(answer) == 0:
        print('No answer found')
        answer = random.choice(['A', 'B', 'C', 'D', 'E'])
    else:
        answer = answer[0]
    map2idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    answer = map2idx[answer]
    return answer

def calc_eval_result(output_path, num_chunks, data_path):
    if num_chunks > 1:
        pred_contents = []
        for _idx in range(num_chunks):
            file = os.path.join(output_path, f"{num_chunks}_{_idx}.json")
            try:
                # pred_contents += [json.loads(line) for line in open(file)]
                for line in open(file):
                    pred_contents += [json.loads(line)]
            except:
                print('Error loading file: line=', line, 'file=', file)
    else:
        file = os.path.join(output_path, f"pred.json")
        pred_contents = [json.loads(line) for line in open(file)]


    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in pred_contents:
        res = extract_answer(sample['pred'])
        if res == sample['answer']:
            acc = "yes"
            score = 1.0
        else:
            acc = "no"
            score = 0.0
        prediction_set[str(sample['id'])] = {
            'acc': acc,
            'score': score,
            **sample
        }
    
    json_path = os.path.join(output_path, 'result.json')
    with open(json_path, "w") as f:
        json.dump(prediction_set, f, indent=4)
    print("[main] All evaluation completed!")

    class ScoreMeter:
        def __init__(self):
            self.score_sum = 0
            self.count = 0
            self.yes_count = 0
            self.no_count = 0
            self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

        def add_score(self, score, pred):
            self.score_sum += score
            self.count += 1
            pred_lower = pred.lower()
            if 'yes' in pred_lower:
                self.yes_count += 1
                self.score_dict['yes'][score] += 1
            elif 'no' in pred_lower:
                self.no_count += 1
                self.score_dict['no'][score] += 1

        def get_average_score(self):
            res = (self.score_sum / self.count) if self.count else 0
            return f"{res * 100:.6f}"

        def get_accuracy(self, response_type):
            if response_type == 'yes':
                res =  (self.yes_count / self.count) if self.count else 0
            elif response_type == 'no':
                res = (self.no_count / self.count) if self.count else 0
            else:
                res = 0
            return f"{res * 100:.6f}"

    meter_dic = {'total': ScoreMeter()}
    for key, result in prediction_set.items():
        # Computing score
        score = result['score']
        acc = result['acc']
        meter_dic["total"].add_score(score, acc)
        if 'a_type' in result and result['a_type'] is not None:
            # typ = str(result[1]['a_type'])
            typ = str(result['a_type'])
            if typ not in meter_dic:
                meter_dic[typ] = ScoreMeter()
            meter_dic[typ].add_score(score, acc)
            if 'next' in output_path:
                typ = typ[0]
                if typ not in meter_dic:
                    meter_dic[typ] = ScoreMeter()
                meter_dic[typ].add_score(score, acc)

    csv_dic = {'acc': meter_dic["total"].get_accuracy('yes'), 'score': meter_dic["total"].get_average_score()}
    output = ""
    output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    output += "\n"
    output += "Total Score Yes/No distribution:\n"
    for key, value in meter_dic["total"].score_dict.items():
        output += f"{key}:\n"
        for k in range(0, 6):
            v = value[k]
            output += f"{k}: {v}\n"
    output += "\n"
    output += "Answer Type Score distribution:\n"
    output += 'Type, Accuracy, Avg_score\n'
    # key_list = sorted([k for k in meter_dic.keys()])
    key_list = [k for k in meter_dic.keys()]
    for key in key_list:
        output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
        csv_dic[key] = meter_dic[key].get_accuracy('yes')

    output += "\n"
    for k in csv_dic.keys():
        output += f"{k}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    for k in csv_dic.keys():
        output += str(csv_dic[k]) + ", "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    # kaggle upload
    if 'egoschema' in output_path:
        upload_path = json_path.replace(".json", "_upload.csv")
        with open(upload_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['q_uid', 'answer'])
            all_qa = json.load(open('data/eval_video/EgoSchema/all_qa.json'))
            info_dic = {}
            for qa in all_qa:
                info_dic[str(qa['id'])] = qa['video_id']
            for key, result in prediction_set.items():
                pred = result['pred']
                q_uid = info_dic[key.split('_')[0]]
                res = extract_answer(pred)
                writer.writerow([q_uid, res])
    elif 'videomme' in output_path:
        score_dic = {
            "duration": {"short": ScoreMeter(), "medium": ScoreMeter(), "long": ScoreMeter()},
            "domain": {"Knowledge": ScoreMeter(), "Film & Television": ScoreMeter(), "Sports Competition": ScoreMeter(), "Artistic Performance": ScoreMeter(), "Life Record": ScoreMeter(), "Multilingual": ScoreMeter()}, 
            "sub_category": {"Humanity & History": ScoreMeter(), "Literature & Art": ScoreMeter(), "Biology & Medicine": ScoreMeter(), "Finance & Commerce": ScoreMeter(), "Astronomy": ScoreMeter(), "Geography": ScoreMeter(), "Law": ScoreMeter(), "Life Tip": ScoreMeter(), "Technology": ScoreMeter(), "Animation": ScoreMeter(), "Movie & TV Show": ScoreMeter(), "Documentary": ScoreMeter(), "News Report": ScoreMeter(), "Esports": ScoreMeter(), "Basketball": ScoreMeter(), "Football": ScoreMeter(), "Athletics": ScoreMeter(), "Other Sports": ScoreMeter(), "Stage Play": ScoreMeter(), "Magic Show": ScoreMeter(), "Variety Show": ScoreMeter(), "Acrobatics": ScoreMeter(), "Handicraft": ScoreMeter(), "Food": ScoreMeter(), "Fashion": ScoreMeter(), "Daily Life": ScoreMeter(), "Travel": ScoreMeter(), "Pet & Animal": ScoreMeter(), "Exercise": ScoreMeter(), "Multilingual": ScoreMeter()}, 
            "task_type": {"Temporal Perception": ScoreMeter(), "Spatial Perception": ScoreMeter(), "Attribute Perception": ScoreMeter(), "Action Recognition": ScoreMeter(), "Object Recognition": ScoreMeter(), "OCR Problems": ScoreMeter(), "Counting Problem": ScoreMeter(), "Temporal Reasoning": ScoreMeter(), "Spatial Reasoning": ScoreMeter(), "Action Reasoning": ScoreMeter(), "Object Reasoning": ScoreMeter(), "Information Synopsis": ScoreMeter()},
        }
        total_dic = ScoreMeter()
        test_qa = json.load(open(data_path))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        level_list = ['duration', 'domain', 'sub_category', 'task_type']
        for key, result in prediction_set.items():
            acc = result['acc']
            qa = info_dic[key.split('_')[0]]
            for level in level_list:
                score_dic[level][qa[level]].add_score(0, acc)
            total_dic.add_score(0, acc)
        output += '\n'
        output += 'Type, Accuracy\n'
        for level in level_list:
            for key, meter_dic in score_dic[level].items():
                output += f"{key}, {float(meter_dic.get_accuracy('yes')):.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')):.02f}\n"
    elif 'lvbench' in output_path:
        score_dic = {
            "key information retrieval": ScoreMeter(),
            "event understanding": ScoreMeter(),
            "summarization": ScoreMeter(),
            "entity recognition": ScoreMeter(),
            "reasoning": ScoreMeter(),
            "temporal grounding": ScoreMeter(),
        }
        total_dic = ScoreMeter()
        test_qa = json.load(open(data_path))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        for key, result in prediction_set.items():
            acc = result['acc']
            qa = info_dic[key]
            for typ in qa['question_type']:
                score_dic[typ].add_score(0, acc)
            total_dic.add_score(0, acc)
        output += '\n'
        output += 'Type, Accuracy\n'
        for key, meter_dic in score_dic.items():
            output += f"{key}, {float(meter_dic.get_accuracy('yes')):.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')):.02f}\n"
    elif 'scalelong' in output_path:
        Granularities = ["Video Clip", "Video Shot", "Video Event", "Video Story"]
        question_types = ["Causal Reasoning", "Object Recognition", "Action Understanding", "Information Summary", "Counting Problem"]
        all_tags = Granularities + question_types
        score_dic = {}
        total_dic = ScoreMeter()
        for tag in all_tags:
            if tag not in score_dic:
                score_dic[tag] = ScoreMeter()
        test_qa = json.load(open(data_path))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        for key, result in prediction_set.items():
            acc = result['acc']
            qa = info_dic[key]
            typ = qa['question_type']
            # All tags: {'Objective Recognition', 'Information Inference', 'Counting Problem', 'Information Summary', 'Counting Problems', 'Action Understanding', 'Casual Reasoning', 'Causal Reasoning', 'Object Recognition', 'information inference'}
            if typ == "information inference" or typ == "Information Inference":
                typ = "Information Summary"
            elif typ == "Counting Problems":
                typ = "Counting Problem"
            elif typ == "Objective Recognition":
                typ = "Object Recognition"
            elif typ == "Casual Reasoning":
                typ = "Causal Reasoning"
            score_dic[typ].add_score(0, acc)
            granu = qa['granularity']
            score_dic[granu].add_score(0, acc)
            total_dic.add_score(0, acc)
        output += '\n'
        output += 'Type, Accuracy\n'
        for key, meter_dic in score_dic.items():
            output += f"{key}, {float(meter_dic.get_accuracy('yes')):.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')):.02f}\n"

    print(output)
    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, 'w') as f:
        f.write(output)

def calc_gpt_based_eval_result(args, output_path, num_chunks, data_path):
    print(f'[main] cal gpt-based eval result')
    print(f'output_path={output_path}')
    print(f'num_chunks={num_chunks}')
    print(f'data_path={data_path}')

    if 'videochatgpt' in output_path:
        assert args.api_key is not None
        assert args.api_base is not None
        assert args.api_type is not None
        assert args.api_version is not None
        code_list = [
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_1_correctness.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_2_detailed_orientation.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_3_context.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_4_temporal.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_5_consistency.py",
        ]
        split_list = ["generic", "generic", "generic", "temporal", "consistency", ]
        short_list = ["1_CI", "2_DO", "3_CU", "4_TU", "5_CO"]
        for split, code, short in zip(split_list, code_list, short_list):
            cmd = ["python", code,
                "--pred_path", output_path + split,
                "--output_dir", os.path.join(output_path + short, "results"),
                "--output_json", os.path.join(output_path + short, "results.json"),
                "--num_chunks", str(num_chunks),
                "--num_tasks", "16",
                "--api_key", args.api_key,
                "--api_base", args.api_base,
                "--api_type", args.api_type,
                "--api_version", args.api_version,
            ]
            exec(cmd)
    else:
        assert args.api_key is not None
        assert args.api_base is not None
        assert args.api_type is not None
        assert args.api_version is not None
        cmd = ["python3", "eval_activitynet_qa.py",
           "--pred_path", os.path.join(output_path),
           "--output_dir", os.path.join(output_path, "results"),
           "--output_json", os.path.join(output_path, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
        exec(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='~')
    parser.add_argument("--evaluation_name", type=str, default='evaluation')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--testtest", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subtitle_frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--flash_memory_dict", type=str, default=None)
    parser.add_argument("--use_high_fps", action="store_true", default=False)
    parser.add_argument("--reproduce", action="store_true", default=False)
    parser.add_argument("--reproduce_total_pixels", type=int, default=None)

    parser.add_argument("--api_key", default=None, type=str, help="OpenAI API key")
    parser.add_argument("--api_type", default=None, type=str, help="OpenAI API type")
    parser.add_argument("--api_version", default=None, type=str, help="OpenAI API version")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI API base")
    args = parser.parse_args()

    info = get_dataset_info(args)
    if info is None:
        print(f'[main] ERROR {args.dataset} dataset was not found!')
        exit(0)
    typ = info.pop('type')
    out_dir = launch_multi_gpu_eval(args, **info, evaluation_name=args.evaluation_name)
    if typ == 'mc':
        print(f'[main] Execute {args.dataset} mcq evaluation')
        calc_eval_result(out_dir, args.num_chunks, info['data_file'])
    elif typ =='oe':
        print(f'[main] Execute {args.dataset} open-ended evaluation')
        calc_gpt_based_eval_result(args, out_dir, args.num_chunks, info['data_file'])