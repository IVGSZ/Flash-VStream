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

import os
import argparse
import subprocess
import multiprocessing

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

# multi gpu, feature
def eval_msvd(args):
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/MSVD-QA/video_features",
                    "--gt_file", "./data/eval_video/MSVD-QA/test_qa.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "msvd"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1"]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "msvd"),
           "--output_dir", os.path.join(model_path, "evaluation", "msvd", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "msvd", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_msrvtt(args):
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/MSRVTT-QA/video_features",
                    "--gt_file", "./data/eval_video/MSRVTT-QA/test_qa.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "msrvtt"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1"]
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "msrvtt"),
           "--output_dir", os.path.join(model_path, "evaluation", "msrvtt", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "msrvtt", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_actnet(args):
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/ActivityNet-QA/video_features",
                    "--gt_file", "./data/eval_video/ActivityNet-QA/test_qa.json", 
                    "--output_dir", os.path.join(model_path, "evaluation", "actnet"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "actnet"),
           "--output_dir", os.path.join(model_path, "evaluation", "actnet", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "actnet", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_nextoe(args):  # follow msvd format, OE follow actnet
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/nextoe/video_features",
                    "--gt_file", "./data/eval_video/nextoe/test_qa.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "nextoe"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                    ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "nextoe"),
           "--output_dir", os.path.join(model_path, "evaluation", "nextoe", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "nextoe", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_vsmovienet(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream/movienet_video_features",
                    "--gt_file", "./data/eval_video/vstream/test_qa_movienet.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "vsmovienet"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "vsmovienet"),
           "--output_dir", os.path.join(model_path, "evaluation", "vsmovienet", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "vsmovienet", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_vsego4d(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream/ego4d_video_features",
                    "--gt_file", "./data/eval_video/vstream/test_qa_ego4d.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "vsego4d"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "vsego4d"),
           "--output_dir", os.path.join(model_path, "evaluation", "vsego4d", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "vsego4d", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_realtime_vsmovienet(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream-realtime/movienet_video_features",
                    "--gt_file", "./data/eval_video/vstream-realtime/test_qa_movienet.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsmovienet"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "realtime_vsmovienet"),
           "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsmovienet", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "realtime_vsmovienet", "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
    exec(cmd)

# multi gpu, feature
def eval_realtime_vsego4d(args):  # follow msvd format
    model_path = args.model_path
    num_chunks = args.num_chunks
    if not args.only_eval:
        processes = []
        for idx in range(0, num_chunks):
            cmd = ["python", "llama_vstream/eval_video/model_msvd_qa_featuresloader.py",
                    "--model-path", model_path,
                    "--video_dir", "./data/eval_video/vstream-realtime/ego4d_video_features",
                    "--gt_file", "./data/eval_video/vstream-realtime/test_qa_ego4d.json",
                    "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsego4d"),
                    "--output_name", "pred",
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(idx),
                    "--conv-mode", "vicuna_v1",
                ]
            
            p = multiprocessing.Process(target=exec, args=(cmd, True, str(idx)))
            processes.append(p)
            p.start() # 启动子进程
        for p in processes:
            p.join()
    cmd = ["python", "llama_vstream/eval_video/eval_activitynet_qa.py",
           "--pred_path", os.path.join(model_path, "evaluation", "realtime_vsego4d"),
           "--output_dir", os.path.join(model_path, "evaluation", "realtime_vsego4d", "results"),
           "--output_json", os.path.join(model_path, "evaluation", "realtime_vsego4d", "results.json"),
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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--api_type", type=str, default=None)
    parser.add_argument("--api_version", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--vizlen", type=int, default=0)
    parser.add_argument("--use_speech", action="store_true", default=False)
    args = parser.parse_args()
    func_dic = {'msvd': eval_msvd,
                'msrvtt': eval_msrvtt,
                'actnet': eval_actnet,
                'nextoe': eval_nextoe,
                'vsmovienet': eval_vsmovienet,
                'vsego4d': eval_vsego4d,
                'realtime_vsmovienet': eval_realtime_vsmovienet,
                'realtime_vsego4d': eval_realtime_vsego4d,
                }
    if args.dataset in func_dic:
        print(f'Execute {args.dataset} evaluation')
        func_dic[args.dataset](args)
