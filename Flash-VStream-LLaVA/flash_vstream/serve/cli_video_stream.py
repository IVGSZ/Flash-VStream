#    This file may have been modified by Flash-VStream Authors (Flash-VStream Modifications”). All Flash-VStream Modifications are Copyright 2024 Flash-VStream Authors. 
#    Based on https://github.com/haotian-liu/LLaVA.
"""
    This file demonstrates an implementation of a multiprocess Real-time Long Video Understanding System. With a multiprocess logging module.
        main process: CLI server I/O, LLM inference
        process-1: logger listener
        process-2: frame generator, 
        process-3: frame memory manager
    Author: Haoji Zhang, Haotian Liu 
    (This code is based on https://github.com/haotian-liu/LLaVA)
"""
import argparse
import requests
import logging
import torch
import numpy as np
import time
import os

from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.conversation import conv_templates, SeparatorStyle
from flash_vstream.model.builder import load_pretrained_model
from flash_vstream.utils import disable_torch_init
from flash_vstream.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer
from decord import VideoReader
from datetime import datetime
from PIL import Image
from io import BytesIO

class _Metric:
    def __init__(self):
        self._latest_value = None
        self._sum = 0.0
        self._max = 0.0
        self._count = 0

    @property
    def val(self):
        return self._latest_value

    @property
    def max(self):
        return self._max

    @property
    def avg(self):
        if self._count == 0:
            return float('nan')
        return self._sum / self._count

    def add(self, value):
        self._latest_value = value
        self._sum += value
        self._count += 1
        if value > self._max:
            self._max = value

    def __str__(self):
        latest_formatted = f"{self.val:.6f}" if self.val is not None else "None"
        average_formatted = f"{self.avg:.6f}"
        max_formatted = f"{self.max:.6f}"
        return f"{latest_formatted} ({average_formatted}, {max_formatted})"
        

class MetricMeter:
    def __init__(self):
        self._metrics = {}

    def add(self, key, value):
        if key not in self._metrics:
            self._metrics[key] = _Metric()
        self._metrics[key].add(value)

    def val(self, key):
        metric = self._metrics.get(key)
        if metric is None or metric.val is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.val

    def avg(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.avg

    def max(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise ValueError(f"No values have been added for key '{key}'.")
        return metric.max

    def __getitem__(self, key):
        metric = self._metrics.get(key)
        if metric is None:
            raise KeyError(f"The key '{key}' does not exist.")
        return str(metric)
    
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def listener(queue, filename):
    ############## Start sub process-1: Listener #############
    import sys, traceback
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # h = logging.StreamHandler(sys.stdout)
    h = logging.FileHandler(filename)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    while True:
        try:
            record = queue.get()
            if record is None:  # None is a signal to finish
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

def video_stream_similator(video_file, frame_queue, log_queue, video_fps=1.0, play_speed=1.0):
    ############## Start sub process-2: Simulator #############
    worker_configurer(log_queue)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    vr = VideoReader(video_file)
    sample_fps = round(vr.get_avg_fps() / video_fps)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    video = vr.get_batch(frame_idx).asnumpy()
    length = video.shape[0]
    sleep_time = 1 / video_fps / play_speed
    time_meter = MetricMeter()
    logger.info(f'Simulator Process: start, length = {length}')
    try:
        for start in range(0, length):
            start_time = time.perf_counter()
            end = min(start + 1, length)
            video_clip = video[start:end]
            frame_queue.put(video_clip)
            if start > 0:
                time_meter.add('real_sleep', start_time - last_start)
                logger.info(f'Simulator: write {end - start} frames,\t{start} to {end},\treal_sleep={time_meter["real_sleep"]}')
            if end < length:
                time.sleep(sleep_time)
            last_start = start_time
        frame_queue.put(None)
    except Exception as e:
        print(f'Simulator Exception: {e}')
        time.sleep(0.1)
    logger.info(f'Simulator Process: end')

def frame_memory_manager(model, image_processor, frame_queue, log_queue):
    ############## Start sub process-3: Memory Manager #############
    worker_configurer(log_queue)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    time_meter = MetricMeter()
    logger.info(f'MemManager Process: start')
    frame_cnt = 0
    while True:
        try:
            video_clip = frame_queue.get()
            start_time = time.perf_counter()
            if video_clip is None:
                logger.info(f'MemManager: Ooops, get None')
                break
            logger.info(f'MemManager: get {video_clip.shape[0]} frames from queue')
            image = image_processor.preprocess(video_clip, return_tensors='pt')['pixel_values']
            image = image.unsqueeze(0)
            image_tensor = image.to(model.device, dtype=torch.float16)
            # time_2 = time.perf_counter()
            logger.info(f'MemManager: Start embedding')
            with torch.inference_mode():
                model.embed_video_streaming(image_tensor)
            logger.info(f'MemManager: End embedding')
            end_time = time.perf_counter()
            if frame_cnt > 0:
                time_meter.add('memory_latency', end_time - start_time)
                logger.info(f'MemManager: embedded {video_clip.shape[0]} frames,\tidx={frame_cnt},\tmemory_latency={time_meter["memory_latency"]}')
            else:
                logger.info(f'MemManager: embedded {video_clip.shape[0]} frames,\tidx={frame_cnt},\tmemory_latency={end_time - start_time:.6f}, not logged')
            frame_cnt += video_clip.shape[0]
        except Exception as e:
            print(f'MemManager Exception: {e}')
            time.sleep(0.1)
    logger.info(f'MemManager Process: end')

def main(args):
    # torch.multiprocessing.log_to_stderr(logging.DEBUG)
    torch.multiprocessing.set_start_method('spawn', force=True)
    disable_torch_init()

    log_queue = Queue()
    frame_queue = Queue(maxsize=10)
    processes = []

    ############## Start listener process #############
    p1 = Process(target=listener, args=(log_queue, args.log_file))
    processes.append(p1)
    p1.start()

    ############## Start main process #############
    worker_configurer(log_queue)
    logger = logging.getLogger(__name__)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    logger.info(f'Using conv_mode={args.conv_mode}')

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    with Manager() as manager:
        image_tensor = None
        model.use_video_streaming_mode = True
        model.video_embedding_memory = manager.list()
        if args.video_max_frames is not None:
            model.config.video_max_frames = args.video_max_frames
            logger.info(f'Important: set model.config.video_max_frames = {model.config.video_max_frames}')

        logger.info(f'Important: set video_fps = {args.video_fps}')
        logger.info(f'Important: set play_speed = {args.play_speed}')

        ############## Start simulator process #############
        p2 = Process(target=video_stream_similator, 
                     args=(args.video_file, frame_queue, log_queue, args.video_fps, args.play_speed))
        processes.append(p2)
        p2.start()

        ############## Start memory manager process #############
        p3 = Process(target=frame_memory_manager, 
                     args=(model, image_processor, frame_queue, log_queue))
        processes.append(p3)
        p3.start()

        # start QA server
        start_time = datetime.now()
        time_meter = MetricMeter()
        conv_cnt = 0
        while True:
            time.sleep(5)
            try:
                # inp = input(f"{roles[0]}: ")
                inp = "what is in the video?"
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break

            # 获取当前时间
            now = datetime.now()
            conv_start_time = time.perf_counter()
            # 将当前时间格式化为字符串
            current_time = now.strftime("%H:%M:%S")
            duration = now.timestamp() - start_time.timestamp()

            # 打印当前时间
            print("\nCurrent Time:", current_time, "Run for:", duration)
            print(f"{roles[0]}: {inp}", end="\n")
            print(f"{roles[1]}: ", end="")
            # every conversation is a new conversation
            conv = conv_templates[args.conv_mode].copy()
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)

            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            llm_start_time = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            llm_end_time = time.perf_counter()

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            conv_end_time = time.perf_counter()
            if conv_cnt > 0:
                time_meter.add('conv_latency', conv_end_time - conv_start_time)
                time_meter.add('llm_latency', llm_end_time - llm_start_time)
                time_meter.add('real_sleep', conv_start_time - last_conv_start_time)
                logger.info(f'CliServer: idx={conv_cnt},\treal_sleep={time_meter["real_sleep"]},\tconv_latency={time_meter["conv_latency"]},\tllm_latency={time_meter["llm_latency"]}')
            else:
                logger.info(f'CliServer: idx={conv_cnt},\tconv_latency={conv_end_time - conv_start_time},\tllm_latency={llm_end_time - llm_start_time}')
            conv_cnt += 1
            last_conv_start_time = conv_start_time

    for p in processes:
        p.terminate()
    print("All processes finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--log-file", type=str, default="tmp_cli.log")
    parser.add_argument("--use_1process", action="store_true")
    parser.add_argument("--video_max_frames", type=int, default=None)
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--play_speed", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
