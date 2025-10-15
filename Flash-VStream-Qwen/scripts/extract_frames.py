import decord
import torch
import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

min_frames = 4
max_frames = 14400
fps = 1
INPUT_DIR = 'data'
OUTPUT_DIR = f'video_{max_frames}frames_fps{fps}'

def process_video(file):
    video_id = file.strip().split('.')[0]
    video_path = f'{OUTPUT_DIR}/{video_id}'
    # if os.path.exists(video_path):
    #     return
    video_reader = decord.VideoReader(f'{INPUT_DIR}/{file}')
    video_len = len(video_reader)
    duration = video_len / video_reader.get_avg_fps()
    nframes = round(duration * fps)
    nframes = min(max(nframes, min_frames), max_frames)
    start_frame_ids = 0
    end_frame_ids = video_len - 1
    idx = torch.linspace(start_frame_ids, end_frame_ids, nframes).round().long().clamp(0, video_len - 1)
    frames = video_reader.get_batch(idx.tolist()).asnumpy()
    os.makedirs(video_path, exist_ok=True)
    for i in range(len(frames)):
        img = Image.fromarray(frames[i])
        img.save(f'{video_path}/{i:06d}.jpg', quality=100)


if __name__ == '__main__':
    files = [f for f in os.listdir(INPUT_DIR)]
    existed_files = [f for f in os.listdir(OUTPUT_DIR)]
    files = [f for f in files if f.strip().split('.')[0] not in existed_files]
    ### multi process
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap_unordered(process_video, files), total=len(files)))

    ### single process
    # for file in tqdm(files):
    #     process_video(file)
