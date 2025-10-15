# 0. create env

conda create -n vstream python=3.10.14
conda activate vstream


# 1. install dependencies

pip install torch==2.6 torchvision==0.21.0
pip install transformers==4.45.0
pip install opencv-python accelerate decord pillow openai
pip install peft deepspeed  # for training


# 2. install flash-attn
# You can install flash-attn online

pip install -U flash-attn --no-build-isolation

# Or you can install flash-attn offline, like following (check your environment, cu12 means cuda12, torch2.6 means pytorch2.6, and cxx11 means c++11, abiFLASE is important, cp310 means python3.10)

# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
