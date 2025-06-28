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
<a href=""><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

We proposed **Flash-VStream**, an efficient VLM with a novel Flash Memory mechanism that enables real-time understanding and Q&A of extremely long video streams. Our model achieves outstanding accuracy and efficiency on EgoSchema, MLVU, LVBench, MVBench and Video-MME Benchmarks.


<p align="center">
<img src="assets/framework.png" width="100%">
</p>

## News
- [2025/6/26] 🔥 [ICCV 2025] Flash-VStream-Qwen is coming! We release the 
[homepage](https://zhang9302002.github.io/vstream-iccv-page/). 
Paper, code and model are on the way.

- [2024/6/15] 🏅 Our team won the 1st Place at [Long-Term Video Question Answering Challenge](https://sites.google.com/view/loveucvpr24/track1) of [LOVEU Workshop@CVPR'24](https://sites.google.com/view/loveucvpr24/home). Here is our [certification](https://github.com/bytedance/Flash-VStream/assets/37479394/e1496dec-52c8-4707-aabe-fd1970c8f874). 
We used a Hierarchical Memory model based on Flash-VStream-7b.

- [2024/06/12] Flash-VStream-LLaVA is coming! We release the 
[homepage](https://invinciblewyq.github.io/vstream-page), 
[paper](https://arxiv.org/abs/2406.08085v1),
[code](https://github.com/IVGSZ/Flash-VStream)
and [model](https://huggingface.co/IVGSZ/Flash-VStream-7b)
for Flash-VStream.
We release the [dataset](https://huggingface.co/datasets/IVGSZ/VStream-QA) for VStream-QA benchmark.


## Contents
- [Flash-VStream-Qwen](#Flash-VStream-Qwen)
- [Flash-VStream-LLaVA](#Flash-VStream-Qwen)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Flash-VStream-Qwen

See [Flash-VStream-Qwen/README.md](Flash-VStream-Qwen/README.md).

## Flash-VStream-LLaVA

See [Flash-VStream-LLaVA/README.md](Flash-VStream-LLaVA/README.md).

## Citation
If you find this project useful in your research, please consider citing:

```
@article{flashvstream,
      title={Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams}, 
      author={Haoji Zhang and Yiqin Wang and Yansong Tang and Yong Liu and Jiashi Feng and Jifeng Dai and Xiaojie Jin},
      year={2024},
      eprint={2406.08085},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
We would like to thank the following repos for their great work:

- This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA).
- This work utilizes LLMs from [Vicuna](https://github.com/lm-sys/FastChat).
- Some code is borrowed from [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID).
- We perform video-based evaluation from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).

## License
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](LICENSE)

This project is licensed under the [Apache-2.0 License](LICENSE).