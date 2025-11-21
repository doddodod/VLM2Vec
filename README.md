# VLM2Vec-V2: Unified Multimodal Embedding for Videos, Images, and Documents

<a target="_blank" href="https://arxiv.org/abs/2507.04590">
<img style="height:22pt" src="https://img.shields.io/badge/-V2 Paper%20-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://arxiv.org/abs/2410.05160">
<img style="height:22pt" src="https://img.shields.io/badge/-V1 Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/VLM2Vec">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/VLM2Vec/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/MMEB-V2">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset(V2)-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/MMEB-eval">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset(V1)-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/VLM2Vec">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Leaderboard-red?style=flat"></a>
<a target="_blank" href="https://x.com/WenhuChen/status/1844577017930694984">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>


## Model
VLM2Vec-V2 fine-tunes a state-of-the-art Vision-Language Model (VLM) using instruction-guided contrastive training. The model learns to produce a single, powerful fixed-dimensional embedding for any combination of text, image, video, and document inputs.

For current V2 models, we use **Qwen2-VL** as the model backbone, which capably handles interleaved sequences of text and visuals, variable resolutions, and long-form inputs like videos and visual documents.

[//]: # (<img width="768" alt="abs" src="assets/train_vlm.png">)

### Released checkpoints
- **[VLM2Vec-v2.0 (Qwen2VL-2B)](https://huggingface.co/VLM2Vec/VLM2Vec-V2.0)**: Our primary model, demonstrating strong, balanced performance across all modalities.

<details>
<summary> V1 checkpoints </summary>

- [VLM2Vec-Qwen2VL (7B)](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-7B)
- [VLM2Vec-Qwen2VL (2B)](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-2B)
- [VLM2Vec-LLaVa-Next](https://huggingface.co/TIGER-Lab/VLM2Vec-LLaVa-Next)
- [VLM2Vec-Phi3.5V](https://huggingface.co/TIGER-Lab/VLM2Vec-Full)
</details>

 
## Set up env
pip install -r requirements.txt

## Data
Download data
Run preprocess_negatives.py to store hard negatives in json file

## Model
Dowload model backbone from backbone official site

## Training
python train_sgg_qwen2vl.py with args

## Evaluation
DDP inference on multiple GPUs is supported. The whole evaluation process is streamlined and can be finished within hours. 

## Heads-up for Reproducing Baseline Models
- GME: requires an older version of the transformers library <=4.51.3.
- MomentSeeker: we recommend using a single GPU with a batch size of 10. This is due to a limitation in baseline processors that cannot handle mixed batches of image and text-only data.

## Citation
```bibtex
@article{jiang2024vlm2vec,
  title={VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks},
  author={Jiang, Ziyan and Meng, Rui and Yang, Xinyi and Yavuz, Semih and Zhou, Yingbo and Chen, Wenhu},
  journal={arXiv preprint arXiv:2410.05160},
  year={2024}
}

@article{meng2025vlm2vecv2,
  title={VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents},
  author={Rui Meng and Ziyan Jiang and Ye Liu and Mingyi Su and Xinyi Yang and Yuepeng Fu and Can Qin and Zeyuan Chen and Ran Xu and Caiming Xiong and Yingbo Zhou and Wenhu Chen and Semih Yavuz},
  journal={arXiv preprint arXiv:2507.04590},
  year={2025}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TIGER-AI-Lab/VLM2Vec&type=Date)](https://star-history.com/#TIGER-AI-Lab/VLM2Vec&Date)
