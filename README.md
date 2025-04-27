# AoTD
Official PyTorch code of "Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation", CVPR 2025.

[[Project page]](https://zhengrongz.github.io/AoTD/) [[Paper]](https://arxiv.org/abs/2412.01694)


## 🔥News
* **[2025.4.27]** AoTD got accepted by CVPR 2025!
* **[2025.4.27]** We change the original title to "Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation".
* **[2025.4.27]** We release the code about Agent system, CoT transfer and filtering.

## TODOs
* Code about distillation and evaluation. (After NIPs deadline)

## Installation
```
conda create -n aotd python=3.10
conda activate aotd
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install flash_attn==2.6.3 --no-build-isolation
```

## Quick Start
You can set all the configs at:

`agent_system/configs/base_config.yaml`.

If the model path is null, it will directly load model from huggingface.

You can add examples at:

`agent_system/prompts`

After setting all the configs and prompts, you can run the Jupyter Notebook `cot.ipynb`. This notebook contains the code to try AoTD on your own videos. The notebook is well documented, and it describes how to use the code.

## Running on Datasets
If you want to construct Chain-of-Thoughts automatically. You can run codes as following:
```
python cot_construction.py
python cot_transfer.py
python cot_filter.py
```
Noted that the configs about `cot_construction.py` are in base_config.yaml, but configs about the other two files are in files themselves.

After running the code, you can get the filtered CoT data and use it in the training process.

## Citation
If you find this paper useful, please consider staring this repo and citing our paper!
```latex
@inproceedings{shi2024aotd,
  title={Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation},
  author={Shi, Yudi and Di, Shangzhe and Chen, Qirui and Xie, Weidi},
  booktitle="Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition",
  year=2025
}
```