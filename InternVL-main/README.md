# GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning
Code for training GeoTikzBridge-Base/Instruct-8B. This repository is built on the InternVL. The core fine-tuning launch script is located in `./internvl_chat/shell/internvl2.0/2nd_finetune/`.

## 1. Environment Setup
### 1.1 Clone Repository
```bash
git clone https://github.com/sjy-1995/GeoTikzBridge-Advancing-Multimodal-Code-Generation-for-Geometric-Perception-and-Reasoning.git
cd GeoTikzBridge-Advancing-Multimodal-Code-Generation-for-Geometric-Perception-and-Reasoning/InternVL-main
```

### 1.2 Create and Activate Virtual Environment, Install Dependencies
```bash
# Conda (recommended)
conda create -n geotikzbridge python=3.10 -y && conda activate geotikzbridge
pip install -r ./requirements/internvl_chat.txt
```

## 2. Run Training Scripts
### 2.1 Script Path
Core training script directory:  
`./internvl_chat/shell/internvl2.0/2nd_finetune/`  
Key launch script: `internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh`

### 2.2 Modify the dataset path, pretrained model path, and the output path
Modify the dataset path in the two json files in
`./internvl_chat/shell/data/`  
Modify the model path and output path in `./internvl_chat/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh`

### Model & Dataset 🤗
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/sjy-1995/GeoTikzBridge)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/sjy-1995/GeoCode-10K)

### 2.3 Run
```bash
cd ./internvl_chat/shell/internvl2.0/2nd_finetune/
bash internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh
```

## 3. Key Hyperparameters
Hyperparameters in the script are:

| Hyperparameter        | Description                                                                 | Recommended Configuration       |
|-----------------------|-----------------------------------------------------------------------------|----------------------------------|
| `--batch_size`        | Batch size                       | 128 |
| `--learning_rate`     | Learning rate                   | 4e-7 |
| `--epochs`            | Training epochs               | 3                 |
| `--max_seq_length`    | Max sequence length               | 12800      |
