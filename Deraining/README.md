
<!-- ## Training

1. To download Rain13K training and testing data

2. To train Restormer with default settings
```
train.sh Deraining/Options/Deraining_Restormer.yml
```

**Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Deraining/Options/Deraining_Restormer.yml](config.yml)
-->

## Pre-requisites
The project was developed using python 3 with the following packages.
- Pytorch
- Opencv
- Numpy
- LMDB
- Scikit-image
- PyYAML
- Einops
- SciPy

1. Install [Pytorch](https://pytorch.org/get-started/locally/)
2. Install with pip:
```bash
pip install -r requirements.txt
```

## Datasets
- Rain 13k - Test: [Here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs)
- Place it in `datasets`

## Pre-trained Models
- Download the pre-trained [model](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u?usp=sharing)
- Place it in `pretrained`

## Evaluation
```
python test.py
```
or
```
python test.py --weights <model_weights> --input_dir <input_path> --result_dir <result_path>
```

<!--
#### To reproduce PSNR/SSIM scores

```
evaluate_PSNR_SSIM.py (not yet)
```
-->
