## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch.nn as nn
import torch
import numpy as np
import argparse
import cv2

from PIL import Image
from torch.nn.functional import pad
from skimage import img_as_ubyte
from tqdm import tqdm
from pathlib import Path

from basicsr.models.archs.restormer_arch import Restormer

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')
parser.add_argument('--input_dir', default='Datasets/test/Test100/input', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='Results', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained/deraining.pth', type=str, help='Path to weights')
parser.add_argument('--config', default='config.yml', type=str, help='Config file YAML')
parser.add_argument('--factor', default=1, type=int, help='Factor')
args = parser.parse_args()

####### Load model #######
config = yaml.load(open(args.config, mode='r'), Loader=Loader)
config['network_g'].pop('type')

model_restoration = Restormer(**config['network_g'])
model_restoration.load_state_dict(torch.load(args.weights)['params'])
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration) #
model_restoration.eval()
##########################

factor = args.factor
Path(args.result_dir).mkdir(exist_ok=True)
input_dir = Path(args.input_dir)
images = []
for ext in ['png', 'jpg']:
    images.extend(input_dir.glob(f'*.{ext}'))

with torch.no_grad():
    for img_path in tqdm(images):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = pad(input_, (0, padw, 0, padh), 'reflect')

        restored = model_restoration(input_)

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored = img_as_ubyte(restored)
        # restored = Image.fromarray(restored) #
        # restored.save(Path(args.result_dir, img_path.name))
        cv2.imwrite(Path(args.result_dir, img_path.name).as_posix(), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))