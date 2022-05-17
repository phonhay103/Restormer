import numpy as np
import cv2
import math
import os
from tqdm import tqdm

def calculate_psnr(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200']
psnr = {ds : [] for ds in datasets}
ssim = {ds : [] for ds in datasets}

for dataset in datasets:
    output_dir = f'./Datasets/test/{dataset}/target/'
    target_dir = f'./Results/{dataset}/'

    out_images = [output_dir + i for i in os.listdir(target_dir) if i.endswith(('jpeg', 'png', 'jpg',"PNG","JPEG","JPG"))]
    tar_images = [target_dir + i for i in os.listdir(target_dir) if i.endswith(('jpeg', 'png', 'jpg',"PNG","JPEG","JPG"))]

    for out_file, tar_file in tqdm(zip(out_images, tar_images)):
        out_img = load_img(out_file)
        tar_img = load_img(tar_file)
        psnr[dataset].append(calculate_psnr(out_img, tar_img))
        ssim[dataset].append(calculate_ssim(out_img, tar_img))
    
    print(dataset, np.mean(psnr[dataset]), np.mean(ssim[dataset]))
    