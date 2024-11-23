import argparse
import os
import torch.nn.functional as F
import torch
import numpy as np
from utils.io_helper import torch_read_image, torch_save_image
from utils.draw_helper import draw_all_circles
import matplotlib.pyplot as plt
from utils.misc_helper import *


def main(args) -> None:
    os.makedirs('outputs', exist_ok=True)
    if args.input_name == 'all':
        run_all(args)
        return
    output_name = f"{
        args.input_name}-{args.ksize}-{args.sigma}-{args.threshold}-{args.n}"
    blob_detection(
        'data/part2/%s.jpg' % args.input_name, 'outputs/%s-blob.jpg' % output_name,
        ksize=args.ksize, sigma=args.sigma, threshold=args.threshold, n=args.n)


def run_all(args) -> None:
    """Run the blob detection on all images."""
    for image_name in [
        'butterfly', 'einstein', 'fishes', 'sunflowers'
    ]:
        input_name = 'data/part2/%s.jpg' % image_name
        # output_name = 'outputs/%s-blob.jpg' % image_name
        output_name = f"{
            image_name}-{args.ksize}-{args.sigma}-{args.threshold}-{args.n}"
        blob_detection(
            input_name, 'outputs/%s-blob.jpg' % output_name,
            ksize=args.ksize, sigma=args.sigma, threshold=args.threshold, n=args.n)


def build_kernel(ksize, sigma):
    x = torch.arange(-ksize//2+1, ksize//2+1,
                     dtype=torch.float32).view(1, -1).repeat(ksize, 1)
    y = torch.arange(-ksize//2+1, ksize//2+1,
                     dtype=torch.float32).view(-1, 1).repeat(1, ksize)
    laplacian_kernel = torch.exp(-(x**2 + y**2) / (2 * sigma ** 2)) * \
        ((x**2 + y**2)/(2*sigma**2)-1) / (torch.pi * sigma ** 4)

    normalized = laplacian_kernel - torch.mean(laplacian_kernel)
    normalized = normalized * (sigma**2)

    return normalized


def apply_kernel(image, kernel):
    padding = int(kernel.shape[-1] - 1)//2
    image = torch.reshape(image, (1, 1, image.shape[1], image.shape[2]))
    kernel = torch.reshape(kernel, (1, 1, kernel.shape[0], kernel.shape[1]))
    image = F.pad(image, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(image, kernel, stride=1)


def nms(scale_space, sigma, factor_k, threshold):
    blobs = []
    n, h, w = np.shape(scale_space)
    for i in range(1, n-1):
        for j in range(1, h-1):
            for k in range(1, w-1):
                val = scale_space[i][j][k]
                neighbors = scale_space[i-1:i+2, j-1:j+2, k-1:k+2]
                if val == np.max(neighbors) and val > threshold:
                    blobs.append((j, k, sigma*(factor_k**i)))  # y, x, scale

    return blobs


def blob_detection(
    input_name: str,
    output_name: str,
    ksize: int,
    sigma: float,
    threshold: float,
    n: int
) -> None:
    # Step 1: Read RGB image as Grayscale
    # Step 2: Build Laplacian kernel
    # Step 3: Build feature pyramid
    image = torch_read_image(input_name, gray=True)
    h = image.shape[1]
    w = image.shape[2]
    print(image.shape)  # [1, h, w]

    current = image.clone()
    scale_space = np.empty((n, h, w))
    factor_k = 1.3

    # LoG
    for i in range(n):
        scale = sigma * factor_k**i  # initial scale is just sigma
        
        print(ksize)
        
        lap_kernel = build_kernel(ksize, scale)  # scale normalized kernel
        # [1, 1, h, w] -> [h, w]
        filtered = apply_kernel(current, lap_kernel).squeeze()

        scale_space[i] = filtered**2
        
        ksize = int(2*np.ceil(3*scale) + 1)
        # ksize = int((ksize*factor_k//2)*2)+1

    # Step 4: Extract and visualize Keypoints
    blobs = nms(scale_space, sigma, factor_k, threshold)

    # VISUALIZE
    draw_all_circles(current, blobs, output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CS59300CVD Assignment 2 Part 2')
    parser.add_argument('-i', '--input_name', required=True,
                        type=str, help='Input image path')
    parser.add_argument('-s', '--sigma', default=1, type=float)
    parser.add_argument('-k', '--ksize', default=7, type=int)
    parser.add_argument('-t', '--threshold', default=0.01, type=int)
    parser.add_argument('-n', default=15, type=int)
    args = parser.parse_args()
    assert (args.ksize % 2 == 1)
    main(args)
