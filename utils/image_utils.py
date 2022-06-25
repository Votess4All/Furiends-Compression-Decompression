import cv2
import numpy as np

import torch

def tensor2uint(input, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225],
    rgb_range=255.0):
    """_summary_
    convert tensor to uint numpy array
    during validation
    Args:
        tensor (_type_): _description_
    """
    input = torch.squeeze(input)
    input = input.transpose(0, 1)
    input = input.transpose(1, 2)
    input = input.detach().cpu().numpy()

    input = input * np.array(std) + np.array(mean)
    input *= rgb_range
    return input.astype(np.uint8)


def tensor2uint_label(input, rgb_range=255.0):
    """_summary_
    convert tensor to uint numpy array
    during validation
    Args:
        tensor (_type_): _description_
    """
    input = torch.squeeze(input)
    input = input.detach().cpu().numpy()

    input *= rgb_range
    return input.astype(np.uint8)


def imsave(img, img_path):
    """save image

    Args:
        img (np.array): numpy array which is in format of N,C,H,W
        img_path (_type_): path to save image
    """
    img = np.squeeze(img)

    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]

    cv2.imwrite(img_path, img)
    return 