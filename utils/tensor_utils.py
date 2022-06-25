import numpy as np

import torch


def tensor2uint(img: torch.tensor):
    """convert tensor to uint8 numpy array

    Args:
        img (torch.tensor): _description_

    Returns:
        _type_: uint8 numpy array
    """
    img = img.data.squeeze().float().clamp_(0, 255).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return np.uint8(img.round())