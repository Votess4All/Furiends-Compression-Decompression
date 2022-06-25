import cv2
import numpy as np
import time

import torch
import torch.nn as nn
from zmq import device

from utils.tensor_utils import tensor2uint
from utils.image_utils import imsave
from model import RFDN


def compress_img(img_path, dst_path, scale=0.25):
    """_summary_
    compress image in resize way, which keeps aspect ratio.
    0.25 scale is only supported right now.
    Args:
        img_path (str): the path of image which needs to be compressed
        scale (float, optional): scale of dst image. Defaults to 0.25.
    """
    if not img_path or not dst_path:
        print("img_path or dst_path should be given.")
        return 
    
    img_arr = cv2.imread(img_path)
    half_img_arr = cv2.resize(img_arr, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dst_path, half_img_arr)
    return 


def preprocess(img_arr, device: torch.device):
    """preprocess image array before send to model
    1. convert color
    2. to tensor

    Args:
        img_arr (np.array): image array 
        device (torch.device): cpu or cuda device
    """
    if img_arr is None:
        print("image array is None")
        return

    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_arr)).permute(2, 0, 1).float().div(1.0).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    return img_tensor


@torch.no_grad()
def decompress_img(img_path, dst_path, sr_model: nn.Module, device):
    """decompress image using image super resolution model.

    Args:
        img_path (str): path of image which needs to be decompressed
        dst_path (str): destination path of result
        sr_model (nn.Module): super resolution model 
    """
    if not img_path or not dst_path:
        print("img_path or dst_path is None.")
        return  

    if not sr_model:
        print("sr_model is None.") 
        return
    
    img_arr = cv2.imread(img_path)
    img_tensor = preprocess(img_arr, device)
    output = sr_model(img_tensor)
    output_arr = tensor2uint(output)
    imsave(output_arr, dst_path)

    return


def init_sr_model(ckpt_path: str, model: str, device: torch.device):
    """init super resolution model

    Args:
        ckpt_path (str): model checkpoint path
        model (str): model name 
        device (torch.device): cpu or cuda device

    Raises:
        NotImplementedError: only support RFDN right now

    Returns:
        _type_: super resolution model
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model == "RFDN":
        model = RFDN()
    else:
        raise NotImplementedError(f"{model} has not been implemented yet.")
    
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model.eval()
    model = model.to(device)

    return model


def main():
    org_img_path = "white_cat_under_tree.jpg"
    downsample_path = "downsample_white_cat_under_tree.jpg"
    upsample_path = "upsample_white_cat_under_tree.jpg"

    ##### 1. compress image first  #####
    compress_img(org_img_path, downsample_path, scale=0.25)


    ##### 2. decompressed it back #####
    ckpt_path = "pretrained_model/RFDN_AIM.pth"
    model = "RFDN"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = init_sr_model(ckpt_path, model, device)
    decompress_img(downsample_path, upsample_path, model, device)


def upsample_profile():
    "Tesla V100S Cuda Mem: 6995MiB 58ms 17.1fps"
    model = "RFDN"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    ckpt_path = "pretrained_model/RFDN_AIM.pth"
    model = init_sr_model(ckpt_path, model, device)

    input = torch.randn(1, 3, 540, 1024).to(device)
    warmp_up_round = 100
    for i in range(warmp_up_round):
        model(input)

    total_round = 500
    start_time = time.time()
    for _ in range(total_round):
        model(input)
        torch.cuda.synchronize()
    duration = time.time() - start_time
    print(f"speed: { duration / total_round }, fps: { total_round / duration }")


if __name__ == "__main__":
    # main()
    upsample_profile()