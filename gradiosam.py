from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import torch
from segment_anything.modeling.prompt_encoder import PromptEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def transform_point_coords(point_coords, original_size, target_size):

    original_width, original_height = original_size
    target_width, target_height = target_size

    transformed_coords = []
    for x, y in point_coords:
        transformed_x = int(x * target_width / original_width)
        transformed_y = int(y * target_height / original_height)
        transformed_coords.append((transformed_x, transformed_y))

    return transformed_coords

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )
    
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions



def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True

def predict_image(args, image, point_coords=None, point_labels=None, boxes=None, original_size=(256,256)):
    device = args.device
    model = sam_model_registry[args.model_type](args).to(device)
    model.eval()

    batched_input = {
        "image": image.to(device).unsqueeze(0), 
        "point_coords": point_coords.to(device).unsqueeze(0) if point_coords is not None else None,
        "point_labels": point_labels.to(device).unsqueeze(0) if point_labels is not None else None,
        "boxes": boxes.to(device).unsqueeze(0) if boxes is not None else None,
        "original_size": original_size
    }

    with torch.no_grad():
        image_embeddings = model.image_encoder(batched_input["image"])

        masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)

    masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)

    return masks.cpu().numpy()  


def predict_interface(image, args):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[123.675/255.0, 116.28/255.0, 103.53/255.0], 
                             std=[58.395/255.0, 57.12/255.0, 57.375/255.0])
    ])
    image_tensor = transform(Image.open(image).convert('RGB'))

    predicted_masks = predict_image(
        image=image_tensor, 
        args=args,
        point_coords=None
    )
    
    output_image = Image.fromarray((predicted_masks[0, 0]).astype(np.uint8), mode='L')
    # binarized_mask = binarize_prediction(predicted_masks[0, 0])  # Assuming the first channel is the one you want
    # output_image = Image.fromarray(binarized_mask, mode='L')

    return output_image

def binarize_prediction(predicted_mask, threshold=0):

    binarized_mask = (predicted_mask > threshold).astype(np.uint8)
    return binarized_mask * 255 

def main():
    args = parse_args()

    css = """
    body { font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif; }
    .input_image, .output_image { border-radius: 10px; }
    .interface { box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    """

    iface = gr.Interface(
        fn=lambda image: predict_interface(image, args),
        inputs=gr.components.Image(type="filepath", label="上传图像"),
        outputs=gr.components.Image(type="pil", label="分割结果"),
        title="医学图像分割模型",
        description="上传图像查看分割结果。",
        theme="default", 
        css=css,
        examples=[['/mnt/SAM/flagged/amos_0507_31.png'], ['/mnt/SAM/flagged/s0114_111.png'], ['/mnt/SAM/flagged/s0619_32.png'],
                  ['/mnt/SAM/flagged/amos_0004_75.png'], ['/mnt/SAM/flagged/amos_0006_90.png']],
    )

    iface.launch(server_name='0.0.0.0', server_port=7860, share=True)

if __name__ == "__main__":
    main()
