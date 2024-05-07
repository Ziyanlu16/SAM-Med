# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    # 类属性定义
    mask_threshold: float = 0.0  # 掩码阈值，用于二值化处理预测的掩码
    image_format: str = "RGB"  # 图像格式，默认为RGB

    # 初始化方法
    def __init__(
        self,
        image_encoder: ImageEncoderViT,  # 图像编码器，使用ViT（Vision Transformer）作为主干网络编码图像
        prompt_encoder: PromptEncoder,  # 提示编码器，用于编码各种类型的输入提示
        mask_decoder: MaskDecoder,  # 掩码解码器，用于从图像嵌入和编码提示中预测掩码
        pixel_mean: List[float] = [123.675, 116.28, 103.53],  # 用于归一化像素值的平均值
        pixel_std: List[float] = [58.395, 57.12, 57.375],  # 用于归一化像素值的标准差
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # 注册缓冲区，用于存储归一化的平均值和标准差，使其能够跟随模型一起移动到不同的设备
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        # 返回模型当前所在的设备
        return self.pixel_mean.device

    @torch.no_grad()  # 不计算梯度，用于推理模式
    def forward(
        self,
        batched_input: List[Dict[str, Any]],  # 批量输入数据，每个元素是一个包含图像和提示信息的字典
        multimask_output: bool,  # 是否预测多个掩码以消除歧义，或仅返回单个掩码
    ) -> List[Dict[str, torch.Tensor]]:
        # 预处理输入图像并堆叠成批量数据
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        # 使用图像编码器编码图像
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            # 根据输入记录提取点提示信息
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            # 使用提示编码器编码提示信息
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            # 使用掩码解码器根据图像嵌入和提示嵌入预测掩码
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            # 后处理掩码，调整大小并二值化
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            # 收集输出结果
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
