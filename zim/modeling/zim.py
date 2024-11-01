"""
Copyright (c) 2024-present Naver Cloud Corp.
This source code is based on code from the Segment Anything Model (SAM)
(https://github.com/facebookresearch/segment-anything).

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List

def gaussian(sigma=6):
    """
    2D Gaussian Kernel Generation.
    """
    size = 6 * sigma + 3
    x = torch.arange(0, size, 1)
    y = x[:, None]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g

class Zim(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        *,
        image_size: int = 1024,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          encoder : The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          decoder : Predicts masks from the image embeddings and given prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_activation = nn.Sigmoid()

        self.image_size = image_size
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.mask_threshold: float = 0.5
        self.image_format: str = "RGB"
        self.num_mask_tokens = decoder.num_mask_tokens
        
        self.encode_stride = 16
        self.encode_kernel = 21
        self.attn_mask_size = 64
        self.g = gaussian(self.encode_kernel)
        
        self.output_conv = nn.Conv2d(
            self.num_mask_tokens, 
            self.num_mask_tokens, 
            kernel_size=1, stride=1, padding=0,
        )

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def cuda(self, device_id=None):
        if type(device_id) == torch.device:
            device_id = device_id.index
            
        if device_id is None:
            device_id = 0
        
        device = torch.device(f"cuda:{device_id}")
        super(Zim, self).cuda(device)
        
        self.encoder.cuda(device_id)
        self.decoder.cuda(device_id)
        
        return self

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: List[int], original_size: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the decoder,
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
            (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def bbox_attn_mask(self, boxes):
        """Prompt-aware Masked Attention: box prompt (binary attn mask) """
        bs = boxes.shape[0]
        attn_mask = torch.zeros((bs, self.attn_mask_size, self.attn_mask_size), device=boxes.device)
            
        # attn_weight = attn_weight.masked_fill(m.logical_not(), -1e4)
        
        for n in range(bs):
            xmin, ymin, xmax, ymax = boxes[n]
            
            xmin, xmax  = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax  = min(ymin, ymax), max(ymin, ymax)
            
            xmin, xmax = int(xmin / self.encode_stride), int(xmax / self.encode_stride)
            ymin, ymax = int(ymin / self.encode_stride), int(ymax / self.encode_stride)
            
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax = min(self.attn_mask_size, xmax+1)
            ymax = min(self.attn_mask_size, ymax+1)
            
            attn_mask[n, ymin:ymax, xmin:xmax] = 1
            
        return attn_mask
    
    def point_attn_mask(self, point_coords):
        """Prompt-aware Masked Attention: point prompt (soft attn mask) """
        bs = point_coords.shape[0]
        attn_mask = torch.zeros((bs, self.attn_mask_size, self.attn_mask_size), device=point_coords.device)
        
        if self.g.device != point_coords.device:
            self.g = self.g.to(point_coords.device)
                
        for n in range(bs):
            for point in point_coords[n]:
                x, y = int(point[0] / self.encode_stride), int(point[1].item() / self.encode_stride)
                
                # outside image boundary
                if x < 0 or y < 0 or x >= self.attn_mask_size or y >= self.attn_mask_size:
                    continue

                # upper left
                ul = int(round(x - 3 * self.encode_kernel - 1)), int(round(y - 3 * self.encode_kernel - 1))
                # bottom right
                br = int(round(x + 3 * self.encode_kernel + 2)), int(round(y + 3 * self.encode_kernel + 2))

                c, d = int(max(0, -ul[0])), int(min(br[0], self.attn_mask_size) - ul[0])
                a, b = int(max(0, -ul[1])), int(min(br[1], self.attn_mask_size) - ul[1])

                cc, dd = int(max(0, ul[0])), int(min(br[0], self.attn_mask_size))
                aa, bb = int(max(0, ul[1])), int(min(br[1], self.attn_mask_size))

                attn_mask[n, aa:bb, cc:dd] = torch.maximum(
                    attn_mask[n, aa:bb, cc:dd], self.g[a:b, c:d]
                )
                
        return attn_mask
    
    