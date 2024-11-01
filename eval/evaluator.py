"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from typing import Any, Dict, List, Tuple
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from zim.build_model import build_zim_model
from zim.predictor import ZimPredictor
from segment_anything import SamPredictor, sam_model_registry

def load_sam_evaluator(config, device):
    sam = sam_model_registry[config.network.encoder](checkpoint=config.eval.sam_weights).cuda(device)
    sam_evaluator = SamEvaluator(sam, config.eval.prompt_type)
    if config.use_ddp:
        sam_evaluator = DDP(
            sam_evaluator,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
        )
    sam_evaluator.eval()
    return sam_evaluator

def load_zim_evaluator(config, device):
    zim = build_zim_model(config.eval.zim_weights).cuda(device)
    zim_evaluator = ZimEvaluator(zim, config.eval.prompt_type)
    
    return zim_evaluator

class SamEvaluator(SamPredictor, nn.Module):
    def __init__(
        self, 
        sam_model, 
        prompt_type: List[str] = None
    ):
        super().__init__(sam_model=sam_model)
        self.prompt_type = prompt_type

    def forward(self, batched_input, multimask_output: bool = False):
        input_images = batched_input["images"]
        
        outputs = {
            prompt: {
                "masks": [],
            } for prompt in self.prompt_type
        }

        with torch.inference_mode():
            for idx, input_image in enumerate(input_images):
                input_image = input_image.cpu().numpy().astype(np.uint8)
                self.set_image(image=input_image)

                for prompt in self.prompt_type:
                    point_coords = None
                    point_labels = None
                    bbox = None

                    if prompt == "point":
                        points = batched_input["points"][idx]
                        points = points[points[:, 2] >= 0] # remove points whose label=-1
                        point_coords = points[:, :2].cpu().numpy()
                        point_labels = points[:, 2].cpu().numpy()

                    elif prompt == "bbox":
                        bbox = batched_input["bboxes"][idx]
                        bbox = bbox.unsqueeze(0).cpu().numpy()

                    masks, _, _ = self.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=bbox,
                        multimask_output=False,
                    )
                    masks = torch.from_numpy(masks).float().unsqueeze(0).to(self.device)

                    outputs[prompt]["masks"].append(masks)
                
        # Concat through batch dimension
        for prompt in self.prompt_type:
            for k, v in outputs[prompt].items():
                if len(v) > 0:
                    outputs[prompt][k] = torch.cat(v, dim=0)

        return outputs


class ZimEvaluator(ZimPredictor, nn.Module):
    def __init__(
        self,
        model,
        prompt_type: List[str] = None
    ) -> None:
        super().__init__(model=model)
        self.prompt_type = prompt_type

    def forward(self, batched_input, multimask_output: bool = False):
        input_images = batched_input["images"]
        
        outputs = {
            prompt: {
                "masks": [],
            } for prompt in self.prompt_type
        }

        with torch.inference_mode():
            for idx, input_image in enumerate(input_images):
                input_image = input_image.cpu().numpy().astype(np.uint8)
                self.set_image(image=input_image)

                for prompt in self.prompt_type:
                    point_coords = None
                    point_labels = None
                    bbox = None

                    if prompt == "point":
                        points = batched_input["points"][idx]
                        points = points[points[:, 2] >= 0] # remove points whose label=-1
                        point_coords = points[:, :2].cpu().numpy()
                        point_labels = points[:, 2].cpu().numpy()

                    elif prompt == "bbox":
                        bbox = batched_input["bboxes"][idx]
                        bbox = bbox.cpu().numpy()

                    masks, _, _ = self.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=bbox,
                        multimask_output=False,
                    )
                    masks = torch.from_numpy(masks).float().unsqueeze(0).to(self.device)

                    outputs[prompt]["masks"].append(masks)

        for prompt in self.prompt_type:
            for k, v in outputs[prompt].items():
                if len(v) > 0:
                    outputs[prompt][k] = torch.cat(v, dim=0)

        return outputs
