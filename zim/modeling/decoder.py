"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from typing import Any, Callable
import onnxruntime
import numpy as np

def np2tensor(np_array, device):
    return torch.from_numpy(np_array).to(device)

def tensor2np(torch_tensor):
    if torch_tensor is None:
        return None
    
    return torch_tensor.detach().cpu().numpy()
    
class ZIM_Decoder():
    def __init__(self, onnx_path, num_threads=16):
        self.onnx_path = onnx_path
        
        sessionOptions = onnxruntime.SessionOptions()
        sessionOptions.intra_op_num_threads = num_threads
        sessionOptions.inter_op_num_threads = num_threads
        providers = ["CPUExecutionProvider"]

        self.ort_session = onnxruntime.InferenceSession(
            onnx_path, sess_options=sessionOptions, providers=providers
        )
        self.num_mask_tokens = 4
        
    def cuda(self, device_id=0):
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                },
            ),
        ]

        self.ort_session.set_providers(providers)
        
    def forward(
        self, 
        interm_feats,
        image_embeddings, 
        points, 
        boxes,
        attn_mask,
    ):
        device = image_embeddings.device
        
        ort_inputs = {
            "feat_D0": tensor2np(interm_feats[0]),
            "feat_D1": tensor2np(interm_feats[1]),
            "feat_D2": tensor2np(interm_feats[2]),
            "image_embeddings": tensor2np(image_embeddings),
            "attn_mask": tensor2np(attn_mask),
        }
        
        if points is not None:
            point_coords, point_labels = points
            ort_inputs["point_coords"] = tensor2np(point_coords.float())
            ort_inputs["point_labels"] = tensor2np(point_labels.float())
            
            # add paddings as done in SAM
            padding_point = np.zeros((ort_inputs["point_coords"].shape[0], 1, 2), dtype=np.float32) - 0.5
            padding_label = -np.ones((ort_inputs["point_labels"].shape[0], 1), dtype=np.float32)
            ort_inputs["point_coords"] = np.concatenate([ort_inputs["point_coords"], padding_point], axis=1)
            ort_inputs["point_labels"] = np.concatenate([ort_inputs["point_labels"], padding_label], axis=1)
            
        if boxes is not None:
            ort_inputs["point_coords"] = tensor2np(boxes.reshape(-1, 2, 2))
            ort_inputs["point_labels"] = np.array([[2, 3]], dtype=np.float32).repeat(boxes.shape[0], 0)
        
        masks, iou_predictions = self.ort_session.run(None, ort_inputs)
        
        masks = np2tensor(masks, device)
        iou_predictions = np2tensor(iou_predictions, device)
        
        return masks, iou_predictions
    
    __call__: Callable[..., Any] = forward
    