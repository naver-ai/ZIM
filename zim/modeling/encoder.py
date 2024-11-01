"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from typing import Any, Callable
import onnxruntime

def np2tensor(np_array, device):
    return torch.from_numpy(np_array).to(device)

def tensor2np(torch_tensor):
    return torch_tensor.detach().cpu().numpy()
    
class ZIM_Encoder():
    def __init__(self, onnx_path, num_threads=16):
        self.onnx_path = onnx_path
        
        sessionOptions = onnxruntime.SessionOptions()
        sessionOptions.intra_op_num_threads = num_threads
        sessionOptions.inter_op_num_threads = num_threads
        providers = ["CPUExecutionProvider"]

        self.ort_session = onnxruntime.InferenceSession(
            onnx_path, sess_options=sessionOptions, providers=providers
        )
        
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
        image, 
    ):
        device = image.device
        
        ort_inputs = {
            "image": tensor2np(image),
        }
        image_embeddings, feat_D0, feat_D1, feat_D2 = self.ort_session.run(None, ort_inputs)
        
        image_embeddings = np2tensor(image_embeddings, device)
        feat_D0 = np2tensor(feat_D0, device)
        feat_D1 = np2tensor(feat_D1, device)
        feat_D2 = np2tensor(feat_D2, device)
        
        return image_embeddings, (feat_D0, feat_D1, feat_D2)
    
    __call__: Callable[..., Any] = forward
