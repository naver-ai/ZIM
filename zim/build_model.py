"""
Copyright (c) 2024-present Naver Cloud Corp.
This source code is based on code from the Segment Anything Model (SAM)
(https://github.com/facebookresearch/segment-anything).

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import torch

from .modeling.zim import Zim
from .modeling.encoder import ZIM_Encoder
from .modeling.decoder import ZIM_Decoder

def build_zim_model(checkpoint):
    
    encoder = ZIM_Encoder(os.path.join(checkpoint, "encoder.onnx"))
    decoder = ZIM_Decoder(os.path.join(checkpoint, "decoder.onnx"))
    net = Zim(encoder, decoder)

    return net

zim_model_registry = {
    "default": build_zim_model,
    "vit_l": build_zim_model,
    "vit_b": build_zim_model,
}

