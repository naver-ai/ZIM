"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

def print_once(message):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(message)

def pretty(d, indent=0):
    for key, value in d.items():
        print_once("\t" * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print_once("\t" * (indent + 1) + str(value))
