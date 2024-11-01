"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
from config.config import config_

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser(verbose=False):
    p = argparse.ArgumentParser("argparser", add_help=False)

    p.add_argument(
        "--data-root", type=str, default=config_.data_root, help="data root directory"
    )
    p.add_argument(
        "--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0"))
    )
    p.add_argument(
        "--amp", type=str2bool, default=True
    )
    p.add_argument(
        "--ddp", action="store_true"
    )
    p.add_argument(
        "--random-seed", type=int, default=config_.random_seed
    )

    # network config
    p.add_argument(
        "--network-encoder",
        type=str,
        default=config_.network.encoder,
        choices=["vit_b", "vit_l"],
    )
    p.add_argument(
        "--network-decoder",
        type=str,
        default=config_.network.decoder,
        choices=["zim", "sam"],
    )
    p.add_argument(
        "--network-encode-kernel",
        type=int,
        default=config_.network.encode_kernel,
    )
    
    # evaluation config
    p.add_argument(
        "--eval-workers", type=int, default=config_.eval.workers,
    )
    p.add_argument(
        "--eval-image-size", type=int, default=config_.eval.image_size,
    )
    p.add_argument(
        "--eval-prompt-type", type=str, default=config_.eval.prompt_type,
    )
    p.add_argument(
        "--eval-model-list", type=str, default=config_.eval.model_list,
    )
    p.add_argument(
        "--eval-zim-weights",
        type=str,
        default=config_.eval.zim_weights,
    )
    p.add_argument(
        "--eval-sam-weights",
        type=str,
        default=config_.eval.sam_weights,
    )
    
    # dataset config
    p.add_argument(
        "--dataset-valset", type=str, default=config_.dataset.valset,
    )
    p.add_argument(
        "--dataset-data-type", type=str, default=config_.dataset.data_type,
    )
    p.add_argument(
        "--dataset-data-list-txt", type=str, default=config_.dataset.data_list_txt,
    )
    
    return p
