"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os, sys
sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings(action="ignore")

import torch
import numpy as np
import random

from zim.utils import get_parser, print_once
from config.config import generate_config
from eval.main_eval import run_eval
from eval.evaluator import load_sam_evaluator, load_zim_evaluator
from eval.eval_loader import get_evalloader

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    return args

def main(args):

    config = generate_config(args)

    # Setup random seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    torch.cuda.set_device(config.local_rank)
    device = torch.device(f"cuda")

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        config.use_ddp = False
    
    # DDP init
    if config.use_ddp:
        torch.distributed.init_process_group(
            backend="nccl", rank=config.local_rank, world_size=n_gpus
        )
        config.world_size = torch.distributed.get_world_size()
        device = torch.device(f"cuda:{config.local_rank}")

    print_once("LOG) Initialization start")

    # Dataset list: string to list
    if isinstance(config.dataset.data_type, str):
        config.dataset.data_type = config.dataset.data_type.split(",")
    if isinstance(config.eval.prompt_type, str):
        config.eval.prompt_type = config.eval.prompt_type.split(",")
            
    # Benchmarking model list: str to list
    if isinstance(config.eval.model_list, str):
        config.eval.model_list = config.eval.model_list.split(",")

    val_loaders = get_evalloader(config)

    # Define SAM
    print_once("LOG) Start loading models")
    
    evaluator_dict = {}
    for model in config.eval.model_list:
        if model == "sam":
            evaluator_dict["sam"] = load_sam_evaluator(config, device)
        elif model == "zim":
            evaluator_dict["zim"] = load_zim_evaluator(config, device)
    
    print_once(f"LOG) Loading model {list(evaluator_dict.keys())}")
    print_once("LOG) Start evaluation")
    run_eval(
        config=config,
        valloader=val_loaders,
        evaluator_dict=evaluator_dict
    )

if __name__ == "__main__":
    args = parse_args()
    main(vars(args))
