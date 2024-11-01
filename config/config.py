"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from easydict import EasyDict as edict

config_ = edict()

"""
    Common configs
"""
config_.data_root = "/mnt/tmp"
config_.use_ddp = True
config_.use_amp = False
config_.local_rank = 0
config_.world_size = 1
config_.random_seed = 3407
"""
    Network configs
"""
config_.network = edict()
config_.network.encoder = "vit_b"
config_.network.decoder = "zim"
config_.network.encode_kernel = 21
"""
    Evaluation configs
"""
config_.eval = edict()
config_.eval.workers = 4
config_.eval.image_size = 1024
config_.eval.prompt_type = "point,bbox"
config_.eval.model_list = "zim,sam"
config_.eval.zim_weights = ""
config_.eval.sam_weights = ""
"""
    Dataset configs
"""
config_.dataset = edict()
config_.dataset.valset = "MicroMat3K"
config_.dataset.data_type = "fine,coarse"
config_.dataset.data_list_txt = "data_list.txt"


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def generate_config(args):
    # merge args & config
    for k, v in args.items():
        if k.startswith("network_"):
            config_["network"][remove_prefix(k, "network_")] = v
        elif k.startswith("eval_"):
            config_["eval"][remove_prefix(k, "eval_")] = v
        elif k.startswith("dataset_"):
            config_["dataset"][remove_prefix(k, "dataset_")] = v
        elif k == "amp":
            config_["use_amp"] = v
        else:
            config_[k] = v
    return config_
