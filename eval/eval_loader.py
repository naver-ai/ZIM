"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import numpy as np
import torch
import json
from PIL import Image
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler

def get_evalloader(config):
    loader_dict = {}
    
    for data_type in config.dataset.data_type:
        dataset = Dataset(
            config.data_root,
            config.dataset,
            data_type,
        )

        if config.local_rank == 0:
            print(f"LOG) ZIM Dataset: {data_type} ({len(dataset)})")

        sampler = None

        if config.use_ddp:
            sampler = DistributedSampler(
                dataset, 
                rank=config.local_rank, 
                num_replicas=config.world_size,
            )

        dataloader = data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=config.eval.workers,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        loader_dict[data_type] = dataloader

    return loader_dict


class Dataset(data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_config,
        data_type,
    ):
        super(Dataset, self).__init__()
        self.root = os.path.join(data_root, dataset_config.valset)
        
        with open(os.path.join(self.root, dataset_config.data_list_txt), "r") as f:
            f_list = f.read().splitlines()
            f_list = [p for p in f_list if data_type in p]
            
        self.images = []
        self.mattes = []
        self.jsons = []

        for fname in f_list:
            img_path, matte_path, json_path, seg_path = fname.split(" ")
            
            img_path = os.path.join(self.root, img_path)
            matte_path = os.path.join(self.root, matte_path)
            json_path = os.path.join(self.root, json_path)
            
            self.images.append(img_path)
            self.mattes.append(matte_path)
            self.jsons.append(json_path)
                
        assert len(self.images) == len(self.mattes) == len(self.jsons)

        
    def __getitem__(self, index):
        fname = os.path.basename(self.mattes[index])
        
        img = Image.open(self.images[index]).convert('RGB')
        matte = Image.open(self.mattes[index]).convert('L')
        orig_w, orig_h = img.size
        
        img = np.float32(img)
        matte = np.float32(matte) / 255.
        
        ratio = (matte > 0.3).sum() / matte.size
        
        with open(self.jsons[index], "r") as f:
            meta_data = json.load(f)
        
        points = meta_data["point"]
        points += [(-1, -1, -1) for _ in range(50-len(points))] # padding
        
        bbox = meta_data["bbox"]
        
        output = {
            "images": torch.tensor(img, dtype=torch.float),
            "mattes": torch.tensor(matte, dtype=torch.float),
            "points": torch.tensor(points, dtype=torch.float),
            "bboxes": torch.tensor(bbox, dtype=torch.float),
            "fname": fname,
            "ratio": ratio,
        }
        
        return output
            
    def __len__(self):
        return len(self.images)

