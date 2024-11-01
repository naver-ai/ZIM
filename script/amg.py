"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import cv2
import glob
from tqdm import tqdm

import torch
from torch.multiprocessing import Process

from zim import zim_model_registry, ZimPredictor, ZimAutomaticMaskGenerator
from zim.utils import show_mat_anns

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

def get_argparser():
    parser = argparse.ArgumentParser()

    # Path option
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--model", type=str, default='zim,sam')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--workers", type=int, default=torch.cuda.device_count())

    parser.add_argument("--backbone", type=str, default='vit_b')
    parser.add_argument("--zim_ckpt", type=str, default=None)
    parser.add_argument("--sam_ckpt", type=str, default=None)
    
    parser.add_argument("--points_per_batch", type=int, default=16)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.6)
    parser.add_argument("--stability_score_thresh", type=float, default=0.9)
    parser.add_argument("--stability_score_offset", type=float, default=0.1)
    parser.add_argument("--box_nms_thresh", type=float, default=0.7)
    parser.add_argument("--crop_nms_thresh", type=float, default=0.7)
    return parser


def load_zim_amg(args):
    zim = zim_model_registry[args.backbone](checkpoint=args.zim_ckpt).cuda()
    mask_generator = ZimAutomaticMaskGenerator(
        zim, 
        pred_iou_thresh=args.pred_iou_thresh, 
        points_per_batch=args.points_per_batch,
        stability_score_thresh=args.stability_score_thresh, 
        stability_score_offset=args.stability_score_offset,
        box_nms_thresh=args.box_nms_thresh, 
        crop_nms_thresh=args.crop_nms_thresh
    )
    return mask_generator

def load_sam_amg(args):
    sam = sam_model_registry[args.backbone](checkpoint=args.sam_ckpt).cuda()
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=args.pred_iou_thresh, 
        points_per_batch=args.points_per_batch,
        stability_score_thresh=args.stability_score_thresh, 
        stability_score_offset=args.stability_score_offset,
        box_nms_thresh=args.box_nms_thresh, 
        crop_nms_thresh=args.crop_nms_thresh
    )
    return mask_generator


def run_amg(pid, args):
    with torch.cuda.device(pid):

        mask_generators = []
        if "zim" in args.model:
            mask_generators.append(load_zim_amg(args))
            
        if "sam" in args.model:
            mask_generators.append(load_sam_amg(args))
        
        for n, img_path in enumerate(tqdm(img_list)):
            if (n+1) % args.workers != pid:
                continue
        
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            with torch.cuda.amp.autocast(enabled=True):
                masks = []
                for mg in mask_generators:
                    m = mg.generate(image)
                    masks.append(m)

            vis = [image]
            for mask in masks:
                vis.append(show_mat_anns(image, mask))
            
            vis = cv2.hconcat(vis)
            
            save_path  = os.path.join(args.save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, vis[:, :, ::-1])


if __name__ == "__main__":
    
    args = get_argparser().parse_args()
    args.model = args.model.split(",")
    
    img_list = glob.glob(f'{args.img_dir}/**', recursive=True)
    img_list = [p for p in img_list if p.endswith((".jpg", ".png", ".jpeg"))]

    os.makedirs(args.save_dir, exist_ok=True)

    processes = []
    for i in range(args.workers):
        proc = Process(target=run_amg, args=(i, args))
        processes.append(proc)
        proc.start()
    for proc in processes:
        proc.join()
        
