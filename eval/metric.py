"""
Copyright (c) 2024-present Naver Cloud Corp.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F
from skimage.measure import label
import numpy as np

def compute_eval_scores(
    preds, gts, grad_filter, 
):
    """
    preds : (B 1 H W)
    gts : (B H W)
    """
    l1_dist_list = []
    l2_dist_list = []
    grad_list = []
    conn_error_list = []
    sad_error_list = []
    
    for pred, gt in zip(preds, gts):
        gt = gt.unsqueeze(0)
        
        l1_dist = F.l1_loss(pred, gt) * 1e3
        l2_dist = F.mse_loss(pred, gt) * 1e3
        grad = compute_grad(pred, gt, grad_filter) * 1e3
        sad_error = compute_sad_loss(pred, gt)
        conn_error = compute_connectivity_error_torch(pred, gt)

        l1_dist_list.append(l1_dist)
        l2_dist_list.append(l2_dist)
        grad_list.append(grad)
        conn_error_list.append(conn_error)
        sad_error_list.append(sad_error)

    l1_dist = torch.stack(l1_dist_list, dim=0)
    l2_dist = torch.stack(l2_dist_list, dim=0)
    grad = torch.stack(grad_list, dim=0)
    conn_error = torch.stack(conn_error_list, dim=0)
    sad_error = torch.stack(sad_error_list, dim=0)

    return {
        "l1": l1_dist.mean().item(),
        "l2": l2_dist.mean().item(),
        "grad": grad.mean().item(),
        "conn": conn_error.mean().item(),
        "sad": sad_error.mean().item(),
    }


def compute_grad(preds, labels, grad_filter):

    if preds.dim() == 3:
        preds = preds.unsqueeze(1)

    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    grad_preds = F.conv2d(preds, weight=grad_filter, padding=1)
    grad_labels = F.conv2d(labels, weight=grad_filter, padding=1)
    grad_preds = torch.sqrt((grad_preds * grad_preds).sum(dim=1, keepdim=True) + 1e-8)
    grad_labels = torch.sqrt(
        (grad_labels * grad_labels).sum(dim=1, keepdim=True) + 1e-8
    )

    return F.l1_loss(grad_preds, grad_labels)


def compute_sad_loss(pred, target):
    error_map = torch.abs((pred - target))
    loss = torch.sum(error_map)

    return loss / 1000.


def getLargestCC(segmentation):
    segmentation = segmentation.cpu().detach().numpy()
    labels = label(segmentation, connectivity=1)
    if labels.max() == 0:
        return np.zeros_like(segmentation, dtype=bool)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1  # Ignore background label
    return largestCC


def compute_connectivity_error_torch(pred, target, step=0.1):
    thresh_steps = list(torch.arange(0, 1 + step, step))
    l_map = torch.ones_like(pred, dtype=torch.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).to(dtype=torch.int)
        target_alpha_thresh = (target >= thresh_steps[i]).to(dtype=torch.int)

        omega = torch.from_numpy(getLargestCC(pred_alpha_thresh * target_alpha_thresh)).to(pred.device, dtype=torch.int)
        flag = ((l_map == -1) & (omega == 0)).to(dtype=torch.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).to(dtype=torch.int)
    target_phi = 1 - target_d * (target_d >= 0.15).to(dtype=torch.int)
    loss = torch.sum(torch.abs(pred_phi - target_phi))

    return loss / 1000.


def get_gradfilter(device):
    """
    generate gradient filter as the conv kernel
    """
    grad_filter = []
    grad_filter.append([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_filter.append([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    grad_filter = np.array(grad_filter)
    grad_filter = np.expand_dims(grad_filter, axis=1)
    grad_filter = grad_filter.astype(np.float32)
    return torch.tensor(grad_filter).to(device)
