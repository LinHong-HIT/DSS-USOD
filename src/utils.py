import torch
import torch.nn as nn
import torch.nn.functional as F


class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter + 1e-6) / (union + 1e-6)
        return iou.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        return dice.mean()


def structure_loss(pred, mask, mode="bce+iou"):
    bce = F.binary_cross_entropy_with_logits(pred, mask)

    if mode == "bce":
        return bce

    iou  = IOULoss()(pred, mask)
    dice = DiceLoss()(pred, mask)

    if mode == "bce+iou":
        return bce + iou
    if mode == "bce+dice":
        return bce + dice
    if mode == "bce+iou+dice":
        return bce + iou + dice

    raise ValueError(f"Unknown mode: {mode}")

 
 

