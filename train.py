import os
import cv2
import copy
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dataset import USODDataset
from src.model2 import USODNet
from src.utils import structure_loss


# =========================================================
# 1. Config
# =========================================================
DATA_ROOT = "/home/ubuntu/path/data/USOD10K"
TRAIN_SIZE = 352

BATCH_SIZE = 46          # per GPU
NUM_WORKERS = 5
EPOCHS = 50

SAVE_DIR = "checkpoints_v1"
DEBUG_DIR = "debug_vis_v1"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# segmentation loss weights
W_final = 4.0
W_coarse = 0.25
W_small = 0.25

STAGE1_EPOCHS = 15
LAMBDA_ROUTER_STAGE1 = 0.50
LAMBDA_EDGE_STAGE1 = 0.10

LAMBDA_ROUTER_STAGE2 = 0.10
LAMBDA_EDGE_STAGE2 = 0.03

USE_AMP = True
EMA_ENABLED = True
EMA_DECAY = 0.999

history = {
    "train_step": [],
    "train_loss": [],
    "router_loss": [],
    "edge_loss": [],
    "val_epoch": [],
    "val_mae_total": [],
    "val_mae_final": [],
    "val_mae_coarse": [],
    "val_mae_small": [],
}


# =========================================================
# 2. DDP helpers
# =========================================================
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    return local_rank, device


def cleanup_ddp():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


# =========================================================
# 3. EMA
# =========================================================
class ModelEMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, module: nn.Module):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, module: nn.Module):
        self.backup = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, module: nn.Module):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name].data)
        self.backup = {}

    @torch.no_grad()
    def state_dict(self, module: nn.Module):
        sd = copy.deepcopy(module.state_dict())
        for name, tensor in sd.items():
            if name in self.shadow:
                sd[name] = self.shadow[name].detach().clone()
        return sd


# =========================================================
# 4. losses
# =========================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return dice.mean()


dice_loss_fn = DiceLoss()


def edge_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss_fn(pred, target)
    return bce + dice


def make_router_target(edge, k_dilate=5, k_smooth=5):
    edge = edge.float()
    edge = F.max_pool2d(edge, kernel_size=k_dilate, stride=1, padding=k_dilate // 2)
    edge = F.avg_pool2d(edge, kernel_size=k_smooth, stride=1, padding=k_smooth // 2)
    return edge.clamp(0, 1)


def make_edge_target(edge, smooth=False):
    edge = edge.float()
    if smooth:
        edge = F.avg_pool2d(edge, kernel_size=3, stride=1, padding=1)
    return edge.clamp(0, 1)


def get_aux_weights(epoch: int):
    if epoch < STAGE1_EPOCHS:
        return LAMBDA_ROUTER_STAGE1, LAMBDA_EDGE_STAGE1
    return LAMBDA_ROUTER_STAGE2, LAMBDA_EDGE_STAGE2


# =========================================================
# 5. visualization / history
# =========================================================
def save_debug_images(rgb, gt_mask, pred_prob, epoch, tag="train"):
    if not is_main_process():
        return

    idx = 0
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(3, 1, 1)

    rgb_img = rgb[idx] * std + mean
    rgb_img = torch.clamp(rgb_img, 0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255
    rgb_img = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

    gt_m = (gt_mask[idx, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    gt_m = cv2.cvtColor(gt_m, cv2.COLOR_GRAY2BGR)

    pr_m = (pred_prob[idx, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    pr_m = cv2.cvtColor(pr_m, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([rgb_img, gt_m, pr_m])
    cv2.putText(
        combined,
        f"{tag}_Ep{epoch}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{tag}_ep{epoch}.jpg"), combined)


def plot_history():
    if not is_main_process():
        return

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_step"], history["train_loss"], label="Train Total Loss", alpha=0.8)
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history["val_epoch"], history["val_mae_total"], "r-o", label="Val Weighted MAE", linewidth=2)
    plt.plot(history["val_epoch"], history["val_mae_final"], "b--o", label="MAE Final", alpha=0.8)
    plt.plot(history["val_epoch"], history["val_mae_coarse"], "g--o", label="MAE Coarse", alpha=0.8)
    plt.plot(history["val_epoch"], history["val_mae_small"], "m--o", label="MAE Small", alpha=0.8)
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    if len(history["router_loss"]) > 0:
        plt.plot(history["train_step"][:len(history["router_loss"])], history["router_loss"], label="Router Loss")
    if len(history["edge_loss"]) > 0:
        plt.plot(history["train_step"][:len(history["edge_loss"])], history["edge_loss"], label="Edge Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"))
    plt.close()


# =========================================================
# 6. validation
# =========================================================
@torch.no_grad()
def validate_three_stage_mae(model, val_loader, device, epoch, amp_enabled):
    model.eval()

    total_abs_final = 0.0
    total_pix_final = 0.0

    total_abs_coarse = 0.0
    total_pix_coarse = 0.0

    total_abs_small = 0.0
    total_pix_small = 0.0

    saved_vis = False
    target_batch_idx = epoch % max(len(val_loader), 1)

    for i, (rgb, gt_mask, gt_edge) in enumerate(val_loader):
        rgb = rgb.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            final_logits, coarse_full_logits, seg_logits_small, aux = model(rgb)

        pred_final = torch.sigmoid(final_logits.float())
        pred_coarse = torch.sigmoid(coarse_full_logits.float())
        pred_small = torch.sigmoid(seg_logits_small.float())

        if pred_final.shape[-2:] != gt_mask.shape[-2:]:
            pred_final = F.interpolate(
                pred_final,
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if pred_coarse.shape[-2:] != gt_mask.shape[-2:]:
            pred_coarse = F.interpolate(
                pred_coarse,
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pred_final = pred_final.clamp(0, 1)
        pred_coarse = pred_coarse.clamp(0, 1)

        gt_small = F.interpolate(gt_mask, size=pred_small.shape[-2:], mode="nearest")
        pred_small = pred_small.clamp(0, 1)

        total_abs_final += torch.abs(pred_final - gt_mask).sum().item()
        total_pix_final += gt_mask.numel()

        total_abs_coarse += torch.abs(pred_coarse - gt_mask).sum().item()
        total_pix_coarse += gt_mask.numel()

        total_abs_small += torch.abs(pred_small - gt_small).sum().item()
        total_pix_small += gt_small.numel()

        if i == target_batch_idx and not saved_vis and is_main_process():
            save_debug_images(rgb, gt_mask, pred_final, epoch + 1, tag="val")
            saved_vis = True

    stats = torch.tensor(
        [
            total_abs_final, total_pix_final,
            total_abs_coarse, total_pix_coarse,
            total_abs_small, total_pix_small,
        ],
        device=device,
        dtype=torch.float64,
    )

    if is_dist_avail_and_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    (
        total_abs_final_all, total_pix_final_all,
        total_abs_coarse_all, total_pix_coarse_all,
        total_abs_small_all, total_pix_small_all,
    ) = stats.tolist()

    mae_final = total_abs_final_all / max(total_pix_final_all, 1.0)
    mae_coarse = total_abs_coarse_all / max(total_pix_coarse_all, 1.0)
    mae_small = total_abs_small_all / max(total_pix_small_all, 1.0)

    mae_total = W_final * mae_final + W_coarse * mae_coarse + W_small * mae_small

    return {
        "mae_total": mae_total,
        "mae_final": mae_final,
        "mae_coarse": mae_coarse,
        "mae_small": mae_small,
    }


# =========================================================
# 7. train
# =========================================================
def train():
    local_rank, device = setup_ddp()
    amp_enabled = USE_AMP and device.type == "cuda"

    if is_main_process():
        print(f"Initializing Dataset (Size={TRAIN_SIZE})...")

    full_dataset = USODDataset(DATA_ROOT, mode="train", size=TRAIN_SIZE)
    val_path = os.path.join(DATA_ROOT, "val")

    if os.path.exists(val_path):
        train_dataset = full_dataset
        val_dataset = USODDataset(DATA_ROOT, mode="val", size=TRAIN_SIZE)
    else:
        val_size = int(0.1 * len(full_dataset))
        generator = torch.Generator().manual_seed(2026)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [len(full_dataset) - val_size, val_size],
            generator=generator,
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    if is_main_process():
        print("Creating Model: USODNet")

    model = USODNet(in_chans=3).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = ModelEMA(model.module, decay=EMA_DECAY) if EMA_ENABLED else None
    bce_logits = nn.BCEWithLogitsLoss()

    if is_main_process():
        print(
            f"Start Training | W_final={W_final}, W_coarse={W_coarse}, W_small={W_small}, "
            f"stage1_router={LAMBDA_ROUTER_STAGE1}, stage1_edge={LAMBDA_EDGE_STAGE1}, "
            f"stage2_router={LAMBDA_ROUTER_STAGE2}, stage2_edge={LAMBDA_EDGE_STAGE2}, "
            f"STAGE1_EPOCHS={STAGE1_EPOCHS}, AMP={amp_enabled}, EMA={EMA_ENABLED}, EMA_DECAY={EMA_DECAY}"
        )
        print("Validation criterion: 3-stage weighted MAE")
        print("Best checkpoint criterion: lowest weighted Val MAE")

    best_val_mae = float("inf")
    global_step = 0

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        lambda_router, lambda_edge = get_aux_weights(epoch)

        for i, (rgb, gt_mask, gt_edge) in enumerate(train_loader):
            rgb = rgb.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            gt_edge = gt_edge.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                final_logits, coarse_full_logits, seg_logits_small, aux = model(rgb)

                # segmentation losses
                loss_final = structure_loss(final_logits, gt_mask)
                loss_coarse = structure_loss(coarse_full_logits, gt_mask)

                gt_small = F.interpolate(gt_mask, size=seg_logits_small.shape[-2:], mode="nearest")
                loss_small = structure_loss(seg_logits_small, gt_small)

                loss_seg = W_final * loss_final + W_coarse * loss_coarse + W_small * loss_small

                # auxiliary supervision
                gt_edge_small = F.interpolate(gt_edge, size=seg_logits_small.shape[-2:], mode="nearest")
                router_target = make_router_target(gt_edge_small, k_dilate=3, k_smooth=3)
                edge_target = make_edge_target(gt_edge_small, smooth=False)

                router_logits = aux["router_logits"]
                edge_logits = aux["edge_logits"]

                loss_router = bce_logits(router_logits, router_target)
                loss_edge = edge_loss(edge_logits, edge_target)

                loss = loss_seg + lambda_router * loss_router + lambda_edge * loss_edge

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model.module)

            if is_main_process():
                history["train_step"].append(global_step)
                history["train_loss"].append(float(loss.item()))
                history["router_loss"].append(float(loss_router.item()))
                history["edge_loss"].append(float(loss_edge.item()))
                global_step += 1

                if i % 20 == 0:
                    print(
                        f"Ep[{epoch + 1}/{EPOCHS}] Step[{i}/{len(train_loader)}] "
                        f"Tot:{loss.item():.4f} Seg:{loss_seg.item():.4f} "
                        f"Router:{loss_router.item():.4f} Edge:{loss_edge.item():.4f} "
                        f"(lambda_router={lambda_router:.3f}, lambda_edge={lambda_edge:.3f})"
                    )

                if i == len(train_loader) - 1:
                    pred_prob = torch.sigmoid(final_logits.float())
                    if pred_prob.shape[-2:] != gt_mask.shape[-2:]:
                        pred_prob = F.interpolate(
                            pred_prob,
                            size=gt_mask.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    pred_prob = pred_prob.clamp(0, 1)
                    save_debug_images(rgb, gt_mask, pred_prob, epoch + 1, tag="train")

        scheduler.step()

        if ema is not None:
            ema.apply_shadow(model.module)

        val_metrics = validate_three_stage_mae(model, val_loader, device, epoch, amp_enabled)

        if ema is not None:
            ema.restore(model.module)

        if is_main_process():
            val_mae_total = float(val_metrics["mae_total"])
            val_mae_final = float(val_metrics["mae_final"])
            val_mae_coarse = float(val_metrics["mae_coarse"])
            val_mae_small = float(val_metrics["mae_small"])

            history["val_epoch"].append(epoch + 1)
            history["val_mae_total"].append(val_mae_total)
            history["val_mae_final"].append(val_mae_final)
            history["val_mae_coarse"].append(val_mae_coarse)
            history["val_mae_small"].append(val_mae_small)

            print(
                f"--- Ep {epoch + 1} Val Weighted MAE (EMA): {val_mae_total:.6f} | "
                f"Final: {val_mae_final:.6f}, Coarse: {val_mae_coarse:.6f}, Small: {val_mae_small:.6f} ---"
            )
            plot_history()

            save_state = ema.state_dict(model.module) if ema is not None else model.module.state_dict()
            torch.save(save_state, os.path.join(SAVE_DIR, "last.pth"))

            if val_mae_total < best_val_mae:
                best_val_mae = val_mae_total
                torch.save(save_state, os.path.join(SAVE_DIR, "best.pth"))
                print(f"*** Best Saved (Weighted Val MAE EMA: {best_val_mae:.6f}) ***")

    cleanup_ddp()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    train()
