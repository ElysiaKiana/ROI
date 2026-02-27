"""
训练脚本 —— 非对称双流 ROI 网络

损失函数组合:
  L_total = w_bce × BCE + w_dice × Dice + w_iou × IoU

使用方式:
  python train.py --config configs/default.yaml
  python train.py --config configs/default.yaml --resume checkpoints/epoch_50.pth
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import yaml
from tqdm import tqdm

from models.roi_network import ROINetwork, build_model_from_config
from data.llvip_dataset import LLVIPDataset


# ========== 损失函数 ==========

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class IoULoss(nn.Module):
    """IoU Loss"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss — 解决正负样本严重不平衡 (GT覆盖率仅~6.5%)

    L_focal = -α_t × (1 - p_t)^γ × log(p_t)

    当 γ>0 时，对"容易"的背景像素降权，对"困难"的前景像素增权。
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # 正样本权重 (前景占比小 → 高α)
        self.gamma = gamma   # 聚焦参数 (γ=2 为 RetinaNet 经典值)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        # p_t = p if target=1 else (1-p)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """组合损失: Focal BCE + Dice + IoU

    v1: BCE + Dice + IoU (正负样本不平衡导致背景主导梯度)
    v2: Focal + Dice + IoU (focal 抑制易分类背景, dice/iou 关注前景)
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, iou_weight=0.5,
                 focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight

        self.focal = FocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.iou = IoULoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)
        iou_loss = self.iou(logits, target)

        total = (
            self.bce_weight * focal_loss
            + self.dice_weight * dice_loss
            + self.iou_weight * iou_loss
        )

        return {
            "total": total,
            "bce": focal_loss.item(),
            "dice": dice_loss.item(),
            "iou": iou_loss.item(),
        }


# ========== 评估指标 ==========

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold=0.5):
    """计算 Precision, Recall, F1, IoU"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = target.float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


# ========== 训练流程 ==========

def train_one_epoch(model, loader, criterion, optimizer, device, log_interval=10):
    model.train()
    total_loss = 0
    total_metrics = {"precision": 0, "recall": 0, "f1": 0, "iou": 0}

    pbar = tqdm(loader, desc="Training")
    for i, batch in enumerate(pbar):
        ir = batch["ir"].to(device)
        rgb = batch["rgb"].to(device)
        gt = batch["saliency_gt"].to(device)

        # Forward
        outputs = model(ir, rgb)
        logits = outputs["saliency_logits"]

        # Loss
        loss_dict = criterion(logits, gt)
        loss = loss_dict["total"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(logits, gt)

        total_loss += loss.item()
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        if (i + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "F1": f"{metrics['f1']:.3f}",
                "IoU": f"{metrics['iou']:.3f}",
            })

    n = len(loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_metrics = {"precision": 0, "recall": 0, "f1": 0, "iou": 0}

    for batch in tqdm(loader, desc="Validation"):
        ir = batch["ir"].to(device)
        rgb = batch["rgb"].to(device)
        gt = batch["saliency_gt"].to(device)

        outputs = model(ir, rgb)
        logits = outputs["saliency_logits"]

        loss_dict = criterion(logits, gt)
        total_loss += loss_dict["total"].item()

        metrics = compute_metrics(logits, gt)
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = len(loader)
    avg_loss = total_loss / n
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ROI Network")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建模型
    model = build_model_from_config(cfg).to(device)
    print(model.count_flops())

    # 构建数据集
    dataset_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})

    full_dataset = LLVIPDataset(
        root=dataset_cfg.get("root", "./data/LLVIP"),
        split="train",
        img_size=tuple(dataset_cfg.get("img_size", [512, 640])),
        augment=True,
        gt_mode=dataset_cfg.get("gt_mode", "gaussian"),
        sigma_ratio=dataset_cfg.get("sigma_ratio", 0.3),
        ir_weighting=dataset_cfg.get("ir_weighting", True),
    )

    # 划分训练/验证集
    train_ratio = dataset_cfg.get("train_split", 0.8)
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # 损失函数
    loss_cfg = train_cfg.get("loss", {})
    criterion = CombinedLoss(
        bce_weight=loss_cfg.get("bce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 1.0),
        iou_weight=loss_cfg.get("iou_weight", 0.5),
        focal_alpha=loss_cfg.get("focal_alpha", 0.75),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    # 学习率调度器
    scheduler_type = train_cfg.get("lr_scheduler", "cosine")
    epochs = train_cfg.get("epochs", 100)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("step_size", 30),
            gamma=train_cfg.get("gamma", 0.1),
        )

    # 恢复训练
    start_epoch = 0
    best_f1 = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0)
        print(f"Resumed from epoch {start_epoch}, best F1={best_f1:.4f}")

    # 输出目录
    output_cfg = cfg.get("output", {})
    ckpt_dir = Path(output_cfg.get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ===== 训练循环 =====
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")

        # 训练
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval=output_cfg.get("log_interval", 10),
        )

        # 验证
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        # 打印结果
        print(f"\n[Train] Loss={train_loss:.4f} | "
              f"P={train_metrics['precision']:.3f} R={train_metrics['recall']:.3f} "
              f"F1={train_metrics['f1']:.3f} IoU={train_metrics['iou']:.3f}")
        print(f"[Val]   Loss={val_loss:.4f} | "
              f"P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
              f"F1={val_metrics['f1']:.3f} IoU={val_metrics['iou']:.3f}")

        # 保存 checkpoint
        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]

        save_interval = output_cfg.get("save_interval", 5)
        if (epoch + 1) % save_interval == 0 or is_best:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch+1}.pth")
            if is_best:
                torch.save(ckpt, ckpt_dir / "best.pth")
                print(f"★ New best model saved! F1={best_f1:.4f}")

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
