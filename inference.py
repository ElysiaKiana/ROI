"""
推理与 ROI 导出脚本

功能:
1. 加载训练好的模型，对 LLVIP 测试集推理生成显著性图
2. 利用空间密度聚类切出 ROI 切片
3. 保存 ROI 切片图像 + 生成 tasks.csv（供第二、三章使用）

tasks.csv 格式:
  frame_id, roi_id, x1, y1, x2, y2, is_crowd, num_targets, data_size_kb, roi_path

使用方式:
  python inference.py --config configs/default.yaml --checkpoint checkpoints/best.pth --data_root ./data/LLVIP
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.roi_network import build_model_from_config
from data.llvip_dataset import LLVIPDataset
from utils.clustering import SpatialDensityClustering


def main():
    parser = argparse.ArgumentParser(description="ROI Inference & Export")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data/LLVIP")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_saliency", action="store_true", help="保存显著性图")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载模型
    model = build_model_from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    if "val_metrics" in ckpt:
        print(f"  Val metrics: {ckpt['val_metrics']}")

    # 数据集
    dataset_cfg = cfg.get("dataset", {})
    img_size = tuple(dataset_cfg.get("img_size", [512, 640]))

    dataset = LLVIPDataset(
        root=args.data_root,
        split=args.split,
        img_size=img_size,
        use_annotations=False,
        augment=False,
    )

    # ROI 生成器 (v2 — 基于连通域 bbox)
    roi_cfg = cfg.get("roi", {})
    roi_generator = SpatialDensityClustering(
        saliency_threshold=roi_cfg.get("saliency_threshold", 0.5),
        min_area=roi_cfg.get("min_area", 100),
        dbscan_eps=roi_cfg.get("dbscan_eps", 80),
        dbscan_min_samples=roi_cfg.get("dbscan_min_samples", 1),
        padding_ratio=roi_cfg.get("padding_ratio", 0.3),
        min_padding=roi_cfg.get("min_padding", 15),
        crowd_padding_ratio=roi_cfg.get("crowd_padding_ratio", 0.25),
        crowd_threshold=roi_cfg.get("crowd_threshold", 3),
        min_roi_size=roi_cfg.get("min_roi_size", 64),
        jpeg_quality=roi_cfg.get("jpeg_quality", 85),
    )

    # 输出目录
    output_dir = Path(args.output_dir)
    roi_dir = output_dir / "roi_slices"
    saliency_dir = output_dir / "saliency_maps"
    roi_dir.mkdir(parents=True, exist_ok=True)
    if args.save_saliency:
        saliency_dir.mkdir(parents=True, exist_ok=True)

    # CSV 输出
    csv_path = output_dir / "tasks.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "roi_id", "x1", "y1", "x2", "y2",
        "is_crowd", "num_targets", "data_size_kb", "roi_path",
    ])

    total_rois = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            sample = dataset[idx]
            stem = sample["stem"]

            ir = sample["ir"].unsqueeze(0).to(device)     # (1, 1, H, W)
            rgb = sample["rgb"].unsqueeze(0).to(device)   # (1, 3, H, W)

            # 模型推理
            outputs = model(ir, rgb)
            saliency_map = outputs["saliency_map"].squeeze().cpu().numpy()  # (H, W)

            # 保存显著性图
            if args.save_saliency:
                sal_vis = (saliency_map * 255).astype(np.uint8)
                sal_color = cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET)
                cv2.imwrite(str(saliency_dir / f"{stem}_saliency.png"), sal_color)

            # 获取原始 RGB 图像 (用于切片)
            rgb_np = sample["rgb"].permute(1, 2, 0).numpy()  # (H, W, 3), float [0,1]
            rgb_uint8 = (rgb_np * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

            # 生成 ROI 切片
            rois = roi_generator.generate_rois(saliency_map, rgb_bgr)

            for roi in rois:
                # 保存 ROI 切片
                roi_crop = rgb_bgr[roi.y1:roi.y2, roi.x1:roi.x2]
                roi_filename = f"{stem}_roi{roi.roi_id}.jpg"
                roi_path = roi_dir / roi_filename
                cv2.imwrite(
                    str(roi_path), roi_crop,
                    [cv2.IMWRITE_JPEG_QUALITY, roi_cfg.get("jpeg_quality", 85)],
                )

                # 写入 CSV
                csv_writer.writerow([
                    stem, roi.roi_id,
                    roi.x1, roi.y1, roi.x2, roi.y2,
                    int(roi.is_crowd), roi.num_targets,
                    f"{roi.data_size_kb:.2f}",
                    str(roi_path.relative_to(output_dir)),
                ])
                total_rois += 1

    csv_file.close()

    print(f"\n{'='*50}")
    print(f"推理完成!")
    print(f"  处理帧数: {len(dataset)}")
    print(f"  生成 ROI: {total_rois}")
    print(f"  平均 ROI/帧: {total_rois / max(len(dataset), 1):.1f}")
    print(f"  CSV 输出: {csv_path}")
    print(f"  ROI 切片: {roi_dir}")
    if args.save_saliency:
        print(f"  显著性图: {saliency_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
