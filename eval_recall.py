"""计算 ROI 的行人覆盖率 (Person Recall)

对 LLVIP 测试集的每个 GT 标注行人 bbox，检查是否被至少一个 ROI 覆盖。
覆盖标准: GT bbox 中心在某个 ROI 内 或 IoU > 阈值。

这是评估 ROI 提取质量的最核心指标，因为漏报代价极高。
"""

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def load_gt_boxes(ann_dir, stem):
    """加载 GT bbox"""
    xml_path = Path(ann_dir) / f"{stem}.xml"
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 原始标注在 1280x1024，推理 resize 到 640x512
    size_elem = root.find("size")
    orig_w = int(size_elem.find("width").text)
    orig_h = int(size_elem.find("height").text)
    scale_x = 640 / orig_w
    scale_y = 512 / orig_h
    
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is not None:
            x1 = int(float(bbox.find("xmin").text) * scale_x)
            y1 = int(float(bbox.find("ymin").text) * scale_y)
            x2 = int(float(bbox.find("xmax").text) * scale_x)
            y2 = int(float(bbox.find("ymax").text) * scale_y)
            boxes.append((x1, y1, x2, y2))
    return boxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


def center_in_box(gt_box, roi_box):
    """GT bbox 中心是否在 ROI 内"""
    cx = (gt_box[0] + gt_box[2]) / 2
    cy = (gt_box[1] + gt_box[3]) / 2
    return roi_box[0] <= cx <= roi_box[2] and roi_box[1] <= cy <= roi_box[3]


def evaluate_recall(csv_path, ann_dir, iou_threshold=0.3):
    """计算行人覆盖率"""
    # 读取 ROI
    rois_by_frame = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row['frame_id']
            box = (int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2']))
            rois_by_frame.setdefault(stem, []).append(box)

    all_stems = sorted(rois_by_frame.keys())
    
    total_gt = 0
    covered_center = 0
    covered_iou = 0
    frames_with_gt = 0
    frames_all_covered = 0

    ann_path = Path(ann_dir)
    for stem in all_stems:
        gt_boxes = load_gt_boxes(ann_path, stem)
        if not gt_boxes:
            continue
        
        frames_with_gt += 1
        rois = rois_by_frame.get(stem, [])
        
        frame_covered = True
        for gt in gt_boxes:
            total_gt += 1
            # 中心覆盖
            if any(center_in_box(gt, roi) for roi in rois):
                covered_center += 1
            else:
                frame_covered = False
            # IoU 覆盖
            if any(compute_iou(gt, roi) >= iou_threshold for roi in rois):
                covered_iou += 1
        
        if frame_covered:
            frames_all_covered += 1

    return {
        "total_gt": total_gt,
        "covered_center": covered_center,
        "recall_center": covered_center / max(total_gt, 1),
        "covered_iou": covered_iou,
        "recall_iou": covered_iou / max(total_gt, 1),
        "frames_with_gt": frames_with_gt,
        "frames_all_covered": frames_all_covered,
        "frame_recall": frames_all_covered / max(frames_with_gt, 1),
    }


if __name__ == "__main__":
    ann_dir = "./data/LLVIP/Annotations"
    
    print("=" * 60)
    print("行人覆盖率评估 (Person Recall)")
    print("=" * 60)
    
    for label, csv_path in [
        ("v1 Model + v2 Clustering", "outputs_v1/tasks.csv"),
        ("v2 Model + v2 Clustering", "outputs_v2/tasks.csv"),
    ]:
        result = evaluate_recall(csv_path, ann_dir, iou_threshold=0.3)
        print(f"\n--- {label} ---")
        print(f"  GT 行人总数:        {result['total_gt']}")
        print(f"  中心覆盖数:         {result['covered_center']} ({result['recall_center']*100:.1f}%)")
        print(f"  IoU≥0.3 覆盖数:     {result['covered_iou']} ({result['recall_iou']*100:.1f}%)")
        print(f"  有标注帧数:         {result['frames_with_gt']}")
        print(f"  全覆盖帧数:         {result['frames_all_covered']} ({result['frame_recall']*100:.1f}%)")
