"""
可视化脚本 —— 展示 ROI 生成效果

生成内容:
1. 原图 + ROI 框可视化（红框=分离目标，蓝框=聚集目标）
2. 显著性热力图叠加
3. 多帧 Grid 展示

使用方式:
  python visualize.py --output_dir ./outputs --data_root ./data/LLVIP --num_samples 8
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def load_tasks_csv(csv_path: str) -> dict[str, list[dict]]:
    """加载 tasks.csv 并按 frame_id 分组"""
    frames = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames[row["frame_id"]].append({
                "roi_id": int(row["roi_id"]),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "is_crowd": row["is_crowd"] == "1",
                "num_targets": int(row["num_targets"]),
                "data_size_kb": float(row["data_size_kb"]),
            })
    return dict(frames)


def draw_rois_on_image(image: np.ndarray, rois: list[dict]) -> np.ndarray:
    """在图像上绘制 ROI 框"""
    vis = image.copy()
    for roi in rois:
        color = (255, 50, 50) if roi["is_crowd"] else (50, 200, 50)  # 蓝=crowd, 绿=isolated
        thickness = 2 if roi["is_crowd"] else 2
        cv2.rectangle(vis, (roi["x1"], roi["y1"]), (roi["x2"], roi["y2"]), color, thickness)

        # 标签
        label = f"R{roi['roi_id']}"
        if roi["is_crowd"]:
            label += f" C({roi['num_targets']})"
        label += f" {roi['data_size_kb']:.1f}KB"

        # 文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (roi["x1"], roi["y1"] - th - 6),
                       (roi["x1"] + tw + 4, roi["y1"]), color, -1)
        cv2.putText(vis, label, (roi["x1"] + 2, roi["y1"] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return vis


def create_single_visualization(
    rgb_path: str,
    ir_path: str,
    saliency_path: str | None,
    rois: list[dict],
    save_path: str,
):
    """创建单帧的完整可视化（4 子图）"""
    rgb = cv2.imread(rgb_path)
    ir = cv2.imread(ir_path)
    if rgb is None or ir is None:
        return

    rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ir_rgb = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

    # ROI 可视化
    roi_vis = draw_rois_on_image(rgb, rois)
    roi_vis_rgb = cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB)

    # 显著性图
    if saliency_path and os.path.exists(saliency_path):
        sal = cv2.imread(saliency_path)
        sal = cv2.resize(sal, (rgb.shape[1], rgb.shape[0]))
        sal_rgb = cv2.cvtColor(sal, cv2.COLOR_BGR2RGB)
        # 叠加
        overlay = cv2.addWeighted(rgb, 0.5, sal, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    else:
        sal_rgb = np.zeros_like(rgb_rgb)
        overlay_rgb = rgb_rgb.copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"ROI Generation Result  |  {len(rois)} ROIs detected", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(rgb_rgb)
    axes[0, 0].set_title("RGB (Visible)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ir_rgb)
    axes[0, 1].set_title("IR (Infrared)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title("Saliency Overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(roi_vis_rgb)
    axes[1, 1].set_title("ROI Boxes (Green=Isolated, Blue=Crowd)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_grid_visualization(
    data_root: str,
    output_dir: str,
    frames_data: dict,
    frame_ids: list[str],
    save_path: str,
):
    """创建多帧 Grid 可视化"""
    n = len(frame_ids)
    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Asymmetric Dual-Stream ROI Generation — Multi-Frame Gallery",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for i, fid in enumerate(frame_ids):
        rois = frames_data.get(fid, [])

        # 寻找图像文件
        rgb_path = None
        ir_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(data_root, "visible", "test", f"{fid}{ext}")
            if os.path.exists(p):
                rgb_path = p
            p = os.path.join(data_root, "infrared", "test", f"{fid}{ext}")
            if os.path.exists(p):
                ir_path = p

        if rgb_path is None or ir_path is None:
            continue

        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(ir_path)
        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ir_rgb = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)

        # ROI 可视化
        roi_vis = draw_rois_on_image(rgb, rois)
        roi_vis_rgb = cv2.cvtColor(roi_vis, cv2.COLOR_BGR2RGB)

        # 显著性叠加
        sal_path = os.path.join(output_dir, "saliency_maps", f"{fid}_saliency.png")
        if os.path.exists(sal_path):
            sal = cv2.imread(sal_path)
            sal = cv2.resize(sal, (rgb.shape[1], rgb.shape[0]))
            overlay = cv2.addWeighted(rgb, 0.5, sal, 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        else:
            overlay_rgb = rgb_rgb.copy()

        axes[i, 0].imshow(rgb_rgb)
        axes[i, 0].set_title(f"[{fid}] RGB", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(overlay_rgb)
        axes[i, 1].set_title(f"[{fid}] Saliency Overlay", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(roi_vis_rgb)
        n_isolated = sum(1 for r in rois if not r["is_crowd"])
        n_crowd = sum(1 for r in rois if r["is_crowd"])
        total_kb = sum(r["data_size_kb"] for r in rois)
        axes[i, 2].set_title(
            f"[{fid}] ROIs: {len(rois)} (iso={n_isolated}, crowd={n_crowd}) | {total_kb:.1f}KB",
            fontsize=9,
        )
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid 可视化已保存: {save_path}")


def create_roi_crop_gallery(
    output_dir: str,
    frames_data: dict,
    frame_id: str,
    save_path: str,
):
    """展示单帧的所有 ROI 切片"""
    rois = frames_data.get(frame_id, [])
    if not rois:
        return

    n = len(rois)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"Frame {frame_id} — ROI Crops ({n} slices)", fontsize=14, fontweight="bold")

    for idx, roi in enumerate(rois):
        r, c = idx // cols, idx % cols
        roi_path = os.path.join(output_dir, "roi_slices", f"{frame_id}_roi{roi['roi_id']}.jpg")
        if os.path.exists(roi_path):
            crop = cv2.imread(roi_path)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            axes[r, c].imshow(crop_rgb)
        tag = "Crowd" if roi["is_crowd"] else "Isolated"
        axes[r, c].set_title(
            f"ROI-{roi['roi_id']} [{tag}]\n{roi['data_size_kb']:.1f}KB, {roi['num_targets']} targets",
            fontsize=9,
        )
        axes[r, c].axis("off")

    # 隐藏空余子图
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROI 切片画廊已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="ROI Visualization")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--data_root", type=str, default="./data/LLVIP")
    parser.add_argument("--num_samples", type=int, default=6, help="Grid 中展示的帧数")
    parser.add_argument("--save_dir", type=str, default="./outputs/visualizations")
    args = parser.parse_args()

    # 加载数据
    csv_path = os.path.join(args.output_dir, "tasks.csv")
    if not os.path.exists(csv_path):
        print(f"未找到 {csv_path}，请先运行 inference.py")
        return

    frames_data = load_tasks_csv(csv_path)
    frame_ids = sorted(frames_data.keys())
    print(f"加载了 {len(frame_ids)} 帧的 ROI 数据")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 选出有代表性的帧（变化多的）
    # 选择 ROI 数量不同的帧，展示多样性
    frames_by_roi_count = sorted(frame_ids, key=lambda f: len(frames_data[f]))
    n = min(args.num_samples, len(frame_ids))
    step = max(len(frame_ids) // n, 1)
    selected = [frames_by_roi_count[i * step] for i in range(n)]

    # 1. 多帧 Grid 可视化
    print("\n[1/3] 生成多帧 Grid 可视化...")
    create_grid_visualization(
        args.data_root, args.output_dir, frames_data, selected,
        str(save_dir / "grid_visualization.png"),
    )

    # 2. 单帧详细可视化（取 ROI 数量中位数的帧）
    print("[2/3] 生成单帧详细可视化...")
    mid_frame = frames_by_roi_count[len(frames_by_roi_count) // 2]
    for ext in [".jpg", ".png", ".jpeg"]:
        rgb_p = os.path.join(args.data_root, "visible", "test", f"{mid_frame}{ext}")
        ir_p = os.path.join(args.data_root, "infrared", "test", f"{mid_frame}{ext}")
        if os.path.exists(rgb_p) and os.path.exists(ir_p):
            sal_p = os.path.join(args.output_dir, "saliency_maps", f"{mid_frame}_saliency.png")
            create_single_visualization(
                rgb_p, ir_p, sal_p, frames_data[mid_frame],
                str(save_dir / f"detail_{mid_frame}.png"),
            )
            print(f"  详细可视化已保存: detail_{mid_frame}.png")
            break

    # 3. ROI 切片画廊
    print("[3/3] 生成 ROI 切片画廊...")
    create_roi_crop_gallery(
        args.output_dir, frames_data, mid_frame,
        str(save_dir / f"roi_crops_{mid_frame}.png"),
    )

    # 统计摘要
    total_rois = sum(len(v) for v in frames_data.values())
    all_sizes = [r["data_size_kb"] for rois in frames_data.values() for r in rois]
    crowd_count = sum(1 for rois in frames_data.values() for r in rois if r["is_crowd"])

    print(f"\n{'='*50}")
    print(f"可视化完成!")
    print(f"  总帧数: {len(frame_ids)}")
    print(f"  总 ROI: {total_rois} (平均 {total_rois/len(frame_ids):.1f}/帧)")
    print(f"  聚集 ROI: {crowd_count} ({100*crowd_count/max(total_rois,1):.1f}%)")
    print(f"  ROI 大小: {np.mean(all_sizes):.1f} ± {np.std(all_sizes):.1f} KB")
    print(f"  保存目录: {save_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
