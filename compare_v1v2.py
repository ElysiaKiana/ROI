"""v1 vs v2 显著性图对比可视化脚本"""

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path

from models.roi_network import build_model_from_config
from data.llvip_dataset import LLVIPDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 加载 v1 模型 =====
with open('configs/quick_train.yaml', 'r', encoding='utf-8') as f:
    cfg_v1 = yaml.safe_load(f)
model_v1 = build_model_from_config(cfg_v1).to(device)
ckpt_v1 = torch.load('checkpoints/best_v1.pth', map_location=device, weights_only=True)
model_v1.load_state_dict(ckpt_v1['model_state_dict'], strict=False)
model_v1.eval()
vm1 = ckpt_v1.get("val_metrics", {})
print(f"v1 model loaded. Val F1={vm1.get('f1', 'N/A')}")

# ===== 加载 v2 模型 =====
with open('configs/train_v2.yaml', 'r', encoding='utf-8') as f:
    cfg_v2 = yaml.safe_load(f)
model_v2 = build_model_from_config(cfg_v2).to(device)
ckpt_v2 = torch.load('checkpoints/best.pth', map_location=device, weights_only=True)
model_v2.load_state_dict(ckpt_v2['model_state_dict'], strict=False)
model_v2.eval()
vm2 = ckpt_v2.get("val_metrics", {})
print(f"v2 model loaded. Val F1={vm2.get('f1', 'N/A')}")

# ===== 数据集 =====
dataset = LLVIPDataset(
    root='./data/LLVIP', split='test', img_size=(512, 640),
    use_annotations=False, augment=False,
)

# ===== 对比可视化 =====
sample_indices = [0, 100, 500, 1000, 2000]
out_dir = Path('outputs/diagnostics')
out_dir.mkdir(parents=True, exist_ok=True)

for idx in sample_indices:
    sample = dataset[idx]
    stem = sample['stem']
    ir = sample['ir'].unsqueeze(0).to(device)
    rgb = sample['rgb'].unsqueeze(0).to(device)

    with torch.no_grad():
        sal_v1 = model_v1(ir, rgb)['saliency_map'].squeeze().cpu().numpy()
        sal_v2 = model_v2(ir, rgb)['saliency_map'].squeeze().cpu().numpy()

    # 原始图像
    rgb_np = sample['rgb'].permute(1, 2, 0).numpy()
    rgb_uint8 = (rgb_np * 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ir_np = sample['ir'].squeeze().numpy()
    ir_uint8 = (ir_np * 255).astype(np.uint8)
    ir_bgr = cv2.cvtColor(ir_uint8, cv2.COLOR_GRAY2BGR)

    # 显著性热图
    sal_v1_vis = cv2.applyColorMap((sal_v1 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    sal_v2_vis = cv2.applyColorMap((sal_v2 * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 叠加到原图
    overlay_v1 = cv2.addWeighted(rgb_bgr, 0.5, sal_v1_vis, 0.5, 0)
    overlay_v2 = cv2.addWeighted(rgb_bgr, 0.5, sal_v2_vis, 0.5, 0)

    # 统计文本
    v1_mean = sal_v1.mean()
    v2_mean = sal_v2.mean()
    v1_cov = (sal_v1 > 0.5).mean() * 100
    v2_cov = (sal_v2 > 0.5).mean() * 100

    cv2.putText(overlay_v1, f'v1 mean={v1_mean:.3f} cov={v1_cov:.1f}%',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay_v2, f'v2 mean={v2_mean:.3f} cov={v2_cov:.1f}%',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 四宫格: IR | RGB | v1 | v2
    H, W = rgb_bgr.shape[:2]
    canvas = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
    canvas[0:H, 0:W] = ir_bgr
    canvas[0:H, W:W * 2] = rgb_bgr
    canvas[H:H * 2, 0:W] = overlay_v1
    canvas[H:H * 2, W:W * 2] = overlay_v2

    # 标签
    for text, pos in [('IR', (10, H - 10)), ('RGB', (W + 10, H - 10)),
                      ('v1 Saliency (Hard GT)', (10, H * 2 - 10)),
                      ('v2 Saliency (Gaussian GT)', (W + 10, H * 2 - 10))]:
        cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    out_path = out_dir / f'v1_vs_v2_{stem}.png'
    cv2.imwrite(str(out_path), canvas)
    print(f'{stem}: v1 mean={v1_mean:.3f} cov={v1_cov:.1f}%  |  v2 mean={v2_mean:.3f} cov={v2_cov:.1f}%')

print('\nDone! Comparison images saved to outputs/diagnostics/')
