"""LLVIP 数据集加载器 (v2 — 高斯椭圆软标签)

LLVIP 数据集结构:
  LLVIP/
    ├── infrared/
    │   ├── train/  └── test/
    ├── visible/
    │   ├── train/  └── test/
    └── Annotations/          # PASCAL VOC 格式标注

GT 显著性图生成策略 (v2 改进):
  v1: bbox 区域直接填充 1.0（硬矩形）
      缺陷 — 矩形内大量背景像素被标为正样本，路灯等热源无负样本监督

  v2: 2D 高斯椭圆热图 + 红外亮度加权
      (1) 每个 bbox → 以其中心为均值的 2D 高斯分布
          σ_x = bbox_w × sigma_ratio,  σ_y = bbox_h × sigma_ratio
      (2) 高斯峰值 = 1.0，边缘自然衰减到 ~0
      (3) 多目标取 element-wise max（不叠加，避免超过 1）
      (4) 可选：与归一化IR亮度做 element-wise multiply
          高斯 × IR亮度 → 人体热源处增强，冷背景处抑制
      (5) 无标注帧：Otsu 自适应阈值作为伪标签

  优势:
      - 中心响应强、边缘弱 → 网络学到的激活更聚焦人体核心
      - 高斯衰减提供了平滑的梯度信号，避免硬边界的学习困难
      - IR加权让网络学会"人体热源"模式而非纯几何位置
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LLVIPDataset(Dataset):
    """LLVIP 多模态数据集。

    Args:
        root: 数据集根目录
        split: 'train' 或 'test'
        img_size: 目标尺寸 (H, W)
        use_annotations: 是否使用 VOC 标注生成 GT saliency
        augment: 是否数据增强
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: tuple[int, int] = (512, 640),
        use_annotations: bool = True,
        augment: bool = False,
        gt_mode: str = "gaussian",
        sigma_ratio: float = 0.3,
        ir_weighting: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size  # (H, W)
        self.use_annotations = use_annotations
        self.augment = augment
        self.gt_mode = gt_mode          # "hard" (v1) 或 "gaussian" (v2)
        self.sigma_ratio = sigma_ratio  # 高斯 σ 与 bbox 尺寸的比例
        self.ir_weighting = ir_weighting  # 是否用 IR 亮度加权 GT

        # 构造路径
        self.ir_dir = self.root / "infrared" / split
        self.rgb_dir = self.root / "visible" / split
        self.ann_dir = self.root / "Annotations"

        # 获取文件列表（取 IR 和 RGB 的交集）
        ir_files = set(self._list_images(self.ir_dir))
        rgb_files = set(self._list_images(self.rgb_dir))
        common = sorted(ir_files & rgb_files)

        if len(common) == 0:
            raise FileNotFoundError(
                f"未找到匹配的图像对。\n"
                f"IR 目录: {self.ir_dir} ({len(ir_files)} 张)\n"
                f"RGB 目录: {self.rgb_dir} ({len(rgb_files)} 张)"
            )

        self.file_stems = common
        print(f"[LLVIPDataset] {split}: {len(self.file_stems)} 对图像")

    @staticmethod
    def _list_images(directory: Path) -> list[str]:
        """列出目录下所有图像的文件名（不含扩展名）"""
        if not directory.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return [f.stem for f in directory.iterdir() if f.suffix.lower() in exts]

    def _load_image(self, directory: Path, stem: str) -> np.ndarray | None:
        """加载图像，尝试多种扩展名"""
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return cv2.imread(str(path))
        return None

    def _parse_voc_annotation(self, stem: str) -> list[tuple[int, int, int, int]]:
        """解析 VOC 格式标注，返回 bounding box 列表"""
        xml_path = self.ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            return []

        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            if bbox is not None:
                x1 = int(float(bbox.find("xmin").text))
                y1 = int(float(bbox.find("ymin").text))
                x2 = int(float(bbox.find("xmax").text))
                y2 = int(float(bbox.find("ymax").text))
                boxes.append((x1, y1, x2, y2))

        return boxes

    @staticmethod
    def _generate_gaussian_heatmap(
        H: int, W: int,
        boxes: list[tuple[int, int, int, int]],
        sigma_ratio: float = 0.3,
    ) -> np.ndarray:
        """为每个 bbox 生成 2D 高斯椭圆热图，多目标取 element-wise max。

        高斯参数:
          μ_x, μ_y = bbox 中心
          σ_x = bbox_width × sigma_ratio
          σ_y = bbox_height × sigma_ratio

        Args:
            H, W: 图像尺寸
            boxes: [(x1, y1, x2, y2), ...]
            sigma_ratio: σ 与 bbox 尺寸的比例 (默认 0.3)

        Returns:
            heatmap (H, W) float32, 值域 [0, 1]
        """
        heatmap = np.zeros((H, W), dtype=np.float32)

        # 预计算网格坐标
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            sx = max(bw * sigma_ratio, 3.0)  # 最小 σ=3 防止退化
            sy = max(bh * sigma_ratio, 3.0)

            # 只在 bbox 附近 ±3σ 范围内计算（性能优化）
            rx = int(3 * sx)
            ry = int(3 * sy)
            roi_y1 = max(0, int(cy) - ry)
            roi_y2 = min(H, int(cy) + ry + 1)
            roi_x1 = max(0, int(cx) - rx)
            roi_x2 = min(W, int(cx) + rx + 1)

            local_yy = yy[roi_y1:roi_y2, roi_x1:roi_x2]
            local_xx = xx[roi_y1:roi_y2, roi_x1:roi_x2]

            gauss = np.exp(
                -((local_xx - cx) ** 2 / (2 * sx ** 2)
                  + (local_yy - cy) ** 2 / (2 * sy ** 2))
            )

            # element-wise max（不叠加）
            heatmap[roi_y1:roi_y2, roi_x1:roi_x2] = np.maximum(
                heatmap[roi_y1:roi_y2, roi_x1:roi_x2], gauss
            )

        return heatmap

    def _generate_saliency_gt(
        self, ir_gray: np.ndarray, boxes: list[tuple[int, int, int, int]]
    ) -> np.ndarray:
        """生成 Ground Truth 显著性图。

        v1 (hard):     bbox 区域直接填充 1.0
        v2 (gaussian): 2D 高斯椭圆热图 + 可选 IR 亮度加权

        Returns:
            saliency_gt (H, W) float32, 值域 [0, 1]
        """
        H, W = ir_gray.shape[:2]
        saliency = np.zeros((H, W), dtype=np.float32)

        if len(boxes) > 0:
            if self.gt_mode == "gaussian":
                # ===== v2: 高斯椭圆热图 =====
                saliency = self._generate_gaussian_heatmap(
                    H, W, boxes, sigma_ratio=self.sigma_ratio
                )
                # 可选：IR 亮度加权
                if self.ir_weighting:
                    if len(ir_gray.shape) == 3:
                        ir_gray = cv2.cvtColor(ir_gray, cv2.COLOR_BGR2GRAY)
                    ir_norm = ir_gray.astype(np.float32) / 255.0
                    # 只在高斯热图区域做加权，避免路灯等无标注热源产生虚假响应
                    # 加权公式: saliency = gaussian × (0.5 + 0.5 × ir_norm)
                    # 保证中心区域不会因 ir 暗而被完全压制
                    saliency = saliency * (0.5 + 0.5 * ir_norm)
            else:
                # ===== v1: 硬矩形填充 (保留用于对比实验) =====
                for x1, y1, x2, y2 in boxes:
                    x1 = max(0, min(x1, W))
                    y1 = max(0, min(y1, H))
                    x2 = max(0, min(x2, W))
                    y2 = max(0, min(y2, H))
                    saliency[y1:y2, x1:x2] = 1.0
        else:
            # 伪标签：红外自适应阈值
            if len(ir_gray.shape) == 3:
                ir_gray = cv2.cvtColor(ir_gray, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                ir_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            saliency = binary.astype(np.float32) / 255.0

        return saliency

    def __len__(self) -> int:
        return len(self.file_stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.file_stems[idx]

        # 加载 RGB 和 IR 图像
        rgb_bgr = self._load_image(self.rgb_dir, stem)
        ir_bgr = self._load_image(self.ir_dir, stem)

        if rgb_bgr is None or ir_bgr is None:
            raise FileNotFoundError(f"无法加载图像: {stem}")

        # 转灰度 IR
        ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)

        # 解析标注 → 生成 GT saliency
        boxes = self._parse_voc_annotation(stem) if self.use_annotations else []
        saliency_gt = self._generate_saliency_gt(ir_gray, boxes)

        # Resize
        H, W = self.img_size
        rgb_bgr = cv2.resize(rgb_bgr, (W, H))
        ir_gray = cv2.resize(ir_gray, (W, H))
        saliency_gt = cv2.resize(saliency_gt, (W, H))

        # 数据增强
        if self.augment:
            if np.random.rand() > 0.5:
                rgb_bgr = cv2.flip(rgb_bgr, 1)
                ir_gray = cv2.flip(ir_gray, 1)
                saliency_gt = cv2.flip(saliency_gt, 1)

        # BGR → RGB, 归一化
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ir = ir_gray.astype(np.float32) / 255.0

        # numpy → tensor
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)     # (3, H, W)
        ir_tensor = torch.from_numpy(ir).unsqueeze(0)            # (1, H, W)
        saliency_tensor = torch.from_numpy(saliency_gt).unsqueeze(0)  # (1, H, W)

        return {
            "rgb": rgb_tensor,
            "ir": ir_tensor,
            "saliency_gt": saliency_tensor,
            "stem": stem,
        }


if __name__ == "__main__":
    # 测试（需要先下载 LLVIP 数据集）
    dataset = LLVIPDataset(
        root="./data/LLVIP",
        split="train",
        img_size=(512, 640),
        augment=True,
    )
    sample = dataset[0]
    print(f"RGB: {sample['rgb'].shape}")
    print(f"IR:  {sample['ir'].shape}")
    print(f"GT:  {sample['saliency_gt'].shape}")
    print(f"Stem: {sample['stem']}")
