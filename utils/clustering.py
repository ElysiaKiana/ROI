"""
空间密度聚类与 ROI 切片生成 (v2)

核心改进 (相比 v1):
- v1 只提取连通域质心，丢失空间范围 → 91% ROI 退化为 min_roi_size 固定方块
- v2 保留每个连通域的完整 bbox，基于 bbox 做聚类和合并
- 输出的 ROI 大小由目标的实际空间范围决定，是自适应可变尺寸

设计思路:
1. 从显著性图中提取目标连通域 → 得到每个目标的 bbox + 质心
2. 用 DBSCAN 对质心做密度聚类:
   - 分离目标 (isolated): 以该目标 bbox 为基础，外扩自适应 padding
   - 聚集目标 (crowd):     合并簇内所有 bbox，外扩更大的 padding
3. padding 按目标大小自适应：大目标匹配大 padding（保留携带物品/武器上下文）
4. 动态计算每个 ROI 切片的 JPEG 压缩后大小 data_size_kb
"""

from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class TargetBlob:
    """单个连通域目标"""
    cx: float           # 质心 x
    cy: float           # 质心 y
    x1: int             # bbox 左上 x
    y1: int             # bbox 左上 y
    x2: int             # bbox 右下 x
    y2: int             # bbox 右下 y
    area: int           # 面积 (像素)


@dataclass
class ROISlice:
    """单个 ROI 切片的描述"""
    roi_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    is_crowd: bool          # 是否为聚集目标
    num_targets: int        # 包含的目标数量
    data_size_kb: float     # JPEG 压缩后的文件大小 (KB)
    center_x: float         # ROI 中心 x
    center_y: float         # ROI 中心 y


class SpatialDensityClustering:
    """空间密度聚类 ROI 生成器 (v2 — 基于连通域 bbox).

    Args:
        saliency_threshold: 显著性二值化阈值
        min_area: 最小目标面积（过滤噪声）
        dbscan_eps: DBSCAN 邻域半径 (基于质心距离)
        dbscan_min_samples: DBSCAN 最小样本数
        padding_ratio: 自适应外扩比例 (相对于目标尺寸)
        min_padding: 最小外扩像素
        crowd_padding_ratio: 聚集目标的外扩比例
        crowd_threshold: >= 此数量视为聚集
        min_roi_size: 最小 ROI 尺寸
        jpeg_quality: JPEG 压缩质量
    """

    def __init__(
        self,
        saliency_threshold: float = 0.5,
        min_area: int = 100,
        dbscan_eps: float = 80.0,
        dbscan_min_samples: int = 1,
        padding_ratio: float = 0.3,
        min_padding: int = 15,
        crowd_padding_ratio: float = 0.25,
        crowd_threshold: int = 3,
        min_roi_size: int = 64,
        jpeg_quality: int = 85,
        # 兼容旧配置中的参数名 (v1)
        isolated_padding: int | None = None,
        crowd_padding: int | None = None,
    ):
        self.saliency_threshold = saliency_threshold
        self.min_area = min_area
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.padding_ratio = padding_ratio
        self.min_padding = min_padding
        self.crowd_padding_ratio = crowd_padding_ratio
        self.crowd_threshold = crowd_threshold
        self.min_roi_size = min_roi_size
        self.jpeg_quality = jpeg_quality

    def extract_target_blobs(self, saliency_map: np.ndarray) -> list[TargetBlob]:
        """从显著性图提取目标连通域，返回完整的 bbox + 质心信息。"""
        # 二值化
        binary = (saliency_map > self.saliency_threshold).astype(np.uint8) * 255

        # 形态学操作：闭运算填充小孔 → 开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        blobs = []
        for i in range(1, num_labels):  # 跳过背景 label=0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue

            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            blobs.append(TargetBlob(
                cx=centroids[i][0],
                cy=centroids[i][1],
                x1=bx, y1=by,
                x2=bx + bw, y2=by + bh,
                area=area,
            ))

        return blobs

    def cluster_targets(
        self, blobs: list[TargetBlob]
    ) -> dict[int, list[int]]:
        """用 DBSCAN 对目标质心做密度聚类。"""
        if len(blobs) == 0:
            return {}
        if len(blobs) == 1:
            return {0: [0]}

        coords = np.array([[b.cx, b.cy] for b in blobs])
        db = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="euclidean",
        ).fit(coords)

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(db.labels_):
            if label == -1:
                new_label = max(clusters.keys(), default=-1) + 1
                clusters[new_label] = [idx]
            else:
                clusters.setdefault(label, []).append(idx)

        return clusters

    def _compute_adaptive_padding(
        self, bbox_w: int, bbox_h: int, is_crowd: bool
    ) -> tuple[int, int]:
        """根据目标尺寸计算自适应外扩 padding。

        大目标 → 大 padding (保留携带物品/武器的周围上下文)
        小目标 → 至少 min_padding
        """
        ratio = self.crowd_padding_ratio if is_crowd else self.padding_ratio
        pad_x = max(self.min_padding, int(bbox_w * ratio))
        pad_y = max(self.min_padding, int(bbox_h * ratio))
        return pad_x, pad_y

    def compute_data_size_kb(
        self, rgb_image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """计算 ROI 切片 JPEG 压缩后大小。"""
        roi_crop = rgb_image[y1:y2, x1:x2]
        if roi_crop.size == 0:
            return 0.0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, buf = cv2.imencode(".jpg", roi_crop, encode_params)
        return len(buf) / 1024.0

    def generate_rois(
        self,
        saliency_map: np.ndarray,
        rgb_image: np.ndarray,
    ) -> list[ROISlice]:
        """完整 ROI 生成流水线。"""
        H, W = saliency_map.shape[:2]

        # Step 1: 提取连通域 (包含完整 bbox)
        blobs = self.extract_target_blobs(saliency_map)
        if len(blobs) == 0:
            return []

        # Step 2: 密度聚类
        clusters = self.cluster_targets(blobs)

        # Step 3: 为每个簇合并 bbox → 自适应外扩 → 生成 ROI
        rois: list[ROISlice] = []
        roi_id = 0

        for cluster_id, indices in clusters.items():
            num_targets = len(indices)
            is_crowd = num_targets >= self.crowd_threshold

            # 合并簇内所有目标的 bbox
            cluster_blobs = [blobs[i] for i in indices]
            merged_x1 = min(b.x1 for b in cluster_blobs)
            merged_y1 = min(b.y1 for b in cluster_blobs)
            merged_x2 = max(b.x2 for b in cluster_blobs)
            merged_y2 = max(b.y2 for b in cluster_blobs)

            bbox_w = merged_x2 - merged_x1
            bbox_h = merged_y2 - merged_y1

            # 自适应 padding
            pad_x, pad_y = self._compute_adaptive_padding(bbox_w, bbox_h, is_crowd)

            # 外扩
            x1 = max(0, merged_x1 - pad_x)
            y1 = max(0, merged_y1 - pad_y)
            x2 = min(W, merged_x2 + pad_x)
            y2 = min(H, merged_y2 + pad_y)

            # 确保最小尺寸 (兜底，正常情况不应触发)
            if (x2 - x1) < self.min_roi_size:
                cx = (x1 + x2) // 2
                x1 = max(0, cx - self.min_roi_size // 2)
                x2 = min(W, x1 + self.min_roi_size)
            if (y2 - y1) < self.min_roi_size:
                cy = (y1 + y2) // 2
                y1 = max(0, cy - self.min_roi_size // 2)
                y2 = min(H, y1 + self.min_roi_size)

            # 计算 JPEG 大小
            data_size_kb = self.compute_data_size_kb(rgb_image, x1, y1, x2, y2)

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            rois.append(ROISlice(
                roi_id=roi_id,
                x1=x1, y1=y1, x2=x2, y2=y2,
                is_crowd=is_crowd,
                num_targets=num_targets,
                data_size_kb=data_size_kb,
                center_x=center_x,
                center_y=center_y,
            ))
            roi_id += 1

        return rois


if __name__ == "__main__":
    # 测试用例
    H, W = 512, 640
    saliency = np.zeros((H, W), dtype=np.float32)

    # 分散目标 (模拟不同大小的人形)
    cv2.ellipse(saliency, (100, 100), (15, 40), 0, 0, 360, 1.0, -1)
    cv2.ellipse(saliency, (500, 400), (20, 50), 0, 0, 360, 1.0, -1)

    # 聚集目标（3个靠近的人形）
    cv2.ellipse(saliency, (300, 250), (12, 35), 0, 0, 360, 1.0, -1)
    cv2.ellipse(saliency, (330, 260), (12, 35), 0, 0, 360, 1.0, -1)
    cv2.ellipse(saliency, (315, 245), (12, 35), 0, 0, 360, 1.0, -1)

    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    generator = SpatialDensityClustering(dbscan_eps=80, crowd_threshold=3)
    rois = generator.generate_rois(saliency, rgb)

    for roi in rois:
        print(f"ROI-{roi.roi_id}: [{roi.x1},{roi.y1},{roi.x2},{roi.y2}] "
              f"size={roi.x2-roi.x1}x{roi.y2-roi.y1} "
              f"crowd={roi.is_crowd} targets={roi.num_targets} "
              f"data={roi.data_size_kb:.1f}KB")
