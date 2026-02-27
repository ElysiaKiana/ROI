"""
非对称特征融合模块

设计思路:
- IR 分支输出热源锚点特征 (B, C_ir, H/4, W/4)
- RGB 分支输出边缘纹理特征 (B, C_rgb, H/4, W/4)
- 通道拼接 → CAA 注意力加权 → 显著性预测头 → Saliency Map
- 最终输出原图尺寸的显著性图 (B, 1, H, W)

融合策略选择「拼接 + 注意力」而非「相加」:
- 拼接保留了两个模态的独立信息，让 CAA 自适应学习权重
- 相加会丢失模态区分度，不利于热源-武器关联学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .caa_module import ContextAnchorAttention


class SaliencyHead(nn.Module):
    """显著性预测头：从融合特征生成单通道显著性图。

    Args:
        in_channels: 输入通道数
        mid_channels: 中间通道数
    """

    def __init__(self, in_channels: int = 64, mid_channels: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出 logits (未经 Sigmoid)"""
        return self.head(x)


class AsymmetricFusion(nn.Module):
    """非对称双流特征融合 + 显著性生成。

    Args:
        ir_channels: IR 分支输出通道数
        rgb_channels: RGB 分支输出通道数
        strip_kernel: CAA 条带卷积核大小
        reduction: CAA 通道缩减比
        mid_channels: 显著性头中间通道数
    """

    def __init__(
        self,
        ir_channels: int = 32,
        rgb_channels: int = 32,
        strip_kernel: int = 11,
        reduction: int = 4,
        mid_channels: int = 32,
    ):
        super().__init__()

        fused_channels = ir_channels + rgb_channels

        # CAA 注意力
        self.caa = ContextAnchorAttention(
            in_channels=fused_channels,
            strip_kernel=strip_kernel,
            reduction=reduction,
        )

        # 显著性预测头
        self.saliency_head = SaliencyHead(
            in_channels=fused_channels,
            mid_channels=mid_channels,
        )

    def forward(
        self,
        ir_feat: torch.Tensor,
        rgb_feat: torch.Tensor,
        original_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ir_feat:  IR 特征  (B, C_ir, H/4, W/4)
            rgb_feat: RGB 特征 (B, C_rgb, H/4, W/4)
            original_size: 原始图像尺寸 (H, W)，用于上采样

        Returns:
            dict:
                "saliency_logits": 显著性 logits (B, 1, H, W) — 用于训练
                "saliency_map":   显著性概率图 (B, 1, H, W) — 用于推理
                "fused_feat":     融合特征 (B, C, H/4, W/4)
        """
        # 通道拼接
        fused = torch.cat([ir_feat, rgb_feat], dim=1)  # (B, C_ir+C_rgb, H/4, W/4)

        # CAA 注意力加权
        fused = self.caa(fused)

        # 显著性预测
        logits = self.saliency_head(fused)  # (B, 1, H/4, W/4)

        # 上采样到原始尺寸
        if original_size is not None:
            logits = F.interpolate(
                logits, size=original_size, mode="bilinear", align_corners=False
            )

        saliency_map = torch.sigmoid(logits)

        return {
            "saliency_logits": logits,
            "saliency_map": saliency_map,
            "fused_feat": fused,
        }


if __name__ == "__main__":
    model = AsymmetricFusion(ir_channels=32, rgb_channels=32)
    ir_feat = torch.randn(1, 32, 128, 160)
    rgb_feat = torch.randn(1, 32, 128, 160)
    outputs = model(ir_feat, rgb_feat, original_size=(512, 640))

    for k, v in outputs.items():
        print(f"{k}: {v.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Fusion params: {params:,}")
