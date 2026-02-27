"""
IR 分支 —— 极浅层 CNN 热源锚点提取器

设计思路:
- 红外图像信息密度低(背景暗，热源亮)，只需极浅层网络即可定位热源区域
- 3 层卷积 + BN + ReLU，逐步下采样提取热源锚点特征
- 输出与 RGB 分支对齐的特征图尺寸 (H/4, W/4)
- 极低算力开销 (~0.02 GFLOPs @512×640 输入)

架构:
  Input(1, H, W)
    → Conv2d(3×3, s=2) → BN → ReLU    # (mid_ch, H/2, W/2)
    → Conv2d(3×3, s=1) → BN → ReLU    # (mid_ch, H/2, W/2)
    → Conv2d(3×3, s=2) → BN → ReLU    # (out_ch, H/4, W/4)
"""

import torch
import torch.nn as nn


class IRBranch(nn.Module):
    """极浅层 CNN：从红外图像提取热源锚点特征。

    Args:
        in_channels: 输入通道数，默认 1（单通道红外）
        mid_channels: 中间层通道数，默认 16
        out_channels: 输出通道数，默认 32
    """

    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 16,
        out_channels: int = 32,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: 下采样 ×2，提取粗粒度热源响应
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 2: 保持分辨率，增强热源边界
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # Layer 3: 再下采样 ×2，输出 (out_ch, H/4, W/4)
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ir: 红外图像 (B, 1, H, W)

        Returns:
            热源锚点特征 (B, out_channels, H/4, W/4)
        """
        return self.features(ir)


if __name__ == "__main__":
    # 快速验证
    model = IRBranch(in_channels=1, mid_channels=16, out_channels=32)
    x = torch.randn(1, 1, 512, 640)
    out = model(x)
    print(f"IR Branch: {x.shape} → {out.shape}")

    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
