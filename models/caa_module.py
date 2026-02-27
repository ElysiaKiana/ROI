"""
上下文锚点注意力机制 (Context Anchor Attention, CAA)

核心创新:
- 利用条带卷积 (1×k, k×1) 替代标准大核卷积，参数量从 k² 降至 2k
- 水平条带捕捉"人-枪"水平共现关系（如：人举枪呈水平分布）
- 垂直条带捕捉"人-刀"垂直共现关系（如：人持刀呈垂直分布）
- 两方向注意力融合生成自适应空间 Saliency Map

架构:
  Input(C, H, W)
    ├→ AvgPool → 1×1 Conv(C→C/r) → 1×k Conv → k×1 Conv → 1×1 Conv(C/r→C) → Sigmoid  [空间路径]
    └→ Identity                                                                         [恒等路径]
    → 逐元素乘法 → Output(C, H, W)

参考: LSKNet 的大核分解思想 + Strip Pooling 的条带感受野思想
"""

import torch
import torch.nn as nn


class ContextAnchorAttention(nn.Module):
    """上下文锚点注意力 (CAA)。

    通过条带卷积学习热源与周围武器的空间语义关联。

    Args:
        in_channels: 输入通道数
        strip_kernel: 条带卷积核大小 k (奇数)
        reduction: 通道缩减比
    """

    def __init__(
        self,
        in_channels: int = 64,
        strip_kernel: int = 11,
        reduction: int = 4,
    ):
        super().__init__()
        assert strip_kernel % 2 == 1, f"strip_kernel 必须为奇数, got {strip_kernel}"

        mid_channels = max(in_channels // reduction, 8)
        pad = strip_kernel // 2

        # ---- 空间注意力路径 ----
        self.spatial_attention = nn.Sequential(
            # 占位层（保持空间尺寸不变）
            nn.Identity(),

            # 通道降维
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 水平条带卷积: 捕捉水平方向的空间关联
            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=(1, strip_kernel),
                padding=(0, pad),
                groups=mid_channels,  # 深度可分离
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 垂直条带卷积: 捕捉垂直方向的空间关联
            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=(strip_kernel, 1),
                padding=(pad, 0),
                groups=mid_channels,  # 深度可分离
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 通道恢复
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 融合特征 (B, C, H, W)

        Returns:
            注意力加权后的特征 (B, C, H, W)
        """
        attn = self.spatial_attention(x)
        return x * attn


if __name__ == "__main__":
    model = ContextAnchorAttention(in_channels=64, strip_kernel=11, reduction=4)
    x = torch.randn(1, 64, 128, 160)
    out = model(x)
    print(f"CAA: {x.shape} → {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
