"""
RGB 分支 —— 基于 MobileNetV3-Small 的轻量级边缘纹理提取器

设计思路:
- 利用 MobileNetV3-Small 的 ImageNet 预训练权重，提取 RGB 图像的高频边缘纹理
- 只取前几层特征（浅层即可捕捉边缘/纹理），大幅减少计算量
- 通过 1×1 卷积将通道数对齐到 out_channels，与 IR 分支拼接
- 输出尺寸与 IR 分支一致：(H/4, W/4)

MobileNetV3-Small 特征层输出尺寸 (输入 512×640):
  Layer 0: (16, 256, 320)    # stride=2
  Layer 1: (16, 128, 160)    # stride=4  ← 我们默认取这层
  Layer 2: (24, 64, 80)      # stride=8
  ...

我们取 stride=4 的层，与 IR 分支的 H/4 × W/4 对齐。
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class RGBBranch(nn.Module):
    """MobileNetV3-Small 浅层特征提取 + 通道对齐。

    Args:
        pretrained: 是否加载 ImageNet 预训练权重
        out_channels: 输出通道数（与 IR 分支对齐）
        feature_layer: 截取到 MobileNetV3 features 的第几层
                       (默认 4，输出 stride=4，通道数 24)
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 32,
        feature_layer: int = 4,
    ):
        super().__init__()

        # 加载 MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = mobilenet_v3_small(weights=weights)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # 截取前 feature_layer 层
        # MobileNetV3.features 是 nn.Sequential，包含多个 InvertedResidual 块
        self.features = nn.Sequential(*list(backbone.features.children())[:feature_layer])

        # 冻结第一层（底层特征通用性强，不需要微调）
        for param in list(self.features.parameters())[:2]:
            param.requires_grad = False

        # 探测截取层的输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 640)
            feat = self.features(dummy)
            feat_channels = feat.shape[1]
            feat_stride = dummy.shape[2] // feat.shape[2]

        # 如果 stride 不是 4，需要调整
        self._need_resize = (feat_stride != 4)
        self._target_stride = 4

        # 1×1 卷积对齐通道数
        self.channel_align = nn.Sequential(
            nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        print(f"[RGBBranch] 截取 MobileNetV3-Small 前 {feature_layer} 层, "
              f"特征通道={feat_channels}, stride={feat_stride}")

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: RGB 图像 (B, 3, H, W)

        Returns:
            边缘纹理特征 (B, out_channels, H/4, W/4)
        """
        feat = self.features(rgb)

        # 尺寸对齐（确保输出为 H/4 × W/4）
        if self._need_resize:
            target_h = rgb.shape[2] // self._target_stride
            target_w = rgb.shape[3] // self._target_stride
            feat = nn.functional.interpolate(
                feat, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        return self.channel_align(feat)


if __name__ == "__main__":
    model = RGBBranch(pretrained=True, out_channels=32, feature_layer=4)
    x = torch.randn(1, 3, 512, 640)
    out = model(x)
    print(f"RGB Branch: {x.shape} → {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {params:,}, Trainable: {trainable:,}")
