"""
完整的 ROI 生成网络 —— 非对称双流 + CAA + 显著性预测

将 IR 分支、RGB 分支、CAA 注意力融合、显著性预测头
组装为一个端到端可训练的网络。

整体架构:
  IR Image (B,1,H,W)  ──→ IRBranch   ──→ (B,32,H/4,W/4)  ─┐
                                                              ├→ Cat → CAA → SaliencyHead → (B,1,H,W)
  RGB Image (B,3,H,W) ──→ RGBBranch  ──→ (B,32,H/4,W/4)  ─┘

总计算量目标: < 1 GFLOPs
"""

import torch
import torch.nn as nn

from .ir_branch import IRBranch
from .rgb_branch import RGBBranch
from .fusion import AsymmetricFusion


class ROINetwork(nn.Module):
    """端到端的轻量级多模态 ROI 显著性预测网络。

    Args:
        ir_in_channels: IR 输入通道数
        ir_mid_channels: IR 中间通道数
        ir_out_channels: IR 输出通道数
        rgb_pretrained: RGB 分支是否预训练
        rgb_out_channels: RGB 输出通道数
        rgb_feature_layer: MobileNetV3 截取层数
        strip_kernel: CAA 条带卷积核大小
        reduction: CAA 通道缩减比
        saliency_mid_channels: 显著性头中间通道数
    """

    def __init__(
        self,
        ir_in_channels: int = 1,
        ir_mid_channels: int = 16,
        ir_out_channels: int = 32,
        rgb_pretrained: bool = True,
        rgb_out_channels: int = 32,
        rgb_feature_layer: int = 4,
        strip_kernel: int = 11,
        reduction: int = 4,
        saliency_mid_channels: int = 32,
    ):
        super().__init__()

        # 双流分支
        self.ir_branch = IRBranch(
            in_channels=ir_in_channels,
            mid_channels=ir_mid_channels,
            out_channels=ir_out_channels,
        )

        self.rgb_branch = RGBBranch(
            pretrained=rgb_pretrained,
            out_channels=rgb_out_channels,
            feature_layer=rgb_feature_layer,
        )

        # 融合 + CAA + 显著性头
        self.fusion = AsymmetricFusion(
            ir_channels=ir_out_channels,
            rgb_channels=rgb_out_channels,
            strip_kernel=strip_kernel,
            reduction=reduction,
            mid_channels=saliency_mid_channels,
        )

    def forward(
        self,
        ir: torch.Tensor,
        rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            ir:  红外图像 (B, 1, H, W)
            rgb: 可见光图像 (B, 3, H, W)

        Returns:
            dict:
                "saliency_logits": (B, 1, H, W) — 训练用
                "saliency_map":   (B, 1, H, W) — 推理用
                "fused_feat":     (B, 64, H/4, W/4)
        """
        original_size = (ir.shape[2], ir.shape[3])

        # 双流特征提取
        ir_feat = self.ir_branch(ir)       # (B, 32, H/4, W/4)
        rgb_feat = self.rgb_branch(rgb)    # (B, 32, H/4, W/4)

        # 融合 + 显著性生成
        outputs = self.fusion(ir_feat, rgb_feat, original_size=original_size)

        return outputs

    def count_flops(self, input_size: tuple = (512, 640)) -> str:
        """统计 FLOPs (需要安装 thop)"""
        try:
            from thop import profile
        except ImportError:
            return "请安装 thop: pip install thop"

        device = next(self.parameters()).device
        ir = torch.randn(1, 1, *input_size, device=device)
        rgb = torch.randn(1, 3, *input_size, device=device)

        flops, params = profile(self, inputs=(ir, rgb), verbose=False)

        # 清理 thop 注入的 buffer，防止污染 state_dict
        for m in self.modules():
            if hasattr(m, 'total_ops'):
                del m.total_ops
            if hasattr(m, 'total_params'):
                del m.total_params

        return (
            f"FLOPs: {flops / 1e9:.3f} GFLOPs | "
            f"Params: {params / 1e6:.3f} M"
        )


def build_model_from_config(cfg: dict) -> ROINetwork:
    """从配置字典构建模型。

    Args:
        cfg: 配置字典（来自 YAML）

    Returns:
        ROINetwork 实例
    """
    model_cfg = cfg.get("model", {})
    ir_cfg = model_cfg.get("ir_branch", {})
    rgb_cfg = model_cfg.get("rgb_branch", {})
    caa_cfg = model_cfg.get("caa", {})
    head_cfg = model_cfg.get("saliency_head", {})

    return ROINetwork(
        ir_in_channels=ir_cfg.get("in_channels", 1),
        ir_mid_channels=ir_cfg.get("mid_channels", 16),
        ir_out_channels=ir_cfg.get("out_channels", 32),
        rgb_pretrained=rgb_cfg.get("pretrained", True),
        rgb_out_channels=rgb_cfg.get("out_channels", 32),
        rgb_feature_layer=rgb_cfg.get("feature_layer", 4),
        strip_kernel=caa_cfg.get("strip_kernel", 11),
        reduction=caa_cfg.get("reduction", 4),
        saliency_mid_channels=head_cfg.get("mid_channels", 32),
    )


if __name__ == "__main__":
    model = ROINetwork()
    ir = torch.randn(2, 1, 512, 640)
    rgb = torch.randn(2, 3, 512, 640)

    outputs = model(ir, rgb)
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")

    print(model.count_flops())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total_params:,} | Trainable: {trainable_params:,}")
