import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
    MultiStepIFNode,
)
from .Learnable_IF import MultiStepLearnableIFNode
from timm.models.layers import to_2tuple

v_reset = 0.0

class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="if_learnable",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat

        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        
        # Initialize neurons based on spike_mode
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "if":
            print("spike_mode = if_hard")
            self.proj_lif = MultiStepIFNode(v_threshold=1.2, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            print("spike_mode = if_soft")
            self.proj_lif = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            print("spike_mode = if_learnable")
            self.proj_lif = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)
          
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "if":
            self.proj_lif1 = MultiStepIFNode(v_threshold=1.2, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            self.proj_lif1 = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            self.proj_lif1 = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)
          
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "if":
            self.proj_lif2 = MultiStepIFNode(v_threshold=1.2, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            self.proj_lif2 = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            self.proj_lif2 = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "if":
            self.proj_lif3 = MultiStepIFNode(v_threshold=1.2, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            self.proj_lif3 = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            self.proj_lif3 = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)
          
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
         # Option 1: Linear
        self.rpe_linear = nn.Linear(embed_dims, embed_dims, bias=False)
        # Option 2: Sinusoidal 
        self.rpe_scale = nn.Parameter(torch.ones(1))
        # Option 3: Learnable Position
        self.rpe_pos_embed_h = nn.Parameter(torch.randn(1, embed_dims//2, 64, 1))
        self.rpe_pos_embed_w = nn.Parameter(torch.randn(1, embed_dims//2, 1, 64))
        # Option 5: Dilated Conv
        self.rpe_dilated = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "if":
            self.rpe_lif = MultiStepIFNode(v_threshold=1.2, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            self.rpe_lif = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            self.rpe_lif = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1
        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()
        # Option 0: Conv2D
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        # Option 1: Linear
        x_linear = x.flatten(2).transpose(-1, -2)  # [T*B, H*W, C]
        x_linear = self.rpe_linear(x_linear)  # [T*B, H*W, C]
        x_linear = x_linear.transpose(-1, -2).reshape_as(x)  # Back to [T*B, C, H, W]
        x_linear = self.rpe_bn(x_linear)
        x_linear = (x_linear + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook