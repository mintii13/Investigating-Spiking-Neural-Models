import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
    MultiStepIFNode,
)
from .Learnable_IF import MultiStepLearnableIFNode
from timm.models.layers import to_2tuple
import math

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
        rpe_mode="conv",  # Add rpe_mode parameter
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        self.rpe_mode = rpe_mode  # Store rpe_mode

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
            self.proj_lif = MultiStepIFNode(v_threshold=1.0, v_reset=v_reset, detach_reset=True, backend="cupy")
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
            self.proj_lif1 = MultiStepIFNode(v_threshold=1.0, v_reset=v_reset, detach_reset=True, backend="cupy")
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
            self.proj_lif2 = MultiStepIFNode(v_threshold=1.0, v_reset=v_reset, detach_reset=True, backend="cupy")
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
            self.proj_lif3 = MultiStepIFNode(v_threshold=1.0, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            self.proj_lif3 = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            self.proj_lif3 = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)
          
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Initialize RPE layers based on rpe_mode
        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        if rpe_mode in ["linear", "all"]:
            self.rpe_linear = nn.Linear(embed_dims, embed_dims, bias=False)
        if rpe_mode in ["sinusoidal", "all"]:
            self.rpe_scale = nn.Parameter(torch.ones(1))
        if rpe_mode in ["learnable", "all"]:
            self.rpe_pos_embed_h = nn.Parameter(torch.randn(1, embed_dims//2, 64, 1))
            self.rpe_pos_embed_w = nn.Parameter(torch.randn(1, embed_dims//2, 1, 64))
        if rpe_mode in ["dilated", "all"]:
            self.rpe_dilated = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "if":
            self.rpe_lif = MultiStepIFNode(v_threshold=1.0, v_reset=v_reset, detach_reset=True, backend="cupy")
        elif spike_mode == "if_soft":
            self.rpe_lif = MultiStepIFNode(v_threshold=1.0, v_reset=None, detach_reset=True, backend="cupy")
        elif spike_mode == "if_learnable":
            self.rpe_lif = MultiStepLearnableIFNode(init_threshold=1.0, v_reset=None, detach_reset=True)

        # Log RPE mode configuration
        print(f"RPE mode = {self.rpe_mode}")

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
        x = x.flatten(0, 1).contiguous()
        
        # ========== RPE SWITCH CASE - Now configurable ==========
        if self.rpe_mode == "conv":
            # Option 0: Original Conv2D
            x = self.rpe_conv(x)
            
        elif self.rpe_mode == "linear":
            # Option 1: Linear
            TB, C, H_cur, W_cur = x.shape
            x_linear = x.view(TB, C, -1).transpose(1, 2)  # [TB, H*W, C]
            x = self.rpe_linear(x_linear).transpose(1, 2).view(TB, C, H_cur, W_cur)
            
        elif self.rpe_mode == "sinusoidal":
            # Option 2: Sinusoidal
            TB, C, H_cur, W_cur = x.shape
            pe = get_sinusoidal_encoding(H_cur, W_cur, C, x.device)
            pe = pe.unsqueeze(0).expand(TB, -1, -1, -1)  # [TB, C, H, W]
            x = self.rpe_scale * pe
            
        elif self.rpe_mode == "learnable":
            # Option 3: Learnable Position
            TB, C, H_cur, W_cur = x.shape
            pos_h = self.rpe_pos_embed_h[:, :, :H_cur, :].expand(-1, -1, -1, W_cur)  # [1, C//2, H, W]
            pos_w = self.rpe_pos_embed_w[:, :, :, :W_cur].expand(-1, -1, H_cur, -1)  # [1, C//2, H, W]
            pos_embed = torch.cat([pos_h, pos_w], dim=1)  # [1, C, H, W]
            pos_embed = pos_embed.expand(TB, -1, -1, -1)  # [TB, C, H, W]
            x = x + pos_embed
            # print(f"Learnable PE applied - Used spatial size: H={H_cur}, W={W_cur}")
            # print(f"PE - mean: {pos_embed.mean().item():.6f} | max: {pos_embed.max().item():.6f} | min: {pos_embed.min().item():.6f}")
            
        elif self.rpe_mode == "dilated":
            # Option 4: Dilated Conv
            x = self.rpe_dilated(x)
            
        else:
            # Default: Original Conv2D
            x = self.rpe_conv(x)

        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook


def get_sinusoidal_encoding(H, W, C, device):
    """Most robust vectorized implementation"""
    # Create position matrices
    pos_h = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2)  # [H, 1, 1]
    pos_w = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)  # [1, W, 1]
    
    # Initialize PE tensor
    pe = torch.zeros(H, W, C, dtype=torch.float32, device=device)
    
    # Fill first half with height-based encoding
    for i in range(0, C // 2, 2):
        freq = math.pow(10000.0, -i / (C // 2))
        pe[:, :, i] = torch.sin(pos_h.squeeze(-1) * freq).expand(-1, W)
        if i + 1 < C // 2:
            pe[:, :, i + 1] = torch.cos(pos_h.squeeze(-1) * freq).expand(-1, W)
    
    # Fill second half with width-based encoding
    for i in range(C // 2, C, 2):
        freq = math.pow(10000.0, -(i - C // 2) / (C - C // 2))
        pe[:, :, i] = torch.sin(pos_w.squeeze(-1) * freq).expand(H, -1)
        if i + 1 < C:
            pe[:, :, i + 1] = torch.cos(pos_w.squeeze(-1) * freq).expand(H, -1)
    
    return pe.permute(2, 0, 1)  # [C, H, W]