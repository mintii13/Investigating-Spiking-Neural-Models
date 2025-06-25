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
        x = x.flatten(0, 1).contiguous()
        
        # ========== RPE SWITCH CASE ==========
        rpe_mode = "conv" 

        if rpe_mode == "conv":
            # Option 0: Original Conv2D
            x_before = x.clone()
            print(f"Input - shape: {x_before.shape} | mean: {x_before.mean().item():.6f} | max: {x_before.max().item():.6f} | min: {x_before.min().item():.6f}")
            x = self.rpe_conv(x)
            print(f"rpe_mode: {rpe_mode} | Conv2D applied | Weight shape: {self.rpe_conv.weight.shape}")
            print(f"Output - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")
            print(f"Diff: {torch.abs(x - x_before).mean().item():.6f}")
            
        elif rpe_mode == "linear":
            # Option 1: Linear
            TB, C, H_cur, W_cur = x.shape
            x_before = x.clone()
            print(f"Input - shape: {x_before.shape} | mean: {x_before.mean().item():.6f} | max: {x_before.max().item():.6f} | min: {x_before.min().item():.6f}")
            x_linear = x.view(TB, C, -1).transpose(1, 2)  # [TB, H*W, C]
            print(f"Flattened - shape: {x_linear.shape} | mean: {x_linear.mean().item():.6f} | max: {x_linear.max().item():.6f} | min: {x_linear.min().item():.6f}")
            x = self.rpe_linear(x_linear).transpose(1, 2).view(TB, C, H_cur, W_cur)
            print(f"rpe_mode: {rpe_mode} | Linear applied | Weight shape: {self.rpe_linear.weight.shape}")
            print(f"Output - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")
            print(f"Diff: {torch.abs(x - x_before).mean().item():.6f}")
            
        elif rpe_mode == "sinusoidal":
            # Option 2: Sinusoidal
            TB, C, H_cur, W_cur = x.shape
            x_before = x.clone()
            print(f"Input - shape: {x_before.shape} | mean: {x_before.mean().item():.6f} | max: {x_before.max().item():.6f} | min: {x_before.min().item():.6f}")
            pe = get_sinusoidal_encoding(H_cur, W_cur, C, x.device)
            pe = pe.unsqueeze(0).expand(TB, -1, -1, -1)  # [TB, C, H, W]
            print(f"PE - shape: {pe.shape} | mean: {pe.mean().item():.6f} | max: {pe.max().item():.6f} | min: {pe.min().item():.6f}")
            print(f"Scale: {self.rpe_scale.item():.6f} | Scaled PE mean: {(self.rpe_scale * pe).mean().item():.6f}")
            x = x + self.rpe_scale * pe
            print(f"rpe_mode: {rpe_mode} | Sinusoidal PE added")
            print(f"Output - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")
            print(f"Diff: {torch.abs(x - x_before).mean().item():.6f}")
            
        elif rpe_mode == "learnable":
            # Option 3: Learnable Position
            TB, C, H_cur, W_cur = x.shape
            x_before = x.clone()
            print(f"Input - shape: {x_before.shape} | mean: {x_before.mean().item():.6f} | max: {x_before.max().item():.6f} | min: {x_before.min().item():.6f}")
            pos_h = self.rpe_pos_embed_h[:, :, :H_cur, :].expand(-1, -1, -1, W_cur)  # [1, C//2, H, W]
            pos_w = self.rpe_pos_embed_w[:, :, :, :W_cur].expand(-1, -1, H_cur, -1)  # [1, C//2, H, W]
            print(f"pos_h - shape: {pos_h.shape} | mean: {pos_h.mean().item():.6f} | max: {pos_h.max().item():.6f} | min: {pos_h.min().item():.6f}")
            print(f"pos_w - shape: {pos_w.shape} | mean: {pos_w.mean().item():.6f} | max: {pos_w.max().item():.6f} | min: {pos_w.min().item():.6f}")
            pos_embed = torch.cat([pos_h, pos_w], dim=1)  # [1, C, H, W]
            pos_embed = pos_embed.expand(TB, -1, -1, -1)  # [TB, C, H, W]
            print(f"PE combined - shape: {pos_embed.shape} | mean: {pos_embed.mean().item():.6f} | max: {pos_embed.max().item():.6f} | min: {pos_embed.min().item():.6f}")
            print(f"Used spatial size: H={H_cur}, W={W_cur}")
            x = x + pos_embed
            print(f"rpe_mode: {rpe_mode} | Learnable PE added")
            print(f"Output - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")
            print(f"Diff: {torch.abs(x - x_before).mean().item():.6f}")
            
        elif rpe_mode == "dilated":
            # Option 4: Dilated Conv
            x_before = x.clone()
            print(f"Input - shape: {x_before.shape} | mean: {x_before.mean().item():.6f} | max: {x_before.max().item():.6f} | min: {x_before.min().item():.6f}")
            x = self.rpe_dilated(x)
            print(f"rpe_mode: {rpe_mode} | Dilated Conv applied | Weight shape: {self.rpe_dilated.weight.shape} | Dilation: 2")
            print(f"Output - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")
            print(f"Diff: {torch.abs(x - x_before).mean().item():.6f}")
            
        else:
            # Default: Original Conv2D
            x_before = x.clone()
            print(f"Input - shape: {x_before.shape} | mean: {x_before.mean().item():.6f} | max: {x_before.max().item():.6f} | min: {x_before.min().item():.6f}")
            x = self.rpe_conv(x)
            print(f"rpe_mode: {rpe_mode} (DEFAULT) | Conv2D applied | Weight shape: {self.rpe_conv.weight.shape}")
            print(f"Output - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")
            print(f"Diff: {torch.abs(x - x_before).mean().item():.6f}")

        print(f"FINAL RPE Summary: mode={rpe_mode}")
        print(f"Final tensor - shape: {x.shape} | mean: {x.mean().item():.6f} | max: {x.max().item():.6f} | min: {x.min().item():.6f}")

        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook


def get_sinusoidal_encoding(H, W, C, device):
    """Generate 2D sinusoidal position encoding"""
    pos_h = torch.arange(H, device=device).unsqueeze(1).float()
    pos_w = torch.arange(W, device=device).unsqueeze(0).float()
    
    div_term = torch.exp(torch.arange(0, C//2, 2, device=device).float() * 
                    -(math.log(10000.0) / (C//2)))
    
    pe_h = torch.zeros(H, W, C//2, device=device)
    pe_h[:, :, 0::2] = torch.sin(pos_h * div_term).unsqueeze(1).expand(-1, W, -1)
    pe_h[:, :, 1::2] = torch.cos(pos_h * div_term).unsqueeze(1).expand(-1, W, -1)
    
    pe_w = torch.zeros(H, W, C//2, device=device)
    pe_w[:, :, 0::2] = torch.sin(pos_w * div_term).unsqueeze(0).expand(H, -1, -1)
    pe_w[:, :, 1::2] = torch.cos(pos_w * div_term).unsqueeze(0).expand(H, -1, -1)
    
    pe = torch.cat([pe_h, pe_w], dim=-1).permute(2, 0, 1)  # [C, H, W]
    return pe