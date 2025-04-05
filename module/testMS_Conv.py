import torch
import torch.nn as nn
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

# Định nghĩa lớp Erode
class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)

# Định nghĩa lớp MS_MLP_Conv
class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        print_tensor("After fc1_lif (MLP LIF Neuron)", x)

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        print_tensor("After fc1_conv and bn (MLP Convolution)", x)

        if self.res:
            x = identity + x
            identity = x

        x = self.fc2_lif(x)
        print_tensor("After fc2_lif (MLP LIF Neuron)", x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        print_tensor("After fc2_conv and bn (MLP Convolution)", x)

        x = x + identity
        return x, hook


# Định nghĩa lớp MS_SSA_Conv
class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125

        # Định nghĩa các convolution và neuron LIF cho q, k, v
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        # Attention LIF
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        # Talking heads
        self.talking_heads = nn.Conv1d(num_heads, num_heads, kernel_size=1, stride=1, bias=False)
        self.talking_heads_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        # Projection
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        # Shortcut LIF
        self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W

        x = self.shortcut_lif(x)
        print_tensor("After shortcut_lif (SSA Shortcut LIF)", x)

        x_for_qkv = x.flatten(0, 1)

        # Query
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        print_tensor("After q_lif (Query LIF)", q_conv_out)

        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # Key
        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        print_tensor("After k_lif (Key LIF)", k_conv_out)

        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # Value
        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        print_tensor("After v_lif (Value LIF)", v_conv_out)

        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # Attention calculation
        kv = k.mul(v)
        print_tensor("After kv (Attention KV)", kv)

        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        print_tensor("After talking_heads_lif (Talking Heads LIF)", kv)

        x = q.mul(kv)
        print_tensor("After q.mul(kv) (Attention Output)", x)

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )
        print_tensor("After proj_conv (Projection Convolution)", x)

        x = x + identity
        return x, v, hook


# Định nghĩa lớp MS_Block_Conv
class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x, hook=None):
        print_tensor("Input to MS_Block_Conv", x)

        x_attn, attn, hook = self.attn(x, hook=hook)
        print_tensor("After MS_SSA_Conv (Self-Attention Block), x + identity", x_attn)

        x, hook = self.mlp(x_attn, hook=hook)
        print_tensor("After MS_MLP_Conv (MLP Block)", x)

        return x, attn, hook


# Hàm hỗ trợ in tensor
def print_tensor(name, tensor):
    print(f"\n{name}:")
    print("Shape:", tensor.shape)
    data = tensor.detach().cpu().numpy()
    print("Data (first 10 rows):\n", data.flatten()[:10])
    print("Min:", tensor.min().item(), "Max:", tensor.max().item())


# Khởi tạo module MS_Block_Conv
block = MS_Block_Conv(
    dim=256,
    num_heads=8,
    mlp_ratio=4.0,
    spike_mode="lif",
    dvs=False,
    layer=0,
)

# Di chuyển model sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
block.to(device)

# Tạo input tensor giả lập
input_tensor = torch.randn(4, 2, 256, 16, 16).to(device)  # Kích thước: (T, B, C, H, W)

# Forward qua MS_Block_Conv
output, attn, hook = block(input_tensor)

print("\nFinal Output shape:", output.shape)