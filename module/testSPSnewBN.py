import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple
import torch.nn.init as init

# Định nghĩa lớp MS_SPS
class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
        spike_mode="lif",
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

        # Conv + BN + LIF layers
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        init.constant_(self.proj_bn.weight, 1.0)  # Khởi tạo gamma = 1.0
        init.constant_(self.proj_bn.bias, 0.0)    # Khởi tạo beta = 0.0

        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Conv1 + BN1 + LIF1
        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        init.constant_(self.proj_bn1.weight, 1.0)  # Khởi tạo gamma = 1.0
        init.constant_(self.proj_bn1.bias, 0.0)    # Khởi tạo beta = 0.0

        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Conv2 + BN2 + LIF2
        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        init.constant_(self.proj_bn2.weight, 1.0)  # Khởi tạo gamma = 1.0
        init.constant_(self.proj_bn2.bias, 0.0)    # Khởi tạo beta = 0.0

        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Conv3 + BN3
        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        init.constant_(self.proj_bn3.weight, 1.0)  # Khởi tạo gamma = 1.0
        init.constant_(self.proj_bn3.bias, 0.0)    # Khởi tạo beta = 0.0

        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # RPE Conv + BN
        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        init.constant_(self.rpe_bn.weight, 1.0)  # Khởi tạo gamma = 1.0
        init.constant_(self.rpe_bn.bias, 0.0)    # Khởi tạo beta = 0.0

        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1

        # Helper function to print tensor details
        def print_tensor(name, tensor):
            print(f"\n{name}:")
            print("Shape:", tensor.shape)
            data = tensor.detach().cpu().numpy()
            print("Data (first 10 rows):\n", data.flatten()[:10])
            print("Min:", tensor.min().item(), "Max:", tensor.max().item())

        print_tensor("Input", x)

        # Step 1: Conv + BN + LIF
        x = self.proj_conv(x.flatten(0, 1))
        print_tensor("After proj_conv1 (Convolution)", x)

        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("After proj_bn1 (BatchNorm)", x)

        x = self.proj_lif(x)
        print_tensor("After proj_lif1 (LIF Neuron)", x)

        # Step 2: MaxPool (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2
            print_tensor("After maxpool1 (MaxPooling)", x)

        # Step 3: Conv1 + BN1 + LIF1
        x = self.proj_conv1(x)
        print("\n====================")
        print_tensor("After proj_conv2 (Convolution)", x)

        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("After proj_bn2 (BatchNorm)", x)

        x = self.proj_lif1(x)
        print_tensor("After proj_lif2 (LIF Neuron)", x)

        # Step 4: MaxPool1 (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2
            print_tensor("After maxpool2 (MaxPooling)", x)

        # Step 5: Conv2 + BN2 + LIF2
        x = self.proj_conv2(x)
        print("\n====================")
        print_tensor("After proj_conv3 (Convolution)", x)

        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("After proj_bn3 (BatchNorm)", x)

        x = self.proj_lif2(x)
        print_tensor("After proj_lif3 (LIF Neuron)", x)

        # Step 6: MaxPool2 (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2
            print_tensor("After maxpool3 (MaxPooling)", x)

        # Step 7: Conv3 + BN3
        x = self.proj_conv3(x)
        print("\n====================")
        print_tensor("After proj_conv4 (Convolution)", x)

        x = self.proj_bn3(x)
        print_tensor("After proj_bn4 (BatchNorm)", x)

        # Step 8: MaxPool3 (optional)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2
            print_tensor("After maxpool4 (MaxPooling) - x_feat", x)

        # Step 9: LIF3 + Residual connection
        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        print("\n====================")
        print_tensor("After rpe_lif (LIF Neuron)", x)

        x = x.flatten(0, 1).contiguous()

        # Step 10: RPE Conv + BN
        x = self.rpe_conv(x)
        print_tensor("After rpe_conv (Relative Positional Encoding Convolution)", x)

        x = self.rpe_bn(x)
        print_tensor("After rpe_bn (BatchNorm for RPE)", x)

        # Final residual addition
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("Final output (x + x_feat)", x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook


# Khởi tạo module MS_SPS
sps = MS_SPS(
    img_size_h=128,
    img_size_w=128,
    patch_size=16,
    in_channels=3,
    embed_dims=256,
    pooling_stat="1111",
    spike_mode="lif",
)

# Di chuyển model sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sps.to(device)

# Tạo input tensor giả lập với giá trị pixel từ 0 đến 255
# Kích thước: (T, B, C, H, W) = (8, 4, 3, 128, 128)
input_tensor = torch.randint(0, 256, (8, 64, 3, 128, 128), dtype=torch.float32).to(device)

# Forward qua SPS
output, (H, W), hook = sps(input_tensor)

# In ra thống kê của các lớp BN
def print_bn_stats(name, bn_layer):
    print(f"\n{name} stats:")
    print("Running mean:", bn_layer.running_mean.detach().cpu().numpy())
    print("Running variance:", bn_layer.running_var.detach().cpu().numpy())

print_bn_stats("proj_bn", sps.proj_bn)
print_bn_stats("proj_bn1", sps.proj_bn1)
print_bn_stats("proj_bn2", sps.proj_bn2)
print_bn_stats("proj_bn3", sps.proj_bn3)

print("Gamma of proj_bn2:", sps.proj_bn2.weight.data)
print("Beta of proj_bn2:", sps.proj_bn2.bias.data)