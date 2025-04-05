import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple

# Định nghĩa lớp MS_SPS (giữ nguyên như đoạn mã bạn cung cấp)
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
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
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
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Các lớp khác giữ nguyên...

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1

        # Step 1: Conv + BN + LIF
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)

        # Step 2: MaxPool (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        # Step 3: Conv1 + BN1 + LIF1
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)

        # Step 4: MaxPool1 (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)  # <-- Đây là input của conv2
            ratio *= 2

        # Lưu input của conv2 vào biến global
        global conv2_input
        conv2_input = x.clone()

        # Tiếp tục forward...
        return x


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
# Kích thước: (T, B, C, H, W) = (4, 2, 3, 128, 128)
input_tensor = torch.randint(0, 256, (4, 2, 3, 128, 128), dtype=torch.float32).to(device)

# Forward qua SPS
output = sps(input_tensor)

# In thông tin về input của conv2
print("\nInput of conv2 shape:", conv2_input.shape)
# print("Input of conv2 data:\n", conv2_input)

# Sử dụng input của conv2 làm input cho đoạn mã BN
# Định nghĩa một lớp Convolutional và Batch Normalization
conv_layer = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
bn_layer = nn.BatchNorm2d(num_features=128)

# Di chuyển các lớp sang GPU nếu có
conv_layer.to(device)
bn_layer.to(device)

# Áp dụng Conv và BN lên input của conv2
conv_output = conv_layer(conv2_input)
print("\nOutput from Conv Layer (Input to BN):")
print("  Min:", conv_output.min().item())
print("  Max:", conv_output.max().item())
print("  Mean:", conv_output.mean().item())
print("  Variance:", conv_output.var().item())

bn_output = bn_layer(conv_output)
print("\nOutput from BatchNorm Layer:")
print("  Min:", bn_output.min().item())
print("  Max:", bn_output.max().item())
print("  Mean:", bn_output.mean().item())
print("  Variance:", bn_output.var().item())

print("Gamma:", bn_layer.weight.data)
print("Beta:", bn_layer.bias.data)