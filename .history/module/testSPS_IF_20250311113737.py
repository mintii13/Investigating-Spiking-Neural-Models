import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

# Định nghĩa lớp IF Neuron
class MultiStepIFNode(nn.Module):
    def __init__(self, v_threshold=1.0, detach_reset=True):
        super().__init__()
        self.v_threshold = v_threshold  # Ngưỡng phát xung
        self.detach_reset = detach_reset  # Có tách gradient khi reset hay không

    def forward(self, x_seq):
        # x_seq: (T, B, C, H, W)
        T, B, C, H, W = x_seq.shape
        spike_seq = []  # Lưu trữ chuỗi xung
        v = torch.zeros(B, C, H, W, device=x_seq.device)  # Điện thế ban đầu

        for t in range(T):
            v += x_seq[t]  # Tích lũy điện thế
            spike = (v >= self.v_threshold).float()  # Phát xung nếu vượt ngưỡng
            v = v - spike * self.v_threshold  # Reset điện thế: V = V - ngưỡng phát xung
            if self.detach_reset:
                v = v.detach()  # Tách gradient khi reset
            spike_seq.append(spike)

        return torch.stack(spike_seq)  # Trả về chuỗi xung


# Định nghĩa lớp MS_SPS với IF Neuron
class MS_SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
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

        # Conv + BN + IF layers
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_if = MultiStepIFNode(v_threshold=1.0, detach_reset=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Conv1 + BN1 + IF1
        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_if1 = MultiStepIFNode(v_threshold=1.0, detach_reset=True)

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Conv2 + BN2 + IF2
        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_if2 = MultiStepIFNode(v_threshold=1.0, detach_reset=True)

        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # Conv3 + BN3 + IF3
        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_if3 = MultiStepIFNode(v_threshold=1.0, detach_reset=True)

        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        # RPE Conv + BN + IF
        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_if = MultiStepIFNode(v_threshold=1.0, detach_reset=True)

    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1
        
        # Function to print number of Spike
        def print_spike_count(name, tensor):
            spike_count = torch.sum(tensor == 1).item()  # Đếm số lượng xung (giá trị 1)
            print(f"{name} - Spike count: {spike_count}")
        # Helper function to print tensor details
        def print_tensor(name, tensor):
            print(f"\n{name}:")
            print("Shape:", tensor.shape)
            data = tensor.detach().cpu().numpy()
            print("Data (first 10 rows):\n", data.flatten()[:10])
            print("Min:", tensor.min().item(), "Max:", tensor.max().item())
            print("Mean:", tensor.mean().item())
            print("Variance:", tensor.var().item())

        print_tensor("Input", x)

        # Step 1: Conv + BN + IF
        x = self.proj_conv(x.flatten(0, 1))
        print_tensor("After proj_conv1 (Convolution)", x)

        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("After proj_bn1 (BatchNorm)", x)

        x = self.proj_if(x)
        print_tensor("After proj_if1 (IF Neuron)", x)
        print_spike_count("After proj_if1 (IF Neuron)", x)

        # Step 2: MaxPool (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2
            print_tensor("After maxpool1 (MaxPooling)", x)

        # Step 3: Conv1 + BN1 + IF1
        x = self.proj_conv1(x)
        print("\n====================")
        print_tensor("After proj_conv2 (Convolution)", x)

        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("After proj_bn2 (BatchNorm)", x)

        x = self.proj_if1(x)
        print_tensor("After proj_if2 (IF Neuron)", x)
        print_spike_count("After proj_if2 (IF Neuron)", x)

        # Step 4: MaxPool1 (optional)
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2
            print_tensor("After maxpool2 (MaxPooling)", x)

        # Step 5: Conv2 + BN2 + IF2
        x = self.proj_conv2(x)
        print("\n====================")
        print_tensor("After proj_conv3 (Convolution)", x)

        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        print_tensor("After proj_bn3 (BatchNorm)", x)

        x = self.proj_if2(x)
        print_tensor("After proj_if3 (IF Neuron)", x)
        print_spike_count("After proj_if3 (IF Neuron)", x)

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

        # Step 9: IF3 + Residual connection
        x_feat = x
        x = self.proj_if3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())
        print("\n====================")
        print_tensor("After rpe_if (IF Neuron)", x)
        print_spike_count("After rpe_if (IF Neuron)", x)

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
)

# Di chuyển model sang GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sps.to(device)

# Tạo input tensor giả lập với giá trị pixel từ 0 đến 255
# Kích thước: (T, B, C, H, W) = (4, 2, 3, 128, 128)
input_tensor = torch.randint(0, 256, (4, 2, 3, 128, 128), dtype=torch.float32).to(device)

# Forward qua SPS
output, (H, W), hook = sps(input_tensor)