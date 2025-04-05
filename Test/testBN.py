import torch
import torch.nn as nn

# Khởi tạo một tensor giả lập cho ảnh đầu vào (batch_size=2, channels=3, height=4, width=4)
input_tensor = torch.randn(2, 3, 4, 4)

# In thông tin về input_tensor
print("Input to Conv Layer:")
print("  Min:", input_tensor.min().item())
print("  Max:", input_tensor.max().item())
print("  Mean:", input_tensor.mean().item())
print("  Variance:", input_tensor.var().item())

# Định nghĩa một lớp Convolutional với 3 kênh đầu vào và 6 kênh đầu ra, kernel size 3x3
conv_layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
# Áp dụng lớp Convolutional lên input_tensor
conv_output = conv_layer(input_tensor)

# In thông tin về conv_output
print("\nOutput from Conv Layer (Input to BN):")
print("  Min:", conv_output.min().item())
print("  Max:", conv_output.max().item())
print("  Mean:", conv_output.mean().item())
print("  Variance:", conv_output.var().item())

# Định nghĩa một lớp Batch Normalization với 6 kênh (tương ứng với số kênh đầu ra của Conv)
bn_layer = nn.BatchNorm2d(num_features=6)
# Áp dụng lớp Batch Normalization lên output của Conv
bn_output = bn_layer(conv_output)

# In thông tin về bn_output
print("\nOutput from BatchNorm Layer:")
print("  Min:", bn_output.min().item())
print("  Max:", bn_output.max().item())
print("  Mean:", bn_output.mean().item())
print("  Variance:", bn_output.var().item())

print("Gamma:", bn_layer.weight.data)
print("Beta:", bn_layer.bias.data)