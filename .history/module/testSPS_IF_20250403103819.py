import torch
from spikingjelly.clock_driven import neuron

# Khởi tạo IFNode với soft reset
if_node = neuron.IFNode(v_threshold=1.0, v_reset=None)
if_node.v = 0.0
# Tạo dữ liệu đầu vào
x = torch.tensor([0.5, 1.2, 0.8, 2.2, 0.1])
spike = if_node(x)
for t in range(len(x)):
    print(f"Step {t}:")
    print(f"  Before accumulation: v = {if_node.v}")
    if_node(x[t])
    print(f"  After accumulation and reset: v = {if_node.v}")
print("Spike:", spike)
print("Voltage after reset:", if_node.v)