from spikingjelly.clock_driven import neuron
import torch

# Khởi tạo MultiStepIFNode với soft reset
multi_step_if_node = neuron.MultiStepIFNode(v_threshold=1.0, v_reset=None)

# Đặt điện thế ban đầu
multi_step_if_node.v = 0.0

# Tạo dữ liệu đầu vào với shape (T, *)
x_seq = torch.tensor([[0.5], [1.2], [0.2], [1.2], [0.1]])  # Shape (T=5, B=1)

# Chạy forward pass
spike_seq = multi_step_if_node(x_seq)

# In ra chuỗi xung
print("Spike sequence:", spike_seq)

# In ra điện thế màng sau reset cuối cùng
print("Voltage after reset (final):", multi_step_if_node.v)

# In ra điện thế màng cho từng timestep
print("Membrane potential at each timestep:")
for t, v_t in enumerate(multi_step_if_node.v_seq):
    print(f"  Timestep {t}: {v_t}")