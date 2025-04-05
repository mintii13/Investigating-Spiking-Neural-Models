import numpy as np

import torch


# class PoissonEncoder:
#     # Poisson processor for encoding images
#     def __init__(self, time_interval):
#         self.time_interval = time_interval
#
#     def __call__(self, images):
#         # images.size: [b, c, h, w]
#         # spiked_images.size: [t, b, c, h, w]
#         b, c, h, w = images.size()
#         r = images.unsqueeze(0).repeat(self.time_interval, 1, 1, 1, 1) / 32.0
#         p = torch.rand(self.time_interval, b, c, h, w)
#         return (p <= r).float()
#
#
# class PoissonEncoder2:
#     def __init__(self, time_interval):
#         self.time_interval = time_interval
#
#     def __call__(self, images):
#         rate = torch.zeros(size)
#         rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)


class Simple:
    """ This is a simple version encoder

        It has to inherit base encoder for consistency.

    """
    def __init__(self, time_interval: int = 100, scale: float = 5.0) -> None:
        """

            scale에 대한 고찰
            scale < 1.0인 경우는 more deterministic 특성을 보이게 된다.
            scale > 1.0인 경우는 more stochastic 특성을 보이게 된다. (more realistic spike train)

        :param time_interval:
        :param scale:
        """
        self.time_interval = time_interval
        self.scale = scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x.size: [1, 28, 28]
        xx = x.unsqueeze(dim=0).repeat(self.time_interval, 1, 1, 1)
        r = torch.rand([self.time_interval] + [_ for _ in x.size()], device=x.device)
        return (xx >= self.scale * r).float()


class SimplePoisson:
    # Poisson processor for encoding images
    def __init__(self, time_interval):
        self.time_interval = time_interval

    def __call__(self, images):
        # images.size: [c, h, w]
        # spiked_images.size: [t, c, h, w]
        c, h, w = images.size()
        r = images.unsqueeze(0).repeat(self.time_interval, 1, 1, 1)
        p = torch.rand(self.time_interval, c, h, w)
        return (p <= r).float()


class Encoder:
    # language=rst
    """
    Base class for spike encodings transforms.

    Calls ``self.enc`` from the subclass and passes whatever arguments were provided.
    ``self.enc`` must be callable with ``torch.Tensor``, ``*args``, ``**kwargs``
    """

    def __init__(self, *args, **kwargs) -> None:
        self.enc_args = args
        self.enc_kwargs = kwargs

    def __call__(self, img):
        return self.enc(img, *self.enc_args, **self.enc_kwargs)


class PoissonEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable PoissonEncoder which encodes as defined in
        ``bindsnet.encoding.poisson`

        :param time: Length of Poisson spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = poisson


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)

def test_encoders(choice):
    # Tạo một generator riêng cho việc tạo dữ liệu đầu vào
    input_generator = torch.Generator()
    input_generator.manual_seed(42)  # Đặt seed cho generator này

    # Tạo dữ liệu đầu vào giả lập với generator riêng
    input_image = torch.randint(
        0, 256, (1, 28, 28), dtype=torch.float32, generator=input_generator
    )  # Giá trị pixel từ 0 đến 255
    print("Input image shape:", input_image.shape)
    print("First few values of input image:")
    print(input_image[0, :5, :5])  # In ra một phần nhỏ của hình ảnh

    # Khởi tạo encoder tương ứng
    time_interval = 10  # Độ dài chuỗi xung
    if choice == 1:
        encoder = Simple(time_interval=time_interval, scale=100.0)
        encoder_name = "Simple Encoder"
    elif choice == 2:
        encoder = SimplePoisson(time_interval=time_interval)
        encoder_name = "SimplePoisson Encoder"
    elif choice == 3:
        encoder = PoissonEncoder(time=time_interval, dt=1.0)
        encoder_name = "PoissonEncoder"

    # Mã hóa dữ liệu
    spiked_images = encoder(input_image)

    # In kết quả mã hóa
    def print_results(encoder_name, spiked_images):
        print(f"\n--- {encoder_name} ---")
        print("Spiked images shape:", spiked_images.shape)
        for t in range(time_interval):
            print(f"\nTime step {t}:")
            print(spiked_images[t, 0, :5, :5])

        # Tính tổng số xung qua tất cả các bước thời gian
        total_spikes = torch.sum(spiked_images, dim=0)[0, :5, :5]
        print("\nNumber of spikes over time interval:")
        print(total_spikes)

    # In kết quả
    print_results(encoder_name, spiked_images)

# Test hết cả 3
def test_all_encoders():
    # Tạo một generator riêng cho việc tạo dữ liệu đầu vào
    input_generator = torch.Generator()
    input_generator.manual_seed(42)  # Đặt seed cho generator này

    # Tạo dữ liệu đầu vào giả lập với generator riêng
    input_image = torch.randint(
        0, 256, (1, 28, 28), dtype=torch.float32, generator=input_generator
    )  # Giá trị pixel từ 0 đến 255
    print("Input image shape:", input_image.shape)
    print("First few values of input image:")
    print(input_image[0, :5, :5])  # In ra một phần nhỏ của hình ảnh

    # KHÔNG đặt seed cho các phép toán ngẫu nhiên khác
    # Các encoder sẽ sử dụng bộ sinh số ngẫu nhiên mặc định của PyTorch

    # Khởi tạo các encoder
    time_interval = 4  # Độ dài chuỗi xung
    simple_encoder = Simple(time_interval=time_interval, scale=100.0)
    poisson_encoder = SimplePoisson(time_interval=time_interval)
    full_poisson_encoder = PoissonEncoder(time=time_interval, dt=1.0)

    # Mã hóa dữ liệu
    simple_spikes = simple_encoder(input_image)
    poisson_spikes = poisson_encoder(input_image)
    full_poisson_spikes = full_poisson_encoder(input_image)

    # Hàm tính và in tổng số xung
    def print_total_spikes(encoder_name, spiked_images):
        total_spikes = torch.sum(spiked_images, dim=0)[0, :5, :5]
        print(f"\n--- {encoder_name} ---")
        print("Number of spikes over time interval:")
        print(total_spikes)

    # In kết quả cho từng encoder
    print_total_spikes("Simple Encoder", simple_spikes)
    print_total_spikes("SimplePoisson Encoder", poisson_spikes)
    print_total_spikes("PoissonEncoder", full_poisson_spikes)


# Main
# test_encoders(3)
test_all_encoders()