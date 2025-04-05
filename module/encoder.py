import torch

class SimplePoisson:
    def __init__(self, time_interval):
        self.time_interval = time_interval

    def __call__(self, images):
        # Xác định kích thước của batch
        batch_size, c, h, w = images.size()
        r = images.unsqueeze(1).repeat(1, self.time_interval, 1, 1, 1) # Repeat theo batch và timestep
        
        # Tạo ma trận ngẫu nhiên trên cùng thiết bị với `images`
        p = torch.rand(batch_size, self.time_interval, c, h, w, device=images.device)
        
        return (p <= r).float()


class Encoder:
    def __init__(self, *args, **kwargs) -> None:
        self.enc_args = args
        self.enc_kwargs = kwargs

    def __call__(self, img):
        return self.enc(img, *self.enc_args, **self.enc_kwargs)


class PoissonEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        super().__init__(time, dt=dt, **kwargs)
        self.enc = poisson


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)


