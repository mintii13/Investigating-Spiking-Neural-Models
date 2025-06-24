# learnable_threshold_neuron.py - New file to add to module/

import torch
import torch.nn as nn

class LearnableThresholdNeuron(nn.Module):
    """
    Learnable threshold neuron với sigmoid constraint
    Shared threshold across all timesteps
    """
    def __init__(self, 
                 dim, 
                 base_threshold=1.0, 
                 learning_range=0.5,
                 spike_mode="if_soft",
                 init_method="zero"):
        super().__init__()
        
        self.dim = dim
        self.base_threshold = base_threshold
        self.learning_range = learning_range
        self.spike_mode = spike_mode
        
        # Core learnable parameter - shared across timesteps
        if init_method == "zero":
            # sigmoid(0) = 0.5 → maps to base_threshold
            self.threshold_raw = nn.Parameter(torch.zeros(dim))
        elif init_method == "normal":
            # Small random initialization
            self.threshold_raw = nn.Parameter(torch.randn(dim) * 0.1)
        elif init_method == "uniform":
            # Uniform in [-1, 1] range
            self.threshold_raw = nn.Parameter(torch.uniform(-1.0, 1.0, (dim,)))
        
        # Memory for membrane potential (if using LIF-style)
        self.register_buffer('v_mem', None)
        
        # Monitoring buffers for stability
        self.register_buffer('spike_rate_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    @property
    def threshold(self):
        """
        Convert raw parameter to actual threshold using sigmoid constraint
        Range: [base_threshold - learning_range, base_threshold + learning_range]
        """
        # Sigmoid maps (-∞, +∞) → (0, 1)
        sigmoid_out = torch.sigmoid(self.threshold_raw)
        
        # Map to learning range: (0, 1) → (-learning_range, +learning_range)
        offset = self.learning_range * (2 * sigmoid_out - 1)
        
        # Final threshold
        constrained_threshold = self.base_threshold + offset
        
        return constrained_threshold
    
    def get_threshold_stats(self):
        """Get statistics about current thresholds"""
        thresh = self.threshold
        return {
            'mean': thresh.mean().item(),
            'std': thresh.std().item(),
            'min': thresh.min().item(),
            'max': thresh.max().item(),
            'raw_mean': self.threshold_raw.mean().item(),
            'raw_std': self.threshold_raw.std().item()
        }
    
    def reset_state(self):
        """Reset membrane potential"""
        if self.v_mem is not None:
            self.v_mem.zero_()
    
    def monitor_stability(self, spikes):
        """Monitor spike rate stability during training"""
        if not self.training:
            return True
            
        current_rate = spikes.mean().item()
        
        # Update circular buffer
        idx = self.history_idx % 100
        self.spike_rate_history[idx] = current_rate
        self.history_idx += 1
        
        # Check stability after collecting enough history
        if self.history_idx >= 50:
            recent_rates = self.spike_rate_history[:min(100, self.history_idx)]
            rate_std = recent_rates.std().item()
            rate_mean = recent_rates.mean().item()
            
            # Stability criteria
            stable = (
                rate_std < 0.15 and  # Low variance
                0.01 < rate_mean < 0.5 and  # Reasonable spike rate
                not torch.isnan(recent_rates).any()  # No NaN values
            )
            
            if not stable:
                print(f"Warning: Unstable spike rate - mean: {rate_mean:.3f}, std: {rate_std:.3f}")
                
            return stable
        
        return True
    
    def forward(self, x):
        """
        Forward pass với shared threshold across timesteps
        
        Args:
            x: Input tensor
               - (T, B, C, H, W) for conv layers
               - (T, B, C) for linear layers
        
        Returns:
            spikes: Binary spike tensor with same shape as input
        """
        if x.dim() < 3:
            raise ValueError(f"Input must have at least 3 dims (T, B, C), got {x.dim()}")
        
        T, B = x.shape[:2]
        device = x.device
        
        # Get shared threshold for all timesteps
        shared_threshold = self.threshold  # Shape: (dim,)
        
        # Expand threshold for broadcasting
        if x.dim() == 5:  # Conv case: (T, B, C, H, W)
            threshold_expanded = shared_threshold.view(1, 1, -1, 1, 1)
        elif x.dim() == 3:  # Linear case: (T, B, C)
            threshold_expanded = shared_threshold.view(1, 1, -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        if self.spike_mode == "if_hard":
            # Hard binary spikes
            spikes = (x >= threshold_expanded).float()
            
        elif self.spike_mode == "if_soft":
            # Soft differentiable spikes
            alpha = 10.0  # Sharpness parameter
            diff = x - threshold_expanded
            spikes = torch.sigmoid(alpha * diff)
            
        elif self.spike_mode == "surrogate":
            # Surrogate gradient approach
            spikes = SurrogateSpike.apply(x, threshold_expanded)
            
        else:
            raise ValueError(f"Unknown spike_mode: {self.spike_mode}")
        
        # Monitor stability
        self.monitor_stability(spikes)
        
        return spikes


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient spike function
    Forward: Binary spikes (realistic)
    Backward: Smooth gradients (trainable)
    """
    @staticmethod
    def forward(ctx, v_mem, threshold, alpha=10.0):
        spikes = (v_mem >= threshold).float()
        ctx.save_for_backward(v_mem, threshold)
        ctx.alpha = alpha
        return spikes
    
    @staticmethod
    def backward(ctx, grad_output):
        v_mem, threshold = ctx.saved_tensors
        alpha = ctx.alpha
        
        # Surrogate gradient: derivative of fast sigmoid
        diff = alpha * (v_mem - threshold)
        surrogate_grad = alpha * torch.sigmoid(diff) * (1 - torch.sigmoid(diff))
        
        return grad_output * surrogate_grad, None, None


class LearnableLIFNeuron(LearnableThresholdNeuron):
    """
    LIF neuron với learnable threshold và membrane dynamics
    """
    def __init__(self, dim, tau=2.0, v_reset=0.0, **kwargs):
        super().__init__(dim, **kwargs)
        
        # LIF-specific parameters
        self.tau = tau
        self.v_reset = v_reset
        
    def forward(self, x):
        """
        LIF dynamics với learnable threshold
        """
        T, B = x.shape[:2]
        device = x.device
        
        # Initialize membrane potential
        if self.v_mem is None or self.v_mem.shape != x.shape[1:]:
            self.v_mem = torch.zeros(x.shape[1:], device=device, dtype=x.dtype)
        
        outputs = []
        shared_threshold = self.threshold
        
        # Expand threshold for broadcasting
        if x.dim() == 5:  # Conv case
            threshold_expanded = shared_threshold.view(1, -1, 1, 1)
        else:  # Linear case
            threshold_expanded = shared_threshold.view(1, -1)
        
        for t in range(T):
            # LIF membrane dynamics
            decay = 1.0 - 1.0 / self.tau
            self.v_mem = self.v_mem * decay + x[t]
            
            # Spike generation với shared threshold
            if self.spike_mode == "surrogate":
                spikes = SurrogateSpike.apply(self.v_mem, threshold_expanded)
            else:
                spikes = (self.v_mem >= threshold_expanded).float()
            
            # Reset membrane potential after spike
            self.v_mem = torch.where(spikes > 0.5, self.v_reset, self.v_mem)
            
            outputs.append(spikes)
        
        result = torch.stack(outputs, dim=0)
        
        # Monitor stability
        self.monitor_stability(result)
        
        return result


# Helper functions
def replace_neurons_with_learnable_threshold(model, 
                                           base_threshold=1.0,
                                           learning_range=0.3,
                                           spike_mode="surrogate"):
    """
    Replace existing spiking neurons với learnable threshold versions
    """
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode, MultiStepIFNode, MultiStepParametricLIFNode
    )
    
    replaced_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (MultiStepLIFNode, MultiStepIFNode, MultiStepParametricLIFNode)):
            # Determine dimension from context (tricky - might need manual specification)
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                
            child_name = name.split('.')[-1]
            
            # Try to infer dimension (heuristic - may need adjustment)
            if hasattr(module, 'dim'):
                dim = module.dim
            else:
                # Default fallback - should be specified per model
                dim = 512  # Adjust based on your model
                
            # Create replacement
            if isinstance(module, MultiStepLIFNode):
                replacement = LearnableLIFNeuron(
                    dim=dim,
                    tau=getattr(module, 'tau', 2.0),
                    base_threshold=base_threshold,
                    learning_range=learning_range,
                    spike_mode=spike_mode
                )
            else:
                replacement = LearnableThresholdNeuron(
                    dim=dim,
                    base_threshold=base_threshold,
                    learning_range=learning_range,
                    spike_mode=spike_mode
                )
            
            setattr(parent, child_name, replacement)
            replaced_count += 1
            print(f"Replaced {name}: {type(module).__name__} -> {type(replacement).__name__}")
    
    print(f"Total replaced: {replaced_count} neurons")
    return model


def create_threshold_optimizer(model, base_lr=1e-3, threshold_lr_ratio=0.1):
    """
    Create optimizer với different learning rates for threshold parameters
    """
    threshold_params = []
    regular_params = []
    
    for name, param in model.named_parameters():
        if 'threshold_raw' in name:
            threshold_params.append(param)
            print(f"Threshold parameter: {name}, shape: {param.shape}")
        else:
            regular_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {
            'params': regular_params, 
            'lr': base_lr,
            'weight_decay': 1e-4
        },
        {
            'params': threshold_params, 
            'lr': base_lr * threshold_lr_ratio,  # Smaller LR for thresholds
            'weight_decay': 0.0  # No weight decay for thresholds
        }
    ])
    
    print(f"Regular parameters: {len(regular_params)}")
    print(f"Threshold parameters: {len(threshold_params)}")
    print(f"Regular LR: {base_lr}, Threshold LR: {base_lr * threshold_lr_ratio}")
    
    return optimizer