import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import BaseNode


class MultiStepLearnableIFNode(nn.Module):
    def __init__(self, 
                 init_threshold=1.0,
                 v_reset=None,
                 detach_reset=True):
        """
        Learnable Threshold IF Neuron
        
        Args:
            init_threshold (float): Initial threshold value
            v_reset (float): Reset voltage (None for soft reset, 0.0 for hard reset)
            detach_reset (bool): Whether to detach reset
        """
        super().__init__()
        
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        
        # Learnable threshold parameter
        # Using log space for stable learning and ensuring positive values
        self.log_threshold = nn.Parameter(torch.log(torch.tensor(init_threshold)))
        
        # Initialize membrane potential
        self.register_buffer('v', None)
        
    def get_threshold(self):
        """Get current threshold value using exp to ensure positive"""
        return torch.exp(self.log_threshold)
    
    def reset_state(self):
        """Reset neuron state"""
        self.v = None
    
    def neuronal_charge(self, x):
        """Charge the neuron"""
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + x
    
    def neuronal_fire(self):
        """Fire the neuron based on learnable threshold"""
        current_threshold = self.get_threshold()
        return (self.v >= current_threshold).float()
    
    def neuronal_reset(self, spike):
        """Reset the neuron after firing"""
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        if self.v_reset is None:
            # Soft reset: subtract threshold
            current_threshold = self.get_threshold()
            self.v = self.v - spike_d * current_threshold
        else:
            # Hard reset: set to v_reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset
    
    def forward(self, x_seq):
        """
        Forward pass for multi-step input
        
        Args:
            x_seq: Input sequence with shape [T, N, *] where T is time steps
            
        Returns:
            spike_seq: Output spike sequence with shape [T, N, *]
        """
        T = x_seq.shape[0]
        y_seq = []
        
        # Reset state at the beginning of each forward pass
        if not self.training or self.v is None:
            self.v = None
        
        for t in range(T):
            self.neuronal_charge(x_seq[t])
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            y_seq.append(spike)
        
        return torch.stack(y_seq)
    
    def reset(self):
        """Reset neuron state - for compatibility with other neurons"""
        self.v = None
    
    def extra_repr(self):
        """Extra representation for printing"""
        with torch.no_grad():
            current_threshold = self.get_threshold().item()
        return f'threshold={current_threshold:.4f}(learnable), v_reset={self.v_reset}, detach_reset={self.detach_reset}'