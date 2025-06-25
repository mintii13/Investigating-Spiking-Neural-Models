import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import BaseNode


class MultiStepLearnableIFNode(BaseNode):
    def __init__(self, 
                 init_threshold=1.0,
                 v_reset=0.0,
                 detach_reset=True,
                 backend='torch',
                 store_v_seq=False):
        """
        Learnable Threshold IF Neuron
        
        Args:
            init_threshold (float): Initial threshold value
            v_reset (float): Reset voltage (0.0 for hard reset, None for soft reset)
            detach_reset (bool): Whether to detach reset
            step_mode (str): 'm' for multi-step
            backend (str): Backend to use
            store_v_seq (bool): Whether to store voltage sequence
        """
        super().__init__(v_threshold=init_threshold, 
                        v_reset=v_reset,
                        detach_reset=detach_reset,
                        backend=backend,
                        store_v_seq=store_v_seq)
        
        # Learnable threshold parameter
        # Using sigmoid to ensure threshold is positive
        self.threshold_param = nn.Parameter(torch.tensor(self.logit_transform(init_threshold)))
        
    def logit_transform(self, x):
        """Transform threshold to logit space for stable learning"""
        # Avoid log(0) by clamping
        x = torch.clamp(x, min=1e-7, max=1-1e-7)
        return torch.log(x / (1 - x))
    
    def get_threshold(self):
        """Get current threshold value using sigmoid"""
        return torch.sigmoid(self.threshold_param)
    
    def neuronal_charge(self, x):
        """Charge the neuron"""
        if self.v_reset is None:
            # Soft reset
            self.v = self.v + x
        else:
            # Hard reset
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
    
    def multi_step_forward(self, x_seq):
        """
        Multi-step forward pass
        
        Args:
            x_seq: Input sequence with shape [T, N, *] where T is time steps
            
        Returns:
            spike_seq: Output spike sequence with shape [T, N, *]
        """
        T = x_seq.shape[0]
        y_seq = []
        
        for t in range(T):
            self.neuronal_charge(x_seq[t])
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            y_seq.append(spike)
            
            if self.store_v_seq:
                if not hasattr(self, 'v_seq'):
                    self.v_seq = []
                self.v_seq.append(self.v.clone())
        
        return torch.stack(y_seq)
    
    def single_step_forward(self, x):
        """Single step forward pass"""
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def forward(self, x):
        """Forward pass"""
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return self.multi_step_forward(x)
        else:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")
    
    def reset(self):
        """Reset neuron state"""
        super().reset()
        if hasattr(self, 'v_seq'):
            delattr(self, 'v_seq')
    
    def extra_repr(self):
        """Extra representation for printing"""
        with torch.no_grad():
            current_threshold = self.get_threshold().item()
        return f'threshold={current_threshold:.4f}(learnable), v_reset={self.v_reset}, detach_reset={self.detach_reset}'