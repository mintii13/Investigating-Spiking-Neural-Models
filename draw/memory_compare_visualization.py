# single_memory_chart.py - Single Memory Breakdown Visualization
# Creates only one chart showing forward memory breakdown like EfficientLIF-Net paper

import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from collections import defaultdict
from spikingjelly.clock_driven import functional
import pandas as pd
import matplotlib.pyplot as plt

# Import your model architecture
sys.path.append('.')
from model.spikeformer import sdt

class SimpleMemoryAnalyzer:
    """
    Simple memory analyzer that creates one chart showing memory breakdown
    """
    
    def __init__(self):
        self.memory_results = {}
        
    def analyze_model_memory(self, model, data_loader, model_name):
        """Analyze memory components for one model"""
        print(f"Analyzing {model_name}...")
        
        # 1. Weight Memory (32-bit parameters)
        weight_memory = self._calculate_weight_memory(model)
        
        # 2. LIF Memory (depends on spike mode and activity)
        lif_memory = self._calculate_lif_memory(model, data_loader, model_name)
        
        # 3. Activation Memory (1-bit spikes)
        activation_memory = self._calculate_activation_memory(model, data_loader, model_name)
        
        return {
            'weight_memory': weight_memory,
            'lif_memory': lif_memory, 
            'activation_memory': activation_memory
        }
    
    def _calculate_weight_memory(self, model):
        """Calculate weight parameter memory (32-bit)"""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return (total_params * 4) / (1024 ** 2)  # Convert to MB
    
    def _calculate_lif_memory(self, model, data_loader, model_name):
        """Calculate LIF neuron memory based on spike mode and activity"""
        lif_neuron_counts = {}
        spike_activities = {}
        
        def lif_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'numel'):
                    batch_size = output.shape[0] if len(output.shape) > 0 else 1
                    neurons_per_sample = output.numel() // batch_size
                    lif_neuron_counts[name] = neurons_per_sample
                    
                    # Calculate spike activity
                    if hasattr(output, 'detach'):
                        spikes = output.detach()
                        spike_rate = spikes.mean().item()
                        spike_activities[name] = spike_rate
            return hook
        
        # Register hooks for LIF neurons
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['lif', 'if']) and 'head' not in name.lower():
                hook = module.register_forward_hook(lif_hook(name))
                hooks.append(hook)
        
        # Run inference to collect data
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= 5:  # Sample a few batches
                    break
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    _ = model(data)
                except Exception:
                    functional.reset_net(model)
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
        
        if not lif_neuron_counts:
            return 0
        
        # Calculate LIF memory based on spike mode
        total_lif_neurons = sum(lif_neuron_counts.values())
        avg_spike_rate = np.mean(list(spike_activities.values())) if spike_activities else 0.1
        
        # Different memory requirements for different spike modes
        if 'lif' in model_name.lower():
            base_memory_per_neuron = 8  # LIF: membrane + leak state (32-bit each)
            activity_factor = 1.0 + avg_spike_rate * 0.5  # LIF affected more by activity
        elif 'hard' in model_name.lower():  
            base_memory_per_neuron = 4  # IF_Hard: just membrane potential
            activity_factor = 1.0 + avg_spike_rate * 0.3  # Less affected by activity
        elif 'soft' in model_name.lower():
            base_memory_per_neuron = 6  # IF_Soft: membrane + threshold tracking
            activity_factor = 1.0 + avg_spike_rate * 0.4  # Intermediate
        else:
            base_memory_per_neuron = 4
            activity_factor = 1.0
        
        # Total LIF memory in MB
        lif_memory_mb = (total_lif_neurons * base_memory_per_neuron * activity_factor) / (1024 ** 2)
        return lif_memory_mb
    
    def _calculate_activation_memory(self, model, data_loader, model_name):
        """Calculate activation memory (1-bit spikes)"""
        total_activations = 0
        activation_count = 0
        
        def activation_hook(name):
            def hook(module, input, output):
                nonlocal total_activations, activation_count
                if hasattr(output, 'numel'):
                    # Count actual spikes (sparse representation)
                    if hasattr(output, 'detach'):
                        spikes = output.detach()
                        active_spikes = torch.count_nonzero(spikes).item()
                        total_activations += active_spikes
                        activation_count += 1
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['conv', 'linear', 'fc']):
                hook = module.register_forward_hook(activation_hook(name))
                hooks.append(hook)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= 5:
                    break
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    _ = model(data)
                except Exception:
                    functional.reset_net(model)
                    continue
        
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
        
        # Convert to MB (1-bit per activation, but stored efficiently)
        if activation_count > 0:
            avg_activations = total_activations / activation_count
            activation_memory_mb = (avg_activations / 8) / (1024 ** 2)  # 1-bit = 1/8 byte
        else:
            activation_memory_mb = 0
            
        return activation_memory_mb
    
    def create_single_memory_chart(self, models, data_loader):
        """Create single memory breakdown chart like EfficientLIF-Net paper Figure 7B"""
        print("Creating memory breakdown analysis...")
        
        # Analyze all models
        results = {}
        for model_name, model in models.items():
            try:
                result = self.analyze_model_memory(model, data_loader, model_name)
                results[model_name] = result
                print(f"  {model_name}: Weight={result['weight_memory']:.1f}MB, "
                      f"LIF={result['lif_memory']:.1f}MB, "
                      f"Activation={result['activation_memory']:.1f}MB")
            except Exception as e:
                print(f"  Error analyzing {model_name}: {e}")
                continue
        
        if not results:
            print("No results to visualize!")
            return
        
        # Prepare data for plotting
        models_list = list(results.keys())
        weight_memory = [results[m]['weight_memory'] for m in models_list]
        lif_memory = [results[m]['lif_memory'] for m in models_list]  
        activation_memory = [results[m]['activation_memory'] for m in models_list]
        
        # Create the single chart
        plt.figure(figsize=(10, 6))
        
        # Set up the bar chart
        x = np.arange(len(models_list))
        width = 0.6
        
        # Colors matching the reference image
        colors = ['#2E8B57', '#4169E1', '#DA70D6']  # Green, Blue, Purple
        
        # Create stacked bars
        bars1 = plt.bar(x, weight_memory, width, label='Weight', color=colors[0], alpha=0.8)
        bars2 = plt.bar(x, lif_memory, width, bottom=weight_memory, label='LIF', color=colors[1], alpha=0.8)
        bars3 = plt.bar(x, activation_memory, width, 
                       bottom=[w+l for w,l in zip(weight_memory, lif_memory)], 
                       label='Activation', color=colors[2], alpha=0.8)
        
        # Customize the chart
        plt.ylabel('FW Memory (MB)', fontsize=12)
        plt.xlabel('Methods', fontsize=12)
        plt.xticks(x, models_list, fontsize=11)
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (w, l, a) in enumerate(zip(weight_memory, lif_memory, activation_memory)):
            total = w + l + a
            # Weight label
            if w > total * 0.05:  # Only show if significant
                plt.text(i, w/2, f'{w:.1f}', ha='center', va='center', 
                        fontweight='bold', color='white', fontsize=10)
            # LIF label  
            if l > total * 0.05:
                plt.text(i, w + l/2, f'{l:.1f}', ha='center', va='center',
                        fontweight='bold', color='white', fontsize=10)
            # Activation label
            if a > total * 0.05:
                plt.text(i, w + l + a/2, f'{a:.1f}', ha='center', va='center',
                        fontweight='bold', color='white', fontsize=10)
            
            # Total label on top
            plt.text(i, total + max(weight_memory + lif_memory + activation_memory) * 0.01,
                    f'{total:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Set y-axis limit with some padding
        max_total = max(w+l+a for w,l,a in zip(weight_memory, lif_memory, activation_memory))
        plt.ylim(0, max_total * 1.15)
        
        plt.tight_layout()
        plt.savefig('memory_breakdown_single.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nMemory breakdown chart saved as 'memory_breakdown_single.png'")
        
        # Print summary
        print(f"\nMemory Breakdown Summary:")
        print("-" * 50)
        for model_name in models_list:
            r = results[model_name]
            total = r['weight_memory'] + r['lif_memory'] + r['activation_memory']
            weight_pct = (r['weight_memory'] / total) * 100
            lif_pct = (r['lif_memory'] / total) * 100  
            activation_pct = (r['activation_memory'] / total) * 100
            
            print(f"{model_name}:")
            print(f"  Weight: {r['weight_memory']:.1f}MB ({weight_pct:.1f}%)")
            print(f"  LIF: {r['lif_memory']:.1f}MB ({lif_pct:.1f}%)")
            print(f"  Activation: {r['activation_memory']:.1f}MB ({activation_pct:.1f}%)")
            print(f"  Total: {total:.1f}MB")
            print()
        
        return results


def load_your_models():
    """Load models for comparison"""
    models = {}
    
    model_configs = {
        'LIF': {  # This represents the standard SNN
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',
            'config': {
                'img_size_h': 32, 'img_size_w': 32, 'patch_size': 4, 'in_channels': 3,
                'num_classes': 10, 'embed_dims': 256, 'num_heads': 8, 'mlp_ratios': 4,
                'depths': 2, 'T': 4, 'spike_mode': 'lif', 'rpe_mode': 'conv'
            }
        },
        'IF_Hard': {  # Layer sharing
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',
            'config': {
                'img_size_h': 32, 'img_size_w': 32, 'patch_size': 4, 'in_channels': 3,
                'num_classes': 10, 'embed_dims': 256, 'num_heads': 8, 'mlp_ratios': 4,
                'depths': 2, 'T': 4, 'spike_mode': 'if', 'rpe_mode': 'conv'
            }
        },
        'IF_Soft': {  # Channel sharing #2
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\IF_soft\b64_Uth1_2\model_best.pth.tar',
            'config': {
                'img_size_h': 32, 'img_size_w': 32, 'patch_size': 4, 'in_channels': 3,
                'num_classes': 10, 'embed_dims': 256, 'num_heads': 8, 'mlp_ratios': 4,
                'depths': 2, 'T': 4, 'spike_mode': 'if_soft', 'rpe_mode': 'conv'
            }
        }
    }
    
    for model_name, info in model_configs.items():
        checkpoint_path = info['checkpoint']
        config = info['config']
        
        try:
            print(f"Loading {model_name}...")
            
            if not os.path.exists(checkpoint_path):
                print(f"   Checkpoint not found: {checkpoint_path}")
                continue
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = sdt(**config)
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model = checkpoint
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            models[model_name] = model
            print(f"   {model_name} loaded successfully")
            
        except Exception as e:
            print(f"   Failed to load {model_name}: {e}")
    
    return models


def setup_data_loader():
    """Setup CIFAR-10 data loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Data loader ready: {len(test_dataset)} test samples")
    return test_loader


def main():
    """Main function to create single memory breakdown chart"""
    
    print("SNN MEMORY BREAKDOWN - SINGLE CHART")
    print("=" * 50)
    
    # Setup
    test_loader = setup_data_loader()
    models = load_your_models()
    
    if not models:
        print("No models loaded!")
        return
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # Create single memory chart
    analyzer = SimpleMemoryAnalyzer()
    results = analyzer.create_single_memory_chart(models, test_loader)
    
    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    results = main()