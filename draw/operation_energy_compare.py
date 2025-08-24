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

class Paper1EnergyAnalyzer:
    """
    Exact implementation following Paper 1 methodology
    Equations (13) and (14) for SNN energy comparison
    """
    
    def __init__(self):
        self.layer_info = {}
    
    def count_snn_synaptic_operations(self, model, data_loader, model_name):
        """
        Paper 1 Equation (14):
        Synaptic operations for SNNs = Σ(t=1 to T) Σ(n=1 to N) f^n_out * s^n
        
        where:
        - T: timesteps (=4 for your models)
        - f^n_out: number of output connections of layer n
        - s^n: average number of spikes per neuron of layer n
        """
        print(f"\n--- Analyzing {model_name} (Paper 1 Equation 14) ---")
        
        # Step 1: Get f^n_out for each layer
        layer_connections = self._get_output_connections(model)
        print(f"   Found {len(layer_connections)} layers with connections")
        
        # Step 2: Measure s^n (average spikes per neuron)
        spike_stats = self._measure_spike_statistics(model, data_loader, model_name)
        print(f"   Measured spike statistics from {len(spike_stats)} spiking layers")
        
        # Step 3: Apply Equation (14)
        T = 4  # Timesteps as defined in your model configs
        total_synaptic_ops = 0
        layer_details = []
        
        # Match spiking layers with connection layers by finding corresponding layers
        for spike_layer_name in spike_stats:
            # Find the corresponding connection layer (remove _lif suffix and match)
            base_name = spike_layer_name.replace('_lif', '')
            
            # Try exact match first
            f_out = None
            connection_layer = None
            
            if base_name in layer_connections:
                f_out = layer_connections[base_name]
                connection_layer = base_name
            else:
                # Try fuzzy matching - find layer with similar structure
                for conn_name in layer_connections:
                    if any(part in conn_name for part in base_name.split('.')):
                        f_out = layer_connections[conn_name]
                        connection_layer = conn_name
                        break
            
            if f_out is not None:
                s_n = spike_stats[spike_layer_name]['avg_spikes_per_neuron']
                
                # Paper 1 Equation (14): T * f^n_out * s^n
                layer_synaptic_ops = T * f_out * s_n
                total_synaptic_ops += layer_synaptic_ops
                
                layer_details.append({
                    'model': model_name,
                    'layer': spike_layer_name,
                    'connection_layer': connection_layer,
                    'f_out': f_out,
                    's_n': s_n,
                    'timesteps': T,
                    'synaptic_ops': layer_synaptic_ops
                })
                
                print(f"     {spike_layer_name} -> {connection_layer}: f_out={f_out}, s_n={s_n:.4f}, ops={layer_synaptic_ops:.0f}")
            else:
                print(f"     WARNING: No connection found for {spike_layer_name}")
        
        # Convert to MOps (Million Operations) as used in Paper 1
        total_mops = total_synaptic_ops / 1_000_000
        
        print(f"   DEBUG - Connection layers: {list(layer_connections.keys())[:5]}...")  # Show first 5
        print(f"   DEBUG - Spike layers: {list(spike_stats.keys())[:5]}...")  # Show first 5
        
        print(f"   Total Synaptic Operations: {total_synaptic_ops:,.0f}")
        print(f"   Total Energy: {total_mops:.2f} MOps")
        
        return {
            'model_name': model_name,
            'total_synaptic_operations': total_synaptic_ops,
            'total_mops': total_mops,
            'layer_details': layer_details,
            'spike_statistics': spike_stats,
            'timesteps': T
        }
    
    def _get_output_connections(self, model):
        """Calculate f^n_out for each layer following Paper 1"""
        connections = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # For SNN: f_out approximated by output features/channels
                if hasattr(module, 'out_features'):  # Linear layers
                    connections[name] = module.out_features
                elif hasattr(module, 'out_channels'):  # Conv layers
                    connections[name] = module.out_channels
                elif hasattr(module, 'embed_dims'):  # Transformer layers
                    connections[name] = module.embed_dims
                else:
                    # Estimate from weight shape
                    weight_shape = module.weight.shape
                    connections[name] = weight_shape[0]  # Output dimension
        
        return connections
    
    def _measure_spike_statistics(self, model, data_loader, model_name):
        """Measure s^n exactly as described in Paper 1"""
        spike_counts = defaultdict(list)
        neuron_counts = {}
        batch_counts = defaultdict(int)
        
        def spike_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'detach'):
                    spikes = output.detach()
                    
                    # DEBUG: Check if spikes are binary (0/1) as expected
                    unique_vals = torch.unique(spikes)
                    if len(unique_vals) > 10:  # Too many unique values, likely not spikes
                        print(f"   WARNING: {name} may not be spikes (unique vals: {len(unique_vals)})")
                        return
                    
                    # Ensure spikes are non-negative (real spikes should be 0 or 1)
                    spikes = torch.clamp(spikes, min=0)
                    
                    # Count spikes following Paper 1 methodology
                    if len(spikes.shape) == 5:  # (T, B, C, H, W)
                        # Sum over time and spatial dimensions, average over batch
                        batch_spike_sum = spikes.sum(dim=(0, 3, 4)).mean(dim=0)  # (C,)
                        total_spikes = batch_spike_sum.sum().item()
                        total_neurons = spikes.shape[2]  # Channels
                    elif len(spikes.shape) == 4:  # (B, C, H, W) or (T, B, C)
                        if spikes.shape[0] <= 16:  # Likely T dimension
                            batch_spike_sum = spikes.sum(dim=0).mean(dim=0)  # (C,)
                            total_spikes = batch_spike_sum.sum().item()
                            total_neurons = spikes.shape[2]
                        else:  # Batch first
                            batch_spike_sum = spikes.sum(dim=(2, 3)).mean(dim=0)  # (C,)
                            total_spikes = batch_spike_sum.sum().item()
                            total_neurons = spikes.shape[1]
                    elif len(spikes.shape) == 3:  # (T, B, C) or (B, T, C)
                        if spikes.shape[0] <= 16:  # T first
                            batch_spike_sum = spikes.sum(dim=0).mean(dim=0)  # (C,)
                        else:  # B first
                            batch_spike_sum = spikes.sum(dim=1).mean(dim=0)  # (C,)
                        total_spikes = batch_spike_sum.sum().item()
                        total_neurons = spikes.shape[-1]
                    else:
                        total_spikes = spikes.sum().item() / max(spikes.shape[0], 1)
                        total_neurons = spikes.numel() // max(spikes.shape[0], 1)
                    
                    # Ensure non-negative spike counts
                    total_spikes = max(0, total_spikes)
                    
                    spike_counts[name].append(total_spikes)
                    neuron_counts[name] = total_neurons
                    batch_counts[name] += 1
            return hook
        
        # Register hooks for all spiking layers
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['lif', 'if']) and 'head' not in name.lower():
                hook = module.register_forward_hook(spike_hook(name))
                hooks.append(hook)
                print(f"   Registered hook for: {name}")
        
        # Run inference to collect spike statistics
        model.eval()
        successful_batches = 0
        
        print(f"   Collecting spike statistics for {model_name}...")
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    _ = model(data)
                    successful_batches += 1
                    
                    # Progress indicator
                    if batch_idx % 20 == 0:
                        print(f"     Processed {batch_idx + 1} batches...")
                    
                    # Limit batches for reasonable measurement time
                    if batch_idx >= 100:  # Process more batches for accurate measurement
                        break
                        
                except Exception as e:
                    print(f"     Warning: Batch {batch_idx} failed: {str(e)[:100]}")
                    functional.reset_net(model)
                    continue
        
        # Remove hooks and final reset
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
        
        # Calculate s^n (average spikes per neuron) following Paper 1
        spike_statistics = {}
        for name in spike_counts:
            if spike_counts[name] and name in neuron_counts:
                avg_total_spikes = np.mean(spike_counts[name])
                total_neurons = neuron_counts[name]
                
                # This is s^n in Equation (14)
                avg_spikes_per_neuron = avg_total_spikes / total_neurons if total_neurons > 0 else 0
                
                spike_statistics[name] = {
                    'avg_spikes_per_neuron': avg_spikes_per_neuron,  # s^n
                    'total_neurons': total_neurons,
                    'avg_total_spikes': avg_total_spikes,
                    'batches_measured': len(spike_counts[name])
                }
        
        print(f"   Collected statistics from {successful_batches} batches")
        return spike_statistics
    
    def compare_spike_modes(self, models, data_loader):
        """Compare energy consumption of different spike modes using Paper 1 methodology"""
        print("=" * 80)
        print("PAPER 1 METHODOLOGY: SNN ENERGY COMPARISON")
        print("Equation (14): Synaptic operations = Σ(t=1 to T) Σ(n=1 to N) f^n_out × s^n")
        print("=" * 80)
        
        results = {}
        
        # Analyze each model
        for model_name, model in models.items():
            try:
                result = self.count_snn_synaptic_operations(model, data_loader, model_name)
                results[model_name] = result
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")
                continue
        
        if not results:
            print("No successful analyses!")
            return None
        
        # Create comparison summary
        print("\n" + "=" * 80)
        print("ENERGY COMPARISON SUMMARY (Paper 1 Format)")
        print("=" * 80)
        
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Synaptic_Operations': f"{result['total_synaptic_operations']:,.0f}",
                'Energy_MOps': f"{result['total_mops']:.2f}",
                'Timesteps': result['timesteps'],
                'Num_Spiking_Layers': len(result['spike_statistics'])
            })
        
        # Find most efficient model
        min_energy = min(results.values(), key=lambda x: x['total_mops'])
        max_energy = max(results.values(), key=lambda x: x['total_mops'])
        
        print(f"Most Energy Efficient: {min_energy['model_name']} ({min_energy['total_mops']:.2f} MOps)")
        print(f"Least Energy Efficient: {max_energy['model_name']} ({max_energy['total_mops']:.2f} MOps)")
        
        if max_energy['total_mops'] > 0:
            efficiency_gain = max_energy['total_mops'] / min_energy['total_mops']
            print(f"Energy Efficiency Gain: {efficiency_gain:.2f}x")
        
        # Print detailed comparison table
        df = pd.DataFrame(comparison_data)
        print(f"\nDetailed Comparison:")
        print(df.to_string(index=False))
        
        return results
    
    def create_energy_visualization(self, results):
        """Create visualizations following Paper 1 style"""
        if not results:
            return
        
        models = list(results.keys())
        energies = [results[m]['total_mops'] for m in models]
        operations = [results[m]['total_synaptic_operations'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Energy comparison (MOps)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars1 = ax1.bar(models, energies, color=colors[:len(models)])
        ax1.set_ylabel('Energy Consumption (MOps)')
        ax1.set_title('SNN Energy Comparison\n(Paper 1 Equation 14)')
        ax1.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars1, energies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energies)*0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Operations comparison
        bars2 = ax2.bar(models, operations, color=colors[:len(models)])
        ax2.set_ylabel('Synaptic Operations')
        ax2.set_title('Total Synaptic Operations\n(Σ f_out × s_n × T)')
        ax2.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, val in zip(bars2, operations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(operations)*0.01,
                    f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('paper1_snn_energy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved as 'paper1_snn_energy_comparison.png'")
    
    def save_detailed_results(self, results, filename='paper1_snn_energy_detailed.csv'):
        """Save detailed results in Paper 1 format"""
        if not results:
            return
        
        # Layer-wise details
        all_layer_data = []
        for model_name, result in results.items():
            for layer_detail in result['layer_details']:
                all_layer_data.append(layer_detail)
        
        layer_df = pd.DataFrame(all_layer_data)
        layer_df.to_csv(filename, index=False)
        
        # Summary data
        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Total_Synaptic_Operations': result['total_synaptic_operations'],
                'Energy_MOps': result['total_mops'],
                'Timesteps': result['timesteps'],
                'Num_Spiking_Layers': len(result['spike_statistics']),
                'Method': 'Paper1_Equation14'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = filename.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"\nDetailed results saved:")
        print(f"   - Layer details: {filename}")
        print(f"   - Summary: {summary_filename}")


def load_your_models():
    """Load the three SNN models with different spike modes"""
    models = {}
    
    model_configs = {
        'LIF': {
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',
            'config': {
                'img_size_h': 32,
                'img_size_w': 32,
                'patch_size': 4,
                'in_channels': 3,
                'num_classes': 10,
                'embed_dims': 256,
                'num_heads': 8,
                'mlp_ratios': 4,
                'depths': 2,
                'T': 4,
                'spike_mode': 'lif',
                'rpe_mode': 'conv'
            }
        },
        'IF_Hard': {
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',
            'config': {
                'img_size_h': 32,
                'img_size_w': 32,
                'patch_size': 4,
                'in_channels': 3,
                'num_classes': 10,
                'embed_dims': 256,
                'num_heads': 8,
                'mlp_ratios': 4,
                'depths': 2,
                'T': 4,
                'spike_mode': 'if',
                'rpe_mode': 'conv'
            }
        },
        'IF_Soft': {
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\IF_soft\b64_Uth1_2\model_best.pth.tar',
            'config': {
                'img_size_h': 32,
                'img_size_w': 32,
                'patch_size': 4,
                'in_channels': 3,
                'num_classes': 10,
                'embed_dims': 256,
                'num_heads': 8,
                'mlp_ratios': 4,
                'depths': 2,
                'T': 4,
                'spike_mode': 'if_soft',
                'rpe_mode': 'conv'
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
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Create model architecture
            model = sdt(**config)
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model = checkpoint
            
            # Move to GPU if available
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
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Data loader ready: {len(test_dataset)} test samples")
    return test_loader


def main():
    """Main function following Paper 1 methodology"""
    
    print("PAPER 1 ENERGY ANALYSIS: SNN SPIKE MODE COMPARISON")
    print("=" * 80)
    print("Implementing Equation (14) from 'A Little Energy Goes a Long Way'")
    print("Comparing LIF vs IF_Hard vs IF_Soft spike modes")
    print("=" * 80)
    
    # Setup
    test_loader = setup_data_loader()
    models = load_your_models()
    
    if not models:
        print("No models loaded! Check checkpoint paths.")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models: {list(models.keys())}")
    
    # Run Paper 1 energy analysis
    analyzer = Paper1EnergyAnalyzer()
    results = analyzer.compare_spike_modes(models, test_loader)
    
    if results:
        # Create visualization
        analyzer.create_energy_visualization(results)
        
        # Save detailed results
        analyzer.save_detailed_results(results, 'paper1_spike_mode_comparison.csv')
        
        print("\n" + "=" * 80)
        print("PAPER 1 ANALYSIS COMPLETE!")
        print("=" * 80)
        print("Files generated:")
        print("   - paper1_snn_energy_comparison.png")
        print("   - paper1_spike_mode_comparison.csv")
        print("   - paper1_spike_mode_comparison_summary.csv")
        print("\nResults based on exact Paper 1 methodology (Equation 14)")
    
    return results


if __name__ == "__main__":
    results = main()