# snn_energy_analysis.py - SNN Energy Analysis using Synaptic Operations
# Synaptic operations = Σ(t=1 to T) Σ(n=1 to N) f^n_out × s^n

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

class SynapticEnergyAnalyzer:
    """
    SNN Energy Analysis based on synaptic operations
    Synaptic operations = Σ(t=1 to T) Σ(n=1 to N) f^n_out × s^n
    """
    
    def __init__(self):
        self.layer_info = {}
    
    def analyze_synaptic_operations(self, model, data_loader, model_name):
        """
        Calculate synaptic operations for SNN energy analysis
        Formula: Synaptic operations = Σ(t=1 to T) Σ(n=1 to N) f^n_out × s^n
        
        where:
        - T: timesteps
        - f^n_out: number of output connections of layer n
        - s^n: average number of spikes per neuron of layer n
        """
        print(f"\n--- Analyzing {model_name} ---")
        
        # Step 1: Calculate actual output connections for each layer
        layer_connections = self._calculate_actual_output_connections(model)
        print(f"   Found {len(layer_connections)} layers with connections")
        
        # Step 2: Measure spike statistics across full dataset
        spike_stats = self._measure_spike_statistics(model, data_loader, model_name)
        print(f"   Measured spike statistics from {len(spike_stats)} spiking layers")
        
        # Step 3: Apply formula exactly as specified
        T = 4  # Timesteps from model configuration
        total_synaptic_ops = 0
        layer_details = []
        
        # For each timestep and each layer, calculate f^n_out × s^n
        for spike_layer_name in spike_stats:
            # Find corresponding connection layer
            base_name = spike_layer_name.replace('_lif', '')
            
            f_out = None
            connection_layer = None
            
            # Match spiking layer with weight layer
            if base_name in layer_connections:
                f_out = layer_connections[base_name]
                connection_layer = base_name
            else:
                # Try pattern matching for layer correspondence
                for conn_name in layer_connections:
                    if any(part in conn_name for part in base_name.split('.')):
                        f_out = layer_connections[conn_name]
                        connection_layer = conn_name
                        break
            
            if f_out is not None:
                s_n = spike_stats[spike_layer_name]['avg_spikes_per_neuron']
                
                # Apply the summation: T timesteps × f^n_out × s^n
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
                
                print(f"     {spike_layer_name}: f_out={f_out}, s_n={s_n:.4f}, ops={layer_synaptic_ops:.0f}")
            else:
                print(f"     WARNING: No connection found for {spike_layer_name}")
        
        print(f"   Total Synaptic Operations: {total_synaptic_ops:,.0f}")
        
        return {
            'model_name': model_name,
            'total_synaptic_operations': total_synaptic_ops,
            'layer_details': layer_details,
            'spike_statistics': spike_stats,
            'timesteps': T
        }
    
    def _calculate_actual_output_connections(self, model):
        """Calculate actual output connections (f^n_out) for each layer"""
        connections = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight_shape = module.weight.shape
                
                # Calculate total number of synaptic connections from this layer
                if hasattr(module, 'out_features'):  # Linear/Dense layers
                    # f_out = out_features × in_features (total connections)
                    connections[name] = module.out_features * module.in_features
                elif hasattr(module, 'out_channels'):  # Convolutional layers
                    # f_out = out_channels × (in_channels × kernel_height × kernel_width)
                    # This represents all synaptic connections from this conv layer
                    connections[name] = weight_shape[0] * np.prod(weight_shape[1:])
                elif hasattr(module, 'embed_dims'):  # Transformer layers
                    # For transformer layers, calculate based on actual weight dimensions
                    connections[name] = np.prod(weight_shape)
                else:
                    # For other layer types, use total weight parameters as connections
                    connections[name] = np.prod(weight_shape)
        
        return connections
    
    def _measure_spike_statistics(self, model, data_loader, model_name):
        """Measure average spikes per neuron (s^n) for each spiking layer across full dataset"""
        spike_counts = defaultdict(list)
        neuron_counts = {}
        total_batches_processed = 0
        
        def spike_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'detach'):
                    spikes = output.detach()
                    
                    # Debug: Print shape and value range for first few layers
                    if name in ['patch_embed.proj_lif', 'block.0.attn.q_lif']:
                        print(f"   DEBUG {name}: shape={spikes.shape}, range=[{spikes.min():.3f}, {spikes.max():.3f}], unique_vals={len(torch.unique(spikes))}")
                    
                    # Verify spike data quality
                    unique_vals = torch.unique(spikes)
                    if len(unique_vals) > 10:
                        print(f"   WARNING: {name} may not be binary spikes (unique vals: {len(unique_vals)})")
                        return
                    
                    # Ensure non-negative spike values
                    spikes = torch.clamp(spikes, min=0)
                    
                    # Count spikes correctly: s_n should be average spikes per neuron per timestep
                    if len(spikes.shape) == 5:  # (T, B, C, H, W)
                        # Average across time first, then sum over spatial dims, then average over batch
                        time_avg_spikes = spikes.mean(dim=0)  # (B, C, H, W) - average per timestep
                        batch_total_spikes = time_avg_spikes.sum(dim=(1, 2, 3))  # (B,) - total per sample
                        avg_spikes_this_batch = batch_total_spikes.mean().item()
                        total_neurons = spikes.shape[2] * spikes.shape[3] * spikes.shape[4]  # C × H × W
                    elif len(spikes.shape) == 4:  # (B, C, H, W) or (T, B, C)
                        if spikes.shape[0] <= 16:  # Likely T dimension first
                            time_avg_spikes = spikes.mean(dim=0)  # (B, C) - average per timestep
                            batch_total_spikes = time_avg_spikes.sum(dim=1)  # (B,)
                            avg_spikes_this_batch = batch_total_spikes.mean().item()
                            total_neurons = spikes.shape[2]
                        else:  # Batch first (B, C, H, W)
                            batch_total_spikes = spikes.sum(dim=(1, 2, 3))  # (B,)
                            avg_spikes_this_batch = batch_total_spikes.mean().item()
                            total_neurons = spikes.shape[1] * spikes.shape[2] * spikes.shape[3]
                    elif len(spikes.shape) == 3:  # (T, B, C) or (B, T, C)
                        if spikes.shape[0] <= 16:  # T first
                            time_avg_spikes = spikes.mean(dim=0)  # (B, C)
                            batch_total_spikes = time_avg_spikes.sum(dim=1)  # (B,)
                        else:  # B first
                            batch_total_spikes = spikes.sum(dim=(1, 2))  # (B,)
                        avg_spikes_this_batch = batch_total_spikes.mean().item()
                        total_neurons = spikes.shape[-1]
                    else:
                        # Fallback - assume last dimension is batch
                        total_spikes_all = spikes.sum().item()
                        batch_size = spikes.shape[0] if len(spikes.shape) > 0 else 1
                        avg_spikes_this_batch = total_spikes_all / batch_size
                        total_neurons = spikes.numel() // batch_size
                    
                    # Store results
                    spike_counts[name].append(avg_spikes_this_batch)
                    neuron_counts[name] = total_neurons
            return hook
        
        # Register hooks for spiking layers
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['lif', 'if']) and 'head' not in name.lower():
                hook = module.register_forward_hook(spike_hook(name))
                hooks.append(hook)
                print(f"   Registered hook for: {name}")
        
        # Process entire dataset for accurate statistics
        model.eval()
        print(f"   Processing full dataset for {model_name}...")
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    _ = model(data)
                    total_batches_processed += 1
                    
                    # Progress indicator
                    if batch_idx % 50 == 0:
                        print(f"     Processed {batch_idx + 1} batches...")
                        
                except Exception as e:
                    print(f"     Warning: Batch {batch_idx} failed: {str(e)[:100]}")
                    functional.reset_net(model)
                    continue
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
        
        # Calculate s^n: average spikes per neuron across entire dataset
        spike_statistics = {}
        for name in spike_counts:
            if spike_counts[name] and name in neuron_counts:
                # Average total spikes across all batches
                avg_total_spikes_per_batch = np.mean(spike_counts[name])
                total_neurons = neuron_counts[name]
                
                # s^n = average spikes per neuron per sample
                avg_spikes_per_neuron = avg_total_spikes_per_batch / total_neurons if total_neurons > 0 else 0
                
                spike_statistics[name] = {
                    'avg_spikes_per_neuron': avg_spikes_per_neuron,  # This is s^n
                    'total_neurons': total_neurons,
                    'avg_total_spikes_per_batch': avg_total_spikes_per_batch,
                    'batches_measured': len(spike_counts[name])
                }
        
        print(f"   Collected statistics from {total_batches_processed} batches")
        return spike_statistics
    
    def compare_energy_efficiency(self, models, data_loader):
        """Compare energy efficiency of different SNN models"""
        print("=" * 80)
        print("SNN ENERGY EFFICIENCY COMPARISON")
        print("Synaptic Operations = Σ(t=1 to T) Σ(n=1 to N) f^n_out × s^n")
        print("=" * 80)
        
        results = {}
        
        # Analyze each model
        for model_name, model in models.items():
            try:
                result = self.analyze_synaptic_operations(model, data_loader, model_name)
                results[model_name] = result
            except Exception as e:
                print(f"Error analyzing {model_name}: {e}")
                continue
        
        if not results:
            print("No successful analyses!")
            return None
        
        # Generate comparison summary
        print("\n" + "=" * 80)
        print("ENERGY EFFICIENCY SUMMARY")
        print("=" * 80)
        
        comparison_data = []
        total_samples = len(data_loader.dataset)
        
        for model_name, result in results.items():
            ops_per_sample = result['total_synaptic_operations'] / total_samples
            comparison_data.append({
                'Model': model_name,
                'Total_Synaptic_Operations': result['total_synaptic_operations'],
                'Operations_Per_Sample': f"{ops_per_sample:.1f}",
                'Timesteps': result['timesteps'],
                'Spiking_Layers': len(result['spike_statistics'])
            })
        
        # Find most and least efficient models
        min_ops = min(results.values(), key=lambda x: x['total_synaptic_operations'])
        max_ops = max(results.values(), key=lambda x: x['total_synaptic_operations'])
        
        print(f"Most Energy Efficient: {min_ops['model_name']} ({min_ops['total_synaptic_operations']:,} operations)")
        print(f"Least Energy Efficient: {max_ops['model_name']} ({max_ops['total_synaptic_operations']:,} operations)")
        
        if min_ops['total_synaptic_operations'] > 0:
            efficiency_ratio = max_ops['total_synaptic_operations'] / min_ops['total_synaptic_operations']
            print(f"Energy Efficiency Ratio: {efficiency_ratio:.2f}x")
        
        # Display detailed comparison
        df = pd.DataFrame(comparison_data)
        print(f"\nDetailed Comparison:")
        print(df.to_string(index=False))
        
        return results
    
    def create_visualization(self, results):
        """Create energy efficiency visualization"""
        if not results:
            return
        
        models = list(results.keys())
        total_samples = 10000  # CIFAR-10 test set size
        
        # Calculate operations per sample
        operations_per_sample = [results[m]['total_synaptic_operations'] / total_samples for m in models]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(models, operations_per_sample, color=colors[:len(models)])
        ax.set_ylabel('Synaptic Operations per Sample')
        ax.set_title('SNN Energy Efficiency Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, operations_per_sample):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(operations_per_sample)*0.01,
                    f'{val:.1f}', ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig('snn_energy_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved as 'snn_energy_efficiency.png'")
    
    def save_results(self, results, filename='snn_energy_analysis.csv'):
        """Save detailed analysis results"""
        if not results:
            return
        
        # Layer-wise details
        all_layer_data = []
        for model_name, result in results.items():
            for layer_detail in result['layer_details']:
                all_layer_data.append(layer_detail)
        
        layer_df = pd.DataFrame(all_layer_data)
        layer_df.to_csv(filename, index=False)
        
        # Summary statistics
        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Total_Synaptic_Operations': result['total_synaptic_operations'],
                'Timesteps': result['timesteps'],
                'Spiking_Layers': len(result['spike_statistics'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = filename.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"\nResults saved:")
        print(f"   - Layer details: {filename}")
        print(f"   - Summary: {summary_filename}")


def load_models():
    """Load SNN models with different spike modes"""
    models = {}
    
    model_configs = {
        'LIF': {
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',
            'config': {
                'img_size_h': 32, 'img_size_w': 32, 'patch_size': 4, 'in_channels': 3,
                'num_classes': 10, 'embed_dims': 256, 'num_heads': 8, 'mlp_ratios': 4,
                'depths': 2, 'T': 4, 'spike_mode': 'lif', 'rpe_mode': 'conv'
            }
        },
        'IF_Hard': {
            'checkpoint': r'D:\FPTU-sourse\Term5\FETC\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',
            'config': {
                'img_size_h': 32, 'img_size_w': 32, 'patch_size': 4, 'in_channels': 3,
                'num_classes': 10, 'embed_dims': 256, 'num_heads': 8, 'mlp_ratios': 4,
                'depths': 2, 'T': 4, 'spike_mode': 'if', 'rpe_mode': 'conv'
            }
        },
        'IF_Soft': {
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
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Create model
            model = sdt(**config)
            
            # Load weights
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model = checkpoint
            
            # Setup for inference
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            models[model_name] = model
            print(f"   {model_name} loaded successfully")
            
        except Exception as e:
            print(f"   Failed to load {model_name}: {e}")
    
    return models


def setup_data_loader():
    """Setup CIFAR-10 test data loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Data loader ready: {len(test_dataset)} test samples")
    return test_loader


def main():
    """Main analysis function"""
    
    print("SNN ENERGY EFFICIENCY ANALYSIS")
    print("=" * 80)
    print("Analyzing energy efficiency using synaptic operations")
    print("=" * 80)
    
    # Setup data and models
    test_loader = setup_data_loader()
    models = load_models()
    
    if not models:
        print("No models loaded! Please check checkpoint paths.")
        return
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # Run energy analysis
    analyzer = SynapticEnergyAnalyzer()
    results = analyzer.compare_energy_efficiency(models, test_loader)
    
    if results:
        # Create visualization
        analyzer.create_visualization(results)
        
        # Save results
        analyzer.save_results(results, 'snn_synaptic_operations.csv')
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("Generated files:")
        print("   - snn_energy_efficiency.png")
        print("   - snn_synaptic_operations.csv")
        print("   - snn_synaptic_operations_summary.csv")
    
    return results


if __name__ == "__main__":
    results = main()