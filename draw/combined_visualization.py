# combined_analysis_visualizer.py
# Combine results from both Paper 1 methodology and run_analyses.py
# Create 3 comprehensive visualizations

import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from spikingjelly.clock_driven import functional

# Import both analyzers
sys.path.append('.')
from model.spikeformer import sdt

class CombinedAnalysisVisualizer:
    """
    Combine Paper 1 methodology with run_analyses.py approach
    Create comprehensive visualizations
    """
    
    def __init__(self):
        self.paper1_results = {}
        self.run_analysis_results = {}
        self.spike_patterns = {}
    
    def run_paper1_analysis(self, models, data_loader):
        """Run Paper 1 methodology analysis"""
        print("\n" + "="*80)
        print("PAPER 1 METHODOLOGY ANALYSIS")
        print("="*80)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n--- Analyzing {model_name} (Paper 1 Equation 14) ---")
            
            # Get connections
            layer_connections = self._get_output_connections(model)
            print(f"   Found {len(layer_connections)} layers with connections")
            
            # Measure spikes
            spike_stats = self._measure_spike_statistics(model, data_loader, model_name)
            print(f"   Measured spike statistics from {len(spike_stats)} spiking layers")
            
            # Calculate synaptic operations
            T = 4
            total_synaptic_ops = 0
            
            for spike_layer_name in spike_stats:
                base_name = spike_layer_name.replace('_lif', '')
                f_out = None
                
                if base_name in layer_connections:
                    f_out = layer_connections[base_name]
                else:
                    for conn_name in layer_connections:
                        if any(part in conn_name for part in base_name.split('.')):
                            f_out = layer_connections[conn_name]
                            break
                
                if f_out is not None:
                    s_n = spike_stats[spike_layer_name]['avg_spikes_per_neuron']
                    layer_synaptic_ops = T * f_out * s_n
                    total_synaptic_ops += layer_synaptic_ops
            
            total_mops = total_synaptic_ops / 1_000_000
            
            results[model_name] = {
                'synaptic_operations': total_synaptic_ops,
                'energy_mops': total_mops,
                'spike_statistics': spike_stats,
                'avg_spike_rate': np.mean([stats['avg_spikes_per_neuron'] 
                                         for stats in spike_stats.values()]) if spike_stats else 0
            }
            
            print(f"   Total Energy: {total_mops:.2f} MOps, Avg Spike Rate: {results[model_name]['avg_spike_rate']:.4f}")
        
        self.paper1_results = results
        return results
    
    def run_efficiency_analysis(self, models, data_loader):
        """Run efficiency analysis (FPS, Latency) similar to run_analyses.py"""
        print("\n" + "="*80)
        print("EFFICIENCY ANALYSIS (FPS, LATENCY)")
        print("="*80)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n--- Analyzing {model_name} Efficiency ---")
            
            # Measure inference time
            timing_stats = self._measure_inference_time(model, data_loader)
            
            # Collect spike patterns for layer-wise analysis
            spike_data = self._collect_spike_patterns(model, data_loader, model_name)
            
            results[model_name] = {
                'fps': timing_stats['fps'],
                'latency_ms': timing_stats['latency_ms'],
                'inference_time': timing_stats['mean_time'],
                'spike_patterns': spike_data
            }
            
            print(f"   FPS: {timing_stats['fps']:.1f}, Latency: {timing_stats['latency_ms']:.2f}ms")
        
        self.run_analysis_results = results
        return results
    
    def create_combined_performance_chart(self):
        """
        Create Chart 1: Combined Performance Metrics
        (FPS, Latency, Average Spike Rate, Energy) - similar to your reference image
        """
        if not self.paper1_results or not self.run_analysis_results:
            print("Missing analysis results!")
            return
        
        models = list(self.paper1_results.keys())
        
        # Prepare data
        fps_values = [self.run_analysis_results[m]['fps'] for m in models]
        latency_values = [self.run_analysis_results[m]['latency_ms'] for m in models]
        spike_rates = [self.paper1_results[m]['avg_spike_rate'] for m in models]
        energy_mops = [self.paper1_results[m]['energy_mops'] for m in models]
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SNN Spike Mode Performance Comparison\n(LIF vs IF_Hard vs IF_Soft)', 
                     fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Chart 1: FPS
        bars1 = axes[0, 0].bar(models, fps_values, color=colors)
        axes[0, 0].set_ylabel('Frames Per Second (FPS)')
        axes[0, 0].set_title('Inference Speed')
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, fps_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.02,
                           f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Latency
        bars2 = axes[0, 1].bar(models, latency_values, color=colors)
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Inference Latency')
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, latency_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(latency_values)*0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: Average Spike Rate
        bars3 = axes[1, 0].bar(models, spike_rates, color=colors)
        axes[1, 0].set_ylabel('Average Spike Rate')
        axes[1, 0].set_title('Spike Activity Level')
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, val in zip(bars3, spike_rates):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spike_rates)*0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 4: Energy (Paper 1)
        bars4 = axes[1, 1].bar(models, energy_mops, color=colors)
        axes[1, 1].set_ylabel('Energy Consumption (MOps)')
        axes[1, 1].set_title('Energy Efficiency (Paper 1 Methodology)')
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, val in zip(bars4, energy_mops):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_mops)*0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('combined_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Chart 1 saved: combined_performance_comparison.png")
    
    def create_layer_wise_activity_chart(self):
        """
        Create Chart 2: Layer-wise Spike Activity Comparison
        """
        if not self.run_analysis_results:
            print("Missing spike pattern data!")
            return
        
        plt.figure(figsize=(14, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (model_name, data) in enumerate(self.run_analysis_results.items()):
            spike_patterns = data['spike_patterns']
            
            if spike_patterns:
                layer_names = list(spike_patterns.keys())
                activities = [np.mean(spikes) if len(spikes) > 0 else 0 
                             for spikes in spike_patterns.values()]
                
                # Create x-axis positions
                x_pos = np.arange(len(layer_names))
                
                plt.plot(x_pos, activities, marker='o', linewidth=2.5, markersize=8,
                        label=model_name, color=colors[idx])
        
        plt.xlabel('Layer Index')
        plt.ylabel('Average Spike Activity')
        plt.title('Layer-wise Spike Activity Comparison\nAcross Different Spike Modes')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set x-axis labels
        if 'layer_names' in locals() and layer_names:
            plt.xticks(range(len(layer_names)), 
                      [name.split('.')[-1][:10] for name in layer_names], 
                      rotation=45)
        
        plt.tight_layout()
        plt.savefig('layer_wise_activity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Chart 2 saved: layer_wise_activity_comparison.png")
    
    def create_firing_rate_distribution_chart(self):
        """
        Create Chart 3: Firing Rate Distribution Comparison
        """
        if not self.run_analysis_results:
            print("Missing spike pattern data!")
            return
        
        num_models = len(self.run_analysis_results)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 6))
        
        if num_models == 1:
            axes = [axes]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, (model_name, data) in enumerate(self.run_analysis_results.items()):
            spike_patterns = data['spike_patterns']
            
            firing_rates = []
            for layer_name, spikes in spike_patterns.items():
                if len(spikes) > 0:
                    layer_firing_rates = np.mean(spikes, axis=0) if len(spikes.shape) > 1 else spikes
                    firing_rates.extend(layer_firing_rates.flatten())
            
            if firing_rates:
                axes[idx].hist(firing_rates, bins=50, alpha=0.7, density=True, 
                              color=colors[idx])
                axes[idx].set_xlabel('Firing Rate')
                axes[idx].set_ylabel('Density')
                axes[idx].set_title(f'{model_name}\nMean: {np.mean(firing_rates):.4f}')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f'Std: {np.std(firing_rates):.4f}\nMax: {np.max(firing_rates):.4f}'
                axes[idx].text(0.7, 0.8, stats_text, transform=axes[idx].transAxes,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center',
                              transform=axes[idx].transAxes, fontsize=14)
        
        plt.suptitle('Firing Rate Distribution Comparison\nAcross Different Spike Modes', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('firing_rate_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Chart 3 saved: firing_rate_distribution_comparison.png")
    
    def save_combined_results(self):
        """Save combined results to CSV"""
        combined_data = []
        
        for model_name in self.paper1_results.keys():
            if model_name in self.run_analysis_results:
                row = {
                    'Model': model_name,
                    'Energy_MOps_Paper1': self.paper1_results[model_name]['energy_mops'],
                    'Avg_Spike_Rate': self.paper1_results[model_name]['avg_spike_rate'],
                    'Synaptic_Operations': self.paper1_results[model_name]['synaptic_operations'],
                    'FPS': self.run_analysis_results[model_name]['fps'],
                    'Latency_ms': self.run_analysis_results[model_name]['latency_ms'],
                    'Inference_Time_s': self.run_analysis_results[model_name]['inference_time']
                }
                combined_data.append(row)
        
        df = pd.DataFrame(combined_data)
        df.to_csv('combined_analysis_results.csv', index=False)
        
        print("\nCombined results saved: combined_analysis_results.csv")
        print("\nSummary Table:")
        print(df.to_string(index=False))
    
    # Helper methods (similar to previous implementations)
    def _get_output_connections(self, model):
        connections = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if hasattr(module, 'out_features'):
                    connections[name] = module.out_features
                elif hasattr(module, 'out_channels'):
                    connections[name] = module.out_channels
                elif hasattr(module, 'embed_dims'):
                    connections[name] = module.embed_dims
                else:
                    weight_shape = module.weight.shape
                    connections[name] = weight_shape[0]
        return connections
    
    def _measure_spike_statistics(self, model, data_loader, model_name):
        spike_counts = defaultdict(list)
        neuron_counts = {}
        
        def spike_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'detach'):
                    spikes = output.detach()
                    spikes = torch.clamp(spikes, min=0)
                    
                    if len(spikes.shape) == 5:
                        batch_spike_sum = spikes.sum(dim=(0, 3, 4)).mean(dim=0)
                        total_spikes = batch_spike_sum.sum().item()
                        total_neurons = spikes.shape[2]
                    elif len(spikes.shape) == 4:
                        if spikes.shape[0] <= 16:
                            batch_spike_sum = spikes.sum(dim=0).mean(dim=0)
                            total_spikes = batch_spike_sum.sum().item()
                            total_neurons = spikes.shape[2]
                        else:
                            batch_spike_sum = spikes.sum(dim=(2, 3)).mean(dim=0)
                            total_spikes = batch_spike_sum.sum().item()
                            total_neurons = spikes.shape[1]
                    else:
                        total_spikes = spikes.sum().item() / max(spikes.shape[0], 1)
                        total_neurons = spikes.numel() // max(spikes.shape[0], 1)
                    
                    total_spikes = max(0, total_spikes)
                    spike_counts[name].append(total_spikes)
                    neuron_counts[name] = total_neurons
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['_lif']) and 'surrogate' not in name:
                hook = module.register_forward_hook(spike_hook(name))
                hooks.append(hook)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 50:  # Limit for faster processing
                    break
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    _ = model(data)
                except:
                    functional.reset_net(model)
                    continue
        
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
        
        spike_statistics = {}
        for name in spike_counts:
            if spike_counts[name] and name in neuron_counts:
                avg_total_spikes = np.mean(spike_counts[name])
                total_neurons = neuron_counts[name]
                avg_spikes_per_neuron = avg_total_spikes / total_neurons if total_neurons > 0 else 0
                spike_statistics[name] = {'avg_spikes_per_neuron': avg_spikes_per_neuron}
        
        return spike_statistics
    
    def _measure_inference_time(self, model, data_loader):
        model.eval()
        times = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 20:  # Limit for faster processing
                    break
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                        torch.cuda.synchronize()
                    
                    import time
                    start_time = time.time()
                    _ = model(data)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
                except:
                    functional.reset_net(model)
                    continue
        
        if times:
            mean_time = np.mean(times)
            batch_size = 32
            return {
                'mean_time': mean_time,
                'fps': batch_size / mean_time,
                'latency_ms': mean_time * 1000
            }
        return {'mean_time': 0, 'fps': 0, 'latency_ms': 0}
    
    def _collect_spike_patterns(self, model, data_loader, model_name):
        spike_data = {}
        
        def spike_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'detach'):
                    spikes = output.detach().cpu().numpy()
                    if name not in spike_data:
                        spike_data[name] = []
                    spike_data[name].append(spikes.mean())
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['_lif']) and 'surrogate' not in name:
                hook = module.register_forward_hook(spike_hook(name))
                hooks.append(hook)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 30:
                    break
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    _ = model(data)
                except:
                    functional.reset_net(model)
                    continue
        
        for hook in hooks:
            hook.remove()
        functional.reset_net(model)
        
        return spike_data


def load_models():
    """Load the three SNN models"""
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
        try:
            if os.path.exists(info['checkpoint']):
                checkpoint = torch.load(info['checkpoint'], map_location='cpu', weights_only=False)
                model = sdt(**info['config'])
                
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
                print(f"✓ {model_name} loaded successfully")
            else:
                print(f"✗ {model_name} checkpoint not found")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    return models


def setup_data_loader():
    """Setup CIFAR-10 data loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    return test_loader


def main():
    """Main function to create combined analysis"""
    
    print("COMBINED SNN ANALYSIS: PAPER 1 + EFFICIENCY METRICS")
    print("=" * 80)
    print("Creating 3 comprehensive visualization charts")
    print("=" * 80)
    
    # Setup
    test_loader = setup_data_loader()
    models = load_models()
    
    if not models:
        print("No models loaded! Check checkpoint paths.")
        return
    
    print(f"\n✓ Successfully loaded {len(models)} models: {list(models.keys())}")
    
    # Run combined analysis
    analyzer = CombinedAnalysisVisualizer()
    
    # Run both analyses
    analyzer.run_paper1_analysis(models, test_loader)
    analyzer.run_efficiency_analysis(models, test_loader)
    
    # Create 3 comprehensive charts
    print(f"\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    
    analyzer.create_combined_performance_chart()  # Chart 1: FPS, Latency, Spike Rate, Energy
    analyzer.create_layer_wise_activity_chart()   # Chart 2: Layer-wise activity
    analyzer.create_firing_rate_distribution_chart()  # Chart 3: Firing rate distributions
    
    # Save combined results
    analyzer.save_combined_results()
    
    print(f"\n" + "="*80)
    print("COMBINED ANALYSIS COMPLETE!")
    print("="*80)
    print("Generated Files:")
    print("📊 Chart 1: combined_performance_comparison.png")
    print("📈 Chart 2: layer_wise_activity_comparison.png")
    print("📉 Chart 3: firing_rate_distribution_comparison.png")
    print("📋 Data: combined_analysis_results.csv")
    print("\n✓ All visualizations saved successfully!")


if __name__ == "__main__":
    main()