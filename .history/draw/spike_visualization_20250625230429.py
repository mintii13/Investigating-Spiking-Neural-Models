# spike_visualization.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict
import pandas as pd
import os

class SpikeVisualizer:
    def __init__(self):
        self.spike_patterns = {}
        
    def collect_spike_patterns(self, model, data_loader, num_samples=100):
        """Thu thập spike patterns từ model"""
        spike_data = {}
        sample_count = 0
        
        def spike_hook(name):
            def hook(module, input, output):
                if name not in spike_data:
                    spike_data[name] = []
                
                # Xử lý output của spiking neurons
                if hasattr(output, 'detach'):
                    spikes = output.detach().cpu()
                    
                    # Nếu có time dimension (T, B, C, H, W)
                    if len(spikes.shape) == 5:
                        # Flatten spatial dimensions và average over time
                        spikes_flat = spikes.mean(0).flatten(1)  # (B, C*H*W)
                    elif len(spikes.shape) == 4:
                        # (B, C, H, W)
                        spikes_flat = spikes.flatten(1)  # (B, C*H*W)
                    elif len(spikes.shape) == 3:
                        # (T, B, C) hoặc (B, C, D)
                        if spikes.shape[0] <= 16:  # Likely T dimension
                            spikes_flat = spikes.mean(0)  # (B, C)
                        else:
                            spikes_flat = spikes.flatten(1)  # (B, C*D)
                    else:
                        spikes_flat = spikes
                    
                    spike_data[name].append(spikes_flat)
            return hook
        
        # Register hooks cho spiking layers
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['lif', 'if']):
                hook = module.register_forward_hook(spike_hook(name))
                hooks.append(hook)
                print(f"   Hooked: {name}")
        
        print(f"Collecting spike patterns from {len(hooks)} layers...")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if sample_count >= num_samples:
                    break
                
                if torch.cuda.is_available():
                    data = data.cuda()
                
                # Forward pass
                _ = model(data)
                sample_count += data.size(0)
                
                if batch_idx % 5 == 0:
                    print(f"   Processed {sample_count}/{num_samples} samples")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to numpy arrays
        for name in spike_data:
            if spike_data[name]:
                spike_data[name] = torch.cat(spike_data[name], dim=0).numpy()
                print(f"   {name}: {spike_data[name].shape}")
            
        return spike_data
    
    def plot_firing_rate_distribution(self, spike_data_dict):
        """So sánh phân phối firing rate giữa các models"""
        num_models = len(spike_data_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
        
        if num_models == 1:
            axes = [axes]
            
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (model_name, spike_data) in enumerate(spike_data_dict.items()):
            firing_rates = []
            
            for layer_name, spikes in spike_data.items():
                if len(spikes) > 0:
                    layer_firing_rates = spikes.mean(axis=0)  # Average over samples
                    firing_rates.extend(layer_firing_rates.flatten())
            
            if firing_rates:
                axes[idx].hist(firing_rates, bins=50, alpha=0.7, density=True, 
                              color=colors[idx % len(colors)])
                axes[idx].set_xlabel('Firing Rate')
                axes[idx].set_ylabel('Density')
                axes[idx].set_title(f'{model_name}\nMean: {np.mean(firing_rates):.4f}')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics
                stats_text = f'Std: {np.std(firing_rates):.4f}\nMax: {np.max(firing_rates):.4f}'
                axes[idx].text(0.7, 0.8, stats_text, transform=axes[idx].transAxes,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_layer_wise_activity(self, spike_data_dict):
        """Visualize activity across layers"""
        plt.figure(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (model_name, spike_data) in enumerate(spike_data_dict.items()):
            layer_names = []
            avg_activities = []
            
            for layer_name, spikes in spike_data.items():
                if len(spikes) > 0:
                    # Extract layer type from name
                    layer_type = layer_name.split('.')[-1]
                    layer_names.append(layer_type)
                    avg_activities.append(spikes.mean())
            
            if avg_activities:
                plt.plot(range(len(layer_names)), avg_activities, 
                        marker='o', linewidth=2, markersize=8,
                        label=model_name, color=colors[idx % len(colors)])
        
        plt.xlabel('Layer Index')
        plt.ylabel('Average Spike Activity')
        plt.title('Layer-wise Spike Activity Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if layer_names:
            plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_spike_sparsity_analysis(self, spike_data_dict):
        """Analyze spike sparsity"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sparsity comparison
        models = list(spike_data_dict.keys())
        sparsities = []
        non_zero_rates = []
        
        for model_name, spike_data in spike_data_dict.items():
            all_spikes = []
            for layer_name, spikes in spike_data.items():
                if len(spikes) > 0:
                    all_spikes.append(spikes.flatten())
            
            if all_spikes:
                all_spikes = np.concatenate(all_spikes)
                sparsity = (all_spikes == 0).mean()
                non_zero_rate = (all_spikes > 0).mean()
                
                sparsities.append(sparsity)
                non_zero_rates.append(non_zero_rate)
            else:
                sparsities.append(0)
                non_zero_rates.append(0)
        
        # Sparsity bar plot
        bars1 = ax1.bar(models, sparsities, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_ylabel('Sparsity (Fraction of Zeros)')
        ax1.set_title('Spike Sparsity Comparison')
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, sparsities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Non-zero rates
        bars2 = ax2.bar(models, non_zero_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_ylabel('Non-zero Rate')
        ax2.set_title('Active Spike Rate')
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, non_zero_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_summary_statistics(self, spike_data_dict):
        """Tạo summary statistics table"""
        summary_data = []
        
        for model_name, spike_data in spike_data_dict.items():
            all_spikes = []
            layer_count = 0
            
            for layer_name, spikes in spike_data.items():
                if len(spikes) > 0:
                    all_spikes.append(spikes.flatten())
                    layer_count += 1
            
            if all_spikes:
                all_spikes = np.concatenate(all_spikes)
                
                stats = {
                    'Model': model_name,
                    'Num_Layers': layer_count,
                    'Total_Neurons': len(all_spikes),
                    'Mean_Activity': f"{all_spikes.mean():.4f}",
                    'Std_Activity': f"{all_spikes.std():.4f}",
                    'Sparsity': f"{(all_spikes == 0).mean():.4f}",
                    'Max_Activity': f"{all_spikes.max():.4f}",
                    'Min_Activity': f"{all_spikes.min():.4f}"
                }
                summary_data.append(stats)
        
        return pd.DataFrame(summary_data)

def run_spike_visualization():
    """Main function để chạy spike visualization"""
    
    # 1. Setup data loader
    print("Setting up data loader...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                   download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 2. Load your trained models
    print("Loading trained models...")
    
    model_paths = {
        'LIF': 'path/to/your/lif_model.pth',
        'IF_Hard': 'path/to/your/if_hard_model.pth', 
        'IF_Soft': 'path/to/your/if_soft_model.pth'
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
            # Load model - thay đổi theo cách bạn save model
            model = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            if torch.cuda.is_available():
                model = model.cuda()
            models[name] = model
            print(f"   {name} model loaded")
        except Exception as e:
            print(f"   Failed to load {name}: {e}")
    
    if not models:
        print("No models loaded! Please check your model paths.")
        return
    
    # 3. Collect spike patterns
    visualizer = SpikeVisualizer()
    all_spike_data = {}
    
    print("\nCollecting spike patterns...")
    for model_name, model in models.items():
        print(f"\n--- Processing {model_name} ---")
        spike_data = visualizer.collect_spike_patterns(model, test_loader, num_samples=50)
        if spike_data:
            all_spike_data[model_name] = spike_data
            print(f"   Collected data from {len(spike_data)} layers")
        else:
            print(f"   No spike data collected for {model_name}")
    
    if not all_spike_data:
        print("No spike data collected! Check your model architecture.")
        return
    
    # 4. Create visualizations
    print("\nCreating visualizations...")
    
    # Firing rate distributions
    print("   Creating firing rate distributions...")
    fig1 = visualizer.plot_firing_rate_distribution(all_spike_data)
    fig1.savefig('firing_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Layer-wise activity
    print("   Creating layer-wise activity plot...")
    fig2 = visualizer.plot_layer_wise_activity(all_spike_data)
    fig2.savefig('layer_activity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Sparsity analysis
    print("   Creating sparsity analysis...")
    fig3 = visualizer.plot_spike_sparsity_analysis(all_spike_data)
    fig3.savefig('spike_sparsity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 5. Generate summary statistics
    print("   Generating summary statistics...")
    summary_df = visualizer.create_summary_statistics(all_spike_data)
    print("\n" + "="*80)
    print("SPIKE PATTERN ANALYSIS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # 6. Save results
    summary_df.to_csv('spike_analysis_summary.csv', index=False)
    
    print(f"\nVisualization complete!")
    print(f"Files saved:")
    print(f"   - firing_rate_comparison.png")
    print(f"   - layer_activity_comparison.png") 
    print(f"   - spike_sparsity_analysis.png")
    print(f"   - spike_analysis_summary.csv")
    
    return all_spike_data, summary_df

if __name__ == "__main__":
    # Chạy spike visualization
    spike_data, summary = run_spike_visualization()