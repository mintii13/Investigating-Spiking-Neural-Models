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
from spikingjelly.clock_driven import functional

class SpikeVisualizer:
    def __init__(self):
        self.spike_patterns = {}
        
    def collect_spike_patterns(self, model, data_loader, use_full_dataset=True, num_samples=100):
        """Thu thập spike patterns từ model với option full dataset"""
        spike_data = {}
        sample_count = 0
        successful_batches = 0
        
        def spike_hook(name):
            def hook(module, input, output):
                try:
                    if name not in spike_data:
                        spike_data[name] = []
                    
                    # Xử lý output của spiking neurons
                    if hasattr(output, 'detach'):
                        spikes = output.detach().cpu()
                        
                        # Handle different tensor shapes
                        if len(spikes.shape) == 5:
                            # (T, B, C, H, W) - flatten spatial and average time
                            spikes_flat = spikes.mean(0).flatten(1)  # (B, C*H*W)
                        elif len(spikes.shape) == 4:
                            # (B, C, H, W) - flatten spatial
                            spikes_flat = spikes.flatten(1)  # (B, C*H*W)
                        elif len(spikes.shape) == 3:
                            # (T, B, C) or (B, C, D)
                            if spikes.shape[0] <= 16:  # Likely T dimension
                                spikes_flat = spikes.mean(0)  # (B, C)
                            else:
                                spikes_flat = spikes.flatten(1)  # (B, C*D)
                        else:
                            spikes_flat = spikes
                        
                        spike_data[name].append(spikes_flat)
                except Exception as e:
                    # Silently handle hook errors to prevent crashes
                    pass
            return hook
        
        # Register hooks cho spiking layers
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['lif', 'if']):
                try:
                    hook = module.register_forward_hook(spike_hook(name))
                    hooks.append(hook)
                    print(f"   Hooked: {name}")
                except:
                    continue
        
        dataset_size = "full dataset" if use_full_dataset else f"{num_samples} samples"
        print(f"Collecting spike patterns from {len(hooks)} layers on {dataset_size}...")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                # Break condition based on mode
                if not use_full_dataset and sample_count >= num_samples:
                    break
                
                try:
                    # Reset model state before each batch
                    functional.reset_net(model)
                    
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    # Forward pass
                    _ = model(data)
                    sample_count += data.size(0)
                    successful_batches += 1
                    
                    # Progress reporting
                    if use_full_dataset:
                        if batch_idx % 50 == 0:
                            print(f"   Processed {sample_count} samples ({batch_idx+1} batches)")
                    else:
                        if batch_idx % 5 == 0:
                            print(f"   Processed {sample_count}/{num_samples} samples")
                            
                except Exception as e:
                    print(f"   Warning: Batch {batch_idx} failed: {str(e)[:100]}...")
                    functional.reset_net(model)
                    continue
        
        # Remove hooks
        for hook in hooks:
            try:
                hook.remove()
            except:
                pass
        
        # Final reset
        functional.reset_net(model)
        
        # Convert to numpy arrays
        processed_spike_data = {}
        for name in spike_data:
            if spike_data[name]:
                try:
                    processed_spike_data[name] = torch.cat(spike_data[name], dim=0).numpy()
                    print(f"   {name}: {processed_spike_data[name].shape}")
                except Exception as e:
                    print(f"   Warning: Failed to process {name}: {e}")
                    continue
        
        print(f"Completed processing {sample_count} samples from {successful_batches} batches")
        return processed_spike_data
    
    def plot_firing_rate_distribution(self, spike_data_dict):
        """So sánh phân phối firing rate giữa các models"""
        if not spike_data_dict:
            print("No spike data to plot!")
            return None
            
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
            else:
                axes[idx].text(0.5, 0.5, 'No Data', ha='center', va='center',
                              transform=axes[idx].transAxes, fontsize=14)
        
        plt.tight_layout()
        return fig
    
    def plot_layer_wise_activity(self, spike_data_dict):
        """Visualize activity across layers"""
        if not spike_data_dict:
            print("No spike data to plot!")
            return None
            
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
        
        # Set x-axis labels if available
        if 'layer_names' in locals() and layer_names:
            plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        
        plt.tight_layout()
        return plt.gcf()
    
    def plot_spike_sparsity_analysis(self, spike_data_dict):
        """Analyze spike sparsity"""
        if not spike_data_dict:
            print("No spike data to plot!")
            return None
            
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
        
        # Color mapping for consistent visualization
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)]
        
        # Sparsity bar plot
        bars1 = ax1.bar(models, sparsities, color=colors)
        ax1.set_ylabel('Sparsity (Fraction of Zeros)')
        ax1.set_title('Spike Sparsity Comparison')
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars1, sparsities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Non-zero rates
        bars2 = ax2.bar(models, non_zero_rates, color=colors)
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

def run_spike_visualization(use_full_dataset=True):
    """Main function để chạy spike visualization với full dataset option"""
    
    # 1. Setup data loader
    print("Setting up data loader...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                   download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Dataset info: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"Analysis mode: {'Full dataset' if use_full_dataset else 'Sample-based (100 samples)'}")
    
    # 2. Load your trained models
    print("Loading trained models...")
    
    model_paths = {
        'LIF': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',
        'IF_Hard': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',
        'IF_Soft': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_soft\b64_Uth1_2\model_best.pth.tar'
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
            # Load model with error handling
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    # Create model and load state dict (adjust according to your model creation)
                    # This part needs to match your model architecture
                    # model = YourModelClass(config)
                    # model.load_state_dict(checkpoint['state_dict'], strict=False)
                    print(f"   Warning: {name} - Need to implement proper model loading")
                    continue
                else:
                    model = checkpoint
                
                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                models[name] = model
                print(f"   {name} model loaded")
            else:
                print(f"   {name} checkpoint not found: {path}")
        except Exception as e:
            print(f"   Failed to load {name}: {e}")
    
    if not models:
        print("No models loaded! Please check your model paths and loading logic.")
        print("Update the model loading section with your specific model architecture.")
        return None, None
    
    # 3. Collect spike patterns
    visualizer = SpikeVisualizer()
    all_spike_data = {}
    
    print(f"\nCollecting spike patterns...")
    for model_name, model in models.items():
        print(f"\n--- Processing {model_name} ---")
        try:
            spike_data = visualizer.collect_spike_patterns(
                model, test_loader, 
                use_full_dataset=use_full_dataset, 
                num_samples=100
            )
            if spike_data:
                all_spike_data[model_name] = spike_data
                print(f"   Collected data from {len(spike_data)} layers")
            else:
                print(f"   No spike data collected for {model_name}")
        except Exception as e:
            print(f"   Error processing {model_name}: {e}")
            continue
    
    if not all_spike_data:
        print("No spike data collected! Check your model architecture and spiking layers.")
        return None, None
    
    # 4. Create visualizations
    print("\nCreating visualizations...")
    
    try:
        # Firing rate distributions
        print("   Creating firing rate distributions...")
        fig1 = visualizer.plot_firing_rate_distribution(all_spike_data)
        if fig1:
            fig1.savefig('spike_firing_rates.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # Layer-wise activity
        print("   Creating layer-wise activity plot...")
        fig2 = visualizer.plot_layer_wise_activity(all_spike_data)
        if fig2:
            fig2.savefig('spike_layer_activity.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # Sparsity analysis
        print("   Creating sparsity analysis...")
        fig3 = visualizer.plot_spike_sparsity_analysis(all_spike_data)
        if fig3:
            fig3.savefig('spike_sparsity.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        print("   Visualizations created successfully!")
        
    except Exception as e:
        print(f"   Error creating visualizations: {e}")
    
    # 5. Generate summary statistics
    print("   Generating summary statistics...")
    try:
        summary_df = visualizer.create_summary_statistics(all_spike_data)
        print("\n" + "="*80)
        print("SPIKE PATTERN ANALYSIS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Save results
        summary_df.to_csv('spike_analysis_summary.csv', index=False)
        
    except Exception as e:
        print(f"   Error generating summary: {e}")
        summary_df = pd.DataFrame()
    
    print(f"\nVisualization complete!")
    print(f"Files saved:")
    print(f"   - spike_firing_rates.png")
    print(f"   - spike_layer_activity.png") 
    print(f"   - spike_sparsity.png")
    print(f"   - spike_analysis_summary.csv")
    
    return all_spike_data, summary_df

if __name__ == "__main__":
    # Run spike visualization
    print("Starting Spike Pattern Visualization")
    print("="*50)
    
    # Full dataset analysis (recommended for paper)
    spike_data, summary = run_spike_visualization(use_full_dataset=True)
    
    # Quick analysis option (uncomment for testing)
    # spike_data, summary = run_spike_visualization(use_full_dataset=False)