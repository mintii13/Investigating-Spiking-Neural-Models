# energy_analysis.py - Fixed version
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from spikingjelly.clock_driven import functional

class QuickEnergyAnalyzer:
    def __init__(self):
        self.spike_counts = defaultdict(list)
        self.operation_counts = defaultdict(int)
        
    def count_spikes(self, model, data_loader):
        """Đếm số spikes trong tất cả test data với proper state reset"""
        spike_stats = {}
        
        def spike_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'detach'):
                    spikes = output.detach()
                    spike_rate = spikes.mean().item()
                    spike_stats[name] = spike_rate
            return hook
        
        # Register hooks cho các spiking neurons
        hooks = []
        for name, module in model.named_modules():
            if any(x in name.lower() for x in ['lif', 'if']):
                hook = module.register_forward_hook(spike_hook(name))
                hooks.append(hook)
        
        model.eval()
        total_spike_rates = defaultdict(list)
        total_batches = 0
        
        print(f"   Processing all test batches...")
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                try:
                    # CRITICAL: Reset neuron states before each batch
                    functional.reset_net(model)
                    
                    if torch.cuda.is_available():
                        data = data.cuda()
                    
                    # Forward pass
                    _ = model(data)
                    
                    # Thu thập spike rates cho batch này
                    for name, rate in spike_stats.items():
                        total_spike_rates[name].append(rate)
                    spike_stats.clear()
                    total_batches += 1
                    
                    # Progress indicator every 50 batches
                    if batch_idx % 50 == 0:
                        print(f"     Processed {batch_idx + 1} batches...")
                        
                except Exception as e:
                    print(f"     Warning: Batch {batch_idx} failed: {str(e)[:100]}...")
                    # Reset model state and continue
                    functional.reset_net(model)
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Final reset
        functional.reset_net(model)
        
        # Tính average spike rates across all successful batches
        avg_spike_rates = {}
        for name, rates in total_spike_rates.items():
            if rates:  # Check if we have valid data
                avg_spike_rates[name] = np.mean(rates)
        
        print(f"   Completed processing {total_batches} batches")
        return avg_spike_rates
    
    def estimate_energy(self, spike_rates, model):
        """Ước tính energy consumption"""
        total_energy = 0
        total_params = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                param_count = module.weight.numel()
                total_params += param_count
                
                # Tìm spike rate tương ứng
                spike_rate = 0
                for spike_name in spike_rates:
                    if any(x in spike_name for x in name.split('.')):
                        spike_rate = spike_rates[spike_name]
                        break
                
                # Energy model: spike operations cost more
                if 'conv' in name.lower():
                    base_ops = param_count * 2  # MAC operations
                    spike_factor = 1 + spike_rate * 0.5  # Spike overhead
                elif 'linear' in name.lower() or 'fc' in name.lower():
                    base_ops = param_count
                    spike_factor = 1 + spike_rate * 0.3
                else:
                    base_ops = param_count * 0.1
                    spike_factor = 1 + spike_rate * 0.1
                
                layer_energy = base_ops * spike_factor
                total_energy += layer_energy
        
        return {
            'total_energy': total_energy,
            'total_params': total_params,
            'energy_per_param': total_energy / max(total_params, 1)
        }
    
    def measure_inference_time(self, model, data_loader):
        """Đo thời gian inference trên TOÀN BỘ dataset để công bằng"""
        model.eval()
        times = []
        
        # Warm up với state reset
        print(f"   Warming up...")
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 3:
                    break
                try:
                    functional.reset_net(model)
                    if torch.cuda.is_available():
                        data = data.cuda()
                    _ = model(data)
                except:
                    continue
        
        # Actual timing trên TOÀN BỘ dataset
        print(f"   Measuring inference time on FULL dataset ({len(data_loader)} batches)...")
        
        with torch.no_grad():
            successful_timings = 0
            for batch_idx, (data, targets) in enumerate(data_loader):
                try:
                    # Reset before timing
                    functional.reset_net(model)
                    
                    if torch.cuda.is_available():
                        data = data.cuda()
                        torch.cuda.synchronize()
                        
                    start_time = time.time()
                    _ = model(data)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                    end_time = time.time()
                    times.append(end_time - start_time)
                    successful_timings += 1
                    
                    # Progress indicator every 50 batches
                    if batch_idx % 50 == 0:
                        print(f"     Timed {batch_idx + 1}/{len(data_loader)} batches...")
                        
                except Exception as e:
                    print(f"     Warning: Timing failed on batch {batch_idx}")
                    functional.reset_net(model)
                    continue
        
        if not times:
            print("     Warning: No successful timing measurements!")
            return {
                'mean_time': 0, 'std_time': 0, 'fps': 0, 'latency_ms': 0,
                'total_batches_tested': 0
            }
        
        batch_size = data.size(0)
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'fps': batch_size / np.mean(times),
            'latency_ms': np.mean(times) * 1000,
            'total_batches_tested': len(times)
        }
    
    def analyze_model_efficiency(self, model, data_loader):
        """Comprehensive efficiency analysis with robust error handling"""
        print(f"Analyzing model efficiency...")
        
        # Initial state reset
        functional.reset_net(model)
        
        # 1. Spike analysis on full dataset
        spike_rates = self.count_spikes(model, data_loader)
        print(f"   Spike rates collected for {len(spike_rates)} layers")
        
        if not spike_rates:
            print("   Warning: No spike data collected!")
            return None
        
        # 2. Energy estimation
        energy_stats = self.estimate_energy(spike_rates, model)
        print(f"   Energy estimation completed")
        
        # 3. Timing analysis - BỎ num_batches parameter
        timing_stats = self.measure_inference_time(model, data_loader)
        print(f"   Timing analysis completed on {timing_stats['total_batches_tested']} batches")
        
        # 4. Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'spike_rates': spike_rates,
            'energy_stats': energy_stats,
            'timing_stats': timing_stats,
            'model_stats': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'avg_spike_rate': np.mean(list(spike_rates.values())) if spike_rates else 0
            }
        }
        
        return results

def plot_efficiency_comparison(results):
    """Tạo plots so sánh efficiency với energy scale bình thường"""
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    models = list(valid_results.keys())
    spike_rates = [valid_results[m]['model_stats']['avg_spike_rate'] for m in models]
    energies = [valid_results[m]['energy_stats']['total_energy'] for m in models]
    fps_values = [valid_results[m]['timing_stats']['fps'] for m in models]
    latencies = [valid_results[m]['timing_stats']['latency_ms'] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Spike rates - bỏ note "(Full Dataset)"
    bars1 = axes[0, 0].bar(models, spike_rates, color=colors[:len(models)])
    axes[0, 0].set_ylabel('Average Spike Rate')
    axes[0, 0].set_title('Spike Activity Comparison')  # Bỏ "(Full Dataset)"
    axes[0, 0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars1, spike_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.4f}', ha='center', va='bottom')
    
    # 2. Energy - KHÔNG dùng log scale, hiển thị bình thường
    bars2 = axes[0, 1].bar(models, energies, color=colors[:len(models)])
    axes[0, 1].set_ylabel('Estimated Energy')  # Bỏ "(log scale)"
    axes[0, 1].set_title('Energy Consumption')  # Bỏ "(Full Dataset)"
    # axes[0, 1].set_yscale('log')  # BỎ dòng này
    axes[0, 1].grid(True, alpha=0.3)
    
    # Thêm giá trị energy lên các cột - FIX POSITIONING
    for bar, val in zip(bars2, energies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energies)*0.02,
                       f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # 3. FPS
    bars3 = axes[1, 0].bar(models, fps_values, color=colors[:len(models)])
    axes[1, 0].set_ylabel('Frames Per Second')
    axes[1, 0].set_title('Inference Speed (FPS)')
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, fps_values):
        if val > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.1f}', ha='center', va='bottom')
    
    # 4. Latency
    bars4 = axes[1, 1].bar(models, latencies, color=colors[:len(models)])
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].set_title('Inference Latency')
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars4, latencies):
        if val > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = 'efficiency_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved as '{output_path}'")

# Run function with better error handling...
def run_energy_analysis():
    """Main function với robust error handling"""
    
    # Setup remains the same...
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                   download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Total batches: {len(test_loader)}")
    
    # Load models (same as before)
    model_paths = {
        'LIF': 'Trained/Origin/Ori_b64/model_best.pth.tar',
        'IF_Hard': 'Trained/IF_hard/b64_Uth1_2/model_best.pth.tar', 
        'IF_Soft': 'Trained/IF_soft/b64_Uth1.2/model_best.pth.tar'
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
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
    
    # Run analysis with error handling
    analyzer = QuickEnergyAnalyzer()
    results = {}
    
    print(f"\nRunning efficiency analysis...")
    for name, model in models.items():
        print(f"\n--- Analyzing {name} ---")
        try:
            result = analyzer.analyze_model_efficiency(model, test_loader)
            if result is not None:
                results[name] = result
            else:
                print(f"   Failed to analyze {name}")
        except Exception as e:
            print(f"   Error analyzing {name}: {e}")
            continue
    
    if not results:
        print("No successful analyses!")
        return
    
    # Create outputs
    print("\n" + "="*80)
    print("EFFICIENCY COMPARISON RESULTS")
    print("="*80)
    
    comparison_data = []
    for name, result in results.items():
        row = {
            'Model': name,
            'Avg_Spike_Rate': f"{result['model_stats']['avg_spike_rate']:.4f}",
            'Est_Energy': f"{result['energy_stats']['total_energy']:.2e}",
            'Energy_per_Param': f"{result['energy_stats']['energy_per_param']:.2e}",
            'Inference_Time_ms': f"{result['timing_stats']['latency_ms']:.2f}",
            'FPS': f"{result['timing_stats']['fps']:.1f}",
            'Total_Params': f"{result['model_stats']['total_parameters']:,}"
        }
        comparison_data.append(row)
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    plot_efficiency_comparison(results)
    df.to_csv('efficiency_comparison.csv', index=False)
    
    print(f"\nAnalysis complete! Results saved.")
    return results

if __name__ == "__main__":
    results = run_energy_analysis()