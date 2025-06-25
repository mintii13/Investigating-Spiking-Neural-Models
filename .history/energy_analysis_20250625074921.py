# energy_analysis.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class QuickEnergyAnalyzer:
    def __init__(self):
        self.spike_counts = defaultdict(list)
        self.operation_counts = defaultdict(int)
        
    def count_spikes(self, model, data_loader):
        """Đếm số spikes trong mỗi layer"""
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
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 10:  # Chỉ test 10 batches
                    break
                
                if torch.cuda.is_available():
                    data = data.cuda()
                
                # Forward pass
                _ = model(data)
                
                # Thu thập spike rates
                for name, rate in spike_stats.items():
                    total_spike_rates[name].append(rate)
                spike_stats.clear()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Tính average spike rates
        avg_spike_rates = {}
        for name, rates in total_spike_rates.items():
            avg_spike_rates[name] = np.mean(rates)
            
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
    
    def measure_inference_time(self, model, data_loader, num_batches=50):
        """Đo thời gian inference"""
        model.eval()
        times = []
        
        # Warm up
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 3:
                    break
                if torch.cuda.is_available():
                    data = data.cuda()
                _ = model(data)
        
        # Actual timing
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                if torch.cuda.is_available():
                    data = data.cuda()
                    torch.cuda.synchronize()
                    
                start_time = time.time()
                _ = model(data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                end_time = time.time()
                times.append(end_time - start_time)
        
        batch_size = data.size(0)
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'fps': batch_size / np.mean(times),
            'latency_ms': np.mean(times) * 1000
        }
    
    def analyze_model_efficiency(self, model, data_loader):
        """Comprehensive efficiency analysis"""
        print(f"🔍 Analyzing model efficiency...")
        
        # 1. Spike analysis
        spike_rates = self.count_spikes(model, data_loader)
        print(f"   ✓ Spike rates collected for {len(spike_rates)} layers")
        
        # 2. Energy estimation
        energy_stats = self.estimate_energy(spike_rates, model)
        print(f"   ✓ Energy estimation completed")
        
        # 3. Timing analysis
        timing_stats = self.measure_inference_time(model, data_loader)
        print(f"   ✓ Timing analysis completed")
        
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

def run_energy_analysis():
    """Main function để chạy energy analysis"""
    
    # 1. Setup data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                   download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # 2. Load your trained models
    print("📂 Loading trained models...")
    
    # Thay thế bằng cách load models của bạn
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
            # Hoặc:
            # model = YourModelClass(config)
            # model.load_state_dict(torch.load(path))
            
            if torch.cuda.is_available():
                model = model.cuda()
            models[name] = model
            print(f"   ✓ {name} model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load {name}: {e}")
    
    if not models:
        print("❌ No models loaded! Please check your model paths.")
        return
    
    # 3. Run analysis
    analyzer = QuickEnergyAnalyzer()
    results = {}
    
    print("\n🔬 Running efficiency analysis...")
    for name, model in models.items():
        print(f"\n--- Analyzing {name} ---")
        results[name] = analyzer.analyze_model_efficiency(model, test_loader)
    
    # 4. Print comparison
    print("\n" + "="*80)
    print("📊 EFFICIENCY COMPARISON RESULTS")
    print("="*80)
    
    # Create comparison table
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
    
    # Print table
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # 5. Create visualizations
    plot_efficiency_comparison(results)
    
    # 6. Save results
    df.to_csv('efficiency_comparison.csv', index=False)
    
    print(f"\n✅ Analysis complete! Results saved to 'efficiency_comparison.csv'")
    return results

def plot_efficiency_comparison(results):
    """Tạo plots so sánh efficiency"""
    
    models = list(results.keys())
    spike_rates = [results[m]['model_stats']['avg_spike_rate'] for m in models]
    energies = [results[m]['energy_stats']['total_energy'] for m in models]
    fps_values = [results[m]['timing_stats']['fps'] for m in models]
    latencies = [results[m]['timing_stats']['latency_ms'] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Spike rates
    bars1 = axes[0, 0].bar(models, spike_rates, color=colors[:len(models)])
    axes[0, 0].set_ylabel('Average Spike Rate')
    axes[0, 0].set_title('🧠 Spike Activity Comparison')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, spike_rates):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{val:.4f}', ha='center', va='bottom')
    
    # 2. Energy (log scale)
    bars2 = axes[0, 1].bar(models, energies, color=colors[:len(models)])
    axes[0, 1].set_ylabel('Estimated Energy (log scale)')
    axes[0, 1].set_title('⚡ Energy Consumption')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. FPS
    bars3 = axes[1, 0].bar(models, fps_values, color=colors[:len(models)])
    axes[1, 0].set_ylabel('Frames Per Second')
    axes[1, 0].set_title('🚀 Inference Speed (FPS)')
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, fps_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}', ha='center', va='bottom')
    
    # 4. Latency
    bars4 = axes[1, 1].bar(models, latencies, color=colors[:len(models)])
    axes[1, 1].set_ylabel('Latency (ms)')
    axes[1, 1].set_title('⏱️ Inference Latency')
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars4, latencies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as 'efficiency_comparison.png'")

if __name__ == "__main__":
    # Chạy energy analysis
    results = run_energy_analysis()