import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Import your model architecture
# from spikeformer import sdt  # Uncomment này và adjust theo project của bạn

def load_your_models():
    """
    Thay thế function này bằng cách load models của bạn
    """
    models = {}
    
    # Method 1: Load từ checkpoint files
    model_configs = {
        'LIF': {
            'checkpoint': 'checkpoints/lif_model_best.pth',
            'spike_mode': 'lif'
        },
        'IF_Hard': {
            'checkpoint': 'checkpoints/if_hard_model_best.pth', 
            'spike_mode': 'if'
        },
        'IF_Soft': {
            'checkpoint': 'checkpoints/if_soft_model_best.pth',
            'spike_mode': 'if_soft'
        }
    }
    
    # Method 2: Recreate models và load state dict
    for model_name, config in model_configs.items():
        try:
            print(f"Loading {model_name}...")
            
            # Option A: Load toàn bộ model (nếu bạn save bằng torch.save(model, path))
            if os.path.exists(config['checkpoint']):
                model = torch.load(config['checkpoint'], map_location='cpu')
                
            # Option B: Recreate model và load state dict
            else:
                # Uncomment và adjust theo model architecture của bạn
                """
                model = sdt(
                    img_size_h=32,
                    img_size_w=32, 
                    patch_size=4,
                    in_channels=3,
                    num_classes=10,
                    embed_dims=256,  # hoặc 512 tùy config
                    num_heads=8,
                    mlp_ratios=4,
                    depths=2,
                    T=4,
                    spike_mode=config['spike_mode']
                )
                
                # Load state dict
                checkpoint = torch.load(config['checkpoint'], map_location='cpu')
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                """
                print(f"   ❌ Checkpoint not found: {config['checkpoint']}")
                continue
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            models[model_name] = model
            print(f"   ✓ {model_name} loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
    
    return models

def setup_data_loader():
    """Setup CIFAR-10 data loader"""
    
    # CIFAR-10 transforms (adjust theo config của bạn)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Test dataset
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
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Data loader ready: {len(test_dataset)} test samples")
    return test_loader

def main():
    """Main function để chạy cả 2 analyses"""
    
    print("🚀 Starting Comprehensive SNN Analysis")
    print("="*60)
    
    # 1. Setup
    print("\n📂 Setting up data and models...")
    test_loader = setup_data_loader()
    models = load_your_models()
    
    if not models:
        print("❌ No models loaded! Please check your model paths and update load_your_models() function")
        return
    
    print(f"✓ Loaded {len(models)} models: {list(models.keys())}")
    
    # 2. Energy & Efficiency Analysis
    print("\n" + "="*60)
    print("⚡ ENERGY & EFFICIENCY ANALYSIS")
    print("="*60)
    
    try:
        from energy_analysis import QuickEnergyAnalyzer, plot_efficiency_comparison
        
        analyzer = QuickEnergyAnalyzer()
        energy_results = {}
        
        for name, model in models.items():
            print(f"\n--- Analyzing {name} ---")
            result = analyzer.analyze_model_efficiency(model, test_loader)
            energy_results[name] = result
        
        # Create comparison plots
        plot_efficiency_comparison(energy_results)
        
        # Save results
        import pandas as pd
        comparison_data = []
        for name, result in energy_results.items():
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
        
        df_energy = pd.DataFrame(comparison_data)
        df_energy.to_csv('energy_efficiency_results.csv', index=False)
        
        print("\n✅ Energy analysis complete!")
        print("📁 Saved: energy_efficiency_results.csv, efficiency_comparison.png")
        
    except Exception as e:
        print(f"❌ Energy analysis failed: {e}")
    
    # 3. Spike Pattern Visualization
    print("\n" + "="*60)
    print("🧠 SPIKE PATTERN VISUALIZATION")
    print("="*60)
    
    try:
        from spike_visualization import SpikeVisualizer
        
        visualizer = SpikeVisualizer()
        all_spike_data = {}
        
        # Collect spike patterns
        for model_name, model in models.items():
            print(f"\n--- Collecting spikes from {model_name} ---")
            spike_data = visualizer.collect_spike_patterns(model, test_loader, num_samples=100)
            if spike_data:
                all_spike_data[model_name] = spike_data
                print(f"   ✓ Collected from {len(spike_data)} layers")
        
        if all_spike_data:
            # Create visualizations
            print("\n📊 Creating visualizations...")
            
            # 1. Firing rate distributions
            fig1 = visualizer.plot_firing_rate_distribution(all_spike_data)
            fig1.savefig('spike_firing_rates.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            # 2. Layer-wise activity
            fig2 = visualizer.plot_layer_wise_activity(all_spike_data)
            fig2.savefig('spike_layer_activity.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            # 3. Sparsity analysis
            fig3 = visualizer.plot_spike_sparsity_analysis(all_spike_data)
            fig3.savefig('spike_sparsity.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            # 4. Summary statistics
            summary_df = visualizer.create_summary_statistics(all_spike_data)
            summary_df.to_csv('spike_pattern_summary.csv', index=False)
            
            print("\n✅ Spike visualization complete!")
            print("📁 Saved: spike_firing_rates.png, spike_layer_activity.png")
            print("📁 Saved: spike_sparsity.png, spike_pattern_summary.csv")
            
        else:
            print("❌ No spike data collected!")
            
    except Exception as e:
        print(f"❌ Spike visualization failed: {e}")
    
    # 4. Combined Summary Report
    print("\n" + "="*60)
    print("📋 GENERATING SUMMARY REPORT")
    print("="*60)
    
    try:
        generate_summary_report(energy_results if 'energy_results' in locals() else None,
                              all_spike_data if 'all_spike_data' in locals() else None)
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    print("\n🎉 Analysis Complete!")
    print("="*60)

def generate_summary_report(energy_results, spike_data):
    """Generate a comprehensive summary report"""
    
    report = []
    report.append("# Comprehensive SNN Analysis Report\n")
    report.append("Generated by: SNN Analysis Suite\n")
    report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    if energy_results:
        report.append("## Energy & Efficiency Analysis\n")
        report.append("| Model | Avg Spike Rate | Est. Energy | FPS | Latency (ms) |\n")
        report.append("|-------|----------------|-------------|-----|-------------|\n")
        
        for name, result in energy_results.items():
            spike_rate = result['model_stats']['avg_spike_rate']
            energy = result['energy_stats']['total_energy']
            fps = result['timing_stats']['fps']
            latency = result['timing_stats']['latency_ms']
            
            report.append(f"| {name} | {spike_rate:.4f} | {energy:.2e} | {fps:.1f} | {latency:.2f} |\n")
        
        report.append("\n")
    
    if spike_data:
        report.append("## Spike Pattern Analysis\n")
        
        for model_name, data in spike_data.items():
            all_spikes = []
            for layer_name, spikes in data.items():
                if len(spikes) > 0:
                    all_spikes.append(spikes.flatten())
            
            if all_spikes:
                all_spikes = np.concatenate(all_spikes)
                mean_activity = all_spikes.mean()
                sparsity = (all_spikes == 0).mean()
                
                report.append(f"### {model_name}\n")
                report.append(f"- Mean Activity: {mean_activity:.4f}\n")
                report.append(f"- Sparsity: {sparsity:.4f}\n")
                report.append(f"- Active Layers: {len(data)}\n\n")
    
    # Save report
    with open('analysis_report.md', 'w') as f:
        f.writelines(report)
    
    print("✅ Summary report saved: analysis_report.md")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    main()