# fixed_run_analyses.py
import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Import your model architecture
sys.path.append('.')  # Add current directory to path
from model.spikeformer import sdt  # Import your model

def load_your_models():
    """
    Load models với proper handling cho checkpoint dict format
    """
    models = {}
    
    # Update paths này với actual checkpoint paths của bạn
    model_configs = {
        'LIF': {
            'checkpoint': 'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',  # Update path này
            'config': {
                'img_size_h': 32,
                'img_size_w': 32,
                'patch_size': 4,
                'in_channels': 3,
                'num_classes': 10,
                'embed_dims': 256,  # Hoặc 512, check config của bạn
                'num_heads': 8,
                'mlp_ratios': 4,
                'depths': 2,  # Hoặc số layers bạn dùng
                'T': 4,  # time steps
                'spike_mode': 'lif'
            }
        },
        'IF_Hard': {
            'checkpoint': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',  # Update path này
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
                'spike_mode': 'if'
            }
        },
        'IF_Soft': {
            'checkpoint': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_soft\b64_Uth1_2\model_best.pth.tar',  # Update path này
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
                'spike_mode': 'if_soft'
            }
        }
    }
    
    for model_name, info in model_configs.items():
        checkpoint_path = info['checkpoint']
        config = info['config']
        
        try:
            print(f"Loading {model_name}...")
            
            if not os.path.exists(checkpoint_path):
                print(f"   ❌ Checkpoint not found: {checkpoint_path}")
                continue
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Debug: Print checkpoint keys để hiểu structure
            print(f"   📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            # Create model architecture
            model = sdt(**config)
            print(f"   ✓ Model architecture created")
            
            # Load state dict - try different possible keys
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                    print(f"   ✓ Loaded state_dict from 'model' key")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"   ✓ Loaded state_dict from 'state_dict' key")
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"   ✓ Loaded state_dict from 'model_state_dict' key")
                else:
                    # Try to load directly if checkpoint IS the state_dict
                    try:
                        model.load_state_dict(checkpoint)
                        print(f"   ✓ Loaded checkpoint as state_dict directly")
                    except Exception as e:
                        print(f"   ❌ Could not load state_dict: {e}")
                        print(f"   Available keys: {list(checkpoint.keys())}")
                        continue
            else:
                # Checkpoint is the model itself
                model = checkpoint
                print(f"   ✓ Loaded model directly")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                print(f"   ✓ Moved to GPU")
            
            model.eval()
            models[model_name] = model
            print(f"   ✅ {model_name} loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return models

def inspect_checkpoint(checkpoint_path):
    """Helper function để inspect checkpoint structure"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"\n🔍 Inspecting {checkpoint_path}:")
        print(f"Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            for key, value in checkpoint.items():
                if hasattr(value, 'keys') and callable(getattr(value, 'keys')):
                    print(f"  {key}: dict with {len(value)} items")
                else:
                    print(f"  {key}: {type(value)}")
        
        return checkpoint
    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        return None

def setup_data_loader():
    """Setup CIFAR-10 data loader"""
    
    # CIFAR-10 transforms (theo config trong yml files của bạn)
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
        num_workers=2,  # Giảm xuống để tránh memory issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Data loader ready: {len(test_dataset)} test samples")
    return test_loader

def main():
    """Main function để chạy cả 2 analyses"""
    
    print("🚀 Starting Comprehensive SNN Analysis")
    print("="*60)
    
    # Debug: First let's inspect your checkpoints
    print("\n🔍 CHECKPOINT INSPECTION")
    print("="*60)
    
    # Update these paths với actual checkpoint paths của bạn
    checkpoint_paths = [
        'checkpoints/lif_model_best.pth',
        'checkpoints/if_hard_model_best.pth', 
        'checkpoints/if_soft_model_best.pth'
    ]
    
    # Inspect first checkpoint to understand structure
    for path in checkpoint_paths:
        if os.path.exists(path):
            inspect_checkpoint(path)
            break
    
    # 1. Setup
    print("\n📂 Setting up data and models...")
    test_loader = setup_data_loader()
    models = load_your_models()
    
    if not models:
        print("❌ No models loaded!")
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("1. Check if checkpoint paths exist")
        print("2. Update model configs in load_your_models()")
        print("3. Check the checkpoint inspection output above")
        return
    
    print(f"✓ Loaded {len(models)} models: {list(models.keys())}")
    
    # 2. Energy & Efficiency Analysis
    print("\n" + "="*60)
    print("⚡ ENERGY & EFFICIENCY ANALYSIS")
    print("="*60)
    
    try:
        # Import analysis classes
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
        
    except ImportError as e:
        print(f"❌ Could not import analysis modules: {e}")
        print("💡 Make sure energy_analysis.py is in the same directory")
    except Exception as e:
        print(f"❌ Energy analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
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
            spike_data = visualizer.collect_spike_patterns(model, test_loader, num_samples=50)
            if spike_data:
                all_spike_data[model_name] = spike_data
                print(f"   ✓ Collected from {len(spike_data)} layers")
        
        if all_spike_data:
            # Create visualizations
            print("\n📊 Creating visualizations...")
            
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
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
            
    except ImportError as e:
        print(f"❌ Could not import spike visualization: {e}")
    except Exception as e:
        print(f"❌ Spike visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()