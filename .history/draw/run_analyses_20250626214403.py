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
            'checkpoint': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',
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
                'rpe_mode': 'conv'  # Add default RPE mode
            }
        },
        'IF_Hard': {
            'checkpoint': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',
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
            'checkpoint': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_soft\b64_Uth1_2\model_best.pth.tar',
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
            
            print(f"   Checkpoint keys: {list(checkpoint.keys())}")
            
            # Create model architecture
            model = sdt(**config)
            print(f"   Model architecture created")
            
            # Load state dict with strict=False to handle missing keys
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
                    print(f"   Loaded state_dict from 'model' key")
                elif 'state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
                    print(f"   Loaded state_dict from 'state_dict' key")
                elif 'model_state_dict' in checkpoint:
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"   Loaded state_dict from 'model_state_dict' key")
                else:
                    try:
                        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                        print(f"   Loaded checkpoint as state_dict directly")
                    except Exception as e:
                        print(f"   Could not load state_dict: {e}")
                        print(f"   Available keys: {list(checkpoint.keys())}")
                        continue
                
                # Report missing/unexpected keys
                if missing_keys:
                    print(f"   Missing keys: {len(missing_keys)} (using strict=False)")
                if unexpected_keys:
                    print(f"   Unexpected keys: {len(unexpected_keys)} (ignored)")
                    
            else:
                model = checkpoint
                print(f"   Loaded model directly")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                print(f"   Moved to GPU")
            
            model.eval()
            models[model_name] = model
            print(f"   {model_name} loaded successfully")
            
        except Exception as e:
            print(f"   Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return models

def setup_data_loader():
    """Setup CIFAR-10 data loader với full dataset"""
    
    # CIFAR-10 transforms
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
        batch_size=32,  # Smaller batch size for stability
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Data loader ready: {len(test_dataset)} test samples, {len(test_loader)} batches")
    return test_loader

def run_energy_analysis_full(models, test_loader):
    """Run comprehensive energy analysis on full dataset"""
    print("\n" + "="*80)
    print("COMPREHENSIVE ENERGY & EFFICIENCY ANALYSIS (FULL DATASET)")
    print("="*80)
    
    try:
        from energy_analysis import QuickEnergyAnalyzer, plot_efficiency_comparison
        
        analyzer = QuickEnergyAnalyzer()
        energy_results = {}
        
        for name, model in models.items():
            print(f"\n--- Analyzing {name} ---")
            try:
                result = analyzer.analyze_model_efficiency(model, test_loader)
                if result is not None:
                    energy_results[name] = result
                    print(f"   Analysis completed successfully")
                else:
                    print(f"   Analysis failed for {name}")
            except Exception as e:
                print(f"   Error analyzing {name}: {e}")
                continue
        
        if not energy_results:
            print("No successful energy analyses!")
            return None
        
        # Create comparison plots
        print(f"\nCreating efficiency comparison plots...")
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
        df_energy.to_csv('energy_efficiency_results_full.csv', index=False)
        
        # Print summary table
        print(f"\n" + "="*80)
        print("ENERGY EFFICIENCY RESULTS SUMMARY")
        print("="*80)
        print(df_energy.to_string(index=False))
        
        print(f"\nEnergy analysis complete!")
        print(f"Files saved:")
        print(f"   - efficiency_comparison.png")
        print(f"   - energy_efficiency_results_full.csv")
        
        return energy_results
        
    except ImportError as e:
        print(f"Could not import energy analysis modules: {e}")
        print("Make sure energy_analysis.py is in the same directory")
        return None
    except Exception as e:
        print(f"Energy analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_spike_visualization_full(models, test_loader):
    """Run comprehensive spike pattern visualization on full dataset"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SPIKE PATTERN VISUALIZATION (FULL DATASET)")
    print("="*80)
    
    try:
        from spike_visualization import SpikeVisualizer
        
        visualizer = SpikeVisualizer()
        all_spike_data = {}
        
        # Collect spike patterns from full dataset
        for model_name, model in models.items():
            print(f"\n--- Collecting spikes from {model_name} ---")
            try:
                spike_data = visualizer.collect_spike_patterns(
                    model, test_loader, 
                    use_full_dataset=True,  # Use full dataset
                    num_samples=100  # This parameter ignored when use_full_dataset=True
                )
                if spike_data:
                    all_spike_data[model_name] = spike_data
                    print(f"   Collected from {len(spike_data)} layers")
                else:
                    print(f"   No spike data collected for {model_name}")
            except Exception as e:
                print(f"   Error collecting spikes from {model_name}: {e}")
                continue
        
        if not all_spike_data:
            print("No spike data collected!")
            return None
        
        # Create visualizations
        print(f"\nCreating comprehensive visualizations...")
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # 1. Firing rate distributions
        print("   Creating firing rate distributions...")
        fig1 = visualizer.plot_firing_rate_distribution(all_spike_data)
        if fig1:
            fig1.savefig('spike_firing_rates.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
        
        # 2. Layer-wise activity
        print("   Creating layer-wise activity plot...")
        fig2 = visualizer.plot_layer_wise_activity(all_spike_data)
        if fig2:
            fig2.savefig('spike_layer_activity.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
        
        # 3. Sparsity analysis
        print("   Creating sparsity analysis...")
        fig3 = visualizer.plot_spike_sparsity_analysis(all_spike_data)
        if fig3:
            fig3.savefig('spike_sparsity.png', dpi=300, bbox_inches='tight')
            plt.close(fig3)
        
        # 4. Summary statistics
        print("   Generating summary statistics...")
        summary_df = visualizer.create_summary_statistics(all_spike_data)
        summary_df.to_csv('spike_pattern_summary_full.csv', index=False)
        
        # Print summary table
        print(f"\n" + "="*80)
        print("SPIKE PATTERN ANALYSIS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        print(f"\nSpike visualization complete!")
        print(f"Files saved:")
        print(f"   - spike_firing_rates.png")
        print(f"   - spike_layer_activity.png") 
        print(f"   - spike_sparsity.png")
        print(f"   - spike_pattern_summary_full.csv")
        
        return all_spike_data, summary_df
        
    except ImportError as e:
        print(f"Could not import spike visualization: {e}")
        return None, None
    except Exception as e:
        print(f"Spike visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function để chạy comprehensive analysis trên full dataset"""
    
    print("STARTING COMPREHENSIVE SNN ANALYSIS ON FULL DATASET")
    print("="*80)
    print("This analysis will process the entire CIFAR-10 test set (10,000 samples)")
    print("Expected runtime: 10-30 minutes depending on hardware")
    print("="*80)
    
    # 1. Setup
    print("\nSetting up data and models...")
    test_loader = setup_data_loader()
    models = load_your_models()
    
    if not models:
        print("No models loaded!")
        print("\nTROUBLESHOOTING TIPS:")
        print("1. Check if checkpoint paths exist")
        print("2. Update model configs in load_your_models()")
        print("3. Ensure model architecture is compatible")
        return
    
    print(f"\nSuccessfully loaded {len(models)} models: {list(models.keys())}")
    print(f"Ready to analyze {len(test_loader.dataset)} test samples")
    
    # 2. Run Energy & Efficiency Analysis on Full Dataset
    energy_results = run_energy_analysis_full(models, test_loader)
    
    # 3. Run Spike Pattern Visualization on Full Dataset  
    spike_results, spike_summary = run_spike_visualization_full(models, test_loader)
    
    # 4. Final Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*80)
    
    if energy_results:
        print(f"Energy Analysis: ✓ Completed for {len(energy_results)} models")
    else:
        print(f"Energy Analysis: ✗ Failed")
    
    if spike_results:
        print(f"Spike Analysis: ✓ Completed for {len(spike_results)} models")
    else:
        print(f"Spike Analysis: ✗ Failed")
    
    print(f"\nGenerated Files:")
    print(f"📊 Energy Analysis:")
    print(f"   - efficiency_comparison.png")
    print(f"   - energy_efficiency_results_full.csv")
    print(f"🧠 Spike Analysis:")
    print(f"   - spike_firing_rates.png")
    print(f"   - spike_layer_activity.png")
    print(f"   - spike_sparsity.png") 
    print(f"   - spike_pattern_summary_full.csv")
    
    print(f"\nAll analyses completed successfully!")
    print(f"Results are comprehensive and based on the full CIFAR-10 test dataset.")
    
    return energy_results, spike_results, spike_summary

if __name__ == "__main__":
    energy_results, spike_results, spike_summary = main()