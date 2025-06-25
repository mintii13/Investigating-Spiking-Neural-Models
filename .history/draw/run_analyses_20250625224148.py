# fixed_run_analyses.py
import sys
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import logging

# Import your model architecture
sys.path.append('.')  # Add current directory to path
from model.spikeformer import sdt  # Import your model

def inspect_checkpoint_detailed(checkpoint_path):
    """Detailed checkpoint inspection"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"\n🔍 DETAILED INSPECTION: {os.path.basename(checkpoint_path)}")
        print("="*80)
        
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Check args if available
            if 'args' in checkpoint:
                args = checkpoint['args']
                print(f"\nModel arguments from checkpoint:")
                relevant_args = ['spike_mode', 'rpe_mode', 'dim', 'layer', 'time_steps', 'num_heads']
                for arg in relevant_args:
                    if hasattr(args, arg):
                        print(f"  {arg}: {getattr(args, arg)}")
            
            # Check state_dict structure
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\nState dict has {len(state_dict)} parameters")
                
                # Look for RPE-related parameters
                rpe_params = [k for k in state_dict.keys() if 'rpe' in k.lower()]
                print(f"RPE-related parameters: {rpe_params}")
                
                # Look for patch_embed parameters
                patch_params = [k for k in state_dict.keys() if 'patch_embed' in k]
                print(f"Patch embed parameters (first 10): {patch_params[:10]}")
                
                # Check for spike mode indicators
                spike_params = [k for k in state_dict.keys() if any(x in k for x in ['lif', 'if'])]
                print(f"Spike-related parameters (first 5): {spike_params[:5]}")
        
        return checkpoint
    except Exception as e:
        print(f"❌ Error inspecting {checkpoint_path}: {e}")
        return None

def create_compatible_model(checkpoint, default_config):
    """Create a model compatible with the checkpoint"""
    
    # Try to get config from checkpoint args
    if 'args' in checkpoint:
        args = checkpoint['args']
        config = {
            'img_size_h': getattr(args, 'img_size', 32),
            'img_size_w': getattr(args, 'img_size', 32),
            'patch_size': getattr(args, 'patch_size', 4),
            'in_channels': getattr(args, 'in_channels', 3),
            'num_classes': getattr(args, 'num_classes', 10),
            'embed_dims': getattr(args, 'dim', 256),
            'num_heads': getattr(args, 'num_heads', 8),
            'mlp_ratios': getattr(args, 'mlp_ratio', 4),
            'depths': getattr(args, 'layer', 2),
            'T': getattr(args, 'time_steps', 4),
            'spike_mode': getattr(args, 'spike_mode', 'if_soft'),
            'pooling_stat': getattr(args, 'pooling_stat', '1111'),
        }
        
        # Handle RPE mode - if not in checkpoint args, use 'conv' (original)
        if hasattr(args, 'rpe_mode'):
            config['rpe_mode'] = args.rpe_mode
        else:
            # Check if checkpoint has rpe_dilated parameters
            state_dict = checkpoint.get('state_dict', {})
            has_dilated = any('rpe_dilated' in k for k in state_dict.keys())
            has_conv = any('rpe_conv' in k for k in state_dict.keys())
            
            if has_dilated:
                config['rpe_mode'] = 'dilated'
            elif has_conv:
                config['rpe_mode'] = 'conv'
            else:
                config['rpe_mode'] = 'conv'  # Default fallback
        
        print(f"  📋 Using config from checkpoint args")
        for k, v in config.items():
            print(f"    {k}: {v}")
            
    else:
        # Use default config
        config = default_config.copy()
        print(f"  📋 Using default config (no args in checkpoint)")
    
    return config

def load_models_with_fallback():
    """Load models with multiple fallback strategies"""
    models = {}
    
    # Update these paths to your actual checkpoint locations
    model_paths = {
        'LIF': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\Origin\Ori_b64\model_best.pth.tar',
        'IF_Hard': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_hard\b64_Uth1_2\model_best.pth.tar',
        'IF_Soft': r'D:\FPTU-sourse\Term4\ResFes\Spike-Driven-Transformer_newSPS\Trained\IF_soft\b64_Uth1_2\model_best.pth.tar'
    }
    
    # Default model configuration
    default_config = {
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
        'pooling_stat': '1111',
        'rpe_mode': 'conv'  # Safe default
    }
    
    for model_name, checkpoint_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"Loading {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(checkpoint_path):
            print(f"   ❌ Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            # Load and inspect checkpoint
            checkpoint = inspect_checkpoint_detailed(checkpoint_path)
            if checkpoint is None:
                continue
            
            # Create compatible model config
            config = create_compatible_model(checkpoint, default_config)
            
            # Override spike_mode based on model name if needed
            if model_name == 'LIF':
                config['spike_mode'] = 'lif'
            elif model_name == 'IF_Hard':
                config['spike_mode'] = 'if'
            elif model_name == 'IF_Soft':
                config['spike_mode'] = 'if_soft'
            
            print(f"\n  🏗️ Creating model with config:")
            print(f"    spike_mode: {config['spike_mode']}")
            print(f"    rpe_mode: {config['rpe_mode']}")
            print(f"    embed_dims: {config['embed_dims']}")
            
            # Create model
            model = sdt(**config)
            print(f"   ✓ Model architecture created")
            
            # Load state dict with error handling
            state_dict = checkpoint['state_dict']
            
            # Try loading with strict=False first
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"   ⚠️ Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"   ⚠️ Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                print(f"   ✓ Loaded state_dict (strict=False)")
                
                # If too many missing keys, try with different rpe_mode
                if len(missing_keys) > 5:
                    print(f"   🔄 Too many missing keys, trying different RPE mode...")
                    
                    # Try with conv mode if dilated failed
                    if config['rpe_mode'] == 'dilated':
                        config['rpe_mode'] = 'conv'
                        model = sdt(**config)
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        print(f"   ✓ Retried with rpe_mode='conv', missing: {len(missing_keys)}")
                
            except Exception as e:
                print(f"   ❌ Failed to load state_dict: {e}")
                continue
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
                print(f"   ✓ Moved to GPU")
            
            model.eval()
            
            # Test model with dummy input
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 32, 32)
                    if torch.cuda.is_available():
                        dummy_input = dummy_input.cuda()
                    
                    output = model(dummy_input)
                    print(f"   ✓ Model test passed - output shape: {output[0].shape}")
                    
                models[model_name] = model
                print(f"   ✅ {model_name} loaded successfully!")
                
            except Exception as e:
                print(f"   ❌ Model test failed: {e}")
                continue
                
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return models

def setup_data_loader():
    """Setup CIFAR-10 data loader"""
    
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
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Data loader ready: {len(test_dataset)} test samples")
    return test_loader

def run_quick_analysis(models, test_loader):
    """Run a quick analysis on loaded models"""
    
    print(f"\n{'='*80}")
    print("🔬 QUICK MODEL ANALYSIS")
    print(f"{'='*80}")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Analyzing {model_name} ---")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Quick accuracy test
        correct = 0
        total = 0
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                if batch_idx >= 5:  # Test only 5 batches
                    break
                
                if torch.cuda.is_available():
                    data, targets = data.cuda(), targets.cuda()
                
                outputs = model(data)[0]
                if len(outputs.shape) > 2:  # If TET output
                    outputs = outputs.mean(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        
        results[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'quick_accuracy': accuracy,
            'test_samples': total
        }
        
        print(f"   Parameters: {total_params:,}")
        print(f"   Quick accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # Create summary table
    print(f"\n{'='*80}")
    print("📊 QUICK ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    import pandas as pd
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'Total_Params': f"{result['total_params']:,}",
            'Quick_Accuracy': f"{result['quick_accuracy']:.2f}%",
            'Test_Samples': result['test_samples']
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    return results

def main():
    """Main function"""
    
    print("🚀 Starting Fixed SNN Analysis")
    print("="*80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Setup data
    print("\n📂 Setting up data loader...")
    test_loader = setup_data_loader()
    
    # 2. Load models with fallback strategies
    print("\n🤖 Loading models with fallback strategies...")
    models = load_models_with_fallback()
    
    if not models:
        print("\n❌ No models could be loaded!")
        print("\n💡 SUGGESTIONS:")
        print("1. Check if checkpoint files exist at specified paths")
        print("2. Verify that model architecture matches checkpoint")
        print("3. Consider using strict=False when loading state_dict")
        print("4. Check if RPE mode in current code matches saved model")
        return
    
    print(f"\n✅ Successfully loaded {len(models)} models: {list(models.keys())}")
    
    # 3. Run quick analysis
    results = run_quick_analysis(models, test_loader)
    
    # 4. Try energy analysis if models loaded successfully
    if len(models) >= 1:
        print(f"\n{'='*80}")
        print("⚡ ATTEMPTING ENERGY ANALYSIS")
        print(f"{'='*80}")
        
        try:
            from energy_analysis import QuickEnergyAnalyzer
            
            analyzer = QuickEnergyAnalyzer()
            
            # Run on first model as test
            test_model_name = list(models.keys())[0]
            test_model = models[test_model_name]
            
            print(f"Testing energy analysis on {test_model_name}...")
            energy_result = analyzer.analyze_model_efficiency(test_model, test_loader)
            
            print(f"✅ Energy analysis test successful!")
            print(f"   Avg spike rate: {energy_result['model_stats']['avg_spike_rate']:.4f}")
            print(f"   Est. energy: {energy_result['energy_stats']['total_energy']:.2e}")
            print(f"   Inference time: {energy_result['timing_stats'][