"""
ResNet18 ImageNet Gradient Inversion Attack
Based on ResNet18 - trained on ImageNet.ipynb

This script uses a pretrained ResNet18 model on ImageNet to perform
gradient inversion attacks, which typically gives much better reconstruction
results than training from scratch.
"""

import torch
import torchvision
import numpy as np
import os
import time
import datetime
from collections import defaultdict

import inversefed

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_PATH = './datasets/imagenet'  # You can use any path or skip if not available
SAVE_IMAGES = True

# Target images to try (from the notebook)
TARGET_IMAGES = {
    'beagle': 8112,
    'owl': 1200, 
    'german_shepherd': 11794,
    'panda': 19449
}

def plot_results(tensor, title="Images", save_path=None):
    """Save images using torchvision instead of matplotlib"""
    print(f"[PLOT] {title} - Shape: {tensor.shape}")
    if save_path and SAVE_IMAGES:
        # Get ImageNet normalization constants
        dm = torch.as_tensor(inversefed.consts.imagenet_mean)[:, None, None].to(tensor.device)
        ds = torch.as_tensor(inversefed.consts.imagenet_std)[:, None, None].to(tensor.device)
        
        # Denormalize and save
        tensor_denorm = torch.clamp(tensor * ds + dm, 0, 1)
        torchvision.utils.save_image(tensor_denorm, save_path)
        print(f"[SAVED] Image saved to {save_path}")

def setup_imagenet_data():
    """Setup ImageNet data loading (with fallback if data not available)"""
    # System setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    try:
        # Try to load ImageNet data
        loss_fn, trainloader, validloader = inversefed.construct_dataloaders(
            'ImageNet', defs, data_path=IMAGENET_PATH
        )
        print("ImageNet data loaded successfully!")
        return setup, loss_fn, trainloader, validloader
    except:
        print("⚠️  ImageNet data not available. Using CIFAR-10 as fallback...")
        print("Note: Results will be different from the original notebook.")
        
        # Fallback to CIFAR-10 but with ImageNet normalization for consistency
        loss_fn, trainloader, validloader = inversefed.construct_dataloaders(
            'CIFAR10', defs, data_path='./datasets/cifar10'
        )
        return setup, loss_fn, trainloader, validloader

def create_resnet18_model(setup, pretrained=True):
    """Create ResNet18 model (pretrained on ImageNet)"""
    print(f"Creating ResNet18 model (pretrained={pretrained})...")
    
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.to(**setup)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet18 parameters: {total_params:,}")
    
    return model

def perform_gradient_inversion(model, setup, loss_fn, validloader, target_idx=8112, config_name="standard"):
    """Perform gradient inversion attack with different configurations"""
    
    print(f"\n=== Gradient Inversion Attack ({config_name}) ===")
    print(f"Target image index: {target_idx}")
    
    # Get normalization constants (use ImageNet constants even for CIFAR-10 for consistency)
    dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
    
    # Select target image
    try:
        img, label = validloader.dataset[target_idx]
        print(f"Target class: {validloader.dataset.classes[label] if hasattr(validloader.dataset, 'classes') else label}")
    except:
        # Fallback to a random image if index is out of range
        target_idx = np.random.randint(len(validloader.dataset))
        img, label = validloader.dataset[target_idx]
        print(f"Using fallback target index: {target_idx}")
    
    labels = torch.as_tensor((label,), device=setup['device'])
    ground_truth = img.to(**setup).unsqueeze(0)
    
    # Save original image
    plot_results(ground_truth, f"Original Image - Index {target_idx}", 
                f"original_{target_idx}.png")
    
    # Compute gradient
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    
    full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
    print(f'Gradient norm: {full_norm:.4e}')
    
    # Define attack configurations
    configs = {
        "standard": dict(
            signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=8,
            max_iterations=24000,
            total_variation=1e-1,  # Higher TV for ImageNet
            init='randn',
            filter='none',
            lr_decay=True,
            scoring_choice='loss'
        ),
        "high_lr": dict(
            signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=1.0,  # Higher learning rate
            optim='adam',
            restarts=8,
            max_iterations=24000,
            total_variation=1e-1,
            init='randn',
            filter='none',
            lr_decay=True,
            scoring_choice='loss'
        ),
        "median_filter": dict(
            signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=1.0,
            optim='adam',
            restarts=8,
            max_iterations=24000,
            total_variation=1e-1,
            init='randn',
            filter='median',  # Median filtering
            lr_decay=True,
            scoring_choice='loss'
        )
    }
    
    config = configs.get(config_name, configs["standard"])
    
    # Determine image shape (ImageNet: 224x224, CIFAR-10: 32x32)
    img_shape = (3, 224, 224) if ground_truth.shape[-1] > 32 else (3, 32, 32)
    
    print("Attack configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Image shape: {img_shape}")
    
    # Perform reconstruction
    print(f"\\nStarting reconstruction...")
    start_time = time.time()
    
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape)
    
    attack_time = time.time() - start_time
    
    # Calculate metrics
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean().item()
    
    # Use correct factor for PSNR calculation
    psnr_factor = 1/ds if img_shape[1] == 224 else 1.0
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=psnr_factor)
    
    # Display results
    print(f"\\nAttack Results ({config_name}):")
    print(f"Time: {attack_time:.2f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"PSNR: {test_psnr:.2f} dB")
    print(f"Feature MSE: {feat_mse:.4e}")
    
    # Interpret results
    if test_psnr > 30:
        quality = "🔴 Excellent reconstruction - privacy severely compromised!"
    elif test_psnr > 20:
        quality = "🟠 Good reconstruction - significant privacy leakage"
    elif test_psnr > 15:
        quality = "🟡 Moderate reconstruction - some privacy leakage"
    else:
        quality = "🟢 Poor reconstruction - limited privacy leakage"
    
    print(f"Quality: {quality}")
    
    # Save reconstructed image
    plot_results(output, f"Reconstructed - {config_name} (PSNR: {test_psnr:.2f} dB)",
                f"reconstructed_{target_idx}_{config_name}.png")
    
    # Save comparison
    comparison = torch.cat([ground_truth, output], dim=0)
    plot_results(comparison, f"Comparison - {config_name}",
                f"comparison_{target_idx}_{config_name}.png")
    
    return {
        'config': config_name,
        'target_idx': target_idx,
        'psnr': test_psnr,
        'mse': test_mse,
        'time': attack_time,
        'reconstruction_loss': stats['opt']
    }

def run_multiple_attacks():
    """Run attacks with different configurations"""
    print("ResNet18 ImageNet Gradient Inversion Attack")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    
    # Setup
    setup, loss_fn, trainloader, validloader = setup_imagenet_data()
    model = create_resnet18_model(setup, pretrained=True)
    
    # Try different target images and configurations
    results = []
    
    # Test with beagle image (most famous from the paper)
    target_idx = 8112 if len(validloader.dataset) > 8112 else 42
    
    configs_to_try = ["standard", "high_lr", "median_filter"]
    
    for config_name in configs_to_try:
        try:
            result = perform_gradient_inversion(
                model, setup, loss_fn, validloader, 
                target_idx=target_idx, 
                config_name=config_name
            )
            results.append(result)
            
            print("\\n" + "="*50)
            
        except Exception as e:
            print(f"Error with {config_name} configuration: {e}")
            continue
    
    # Summary
    print("\\n=== SUMMARY ===")
    print(f"Tested {len(results)} configurations:")
    for result in results:
        print(f"{result['config']:15s}: PSNR={result['psnr']:6.2f} dB, "
              f"Time={result['time']:6.1f}s, Loss={result['reconstruction_loss']:.4f}")
    
    if results:
        best_result = max(results, key=lambda x: x['psnr'])
        print(f"\\nBest result: {best_result['config']} with PSNR={best_result['psnr']:.2f} dB")
    
    print(f"\\nCompleted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results

if __name__ == "__main__":
    # Set deterministic for reproducibility
    inversefed.utils.set_deterministic()
    
    # Run the attacks
    results = run_multiple_attacks()