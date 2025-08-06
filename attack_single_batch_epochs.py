"""
Attack gradients from different epochs to reconstruct the single batch of 100 images
Matches the approach from ResNet32-10 - Recovering 100 CIFAR-100 images.ipynb
"""

import torch
import torchvision
import numpy as np
import os
import time

import inversefed

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'ResNet32-10'
NUM_IMAGES = 100

# Attack configuration from the notebook
ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=0.1,
    optim='adam',
    restarts=4,
    max_iterations=24000,
    total_variation=1e-2,  # Same as notebook for batch
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

def load_ground_truth(validloader):
    """Load the original batch of 100 images"""
    batch_info = torch.load('single_batch_gradients/batch_info.pt')
    batch_indices = batch_info['batch_indices']
    
    # Reconstruct the original batch
    images = []
    labels = []
    for idx in batch_indices:
        img, label = validloader.dataset[idx]
        images.append(img)
        labels.append(label)
    
    ground_truth = torch.stack(images)
    labels = torch.tensor(labels)
    
    return ground_truth, labels, batch_indices

def attack_epoch(epoch):
    """Perform gradient inversion attack for a specific epoch"""
    print(f"\n{'='*60}")
    print(f"=== Attacking Epoch {epoch} ===")
    print(f"{'='*60}")
    
    # Load saved data
    epoch_data = torch.load(f'single_batch_gradients/epoch_{epoch}_gradient.pt')
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data for ground truth comparison
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path='./datasets/cifar10'
    )
    
    # Get ground truth
    ground_truth, true_labels, _ = load_ground_truth(validloader)
    ground_truth = ground_truth.to(**setup)
    true_labels = true_labels.to(device=setup['device'])
    
    # Create model and load state
    model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.load_state_dict(epoch_data['model_state'])
    model.to(**setup)
    model.eval()
    
    # Load gradients
    input_gradient = [g.to(**setup) for g in epoch_data['gradients']]
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Print epoch statistics
    print(f"Epoch {epoch} Statistics:")
    print(f"  Original loss: {epoch_data['loss']:.4f}")
    print(f"  Original accuracy: {epoch_data['accuracy']:.2f}%")
    print(f"  Gradient norm: {epoch_data['total_norm']:.4f}")
    
    # Perform reconstruction
    print(f"\nStarting reconstruction with {ATTACK_CONFIG['restarts']} restarts...")
    print(f"Max iterations: {ATTACK_CONFIG['max_iterations']}")
    
    start_time = time.time()
    
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), ATTACK_CONFIG, num_images=NUM_IMAGES)
    output, stats = rec_machine.reconstruct(input_gradient, true_labels, img_shape=(3, 32, 32))
    
    attack_time = time.time() - start_time
    
    # Calculate metrics
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    
    # Per-image PSNR
    per_image_psnr = []
    for i in range(NUM_IMAGES):
        psnr = inversefed.metrics.psnr(
            output[i:i+1], 
            ground_truth[i:i+1], 
            factor=1/ds
        )
        per_image_psnr.append(psnr)
    
    # Results summary
    print(f"\n=== Reconstruction Results ===")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Final reconstruction loss: {stats['opt']:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    print(f"Feature MSE: {feat_mse:.4e}")
    print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
    print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    
    # Save reconstructed images
    os.makedirs('reconstructions_batch', exist_ok=True)
    
    # Save grid of reconstructed images
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(
        output_denorm,
        f'reconstructions_batch/epoch_{epoch}_reconstructed_grid.png',
        nrow=10
    )
    
    # Save comparison (first 20 images)
    comparison_count = min(20, NUM_IMAGES)
    original_subset = ground_truth[:comparison_count]
    reconstructed_subset = output[:comparison_count]
    
    # Create alternating grid: original, reconstructed, original, reconstructed...
    comparison = []
    for i in range(comparison_count):
        comparison.append(original_subset[i])
        comparison.append(reconstructed_subset[i])
    comparison = torch.stack(comparison)
    
    comparison_denorm = torch.clamp(comparison * ds + dm, 0, 1)
    torchvision.utils.save_image(
        comparison_denorm,
        f'reconstructions_batch/epoch_{epoch}_comparison.png',
        nrow=10  # 5 pairs per row
    )
    
    # Save metrics
    results = {
        'epoch': epoch,
        'psnr': test_psnr,
        'mse': test_mse,
        'feat_mse': feat_mse,
        'reconstruction_loss': stats['opt'],
        'attack_time': attack_time,
        'gradient_norm': epoch_data['total_norm'],
        'original_loss': epoch_data['loss'],
        'original_accuracy': epoch_data['accuracy'],
        'per_image_psnr': per_image_psnr
    }
    
    return results

def analyze_all_epochs():
    """Attack all saved epochs and compare results"""
    print("=== Batch Gradient Inversion Attack Analysis ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {NUM_IMAGES}")
    
    # Get list of saved epochs
    saved_epochs = []
    for file in os.listdir('single_batch_gradients'):
        if file.startswith('epoch_') and file.endswith('_gradient.pt'):
            epoch = int(file.split('_')[1])
            saved_epochs.append(epoch)
    saved_epochs.sort()
    
    print(f"Found gradients for epochs: {saved_epochs}")
    
    # Attack each epoch
    all_results = []
    for epoch in saved_epochs:
        try:
            results = attack_epoch(epoch)
            all_results.append(results)
        except Exception as e:
            print(f"Error attacking epoch {epoch}: {e}")
            continue
    
    # Save all results
    torch.save({
        'results': all_results,
        'attack_config': ATTACK_CONFIG,
        'model_name': MODEL_NAME,
        'num_images': NUM_IMAGES
    }, 'reconstructions_batch/all_results.pt')
    
    # Print summary table
    print(f"\n\n{'='*80}")
    print("=== SUMMARY: Reconstruction Quality vs Training Epoch ===")
    print(f"{'='*80}")
    print(f"{'Epoch':<8} {'PSNR (dB)':<12} {'MSE':<12} {'Grad Norm':<12} {'Time (min)':<12}")
    print(f"{'-'*60}")
    
    for r in all_results:
        print(f"{r['epoch']:<8} {r['psnr']:<12.2f} {r['mse']:<12.4f} "
              f"{r['gradient_norm']:<12.4f} {r['attack_time']/60:<12.1f}")
    
    # Analyze trend
    if len(all_results) > 1:
        early_psnr = [r['psnr'] for r in all_results if r['epoch'] <= 5]
        late_psnr = [r['psnr'] for r in all_results if r['epoch'] >= 20]
        
        if early_psnr and late_psnr:
            print(f"\n{'='*60}")
            print("=== ANALYSIS ===")
            print(f"Average PSNR (early epochs ≤5): {np.mean(early_psnr):.2f} dB")
            print(f"Average PSNR (late epochs ≥20): {np.mean(late_psnr):.2f} dB")
            print(f"PSNR decrease: {np.mean(early_psnr) - np.mean(late_psnr):.2f} dB")
            
            if np.mean(early_psnr) - np.mean(late_psnr) > 3:
                print("\n✅ Confirmed: Early training epochs are more vulnerable to gradient attacks!")
            else:
                print("\n⚠️  Minimal difference observed between early and late epochs")

if __name__ == "__main__":
    # Set random seed for reproducibility
    inversefed.utils.set_deterministic()
    
    # Run analysis
    analyze_all_epochs()
    
    print("\n✅ Analysis complete!")
    print("Check 'reconstructions_batch/' folder for reconstructed images")
    print("\nKey files:")
    print("  - epoch_X_reconstructed_grid.png: All 100 reconstructed images")
    print("  - epoch_X_comparison.png: Side-by-side comparison of first 20 images")
    print("  - all_results.pt: Detailed metrics for all epochs")