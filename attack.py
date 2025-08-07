"""
Attack using fresh gradients computed on the spot (like the notebook)
This should give much better reconstruction results
"""

import torch
import torchvision
import numpy as np
import os
import time
import sys
import inversefed

# Force unbuffered output for nohup
os.environ['PYTHONUNBUFFERED'] = '1'

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'ConvNet'
NUM_IMAGES = 10

# Attack configuration optimized for 10 images
ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=0.1,
    optim='adam',
    restarts=2,          # Reduced restarts for faster execution
    max_iterations=12000,   # Reduced iterations for faster execution
    total_variation=1e-2,  # Same as notebook for smoothness
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

def load_ground_truth(validloader):
    """Load the original batch of 10 images"""
    batch_info = torch.load('gradients/batch_info.pt')
    batch_indices = batch_info['batch_indices']
    
    images = []
    labels = []
    for idx in batch_indices:
        img, label = validloader.dataset[idx]
        images.append(img)
        labels.append(label)
    
    ground_truth = torch.stack(images)
    labels = torch.tensor(labels)
    
    return ground_truth, labels, batch_indices

def attack_epoch_fresh(epoch, save_individual=True):
    """Perform gradient inversion attack using fresh gradients (like notebook)"""
    print(f"\n{'='*60}")
    print(f"=== Attacking Epoch {epoch} with FRESH gradients ===")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    # Load saved epoch data (for model state and stats)
    epoch_data = torch.load(f'gradients/epoch_{epoch}_gradient.pt')
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path='./datasets/cifar10'
    )
    
    # Get ground truth
    print("Loading ground truth images...")
    ground_truth, true_labels, _ = load_ground_truth(validloader)
    ground_truth = ground_truth.to(**setup)
    true_labels = true_labels.to(device=setup['device'])
    
    # Create model and load the EXACT state from that epoch
    print(f"Setting up model from epoch {epoch}...")
    model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.load_state_dict(epoch_data['model_state'])  # Load model from that specific epoch
    model.to(**setup)
    model.eval()
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Print epoch statistics
    print(f"Epoch {epoch} Statistics:")
    print(f"  Original loss: {epoch_data['loss']:.4f}")
    print(f"  Original accuracy: {epoch_data['accuracy']:.2f}%")
    print(f"  Saved gradient norm: {epoch_data['total_norm']:.4f}")
    
    # *** KEY DIFFERENCE: Compute FRESH gradients like the notebook ***
    print(f"\nComputing FRESH gradients on model from epoch {epoch}...")
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), true_labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]  # Detach like notebook
    
    # Calculate fresh gradient norm for comparison
    fresh_gradient_norm = torch.stack([g.norm() for g in input_gradient]).mean().item()
    print(f"  Fresh gradient norm: {fresh_gradient_norm:.4f}")
    print(f"  Fresh loss: {target_loss.item():.4f}")
    sys.stdout.flush()
    
    print(f"\nStarting reconstruction with {ATTACK_CONFIG['restarts']} restarts...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Perform reconstruction using FRESH gradients
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), ATTACK_CONFIG, num_images=NUM_IMAGES)
    output, stats = rec_machine.reconstruct(input_gradient, true_labels, img_shape=(3, 32, 32))
    
    attack_time = time.time() - start_time
    
    # Calculate metrics
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
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
    
    # Results
    print(f"\n=== Results for Epoch {epoch} ===")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
    print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    print(f"MSE: {test_mse:.4f}")
    sys.stdout.flush()
    
    # Save reconstructed images
    os.makedirs('reconstructions', exist_ok=True)
    
    # Save grid
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(
        output_denorm,
        f'reconstructions/epoch_{epoch}_reconstructed.png',
        nrow=5
    )
    print(f"Saved reconstructed grid to reconstructions/epoch_{epoch}_reconstructed.png")
    
    # Save comparison (original on top, reconstructed on bottom)
    comparison = torch.cat([ground_truth, output], dim=0)
    comparison_denorm = torch.clamp(comparison * ds + dm, 0, 1)
    torchvision.utils.save_image(
        comparison_denorm,
        f'reconstructions/epoch_{epoch}_comparison.png',
        nrow=10  # Top row: original, Bottom row: reconstructed
    )
    print(f"Saved comparison to reconstructions/epoch_{epoch}_comparison.png")
    
    # Save individual images if requested
    if save_individual:
        for i in range(NUM_IMAGES):
            # Original
            orig = torch.clamp(ground_truth[i] * ds + dm, 0, 1)
            torchvision.utils.save_image(
                orig,
                f'reconstructions/epoch_{epoch}_img_{i}_original.png'
            )
            # Reconstructed
            recon = torch.clamp(output[i] * ds + dm, 0, 1)
            torchvision.utils.save_image(
                recon,
                f'reconstructions/epoch_{epoch}_img_{i}_reconstructed.png'
            )
        print(f"Saved individual images for epoch {epoch}")
    
    sys.stdout.flush()
    
    return {
        'epoch': epoch,
        'psnr': test_psnr,
        'mse': test_mse,
        'attack_time': attack_time,
        'fresh_gradient_norm': fresh_gradient_norm,
        'fresh_loss': target_loss.item(),
        'reconstruction_loss': stats['opt'],
        'per_image_psnr': per_image_psnr
    }

def main():
    """Attack all specified epochs using fresh gradients"""
    print("=== Fresh Gradient Inversion Attack on 10 Images ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Using FRESH gradients computed on-the-fly (like notebook)")
    print(f"Attack configuration:")
    print(f"  Restarts: {ATTACK_CONFIG['restarts']}")
    print(f"  Max iterations: {ATTACK_CONFIG['max_iterations']}")
    print(f"  Total variation: {ATTACK_CONFIG['total_variation']}")
    sys.stdout.flush()
    
    # Attack epochs 1, 2, 5, and 10
    epochs_to_attack = [1, 2, 5, 10]
    results = []
    
    for epoch in epochs_to_attack:
        try:
            result = attack_epoch_fresh(epoch)
            results.append(result)
            
            # Save intermediate results
            torch.save({
                'results': results,
                'attack_config': ATTACK_CONFIG,
                'model_name': MODEL_NAME,
                'num_images': NUM_IMAGES,
                'method': 'fresh_gradients'
            }, 'reconstructions_10_images_fresh/results_fresh.pt')
            
        except FileNotFoundError:
            print(f"Gradient file for epoch {epoch} not found. Skipping...")
            continue
        except Exception as e:
            print(f"Error attacking epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print(f"\n{'='*80}")
    print("=== FINAL SUMMARY (Fresh Gradients Method) ===")
    print(f"{'='*80}")
    print(f"{'Epoch':<8} {'PSNR (dB)':<12} {'Time (min)':<12} {'Fresh Grad Norm':<15} {'Fresh Loss':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['epoch']:<8} {r['psnr']:<12.2f} {r['attack_time']/60:<12.1f} "
              f"{r['fresh_gradient_norm']:<15.4f} {r['fresh_loss']:<12.4f}")
    
    # Analysis
    if len(results) > 1:
        early_psnr = [r['psnr'] for r in results if r['epoch'] <= 2]
        late_psnr = [r['psnr'] for r in results if r['epoch'] >= 10]
        
        if early_psnr and late_psnr:
            print(f"\nAverage PSNR (epochs 1-2): {np.mean(early_psnr):.2f} dB")
            print(f"Average PSNR (epoch 10): {np.mean(late_psnr):.2f} dB")
            print(f"PSNR difference: {np.mean(early_psnr) - np.mean(late_psnr):.2f} dB")
            
            if np.mean(early_psnr) - np.mean(late_psnr) > 3:
                print("\n✅ Early epochs are more vulnerable to gradient attacks!")
            else:
                print("\n⚠️  Similar vulnerability across epochs")
    
    print(f"\n✅ Fresh gradient attack complete!")
    print("Check 'reconstructions_10_images_fresh/' for results")
    print("This method computes gradients fresh like the original notebook.")
    sys.stdout.flush()

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    main()