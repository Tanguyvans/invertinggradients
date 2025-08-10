"""
Smart Federated Attack - Prioritizes gradients with highest vulnerability
Targets rounds/clients with highest loss and gradient norms for best reconstruction
"""

import torch
import torchvision
import numpy as np
import os
import time
import sys
import inversefed
from collections import defaultdict

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'ConvNet'
NUM_IMAGES = 10
NUM_CLIENTS = 2

# Enhanced attack config for vulnerable gradients
SMART_ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=0.15,  # Higher LR for high-loss gradients
    optim='adam',
    restarts=3,  # More restarts for better results
    max_iterations=30000,  # More iterations for high-loss cases
    total_variation=1e-2,
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

# Standard config for low-vulnerability gradients
STANDARD_ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=0.1,
    optim='adam',
    restarts=2,
    max_iterations=24000,
    total_variation=1e-2,
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

def analyze_gradient_vulnerability():
    """Analyze all available gradients to find most vulnerable targets"""
    vulnerability_scores = []
    available_files = []
    
    for round_num in [1, 5, 10]:
        for client_id in [0, 1]:
            filename = f'federated_simple_gradients/round_{round_num}_client_{client_id}.pt'
            if os.path.exists(filename):
                try:
                    data = torch.load(filename)
                    
                    # Calculate gradient norm
                    gradients = data['gradients']
                    grad_norm = torch.stack([g.norm() for g in gradients]).mean().item()
                    
                    # Vulnerability score = loss * gradient_norm (higher = more vulnerable)
                    vulnerability = data['loss'] * grad_norm
                    
                    vulnerability_scores.append({
                        'round': round_num,
                        'client_id': client_id,
                        'loss': data['loss'],
                        'accuracy': data['accuracy'],
                        'grad_norm': grad_norm,
                        'vulnerability_score': vulnerability,
                        'filename': filename
                    })
                    available_files.append(filename)
                    
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")
                    continue
    
    # Sort by vulnerability score (highest first)
    vulnerability_scores.sort(key=lambda x: x['vulnerability_score'], reverse=True)
    
    return vulnerability_scores

def select_attack_config(vulnerability_score, loss):
    """Select attack configuration based on vulnerability"""
    if vulnerability_score > 5.0 or loss > 3.0:  # High vulnerability
        return SMART_ATTACK_CONFIG, "SMART"
    else:
        return STANDARD_ATTACK_CONFIG, "STANDARD"

def load_federated_data(round_num, client_id):
    """Load federated gradient data for specific round and client"""
    filename = f'federated_simple_gradients/round_{round_num}_client_{client_id}.pt'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Gradient file not found: {filename}")
    
    data = torch.load(filename)
    return data

def load_ground_truth_from_federated(round_num, client_id, validloader):
    """Load the original batch used by the federated client"""
    fed_data = load_federated_data(round_num, client_id)
    batch_indices = fed_data['batch_indices']
    
    images = []
    labels = []
    for idx in batch_indices:
        img, label = validloader.dataset[idx]
        images.append(img)
        labels.append(label)
    
    ground_truth = torch.stack(images)
    labels = torch.tensor(labels)
    
    return ground_truth, labels, batch_indices

def smart_attack_client(target, save_individual=True):
    """Smart attack on a specific target with optimized config"""
    round_num = target['round']
    client_id = target['client_id']
    vulnerability = target['vulnerability_score']
    
    # Select attack configuration based on vulnerability
    attack_config, config_type = select_attack_config(vulnerability, target['loss'])
    
    print(f"\n{'='*80}")
    print(f"=== SMART ATTACK: Round {round_num}, Client {client_id} ===")
    print(f"Vulnerability Score: {vulnerability:.4f} (Loss: {target['loss']:.4f}, Grad Norm: {target['grad_norm']:.4f})")
    print(f"Attack Config: {config_type}")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path='./datasets/cifar10'
    )
    
    # Load federated data
    print(f"Loading federated data for round {round_num}, client {client_id}...")
    fed_data = load_federated_data(round_num, client_id)
    
    # Get ground truth for this client
    print("Loading ground truth images...")
    ground_truth, true_labels, batch_indices = load_ground_truth_from_federated(
        round_num, client_id, validloader
    )
    ground_truth = ground_truth.to(**setup)
    true_labels = true_labels.to(device=setup['device'])
    
    # Create model and load the state from federated training
    print(f"Setting up model from round {round_num}, client {client_id}...")
    model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.load_state_dict(fed_data['model_state'])
    model.to(**setup)
    model.eval()
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Load saved gradients
    saved_gradients = [g.to(**setup) for g in fed_data['gradients']]
    
    print(f"Target Statistics:")
    print(f"  Batch indices: {batch_indices}")
    print(f"  Classes: {sorted(true_labels.cpu().tolist())}")
    print(f"  Loss: {fed_data['loss']:.4f}")
    print(f"  Accuracy: {fed_data['accuracy']:.2f}%")
    print(f"  Gradient norm: {target['grad_norm']:.4f}")
    print(f"  Vulnerability: {vulnerability:.4f}")
    
    # Print attack configuration
    print(f"\nAttack Configuration ({config_type}):")
    print(f"  Learning rate: {attack_config['lr']}")
    print(f"  Restarts: {attack_config['restarts']}")
    print(f"  Max iterations: {attack_config['max_iterations']}")
    sys.stdout.flush()
    
    print(f"\nStarting smart reconstruction...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Perform reconstruction using saved gradients with smart config
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), attack_config, num_images=NUM_IMAGES)
    output, stats = rec_machine.reconstruct(saved_gradients, true_labels, img_shape=(3, 32, 32))
    
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
    print(f"\n=== SMART ATTACK RESULTS ===")
    print(f"Round {round_num}, Client {client_id} ({config_type} config)")
    print(f"Vulnerability Score: {vulnerability:.4f}")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
    print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    print(f"MSE: {test_mse:.4f}")
    
    # Quality assessment
    if test_psnr > 25:
        print("🎯 EXCELLENT reconstruction!")
    elif test_psnr > 20:
        print("✅ GOOD reconstruction!")
    elif test_psnr > 15:
        print("⚠️  Fair reconstruction")
    else:
        print("❌ Poor reconstruction")
    
    sys.stdout.flush()
    
    # Save reconstructed images
    os.makedirs('smart_federated_reconstructions', exist_ok=True)
    
    # Save grid with vulnerability info
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(
        output_denorm,
        f'smart_federated_reconstructions/vuln_{vulnerability:.2f}_round_{round_num}_client_{client_id}_reconstructed.png',
        nrow=5
    )
    
    # Save comparison
    comparison = torch.cat([ground_truth, output], dim=0)
    comparison_denorm = torch.clamp(comparison * ds + dm, 0, 1)
    torchvision.utils.save_image(
        comparison_denorm,
        f'smart_federated_reconstructions/vuln_{vulnerability:.2f}_round_{round_num}_client_{client_id}_comparison.png',
        nrow=10
    )
    
    print(f"Saved to smart_federated_reconstructions/vuln_{vulnerability:.2f}_round_{round_num}_client_{client_id}_*.png")
    sys.stdout.flush()
    
    return {
        'round': round_num,
        'client_id': client_id,
        'vulnerability_score': vulnerability,
        'config_type': config_type,
        'psnr': test_psnr,
        'mse': test_mse,
        'attack_time': attack_time,
        'gradient_norm': target['grad_norm'],
        'loss': fed_data['loss'],
        'reconstruction_loss': stats['opt'],
        'per_image_psnr': per_image_psnr,
        'batch_indices': batch_indices
    }

def main():
    """Smart federated attack targeting most vulnerable gradients"""
    print("=== SMART FEDERATED GRADIENT ATTACK ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print("Analyzing gradient vulnerability and optimizing attack strategy...")
    sys.stdout.flush()
    
    # Analyze vulnerability of all available gradients
    print("\n=== VULNERABILITY ANALYSIS ===")
    vulnerability_data = analyze_gradient_vulnerability()
    
    if not vulnerability_data:
        print("No gradient files found! Run federated_simple.py first.")
        return
    
    print(f"{'Rank':<5} {'Round':<6} {'Client':<7} {'Loss':<8} {'Acc%':<6} {'Grad Norm':<12} {'Vulnerability':<13} {'Priority':<10}")
    print(f"{'-'*85}")
    
    for i, target in enumerate(vulnerability_data):
        config_type = "SMART" if target['vulnerability_score'] > 5.0 or target['loss'] > 3.0 else "STANDARD"
        print(f"{i+1:<5} {target['round']:<6} {target['client_id']:<7} {target['loss']:<8.4f} "
              f"{target['accuracy']:<6.1f} {target['grad_norm']:<12.4f} {target['vulnerability_score']:<13.4f} {config_type:<10}")
    
    # Attack all targets, prioritizing by vulnerability
    print(f"\n=== SMART ATTACK EXECUTION ===")
    results = []
    
    for i, target in enumerate(vulnerability_data):
        try:
            print(f"\n[{i+1}/{len(vulnerability_data)}] Attacking target...")
            result = smart_attack_client(target)
            results.append(result)
            
            # Save intermediate results
            torch.save({
                'results': results,
                'vulnerability_analysis': vulnerability_data,
                'smart_config': SMART_ATTACK_CONFIG,
                'standard_config': STANDARD_ATTACK_CONFIG,
                'model_name': MODEL_NAME,
                'num_images': NUM_IMAGES,
                'method': 'smart_federated_attack'
            }, 'smart_federated_reconstructions/smart_attack_results.pt')
            
        except Exception as e:
            print(f"Error attacking round {target['round']}, client {target['client_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final analysis
    print(f"\n{'='*90}")
    print("=== SMART ATTACK SUMMARY ===")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Round':<6} {'Client':<7} {'Vuln':<10} {'Config':<9} {'PSNR':<8} {'Time(m)':<8} {'Quality':<12}")
    print(f"{'-'*90}")
    
    for i, r in enumerate(results):
        quality = "EXCELLENT" if r['psnr'] > 25 else "GOOD" if r['psnr'] > 20 else "FAIR" if r['psnr'] > 15 else "POOR"
        print(f"{i+1:<5} {r['round']:<6} {r['client_id']:<7} {r['vulnerability_score']:<10.4f} "
              f"{r['config_type']:<9} {r['psnr']:<8.2f} {r['attack_time']/60:<8.1f} {quality:<12}")
    
    if results:
        # Best reconstruction
        best_result = max(results, key=lambda x: x['psnr'])
        print(f"\n🏆 BEST RECONSTRUCTION:")
        print(f"   Round {best_result['round']}, Client {best_result['client_id']}")
        print(f"   Vulnerability: {best_result['vulnerability_score']:.4f}")
        print(f"   PSNR: {best_result['psnr']:.2f} dB")
        print(f"   Config: {best_result['config_type']}")
        
        # Vulnerability vs PSNR correlation
        high_vuln = [r for r in results if r['vulnerability_score'] > 5.0]
        if high_vuln:
            avg_psnr_high = np.mean([r['psnr'] for r in high_vuln])
            low_vuln = [r for r in results if r['vulnerability_score'] <= 5.0]
            if low_vuln:
                avg_psnr_low = np.mean([r['psnr'] for r in low_vuln])
                print(f"\n📊 VULNERABILITY EFFECTIVENESS:")
                print(f"   High vulnerability (>5.0): Avg PSNR = {avg_psnr_high:.2f} dB")
                print(f"   Low vulnerability (≤5.0): Avg PSNR = {avg_psnr_low:.2f} dB")
                print(f"   Smart targeting gain: {avg_psnr_high - avg_psnr_low:.2f} dB")
    
    print(f"\n✅ Smart federated attack complete!")
    print("Check 'smart_federated_reconstructions/' for results")
    print("Files are named with vulnerability scores for easy identification")
    sys.stdout.flush()

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    main()