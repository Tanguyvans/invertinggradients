"""
Attack federated learning using weight updates from individual clients
Attack rounds 1 and 5 for each client
"""

import torch
import torchvision
import numpy as np
import os
import time
import sys
import inversefed

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'ConvNet'

# Attack configuration for federated learning
ATTACK_CONFIG = dict(
    signed=True,
    boxed=True,
    cost_fn='sim',
    indices='def',
    weights='equal',
    lr=1.0,  # Higher learning rate for reconstruction
    optim='adam',
    restarts=1,  # More restarts for better results
    max_iterations=2000,  # More iterations
    total_variation=1e-4,  # Adjusted TV regularization
    init='randn',
    filter='none',
    lr_decay=True,
    scoring_choice='loss'
)

def load_experiment_info():
    """Load experiment configuration"""
    return torch.load('federated_models/experiment_info.pt')

def load_client_ground_truth(client_data, client_id, validloader, setup):
    """Load ground truth images for a specific client"""
    indices = client_data[client_id]['indices']
    
    images = []
    labels = []
    for idx in indices:
        img, label = validloader.dataset[idx]
        images.append(img)
        labels.append(label)
    
    ground_truth = torch.stack(images).to(**setup)
    labels = torch.tensor(labels).to(device=setup['device'])
    
    return ground_truth, labels, indices

def attack_client_round(round_num, client_id, save_individual=True):
    """Attack a specific client's updates from a specific round"""
    print(f"\n{'='*70}")
    print(f"=== Attacking Round {round_num}, Client {client_id} ===")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # Load experiment info
    exp_info = load_experiment_info()
    LOCAL_LR = exp_info['local_lr']
    LOCAL_STEPS = exp_info['local_steps']
    IMAGES_PER_CLIENT = exp_info['images_per_client']
    
    # Load client update data
    try:
        client_data_file = f'client_updates/round_{round_num}_client_{client_id}.pt'
        client_update_data = torch.load(client_data_file)
    except FileNotFoundError:
        print(f"Client update file not found: {client_data_file}")
        return None
    
    # Load global model from the PREVIOUS round (what client started with)
    previous_round = round_num - 1
    global_model_data = torch.load(f'federated_models/global_model_round_{previous_round}.pt')
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load dataset
    print("Loading CIFAR-10 dataset...")
    loss_fn, _, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path='./datasets/cifar10'
    )
    
    # Create model and load global model state from previous round
    print(f"Setting up global model from round {previous_round}...")
    model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.load_state_dict(global_model_data['model_state'])
    model.to(**setup)
    model.eval()
    
    # Load ground truth for this client
    print(f"Loading ground truth for client {client_id}...")
    ground_truth, true_labels, indices = load_client_ground_truth(
        exp_info['client_data'], client_id, validloader, setup
    )
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Load client's weight updates
    weight_updates = [w.to(**setup) for w in client_update_data['weight_updates']]
    
    # Print attack info
    print(f"Round {round_num}, Client {client_id} Statistics:")
    print(f"  Images: {IMAGES_PER_CLIENT}")
    print(f"  Ground truth indices: {indices}")
    print(f"  Ground truth labels: {client_update_data['client_labels'].tolist()}")
    print(f"  Local LR: {LOCAL_LR}")
    print(f"  Local steps: {LOCAL_STEPS}")
    print(f"  Update norm: {client_update_data['update_norm']:.4f}")
    print(f"  Client accuracy: {client_update_data['accuracy']:.2f}%")
    
    print(f"\nStarting reconstruction with {ATTACK_CONFIG['restarts']} restarts...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Perform reconstruction using FedAvgReconstructor
    rec_machine = inversefed.FedAvgReconstructor(
        model, (dm, ds), LOCAL_STEPS, LOCAL_LR, ATTACK_CONFIG, 
        num_images=IMAGES_PER_CLIENT, use_updates=True
    )
    output, stats = rec_machine.reconstruct(weight_updates, true_labels, img_shape=(3, 32, 32))
    
    attack_time = time.time() - start_time
    
    # Calculate metrics
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    
    # Per-image PSNR
    per_image_psnr = []
    for i in range(IMAGES_PER_CLIENT):
        psnr = inversefed.metrics.psnr(
            output[i:i+1], 
            ground_truth[i:i+1], 
            factor=1/ds
        )
        per_image_psnr.append(psnr)
    
    # Results
    print(f"\n=== Attack Results ===")
    print(f"Time: {attack_time:.1f} seconds ({attack_time/60:.1f} minutes)")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    print(f"Best image PSNR: {max(per_image_psnr):.2f} dB")
    print(f"Worst image PSNR: {min(per_image_psnr):.2f} dB")
    print(f"MSE: {test_mse:.4f}")
    print(f"Feature MSE: {feat_mse:.4e}")
    sys.stdout.flush()
    
    # Create results directory
    results_dir = f'federated_attacks'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save reconstructed images
    output_denorm = torch.clamp(output * ds + dm, 0, 1)
    torchvision.utils.save_image(
        output_denorm,
        f'{results_dir}/round_{round_num}_client_{client_id}_reconstructed.png',
        nrow=5
    )
    
    # Save comparison (original on top, reconstructed on bottom)
    comparison = torch.cat([ground_truth, output], dim=0)
    comparison_denorm = torch.clamp(comparison * ds + dm, 0, 1)
    torchvision.utils.save_image(
        comparison_denorm,
        f'{results_dir}/round_{round_num}_client_{client_id}_comparison.png',
        nrow=IMAGES_PER_CLIENT
    )
    
    print(f"Saved results to {results_dir}/round_{round_num}_client_{client_id}_*")
    
    # Save individual images if requested
    if save_individual:
        for i in range(IMAGES_PER_CLIENT):
            # Original
            orig = torch.clamp(ground_truth[i] * ds + dm, 0, 1)
            torchvision.utils.save_image(
                orig,
                f'{results_dir}/round_{round_num}_client_{client_id}_img_{i}_original.png'
            )
            # Reconstructed
            recon = torch.clamp(output[i] * ds + dm, 0, 1)
            torchvision.utils.save_image(
                recon,
                f'{results_dir}/round_{round_num}_client_{client_id}_img_{i}_reconstructed.png'
            )
    
    sys.stdout.flush()
    
    return {
        'round': round_num,
        'client_id': client_id,
        'psnr': test_psnr,
        'mse': test_mse,
        'feat_mse': feat_mse,
        'attack_time': attack_time,
        'update_norm': client_update_data['update_norm'],
        'client_accuracy': client_update_data['accuracy'],
        'reconstruction_loss': stats['opt'],
        'per_image_psnr': per_image_psnr,
        'local_lr': LOCAL_LR,
        'local_steps': LOCAL_STEPS,
        'indices': indices
    }

def main():
    """Attack federated learning - rounds 1, 2, and 5 for client 0"""
    print("=== Federated Learning Attack ===")
    print(f"Device: {DEVICE}")
    
    # Attack rounds 1, 2, and 5 on client 0 to show vulnerability progression
    rounds_to_attack = [1, 2, 5]
    target_client = 0  # Focus on client 0
    
    # Load experiment info
    exp_info = load_experiment_info()
    NUM_CLIENTS = exp_info['num_clients']
    
    print(f"Attacking client {target_client} on rounds {rounds_to_attack} to show vulnerability progression")
    print(f"Attack configuration:")
    print(f"  Restarts: {ATTACK_CONFIG['restarts']}")
    print(f"  Max iterations: {ATTACK_CONFIG['max_iterations']}")
    print(f"  Total variation: {ATTACK_CONFIG['total_variation']}")
    sys.stdout.flush()
    
    all_results = []
    
    for round_num in rounds_to_attack:
        print(f"\n{'='*60}")
        print(f"ATTACKING ROUND {round_num}, CLIENT {target_client}")
        print(f"{'='*60}")
        try:
            result = attack_client_round(round_num, target_client)
            if result:
                all_results.append(result)
                    
                # Save intermediate results
                torch.save({
                    'results': all_results,
                    'attack_config': ATTACK_CONFIG,
                    'experiment_info': exp_info,
                    'method': 'federated_learning_attack'
                }, 'federated_attacks/attack_results.pt')
            
        except Exception as e:
            print(f"Error attacking round {round_num}, client {target_client}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print final summary
    print(f"\n{'='*90}")
    print("=== FINAL FEDERATED ATTACK SUMMARY ===")
    print(f"{'='*90}")
    print(f"{'Round':<8} {'Client':<8} {'PSNR (dB)':<12} {'Time (min)':<12} {'Update Norm':<12} {'Client Acc':<12}")
    print(f"{'-'*90}")
    
    for r in all_results:
        print(f"{r['round']:<8} {r['client_id']:<8} {r['psnr']:<12.2f} {r['attack_time']/60:<12.1f} "
              f"{r['update_norm']:<12.4f} {r['client_accuracy']:<12.2f}")
    
    # Analysis by round progression
    if len(all_results) >= 2:
        # Sort by round
        sorted_results = sorted(all_results, key=lambda x: x['round'])
        
        print(f"\n=== VULNERABILITY PROGRESSION ANALYSIS ===")
        print("Expected: Earlier rounds should be more vulnerable")
        
        for i, r in enumerate(sorted_results):
            print(f"Round {r['round']}: PSNR = {r['psnr']:.2f} dB, Update norm = {r['update_norm']:.4f}")
        
        # Compare first and last rounds
        first_round = sorted_results[0]
        last_round = sorted_results[-1]
        
        psnr_diff = first_round['psnr'] - last_round['psnr']
        print(f"\nPSNR difference (Round {first_round['round']} - Round {last_round['round']}): {psnr_diff:.2f} dB")
        
        if psnr_diff > 3:
            print("✅ CONFIRMED: Early rounds are significantly more vulnerable!")
        elif psnr_diff > 1:
            print("📈 Early rounds are somewhat more vulnerable")
        elif psnr_diff < -1:
            print("⚠️  Later rounds appear more vulnerable (unexpected)")
        else:
            print("📊 Similar vulnerability across training progression")
    
    print(f"\n✅ Federated learning attack complete!")
    print("Check 'federated_attacks/' for all results")
    print("This demonstrates privacy vulnerabilities in federated learning!")
    sys.stdout.flush()

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    main()