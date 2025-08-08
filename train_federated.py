"""
Federated Learning Training with 2 clients, 10 images each
Save global models and client weight updates for each round
"""

import torch
import numpy as np
import os
import inversefed
from collections import OrderedDict
import copy
import torchvision.transforms as transforms

# Configuration
MODEL_NAME = 'ConvNet'
DATASET = 'CIFAR10'
NUM_CLIENTS = 2
IMAGES_PER_CLIENT = 50  # Increased from 10 to 50
NUM_ROUNDS = 5  # More rounds for better convergence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Federated learning parameters
LOCAL_LR = 0.1  # Higher learning rate for 50 images
LOCAL_EPOCHS = 5  # Number of local training epochs before aggregation
GRADIENT_CLIP = 5.0  # Gradient clipping value

def select_images_per_client(dataset, num_clients=2, images_per_client=10):
    """Select different images for each client with balanced label distribution"""
    np.random.seed(42)  # For reproducible client data splits
    
    # Group indices by label
    label_to_indices = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    # Shuffle indices within each label
    for label in label_to_indices:
        np.random.shuffle(label_to_indices[label])
    
    client_data = {}
    
    # Distribute samples to ensure each client gets diverse labels
    for client_id in range(num_clients):
        client_indices = []
        client_labels = []
        
        # Try to get at least one sample from each class
        labels_per_client = min(images_per_client, len(label_to_indices))
        samples_per_label = max(1, images_per_client // labels_per_client)
        
        # First pass: get samples from different labels
        for label_idx, (label, indices) in enumerate(label_to_indices.items()):
            if len(client_indices) < images_per_client:
                # Take samples from this label for this client
                start_idx = client_id * samples_per_label
                end_idx = start_idx + samples_per_label
                
                for idx in indices[start_idx:end_idx]:
                    if len(client_indices) < images_per_client:
                        client_indices.append(idx)
                        client_labels.append(label)
        
        # Second pass: fill remaining slots with random samples
        all_indices = [idx for indices in label_to_indices.values() for idx in indices]
        np.random.shuffle(all_indices)
        
        for idx in all_indices:
            if len(client_indices) < images_per_client and idx not in client_indices:
                _, label = dataset[idx]
                client_indices.append(idx)
                client_labels.append(label)
        
        client_data[client_id] = {
            'indices': client_indices[:images_per_client],
            'labels': client_labels[:images_per_client]
        }
        
        print(f"Client {client_id}:")
        print(f"  Indices: {client_indices[:images_per_client]}")
        print(f"  Labels: {sorted(client_labels[:images_per_client])}")
        print(f"  Unique labels: {len(set(client_labels[:images_per_client]))}")
        print()
    
    return client_data

def get_client_batch(dataset, client_data, client_id, setup, augment=True):
    """Extract batch for a specific client with optional augmentation"""
    indices = client_data[client_id]['indices']
    
    # Data augmentation for training
    if augment:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform = None
    
    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        
        # Apply augmentation if enabled
        if transform is not None:
            # Convert to PIL for transforms
            img_pil = transforms.ToPILImage()(img)
            img = transform(img_pil)
            img = transforms.ToTensor()(img)
        
        images.append(img)
        labels.append(label)
    
    batch_images = torch.stack(images).to(**setup)
    batch_labels = torch.tensor(labels).to(device=setup['device'], dtype=torch.long)
    
    return batch_images, batch_labels

def federated_averaging(client_updates):
    """Perform federated averaging of client updates"""
    if len(client_updates) == 0:
        return None
    
    # Initialize averaged update with zeros like the first client
    avg_update = []
    for param in client_updates[0]:
        avg_update.append(torch.zeros_like(param))
    
    # Sum all client updates
    for client_update in client_updates:
        for i, param in enumerate(client_update):
            avg_update[i] += param
    
    # Average by number of clients
    for i in range(len(avg_update)):
        avg_update[i] /= len(client_updates)
    
    return avg_update

def apply_update_to_model(model, update):
    """Apply federated update to global model"""
    with torch.no_grad():
        for param, update_val in zip(model.parameters(), update):
            param.add_(update_val)

def train_federated():
    """Train federated learning with multiple clients"""
    print("=== Federated Learning Training ===")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Images per client: {IMAGES_PER_CLIENT}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"Local LR: {LOCAL_LR}, Local Epochs: {LOCAL_EPOCHS}")
    print()
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(
        DATASET, defs, data_path='./datasets/cifar10'
    )
    
    # Split data among clients
    client_data = select_images_per_client(validloader.dataset, NUM_CLIENTS, IMAGES_PER_CLIENT)
    
    # Create initial global model
    global_model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    global_model.to(**setup)
    
    print(f"Model: {MODEL_NAME}")
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create directories
    os.makedirs('federated_models', exist_ok=True)
    os.makedirs('client_updates', exist_ok=True)
    
    # Save initial global model and client data info
    torch.save({
        'model_state': global_model.state_dict(),
        'round': 0,
        'model_name': MODEL_NAME,
        'num_clients': NUM_CLIENTS,
        'images_per_client': IMAGES_PER_CLIENT,
        'local_lr': LOCAL_LR,
        'local_epochs': LOCAL_EPOCHS,
        'local_steps': LOCAL_EPOCHS
    }, 'federated_models/global_model_round_0.pt')
    
    torch.save({
        'client_data': client_data,
        'model_name': MODEL_NAME,
        'dataset': DATASET,
        'num_clients': NUM_CLIENTS,
        'images_per_client': IMAGES_PER_CLIENT,
        'local_lr': LOCAL_LR,
        'local_epochs': LOCAL_EPOCHS,
        'local_steps': LOCAL_EPOCHS,
        'num_rounds': NUM_ROUNDS
    }, 'federated_models/experiment_info.pt')
    
    print("Starting federated learning rounds...")
    print("=" * 50)
    
    # Federated learning rounds
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")
        
        client_updates = []
        round_stats = {'clients': {}}
        
        # Each client trains locally
        for client_id in range(NUM_CLIENTS):
            print(f"\nClient {client_id} training...")
            
            # Create local copy of global model
            local_model = copy.deepcopy(global_model)
            
            # Get client's data batch (no augmentation for cleaner gradients)
            client_images, client_labels = get_client_batch(
                validloader.dataset, client_data, client_id, setup, augment=False
            )
            
            # Store original parameters
            original_params = [p.clone().detach() for p in local_model.parameters()]
            
            # Check initial loss
            local_model.eval()
            with torch.no_grad():
                initial_outputs = local_model(client_images)
                initial_loss, _, _ = loss_fn(initial_outputs, client_labels)
                print(f"  Initial loss: {initial_loss.item():.4f}")
            
            # Local training with SGD for multiple epochs (single batch)
            local_model.train()
            # Use fixed learning rate (no decay for now)
            current_lr = LOCAL_LR  # Keep constant for debugging
            print(f"  Using LR: {current_lr}")
            optimizer = torch.optim.SGD(local_model.parameters(), lr=current_lr, momentum=0.9, weight_decay=5e-4)
            
            for epoch in range(LOCAL_EPOCHS):
                optimizer.zero_grad()
                outputs = local_model(client_images)
                loss, _, _ = loss_fn(outputs, client_labels)
                
                # Check for NaN or very large loss
                if torch.isnan(loss) or loss.item() > 1e6:
                    print(f"  WARNING: Loss explosion detected ({loss.item():.2e}), skipping update")
                    break
                
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), GRADIENT_CLIP)
                
                optimizer.step()
                
                if epoch == LOCAL_EPOCHS - 1:  # Last epoch
                    print(f"  Final epoch loss: {loss.item():.4f}")
            
            # Calculate weight updates (difference between trained and original)
            weight_updates = []
            with torch.no_grad():
                for new_param, old_param in zip(local_model.parameters(), original_params):
                    update = new_param - old_param
                    weight_updates.append(update)
                    # Debug: check if updates are meaningful
                    if len(weight_updates) == 1:  # First layer only
                        print(f"  First layer update max: {update.abs().max().item():.6f}")
            
            # Calculate statistics
            update_norm = torch.stack([w.norm() for w in weight_updates]).mean().item()
            
            # Check if any updates are non-zero
            non_zero_updates = sum(1 for w in weight_updates if w.abs().max().item() > 1e-8)
            print(f"  Non-zero update tensors: {non_zero_updates}/{len(weight_updates)}")
            
            # Test accuracy on client's data
            local_model.eval()
            with torch.no_grad():
                outputs = local_model(client_images)
                _, predicted = outputs.max(1)
                correct = predicted.eq(client_labels).sum().item()
                accuracy = 100. * correct / len(client_labels)
            
            print(f"  Update norm: {update_norm:.4f}")
            print(f"  Local accuracy: {accuracy:.2f}%")
            
            # Save client update
            client_updates.append(weight_updates)
            round_stats['clients'][client_id] = {
                'update_norm': update_norm,
                'accuracy': accuracy,
                'indices': client_data[client_id]['indices'],
                'labels': client_data[client_id]['labels']
            }
            
            # Save individual client update
            torch.save({
                'round': round_num,
                'client_id': client_id,
                'weight_updates': [w.cpu() for w in weight_updates],
                'original_params': [p.cpu() for p in original_params],
                'update_norm': update_norm,
                'accuracy': accuracy,
                'client_images': client_images.cpu(),
                'client_labels': client_labels.cpu(),
                'local_lr': LOCAL_LR,
                'local_epochs': LOCAL_EPOCHS
            }, f'client_updates/round_{round_num}_client_{client_id}.pt')
        
        # Federated averaging
        print(f"\nPerforming federated averaging...")
        avg_update = federated_averaging(client_updates)
        avg_update_norm = torch.stack([w.norm() for w in avg_update]).mean().item()
        print(f"Average update norm: {avg_update_norm:.4f}")
        
        # Apply averaged update to global model
        print(f"Applying update to global model...")
        apply_update_to_model(global_model, avg_update)
        
        # Check if global model actually changed
        with torch.no_grad():
            test_param = next(global_model.parameters())
            print(f"Global model first param mean: {test_param.mean().item():.6f}")
        
        # Test global model accuracy on all client data
        global_model.eval()
        total_correct = 0
        total_samples = 0
        
        for client_id in range(NUM_CLIENTS):
            client_images, client_labels = get_client_batch(
                validloader.dataset, client_data, client_id, setup, augment=False
            )
            
            with torch.no_grad():
                outputs = global_model(client_images)
                _, predicted = outputs.max(1)
                correct = predicted.eq(client_labels).sum().item()
                total_correct += correct
                total_samples += len(client_labels)
        
        global_accuracy = 100. * total_correct / total_samples
        print(f"Global model accuracy: {global_accuracy:.2f}%")
        
        # Save global model and round statistics
        round_stats['round'] = round_num
        round_stats['avg_update_norm'] = avg_update_norm
        round_stats['global_accuracy'] = global_accuracy
        
        torch.save({
            'model_state': global_model.state_dict(),
            'round': round_num,
            'avg_update': [w.cpu() for w in avg_update],
            'avg_update_norm': avg_update_norm,
            'global_accuracy': global_accuracy,
            'round_stats': round_stats
        }, f'federated_models/global_model_round_{round_num}.pt')
    
    print(f"\n✅ Federated learning complete!")
    print(f"Global models saved in 'federated_models/' for rounds 0-{NUM_ROUNDS}")
    print(f"Client updates saved in 'client_updates/' for each round")
    print("Ready for federated learning attacks!")

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    train_federated()