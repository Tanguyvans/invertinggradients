"""
Train ResNet32-10 on a single batch of 100 CIFAR-10 images and save gradients at different epochs
This matches the setup from ResNet32-10 - Recovering 100 CIFAR-100 images.ipynb
"""

import torch
import torchvision
import numpy as np
import os
import time

import inversefed

# Configuration matching the notebook
MODEL_NAME = 'ResNet32-10'
DATASET = 'CIFAR10'  # Using CIFAR-10 instead of CIFAR-100
BATCH_SIZE = 100  # Single batch of 100 images
EPOCHS_TO_SAVE = [1, 2, 5, 10, 20, 50, 100, 200]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Select 100 unique images with different labels (like in the notebook)
def select_batch_indices(dataset, num_images=100):
    """Select num_images with unique labels"""
    selected_indices = []
    selected_labels = []
    idx = 25  # Start from index 25 like in the notebook
    
    while len(selected_indices) < num_images:
        _, label = dataset[idx]
        if label not in selected_labels:
            selected_indices.append(idx)
            selected_labels.append(label)
        idx += 1
        
        # For CIFAR-10 we only have 10 classes, so after 10 images we allow duplicates
        if len(selected_indices) == 10 and len(selected_indices) < num_images:
            # Fill the rest with sequential images
            while len(selected_indices) < num_images:
                selected_indices.append(idx)
                idx += 1
            break
    
    return selected_indices

def setup_single_batch_data():
    """Setup dataset and select single batch"""
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(
        DATASET, defs, data_path='./datasets/cifar10'
    )
    
    # Select our fixed batch of 100 images
    batch_indices = select_batch_indices(validloader.dataset, BATCH_SIZE)
    
    # Extract the batch
    images = []
    labels = []
    for idx in batch_indices:
        img, label = validloader.dataset[idx]
        images.append(img)
        labels.append(label)
    
    # Convert to tensors
    batch_images = torch.stack(images)
    batch_labels = torch.tensor(labels)
    
    print(f"Selected {len(batch_indices)} images")
    print(f"Unique labels in batch: {len(set(labels))}")
    print(f"Label distribution: {sorted(set(labels))}")
    
    return setup, loss_fn, batch_images, batch_labels, batch_indices

def train_on_single_batch():
    """Train model on single batch and save gradients at different epochs"""
    print("=== Training ResNet32-10 on Single Batch ===")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs to save: {EPOCHS_TO_SAVE}")
    print()
    
    # Setup
    setup, loss_fn, batch_images, batch_labels, batch_indices = setup_single_batch_data()
    
    # Move batch to device
    batch_images = batch_images.to(**setup)
    batch_labels = batch_labels.to(device=setup['device'], dtype=torch.long)
    
    # Create model
    model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.to(**setup)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()
    
    # Training setup - using same optimizer as the notebook but with lower learning rate for stability
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    # Normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Create directory for saving
    os.makedirs('single_batch_gradients', exist_ok=True)
    
    # Save initial batch info
    torch.save({
        'batch_indices': batch_indices,
        'batch_labels': batch_labels.cpu(),
        'model_name': MODEL_NAME,
        'dataset': DATASET,
        'batch_size': BATCH_SIZE
    }, 'single_batch_gradients/batch_info.pt')
    
    # Training loop - only on this single batch!
    print("Training on single batch of 100 images...")
    print("-" * 60)
    
    for epoch in range(1, max(EPOCHS_TO_SAVE) + 1):
        model.train()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss, _, _ = loss_fn(outputs, batch_labels)
        
        # Compute accuracy
        _, predicted = outputs.max(1)
        correct = predicted.eq(batch_labels).sum().item()
        accuracy = 100. * correct / BATCH_SIZE
        
        # Save gradients at specified epochs BEFORE backprop
        if epoch in EPOCHS_TO_SAVE:
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            
            # Compute gradients (retain graph for backward later)
            input_gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            input_gradient = [grad.detach() for grad in input_gradient]
            
            # Calculate gradient statistics
            gradient_norms = [g.norm().item() for g in input_gradient]
            total_norm = torch.stack([g.norm() for g in input_gradient]).mean().item()
            print(f"  Gradient norm: {total_norm:.4f}")
            
            # Save gradient and model state
            save_path = f'single_batch_gradients/epoch_{epoch}_gradient.pt'
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'gradients': [g.cpu() for g in input_gradient],
                'gradient_norms': gradient_norms,
                'total_norm': total_norm,
                'loss': loss.item(),
                'accuracy': accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, save_path)
            print(f"  Saved to {save_path}")
        
        # Backward pass and update
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Print progress occasionally
        if epoch % 25 == 0 and epoch not in EPOCHS_TO_SAVE:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={accuracy:.2f}%")
    
    print("\n=== Training Complete ===")
    print(f"Final accuracy on the single batch: {accuracy:.2f}%")
    print(f"Final loss: {loss.item():.4f}")
    
    # Analyze gradient evolution
    print("\n=== Gradient Norm Evolution ===")
    print("-" * 40)
    print(f"{'Epoch':<10} {'Gradient Norm':<20} {'Loss':<10}")
    print("-" * 40)
    
    for epoch in EPOCHS_TO_SAVE:
        data = torch.load(f'single_batch_gradients/epoch_{epoch}_gradient.pt')
        print(f"{epoch:<10} {data['total_norm']:<20.4f} {data['loss']:<10.4f}")
    
    return model

def verify_reconstruction_setup():
    """Verify our setup matches the notebook approach"""
    print("\n=== Verifying Setup ===")
    
    # Load batch info
    batch_info = torch.load('single_batch_gradients/batch_info.pt')
    print(f"Batch size: {batch_info['batch_size']}")
    print(f"Model: {batch_info['model_name']}")
    
    # Load an epoch's gradient
    epoch_1 = torch.load('single_batch_gradients/epoch_1_gradient.pt')
    print(f"\nEpoch 1 gradient norm: {epoch_1['total_norm']:.4f}")
    print(f"Number of gradient tensors: {len(epoch_1['gradients'])}")
    
    print("\nReady for gradient inversion attacks!")
    print("Next step: Use these gradients to attempt reconstruction at different epochs")

if __name__ == "__main__":
    # Set random seed for reproducibility
    inversefed.utils.set_deterministic()
    
    # Train on single batch
    model = train_on_single_batch()
    
    # Verify setup
    verify_reconstruction_setup()
    
    print("\n✅ Complete! Gradients saved in 'single_batch_gradients/' directory")