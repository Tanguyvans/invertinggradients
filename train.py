"""
Train ConvNet on a batch of 10 CIFAR-10 images (one per class)
and save gradients at epochs 1, 2, 5, and 10
"""

import torch
import numpy as np
import os
import inversefed

# Configuration
MODEL_NAME = 'ConvNet'
DATASET = 'CIFAR10'
BATCH_SIZE = 10  # One image per class
EPOCHS_TO_SAVE = [1, 2, 5, 10]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def select_one_per_class(dataset, num_classes=10):
    """Select one image per class for unique labels"""
    selected_indices = []
    selected_labels = set()
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label not in selected_labels:
            selected_indices.append(idx)
            selected_labels.add(label)
            if len(selected_indices) == num_classes:
                break
    
    print(f"Selected indices: {selected_indices}")
    print(f"Labels: {sorted(selected_labels)}")
    return selected_indices

def train_on_10_images():
    """Train model on 10 images and save gradients"""
    print("=== Training ResNet32-10 on 10 CIFAR-10 Images ===")
    print(f"Device: {DEVICE}")
    print(f"One image per class for maximum gradient information")
    print()
    
    # Setup
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    
    # Load data
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(
        DATASET, defs, data_path='./datasets/cifar10'
    )
    
    # Select 10 images (one per class)
    batch_indices = select_one_per_class(validloader.dataset)
    
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
    
    # Move to device
    batch_images = batch_images.to(**setup)
    batch_labels = batch_labels.to(device=setup['device'], dtype=torch.long)
    
    # Create model
    model, _ = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.to(**setup)
    
    print(f"Model: {MODEL_NAME}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (much smaller than ResNet!)")
    
    # Training setup
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Create directory
    os.makedirs('gradients', exist_ok=True)
    
    # Save batch info
    torch.save({
        'batch_indices': batch_indices,
        'batch_labels': batch_labels.cpu(),
        'model_name': MODEL_NAME,
        'dataset': DATASET,
        'batch_size': BATCH_SIZE
    }, 'gradients/batch_info.pt')
    
    print("\nTraining on batch of 10 images...")
    print("-" * 40)
    
    # Training loop
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
        
        # Save gradients at specified epochs
        if epoch in EPOCHS_TO_SAVE:
            print(f"\nEpoch {epoch}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            
            # Compute gradients
            input_gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
            input_gradient = [grad.detach() for grad in input_gradient]
            
            # Calculate gradient norm
            total_norm = torch.stack([g.norm() for g in input_gradient]).mean().item()
            print(f"  Gradient norm: {total_norm:.4f}")
            
            # Save
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'gradients': [g.cpu() for g in input_gradient],
                'total_norm': total_norm,
                'loss': loss.item(),
                'accuracy': accuracy,
            }, f'gradients/epoch_{epoch}_gradient.pt')
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    print(f"\nFinal accuracy: {accuracy:.2f}%")
    print(f"Gradients saved in 'gradients/' for epochs: {EPOCHS_TO_SAVE}")

if __name__ == "__main__":
    inversefed.utils.set_deterministic()
    train_on_10_images()