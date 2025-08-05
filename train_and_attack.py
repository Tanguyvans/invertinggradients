"""
Train a simple ConvNet on CIFAR-10 and perform gradient inversion attack.
This script demonstrates the privacy vulnerability in federated learning.
"""

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import datetime

import inversefed

# Configuration
DATASET_DIR = './datasets'  # Local dataset directory
CIFAR10_PATH = os.path.join(DATASET_DIR, 'cifar10')
MODEL_NAME = 'ResNet20-4'  # Changed from ConvNet to ResNet20-4
MODEL_SAVE_PATH = f'models/{MODEL_NAME.lower()}_cifar10.pth'
EPOCHS = 20  # Increased for better training with ResNet
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_cifar10_if_needed():
    """Download CIFAR-10 dataset if it doesn't exist."""
    if not os.path.exists(CIFAR10_PATH):
        print(f"CIFAR-10 dataset not found at {CIFAR10_PATH}")
        print("Downloading CIFAR-10 dataset...")
        os.makedirs(CIFAR10_PATH, exist_ok=True)
        
        # This will download CIFAR-10 to the specified directory
        _ = torchvision.datasets.CIFAR10(
            root=CIFAR10_PATH, 
            train=True, 
            download=True
        )
        _ = torchvision.datasets.CIFAR10(
            root=CIFAR10_PATH, 
            train=False, 
            download=True
        )
        print("CIFAR-10 dataset downloaded successfully!")
    else:
        print(f"CIFAR-10 dataset found at {CIFAR10_PATH}")

def plot_images(tensor, title="Images", save_path=None):
    """Plot and optionally save images."""
    # Get data stats for denormalization
    dm = torch.as_tensor(inversefed.consts.cifar10_mean)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std)[:, None, None]
    
    # Clone and denormalize
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * ds + dm
    tensor = torch.clamp(tensor, 0, 1)
    
    # Plot
    if tensor.shape[0] == 1:
        plt.figure(figsize=(5, 5))
        plt.imshow(tensor[0].permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')
    else:
        fig, axes = plt.subplots(1, min(tensor.shape[0], 8), figsize=(16, 4))
        if tensor.shape[0] == 1:
            axes = [axes]
        for i, (im, ax) in enumerate(zip(tensor[:8], axes)):
            ax.imshow(im.permute(1, 2, 0))
            ax.set_title(f"Image {i}")
            ax.axis('off')
        fig.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def train_model():
    """Train a simple ConvNet on CIFAR-10."""
    print("=== Phase 1: Training Model ===")
    
    # Setup
    setup = {'device': DEVICE, 'dtype': torch.float32}
    
    # Load data
    defs = inversefed.training_strategy('conservative')
    defs.epochs = EPOCHS
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(
        'CIFAR10', defs, data_path=CIFAR10_PATH
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(trainloader.dataset)}")
    print(f"Validation samples: {len(validloader.dataset)}")
    print(f"Batch size: {defs.batch_size}")
    print(f"Number of classes: 10")
    print()
    
    # Create model
    model, model_seed = inversefed.construct_model(MODEL_NAME, num_classes=10, num_channels=3)
    model.to(**setup)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Check if model already exists
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print(f"Training new model for {EPOCHS} epochs...")
        print("-" * 70)
        
        # Custom training loop with detailed progress
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
        
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Learning rate: {optimizer.param_groups[0]['lr']:.4f}")
            
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs = inputs.to(**setup)
                targets = targets.to(device=setup['device'], dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, _, _ = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f"  Batch [{batch_idx}/{len(trainloader)}] - "
                          f"Loss: {loss.item():.4f}, "
                          f"Acc: {100.*train_correct/train_total:.2f}%")
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in validloader:
                    inputs = inputs.to(**setup)
                    targets = targets.to(device=setup['device'], dtype=torch.long)
                    outputs = model(inputs)
                    loss, _, _ = loss_fn(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Print epoch summary
            print(f"  Training   - Loss: {train_loss/len(trainloader):.4f}, "
                  f"Accuracy: {100.*train_correct/train_total:.2f}%")
            print(f"  Validation - Loss: {val_loss/len(validloader):.4f}, "
                  f"Accuracy: {100.*val_correct/val_total:.2f}%")
            print("-" * 70)
            
            scheduler.step()
        
        # Save the model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    # Final validation
    print("\nFinal Model Evaluation:")
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in validloader:
            images = images.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f"Overall accuracy: {100 * correct / total:.2f}%")
    print("\nPer-class accuracy:")
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        if class_total[i] > 0:
            print(f"  {classes[i]:8s}: {100 * class_correct[i]/class_total[i]:5.2f}%")
    print()
    
    return model, validloader

def perform_attack(model, validloader, target_id=42, num_restarts=8):
    """Perform gradient inversion attack on a single image."""
    print("\n=== Phase 2: Gradient Inversion Attack ===")
    
    # Setup
    setup = {'device': DEVICE, 'dtype': torch.float32}
    model.eval()
    
    # Get normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Select target image
    ground_truth, true_label = validloader.dataset[target_id]
    ground_truth = ground_truth.unsqueeze(0).to(**setup)
    labels = torch.tensor([true_label], device=setup['device'])
    
    print(f"Target image: Class {validloader.dataset.classes[true_label]} (index {target_id})")
    
    # Plot original image
    plot_images(ground_truth, title=f"Original Image - {validloader.dataset.classes[true_label]}")
    
    # Compute gradient
    model.zero_grad()
    from inversefed.data.loss import Classification
    loss_fn = Classification()
    outputs = model(ground_truth)
    loss, _, _ = loss_fn(outputs, labels)
    input_gradient = torch.autograd.grad(loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    
    print(f"Gradient norm: {torch.stack([g.norm() for g in input_gradient]).mean():.4f}")
    
    # Configure reconstruction
    config = dict(
        signed=True,
        boxed=True,
        cost_fn='sim',  # Use cosine similarity
        indices='def',
        weights='equal',
        lr=0.1,
        optim='adam',
        restarts=num_restarts,
        max_iterations=8000,  # Fewer iterations for CIFAR
        total_variation=1e-6,
        init='randn',
        filter='none',
        lr_decay=True,
        scoring_choice='loss'
    )
    
    # Perform reconstruction
    print(f"\nStarting reconstruction with {num_restarts} restarts...")
    print("Attack configuration:")
    print(f"  - Cost function: {config['cost_fn']}")
    print(f"  - Optimizer: {config['optim']}")
    print(f"  - Learning rate: {config['lr']}")
    print(f"  - Max iterations: {config['max_iterations']}")
    print(f"  - Total variation: {config['total_variation']}")
    print()
    
    start_time = time.time()
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))
    attack_time = time.time() - start_time
    
    # Calculate metrics
    test_mse = (output - ground_truth).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    
    print(f"\nAttack Results:")
    print(f"Attack completed in {attack_time:.2f} seconds")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"MSE: {test_mse:.4f}")
    print(f"PSNR: {test_psnr:.2f} dB")
    
    # Interpret results
    if test_psnr > 30:
        print("Result: Excellent reconstruction - privacy severely compromised!")
    elif test_psnr > 20:
        print("Result: Good reconstruction - significant privacy leakage")
    elif test_psnr > 15:
        print("Result: Moderate reconstruction - some privacy leakage")
    else:
        print("Result: Poor reconstruction - limited privacy leakage")
    
    # Plot reconstructed image
    plot_images(output, title=f"Reconstructed Image - PSNR: {test_psnr:.2f} dB")
    
    # Save comparison
    comparison = torch.cat([ground_truth, output], dim=0)
    plot_images(comparison, title="Original (left) vs Reconstructed (right)", 
                save_path="attack_comparison.png")
    
    return output, ground_truth, stats

def perform_batch_attack(model, validloader, batch_size=8, num_restarts=4):
    """Perform gradient inversion attack on multiple images."""
    print("\n=== Phase 3: Batch Gradient Inversion Attack ===")
    
    # Setup
    setup = {'device': DEVICE, 'dtype': torch.float32}
    model.eval()
    
    # Get normalization constants
    dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    
    # Select multiple images with different labels
    ground_truth, labels = [], []
    idx = 100
    while len(labels) < batch_size:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in [l.item() for l in labels]:
            labels.append(torch.tensor([label], device=setup['device']))
            ground_truth.append(img.to(**setup))
    
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    
    print(f"Target images: {[validloader.dataset.classes[l] for l in labels]}")
    
    # Plot original images
    plot_images(ground_truth[:4], title="Original Images (first 4)")
    
    # Compute gradient
    model.zero_grad()
    from inversefed.data.loss import Classification
    loss_fn = Classification()
    outputs = model(ground_truth)
    loss, _, _ = loss_fn(outputs, labels)
    input_gradient = torch.autograd.grad(loss, model.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    
    # Configure reconstruction for batch
    config = dict(
        signed=True,
        boxed=True,
        cost_fn='sim',
        indices='def',
        weights='equal',
        lr=0.01,  # Lower learning rate for batch
        optim='adam',
        restarts=num_restarts,
        max_iterations=12000,
        total_variation=1e-2,  # Higher TV for batch
        init='randn',
        filter='none',
        lr_decay=True,
        scoring_choice='loss'
    )
    
    # Perform reconstruction
    print(f"Starting batch reconstruction with {num_restarts} restarts...")
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=batch_size)
    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))
    
    # Calculate metrics
    test_mse = (output - ground_truth).pow(2).mean().item()
    test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)
    
    print(f"\nBatch Attack Results:")
    print(f"Reconstruction loss: {stats['opt']:.4f}")
    print(f"Average MSE: {test_mse:.4f}")
    print(f"Average PSNR: {test_psnr:.2f} dB")
    
    # Plot reconstructed images
    plot_images(output[:4], title=f"Reconstructed Images (first 4) - Avg PSNR: {test_psnr:.2f} dB")
    
    return output, ground_truth, stats

def main():
    """Main function to run training and attacks."""
    print("Gradient Inversion Attack Demonstration on CIFAR-10")
    print("=" * 50)
    print(f"Using device: {DEVICE}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"CIFAR-10 data path: {CIFAR10_PATH}")
    print()
    
    # Download CIFAR-10 if needed
    download_cifar10_if_needed()
    print()
    
    # Set random seed for reproducibility
    inversefed.utils.set_deterministic()
    
    # Phase 1: Train model
    model, validloader = train_model()
    
    # Phase 2: Single image attack
    print("\n" + "=" * 50)
    input("Press Enter to perform single image attack...")
    perform_attack(model, validloader, target_id=42, num_restarts=8)
    
    # Phase 3: Batch attack (optional)
    print("\n" + "=" * 50)
    response = input("Perform batch attack on multiple images? (y/n): ")
    if response.lower() == 'y':
        perform_batch_attack(model, validloader, batch_size=8, num_restarts=4)
    
    print("\n" + "=" * 50)
    print("Attack demonstration complete!")
    print(f"Time: {datetime.datetime.now().strftime('%A, %d. %B %Y %I:%M%p')}")

if __name__ == "__main__":
    main()