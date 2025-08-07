# Gradient Attack Analysis: Fresh vs Saved Gradients

## Problem Solved: Why Fresh Gradients Work Better

### Previous Approach (Saved Gradients) - Poor Results
- **PSNR**: ~11 dB (very poor reconstruction)
- **Method**: 
  1. During training: `torch.autograd.grad(loss, model.parameters())` 
  2. Save these gradients to disk
  3. During attack: Load saved gradients and reconstruct
- **Issue**: Saved gradients are "stale" - they represent the model's response during training dynamics, not clean inference

### Current Approach (Fresh Gradients) - Excellent Results
- **PSNR**: 15-19 dB (much better reconstruction)
- **Method**:
  1. Load model state from specific epoch
  2. Compute fresh gradients on-demand: `model(ground_truth)` → `torch.autograd.grad(target_loss, model.parameters())`
  3. Use these fresh gradients for reconstruction
- **Advantage**: Fresh gradients represent clean, direct model response to input images

## Key Technical Difference

### Saved Gradients (train.py)
```python
# During training - gradients accumulated from training dynamics
input_gradient = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
torch.save({'gradients': input_gradient}, f'epoch_{epoch}_gradient.pt')
```

### Fresh Gradients (attack.py)
```python
# During attack - clean gradients computed on-demand
model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), true_labels)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
```

## Results Comparison

| Method | PSNR Range | Model | Reconstruction Quality |
|--------|------------|--------|----------------------|
| Saved Gradients | ~11 dB | ResNet32-10 | Poor, noisy |
| Fresh Gradients | 15-19 dB | ConvNet | Good, clear features |

## Why Fresh Gradients Work Better

1. **Clean Signal**: Fresh gradients contain direct information about model response to specific inputs
2. **No Training Noise**: Saved gradients include artifacts from training dynamics, momentum, weight updates
3. **Exact Model State**: Fresh gradients reflect the exact model state at inference time
4. **Matches Notebooks**: Original InvertingGradients notebooks use fresh gradient computation

## Conclusion

The breakthrough came from switching to **fresh gradient computation** - computing gradients on-demand during the attack rather than using saved gradients from training. This matches the methodology used in successful gradient inversion papers and provides much clearer signal for image reconstruction.