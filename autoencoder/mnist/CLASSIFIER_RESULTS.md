# MNIST Classifier Results

Comparison of different training approaches for MNIST digit classification.

## Results Summary (1 Epoch)

| Approach | Test Accuracy | Train Time | Trainable Params | Total Params |
|----------|--------------|------------|------------------|--------------|
| **From Scratch** | 94.14% | 8.7s | 55,798 | 55,798 |
| **Transfer (Frozen)** | 95.48% | 3.0s | 20,490 | 60,410 |
| **Transfer (Fine-tuned)** | 97.34% | 4.3s | 60,410 | 60,410 |

## Key Findings

### 1. Training from Scratch
- Achieves good baseline accuracy (94.14%)
- Requires training all parameters
- Slower training time

### 2. Transfer Learning with Frozen Encoder
- **+1.34% improvement** over scratch
- **65% faster** training (3.0s vs 8.7s)
- **Only 37% of parameters** are trainable (20,490 / 55,798)
- Pre-trained encoder provides good features for classification

### 3. Transfer Learning with Fine-tuning
- **+3.20% improvement** over scratch
- **+1.86% improvement** over frozen
- Best overall performance (97.34%)
- Still faster than training from scratch (4.3s vs 8.7s)
- Fine-tuning the encoder improves classification accuracy

## Conclusions

1. **Pre-trained encoder helps significantly**: Even with just 1 epoch, transfer learning improves accuracy by 1-3%

2. **Frozen encoder is most efficient**:
   - Fastest training
   - Minimal parameters to train
   - Good for quick prototyping or limited compute

3. **Fine-tuning achieves best results**:
   - Best accuracy (97.34% in 1 epoch)
   - Still faster than training from scratch
   - Recommended for final models

4. **The autoencoder learned useful features**: The encoder successfully extracts features useful for digit classification, validating the autoencoder training

## Architecture Details

### Encoder (from Autoencoder)
- 3 convolutional layers with BatchNorm
- Progressive downsampling: 28×28 → 14×14 → 7×7 → 4×4
- Channel expansion: 1 → 16 → 32 → 64
- Linear projection to latent_dim (16 in this test)
- Parameters: 39,920

### Classifier Head
- Linear: latent_dim → 128 (hidden_dim)
- ReLU + BatchNorm + Dropout(0.3)
- Linear: 128 → 128
- ReLU + BatchNorm + Dropout(0.3)
- Linear: 128 → 10 (num_classes)
- Parameters: 20,490

### Loss Function
- CrossEntropyLoss (includes softmax)

### Optimizer
- Adam with lr=0.001

## Expected Performance (10 Epochs)

Based on these 1-epoch results, with 10 epochs we can expect:

- **From Scratch**: ~98-99% test accuracy
- **Transfer (Frozen)**: ~97-98% test accuracy
- **Transfer (Fine-tuned)**: ~99%+ test accuracy

## Usage Examples

### Train from Scratch
```bash
python train_classifier.py --epochs 10 --latent-dim 16
```

### Transfer Learning (Frozen Encoder)
```bash
python train_classifier.py \
  --epochs 10 \
  --latent-dim 16 \
  --use-pretrained \
  --encoder-checkpoint test_checkpoint.pth \
  --freeze-encoder
```

### Transfer Learning (Fine-tuning)
```bash
python train_classifier.py \
  --epochs 10 \
  --latent-dim 16 \
  --use-pretrained \
  --encoder-checkpoint test_checkpoint.pth
```

## Notes

- All tests used the same configuration:
  - Batch size: 128
  - Learning rate: 0.001
  - Latent dimension: 16
  - Hidden dimension: 128
  - Random seed: 42
  - Device: MPS (Mac GPU)

- The autoencoder was trained for only 2 epochs, yet still provides useful features for classification

- Fine-tuning allows the encoder to adapt specifically for classification while preserving the general feature extraction ability learned during autoencoder training
