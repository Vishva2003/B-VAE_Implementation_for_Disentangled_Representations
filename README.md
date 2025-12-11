# β-VAE: Learning Disentangled Representations

A PyTorch implementation of β-Variational Autoencoder (β-VAE) for learning interpretable and disentangled latent representations. This implementation allows you to train models that separate underlying factors of variation in your data.

## Resources

- **Webpage**: [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://beta-vae-tutorial.vercel.app/)
- **Dataset**: [CelebA]([https://beta-vae-tutorial.vercel.app/](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


## 📋 Overview

β-VAE extends the standard VAE by introducing a hyperparameter β that controls the trade-off between reconstruction quality and disentanglement in the latent space. Higher β values encourage more disentangled representations, where individual latent dimensions correspond to independent factors of variation.

## ✨ Features

- **Complete β-VAE Implementation**: Encoder, decoder, and reparameterization trick
- **Flexible Architecture**: Customizable hidden dimensions and latent space size
- **Training Pipeline**: Full training loop with validation and checkpointing
- **Visualization Tools**: 
  - Reconstruction comparison
  - Random sampling from latent space
  - Latent space traversal (manipulate individual dimensions)
  - Latent interpolation between images
- **TensorBoard Integration**: Real-time training monitoring
- **Custom Dataset Support**: Easy integration with your own image datasets

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy tqdm pillow tensorboard
```

### Basic Usage

1. **Prepare Your Data**: Place your images in a folder structure:
```
/CelebA
└── /img_align_celeba
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2. **Configure Training**: Modify the configuration dictionary:
```python
config = {
    'data_path': './CelebA/img_align_celeba',
    'batch_size': 32,
    'img_size': 64,
    'latent_dim': 128,
    'hidden_dims': [32, 64, 128, 256],
    'beta': 4.0,  # Disentanglement strength
    'lr': 1e-4,
    'epochs': 50,
}
```

3. **Run Training**: Execute the notebook cells sequentially or convert to a Python script.

4. **Monitor Progress**: View training metrics with TensorBoard:
```bash
tensorboard --logdir=./runs
```

## 🏗️ Architecture

### Encoder
- Convolutional layers with BatchNorm and LeakyReLU
- Outputs mean (μ) and log-variance (log σ²) for latent distribution
- Default: 4 conv layers → 128D latent space

### Decoder
- Transposed convolutional layers
- Reconstructs images from latent codes
- Sigmoid activation for output normalization

### Loss Function
```
L = Reconstruction Loss + β × KL Divergence
```
- **Reconstruction Loss**: MSE between input and output
- **KL Divergence**: Regularization term encouraging Gaussian latent distribution
- **β**: Controls disentanglement (typical range: 1-10)

## 📊 Key Parameters

| Parameter | Description | Default | Tuning Tips |
|-----------|-------------|---------|-------------|
| `beta` | Disentanglement strength | 4.0 | Higher → more disentangled but worse reconstruction |
| `latent_dim` | Latent space dimensions | 128 | More dims → more capacity but harder to interpret |
| `hidden_dims` | Encoder/decoder layer sizes | [32,64,128,256] | Adjust based on image complexity |
| `learning_rate` | Optimizer learning rate | 1e-4 | Reduce if training is unstable |

## 🎨 Visualization Examples

### 1. Reconstruction Quality
Compare original images with their reconstructions to evaluate model performance.

### 2. Latent Space Traversal
Manipulate individual latent dimensions to discover learned features:
- Dimension 5 might control lighting
- Dimension 10 might control rotation
- Dimension 15 might control expression

### 3. Interpolation
Smoothly transition between two images by interpolating in latent space.

## 📁 Project Structure

```
├── Disentanglement_Bvae.ipynb   # Main implementation notebook
├── CelebA/                       # Dataset directory
├── checkpoints/                  # Saved models
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
├── runs/                         # TensorBoard logs
└── README.md                     # This file
```

## 🔧 Advanced Usage

### Custom Dataset
```python
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load your images here
    
    def __getitem__(self, idx):
        # Return transformed image
        pass
```

### Loading Pre-trained Models
```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Generating New Images
```python
# Sample from prior distribution
num_samples = 16
samples = model.sample(num_samples, device)
```

## 📈 Training Tips

1. **Start with β=1**: Train a standard VAE first, then gradually increase β
2. **Monitor KL Divergence**: Should stabilize after initial epochs
3. **Adjust Learning Rate**: Use ReduceLROnPlateau scheduler for adaptive learning
4. **Checkpoint Regularly**: Save models every 10 epochs
5. **Visualize Early**: Check reconstructions after 5-10 epochs

## 🐛 Troubleshooting

**Poor Reconstructions:**
- Decrease β value
- Increase latent dimensions
- Train for more epochs
- Check learning rate

**Not Disentangled:**
- Increase β gradually (4 → 6 → 8)
- Ensure diverse training data
- Increase model capacity
- Train longer

**Training Instability:**
- Reduce learning rate
- Add gradient clipping
- Check data normalization
- Reduce batch size

## 📚 References

- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) (Higgins et al., 2017)
- [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599) (Burgess et al., 2018)
- [Variational Autoencoders | Generative AI Animated](https://youtu.be/qJeaCHQ1k2w). (Deepia, 2024)

## 📝 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{bvae-implementation,
  author = Vishva MV,
  title = {β-VAE Implementation for Disentangled Representations},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Vishva2003/beta-vae}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original β-VAE authors for the groundbreaking research
- Community contributors and researchers in disentangled representation learning

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact dev.vishvamv@mail.com

---

**Happy Learning! 🚀** If you find this useful, please consider starring ⭐ the repository!
