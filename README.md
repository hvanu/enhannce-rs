# enha*nn*ce-rs 🔍️

Enhannce-rs is a pure Rust CLI tool for image upscaling using neural networks.

It performs image upscaling using super-resolution neural networks, delivering fast inference through the Candle library. Supports both CPU and GPU execution, with optional acceleration via CUDA or Apple Metal. Model weights are by default embedded to have a single binary, but configuring custom model weights is also possible through cli args.

## Architecture

The network is a lightweight convolutional super-resolution model, based on some ideas in the [NTIRE efficient super-resolution challenge](https://arxiv.org/abs/2504.10686).

```
→ Input (RGB)
→ Conv2D(3 → n_channels, 3×3)            
→ n_blocks ×     
    → Conv2D(n_channels → n_channels, 3×3)       
    → SiLU                                   
    → Conv2D(n_channels → n_channels, 3×3)
    → sigmoid       
→ Conv2D(n_channels → 32, 3×3)                
→ Conv2D(32 → r²×3, 3×3)                      
→ Output (Upscaled RGB)
```

Where `r` is the upscaling factor (e.g., 2 for 2x upscaling).

## Installation

### Prerequisites

- Rust 1.85 or higher

### Build from Source

**Standard build (CPU only):**
```bash
cd enhannce-rs
cargo build --release
```

**Build with Metal support (macOS):**
```bash
cargo build --release --features metal
```

## Usage

### Basic Usage

```bash
enhannce-rs --input input.jpg --output upscaled.jpg --model weights.safetensors
```

### Command-Line Options

```
Options:
  -i, --input <INPUT>
          Path to the input image file

  -o, --output <OUTPUT>
          Path to save the upscaled output image

  -m, --model <MODEL>
          Path to the model weights file (safetensors format)

  -r, --upscale-factor <UPSCALE_FACTOR>
          Upscaling factor (e.g., 2 for 2x upscaling, 3 for 3x, etc.)
          [default: 2]

      --cuda
          Use CUDA GPU for acceleration (if available)

      --metal
          Use Apple Metal GPU for acceleration (macOS only)

  -h, --help
          Print help information

  -V, --version
          Print version information
```

### Examples

#### 2x Upscaling on CPU
```bash
enhannce-rs -i input.png -o output.png -m model_2x.safetensors -r 2
```

#### 3x Upscaling with CUDA GPU
```bash
enhannce-rs -i photo.jpg -o enhanced.jpg -m model_3x.safetensors -r 3 --cuda
```

#### 2x Upscaling with Apple Metal (macOS)
```bash
enhannce-rs -i photo.jpg -o enhanced.jpg -m model_2x.safetensors -r 2 --metal
```

#### Batch Processing
```bash
for img in images/*.jpg; do
    enhannce-rs -i "$img" -o "upscaled/$(basename "$img")" -m model.safetensors
done
```

## Model Weights

### Converting PyTorch Models to SafeTensors

If you have a PyTorch model, you can convert it to SafeTensors format:

```python
import torch
from safetensors.torch import save_file

# Load your PyTorch model
model = SuperResolutionNet(r=2)
model.load_state_dict(torch.load('model.pth'))

# Extract state dict
state_dict = model.state_dict()

# Save as SafeTensors
save_file(state_dict, 'model.safetensors')
```


## Acknowledgments


- [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092)
- [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://arxiv.org/pdf/1609.05158)
- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802)
- [Swift Parameter-free Attention Network for Efficient Super-Resolution](https://arxiv.org/pdf/2311.12770)