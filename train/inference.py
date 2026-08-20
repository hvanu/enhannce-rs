import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from srnet import SuperResolutionNet
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torchvision


def upscale_image(
    model_path: str,
    input_image_path: str,
    output_image_path: str,
    visualize: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Upscale a single image using trained model"""
    
    model = SuperResolutionNet.load_for_inference(model_path, device, r=4)
    
    # Load and preprocess image
    original_image = Image.open(input_image_path).convert('RGB')
    
    transform = transforms.ToTensor()
    image_tensor = transform(original_image).to(device)
    
    # Perform upscaling
    upscaled_tensor = model.upscale_image(image_tensor)
    
    # Convert to PIL image
    upscaled_tensor = upscaled_tensor.clamp(0, 1).cpu()
    upscaled_image = transforms.ToPILImage()(upscaled_tensor).convert('RGB')

    if visualize:
        # Create bicubic upsampled version
        target_size = (upscaled_image.width, upscaled_image.height)
        bicubic_upsampled = original_image.resize(target_size, Image.BICUBIC)
        
        # Create center crops for detail inspection
        crop_size = min(upscaled_image.width, upscaled_image.height) // 3
        center_x = upscaled_image.width // 2
        center_y = upscaled_image.height // 2
        left = center_x - crop_size // 2
        top = center_y - crop_size // 2
        right = left + crop_size
        bottom = top + crop_size
        
        original_crop = original_image.crop((left, top, right, bottom))
        bicubic_crop = bicubic_upsampled.crop((left, top, right, bottom))
        nn_crop = upscaled_image.crop((left, top, right, bottom))
        
        # Create visualization with 2 rows
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row - full images
        axes[0, 0].imshow(np.array(original_image))
        axes[0, 0].set_title(f'Original (High Resolution)\n{original_image.width}x{original_image.height}', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(np.array(bicubic_upsampled))
        axes[0, 1].set_title(f'Bicubic Upsampling\n{bicubic_upsampled.width}x{bicubic_upsampled.height}', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(np.array(upscaled_image))
        axes[0, 2].set_title(f'Neural Network Upsampling\n{upscaled_image.width}x{upscaled_image.height}', fontsize=14)
        axes[0, 2].axis('off')
        
        # Bottom row - center crops at true size
        axes[1, 0].imshow(np.array(original_crop))
        axes[1, 0].set_title(f'Original - Center Crop\n{original_crop.width}x{original_crop.height}', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(np.array(bicubic_crop))
        axes[1, 1].set_title(f'Bicubic - Center Crop\n{bicubic_crop.width}x{bicubic_crop.height}', fontsize=14)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(np.array(nn_crop))
        axes[1, 2].set_title(f'Neural Network - Center Crop\n{nn_crop.width}x{nn_crop.height}', fontsize=14)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Save
    upscaled_image.save(output_image_path)
    print(f"Upscaled image saved to: {output_image_path}")
    
    return upscaled_image


def batch_upscale(
    model_path: str,
    input_dir: str,
    output_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """Upscale all images in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model once
    model = SuperResolutionNet.load_for_inference(model_path, device, r=4)
    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    # Process all images
    image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    
    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).to(device)
        
        upscaled_tensor = model.upscale_image(image_tensor)
        
        upscaled_tensor = upscaled_tensor.cpu().clamp(0, 1)
        upscaled_image = to_pil(upscaled_tensor).convert('RGB')
        
        output_file = output_path / f"{img_path.stem}_upscaled{img_path.suffix}"
        upscaled_image.save(output_file)
    
    print(f"\nProcessed {len(image_files)} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super Resolution Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image or directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch', action='store_true', help='Batch process directory')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_upscale(args.checkpoint, args.input, args.output, args.device)
    else:
        upscale_image(args.checkpoint, args.input, args.output, args.device)