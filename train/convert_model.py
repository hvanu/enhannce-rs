"""
Script to convert PyTorch Super Resolution model to SafeTensors format
for use with enhannce-rs
"""

import torch
from srnet import SuperResolutionNet
import torch.nn as nn
from safetensors.torch import save_file



def convert_model(input_path, output_path, upscale_factor=4, dtype=None):
    """
    Convert a PyTorch model to SafeTensors format
    
    Args:
        input_path: Path to PyTorch model (.pth or .pt)
        output_path: Path to save SafeTensors file (.safetensors)
        upscale_factor: Upscaling factor of the model
        dtype: Target dtype for weights (None, 'fp32', 'fp16', 'bf16')
    """
    print(f"Loading PyTorch model from: {input_path}")
    
    # Create model instance
    model = SuperResolutionNet(r=upscale_factor)
    
    # Load weights
    state_dict = torch.load(input_path, map_location='cpu')
    
    # Handle different state dict formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Converting to SafeTensors format...")
    
    # Get the state dict
    state_dict = model.state_dict()
    
    # Analyze dtypes
    dtypes = {}
    total_params = 0
    for name, tensor in state_dict.items():
        dtype_str = str(tensor.dtype).replace('torch.', '')
        if dtype_str not in dtypes:
            dtypes[dtype_str] = {'count': 0, 'params': 0}
        dtypes[dtype_str]['count'] += 1
        dtypes[dtype_str]['params'] += tensor.numel()
        total_params += tensor.numel()
    
    print(f"\nOriginal weight dtypes:")
    for dtype_name, info in sorted(dtypes.items()):
        percentage = (info['params'] / total_params) * 100
        print(f"  {dtype_name}: {info['count']} tensors, {info['params']:,} params ({percentage:.1f}%)")
    
    # Convert dtype if requested
    if dtype:
        dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Invalid dtype: {dtype}. Choose from: {list(dtype_map.keys())}")
        
        target_dtype = dtype_map[dtype]
        print(f"\nConverting weights to {dtype}...")
        
        state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}
    
    # Save as SafeTensors
    save_file(state_dict, output_path)
    
    print(f"\n✓ Model saved to: {output_path}")
    print(f"  Upscale factor: {upscale_factor}x")
    print(f"  Parameters: {total_params:,}")
    if dtype:
        print(f"  Export dtype: {dtype}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Super Resolution model to SafeTensors"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input PyTorch model file (.pth or .pt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output SafeTensors file (.safetensors)",
    )
    parser.add_argument(
        "--upscale-factor",
        "-r",
        type=int,
        default=4,
        help="Upscaling factor (default: 4)",
    )
    parser.add_argument(
        "--dtype",
        "-d",
        type=str,
        choices=['fp32', 'fp16', 'bf16'],
        default=None,
        help="Convert weights to specified dtype (default: keep original)",
    )
    
    args = parser.parse_args()
    
    convert_model(args.input, args.output, args.upscale_factor, args.dtype)
