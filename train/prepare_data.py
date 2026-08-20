import os
from pathlib import Path
from PIL import Image
import random

def process_hiertext_dataset(input_folder, output_folder):
    """
    Process HierText dataset:
    - Ignore images with any dimension < 512
    - Downscale 2x if any dimension > 1200 (using Lanczos)
    - Generate one random 512x512 crop per image
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    processed_count = 0
    skipped_count = 0
    
    for img_file in input_path.rglob('*'):
        if img_file.suffix.lower() not in image_extensions:
            continue
            
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                
                # Skip if any dimension < 512
                if width < 512 or height < 512:
                    print(f"Skipped {img_file.name}: dimensions too small ({width}x{height})")
                    skipped_count += 1
                    continue
                
                # Downscale 2x if any dimension > 1200
                if width > 1200 or height > 1200:
                    new_width = width // 2
                    new_height = height // 2
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    width, height = new_width, new_height
                    print(f"Downscaled {img_file.name} to {width}x{height}")
                
                # Random crop 512x512
                max_x = width - 512
                max_y = height - 512
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                cropped = img.crop((x, y, x + 512, y + 512))
                
                # Save with original filename
                output_file = output_path / img_file.name
                cropped.save(output_file, quality=98)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} images...")
                    
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue
    
    print(f"\nComplete! Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    input_folder = "data/hiertext/validation"
    output_folder = "data/hiertext/processed_validation"
    
    process_hiertext_dataset(input_folder, output_folder)