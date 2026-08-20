use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use tracing::{debug, info};

use crate::model::SuperResolutionNet;

// Tile size for processing large images (in pixels)
// Images larger than this will be processed in overlapping tiles
const TILE_SIZE: u32 = 512;
const TILE_OVERLAP: u32 = 16; // Overlap to avoid edge artifacts

pub fn upscale_image(model: &SuperResolutionNet, image: DynamicImage) -> Result<DynamicImage> {
    let total_start = Instant::now();
    let (input_width, input_height) = image.dimensions();
    debug!(width = input_width, height = input_height, "Starting upscale_image");
    
    if input_width > TILE_SIZE || input_height > TILE_SIZE {
        debug!(tile_size = TILE_SIZE, "Using tiled processing for large image");
        return upscale_image_tiled(model, image);
    }
    
    let pb = ProgressBar::new(4);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    pb.set_message("Preprocessing image...");
    let preprocess_start = Instant::now();
    let input_tensor = image_to_tensor(&image, model.device())?;
    let preprocess_time = preprocess_start.elapsed().as_millis();
    debug!(elapsed_ms = preprocess_time, tensor_shape = ?input_tensor.dims(), "Image preprocessing complete");
    pb.inc(1);

    pb.set_message("Running inference...");
    let inference_start = Instant::now();
    let output_tensor = model.forward(&input_tensor)?;
    let inference_time = inference_start.elapsed().as_millis();
    debug!(elapsed_ms = inference_time, output_shape = ?output_tensor.dims(), "Inference complete");
    pb.inc(1);

    pb.set_message("Postprocessing result...");
    let postprocess_start = Instant::now();
    let output_image = tensor_to_image(&output_tensor)?;
    let postprocess_time = postprocess_start.elapsed().as_millis();
    let (output_width, output_height) = output_image.dimensions();
    debug!(elapsed_ms = postprocess_time, width = output_width, height = output_height, "Postprocessing complete");
    pb.inc(1);

    pb.set_message("Done!");
    pb.inc(1);
    pb.finish();

    let total_time = total_start.elapsed().as_millis();
    info!(
        total_elapsed_ms = total_time,
        preprocess_ms = preprocess_time,
        inference_ms = inference_time,
        postprocess_ms = postprocess_time,
        input_dims = format!("{}x{}", input_width, input_height),
        output_dims = format!("{}x{}", output_width, output_height),
        "Upscaling complete"
    );

    Ok(output_image)
}

fn image_to_tensor(image: &DynamicImage, device: &Device) -> Result<Tensor> {
    let convert_start = Instant::now();
    let img = image.to_rgb8();
    let (width, height) = img.dimensions();
    debug!(width = width, height = height, device = ?device, "Converting image to tensor");

    // Convert image to f32 vec (raw pixel values 0-255)
    let mut data = Vec::with_capacity((width * height * 3) as usize);
    
    // Process in CHW format (channels, height, width)
    // Channel order: R, G, B
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let value = pixel[channel] as f32;
                data.push(value);
            }
        }
    }

    // Create tensor with shape [1, 3, H, W] and normalize using affine (x * 1/255 + 0)
    let tensor = Tensor::from_vec(data, (1, 3, height as usize, width as usize), device)?;
    let tensor = tensor.affine(1.0 / 255.0, 0.0)?;
    
    // TODO: Add model.to_dtype(BF16) support for full BF16 inference pipeline
    
    let convert_time = convert_start.elapsed().as_millis();
    debug!(elapsed_ms = convert_time, shape = ?tensor.dims(), dtype = ?tensor.dtype(), "Tensor created and normalized with affine");

    Ok(tensor)
}

fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    let convert_start = Instant::now();
    let shape = tensor.dims();
    debug!(shape = ?shape, dtype = ?tensor.dtype(), "Converting tensor to image");
    
    // Expected shape: [1, 3, H, W]
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
        anyhow::bail!(
            "Unexpected tensor shape: {:?}. Expected [1, 3, H, W]",
            shape
        );
    }

    let height = shape[2] as u32;
    let width = shape[3] as u32;

    let cpu_transfer_start = Instant::now();
    let tensor_cpu = tensor.to_device(&Device::Cpu)?;
    let cpu_transfer_time = cpu_transfer_start.elapsed().as_millis();
    debug!(elapsed_ms = cpu_transfer_time, "Tensor transferred to CPU");
    
    // Squeeze batch dimension and denormalize: [1, 3, H, W] -> [3, H, W], then scale to [0, 255]
    let squeeze_start = Instant::now();
    let tensor_squeezed = tensor_cpu.squeeze(0)?;
    // Use affine to denormalize in one kernel: x * 255 + 0
    let tensor_denorm = tensor_squeezed.affine(255.0, 0.0)?;
    let data = tensor_denorm.to_vec3::<f32>()?;
    let squeeze_time = squeeze_start.elapsed().as_millis();
    debug!(elapsed_ms = squeeze_time, "Tensor data extracted and denormalized with affine");

    let pixel_start = Instant::now();
    let mut img_buffer = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = data[0][y as usize][x as usize].clamp(0.0, 255.0) as u8;
            let g = data[1][y as usize][x as usize].clamp(0.0, 255.0) as u8;
            let b = data[2][y as usize][x as usize].clamp(0.0, 255.0) as u8;
            
            img_buffer.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    let pixel_time = pixel_start.elapsed().as_millis();
    debug!(elapsed_ms = pixel_time, "Pixel conversion complete");

    let total_convert_time = convert_start.elapsed().as_millis();
    debug!(
        total_elapsed_ms = total_convert_time,
        cpu_transfer_ms = cpu_transfer_time,
        data_extract_ms = squeeze_time,
        pixel_convert_ms = pixel_time,
        width = width,
        height = height,
        "Tensor to image conversion complete"
    );

    Ok(DynamicImage::ImageRgb8(img_buffer))
}

fn upscale_image_tiled(model: &SuperResolutionNet, image: DynamicImage) -> Result<DynamicImage> {
    let total_start = Instant::now();
    let (input_width, input_height) = image.dimensions();
    let upscale_factor = model.upscale_factor();
    
    info!(
        width = input_width, 
        height = input_height, 
        tile_size = TILE_SIZE,
        overlap = TILE_OVERLAP,
        "Processing large image with tiles"
    );
    
    let output_width = input_width * upscale_factor as u32;
    let output_height = input_height * upscale_factor as u32;
    
    let mut output_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(output_width, output_height);
    
    let tiles_x = input_width.div_ceil(TILE_SIZE).max(1);
    let tiles_y = input_height.div_ceil(TILE_SIZE).max(1);
    let total_tiles = tiles_x * tiles_y;
    
    debug!(tiles_x = tiles_x, tiles_y = tiles_y, total_tiles = total_tiles, "Tile layout calculated");
    
    let pb = ProgressBar::new(total_tiles as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} tiles {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    
    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let tile_process_start = Instant::now();
            
            // Calculate tile boundaries with overlap
            let x_start = (tile_x * TILE_SIZE).saturating_sub(TILE_OVERLAP);
            let y_start = (tile_y * TILE_SIZE).saturating_sub(TILE_OVERLAP);
            let x_end = ((tile_x + 1) * TILE_SIZE + TILE_OVERLAP).min(input_width);
            let y_end = ((tile_y + 1) * TILE_SIZE + TILE_OVERLAP).min(input_height);
            
            let tile_width = x_end - x_start;
            let tile_height = y_end - y_start;
            
            let tile_image = image.crop_imm(x_start, y_start, tile_width, tile_height);
            
            let tile_tensor = image_to_tensor(&tile_image, model.device())?;
            let output_tensor = model.forward(&tile_tensor)?;
            let output_tile = tensor_to_image(&output_tensor)?;
            
            let out_x_start = tile_x * TILE_SIZE * upscale_factor as u32;
            let out_y_start = tile_y * TILE_SIZE * upscale_factor as u32;
            let out_x_end = ((tile_x + 1) * TILE_SIZE).min(input_width) * upscale_factor as u32;
            let out_y_end = ((tile_y + 1) * TILE_SIZE).min(input_height) * upscale_factor as u32;
            
            let crop_x_start = if tile_x > 0 { TILE_OVERLAP * upscale_factor as u32 } else { 0 };
            let crop_y_start = if tile_y > 0 { TILE_OVERLAP * upscale_factor as u32 } else { 0 };
            
            let tile_rgb = output_tile.to_rgb8();
            for y in 0..(out_y_end - out_y_start) {
                for x in 0..(out_x_end - out_x_start) {
                    let tile_x_coord = crop_x_start + x;
                    let tile_y_coord = crop_y_start + y;
                    
                    if tile_x_coord < tile_rgb.width() && tile_y_coord < tile_rgb.height() {
                        let pixel = tile_rgb.get_pixel(tile_x_coord, tile_y_coord);
                        let out_x = out_x_start + x;
                        let out_y = out_y_start + y;
                        
                        if out_x < output_width && out_y < output_height {
                            output_buffer.put_pixel(out_x, out_y, *pixel);
                        }
                    }
                }
            }
            
            let tile_time = tile_process_start.elapsed().as_millis();
            debug!(
                tile_x = tile_x,
                tile_y = tile_y,
                elapsed_ms = tile_time,
                "Tile complete"
            );
            pb.inc(1);
        }
    }
    
    pb.finish_with_message("All tiles processed");
    
    let total_time = total_start.elapsed().as_millis();
    info!(
        total_elapsed_ms = total_time,
        tiles_processed = total_tiles,
        avg_ms_per_tile = total_time / total_tiles as u128,
        "Tiled upscaling complete"
    );
    
    Ok(DynamicImage::ImageRgb8(output_buffer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_tensor_conversion() {
        let img = DynamicImage::new_rgb8(10, 10);
        let device = Device::Cpu;
        
        let tensor = image_to_tensor(&img, &device);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        let shape = tensor.dims();
        assert_eq!(shape, &[1, 3, 10, 10]);
    }

    #[test]
    fn test_tensor_to_image_conversion() {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1f32, (1, 3, 10, 10), &device).unwrap();
        
        let result = tensor_to_image(&tensor);
        assert!(result.is_ok());
        
        let image = result.unwrap();
        assert_eq!(image.width(), 10);
        assert_eq!(image.height(), 10);
    }
}
