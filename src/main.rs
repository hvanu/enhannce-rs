mod inference;
mod model;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::Parser;
use image::GenericImageView;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, Level};
use tracing_subscriber::FmtSubscriber;

use inference::upscale_image;
use model::SuperResolutionNet;

const DEFAULT_MODEL: &[u8] = include_bytes!("../model.safetensors");

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Path to the input image file
    #[arg(short, long)]
    input: PathBuf,

    // Path to save the upscaled output image
    #[arg(short, long)]
    output: PathBuf,

    // Path to the model weights file (safetensors format). If not specified, uses the embedded default model.
    #[arg(short, long)]
    model: Option<PathBuf>,

    // Upscaling factor (e.g., 2 for 2x upscaling, 3 for 3x, etc.)
    #[arg(short = 'r', long, default_value = "4")]
    upscale_factor: usize,

    // Number of feature channels (default: 48)
    #[arg(long, default_value = "48")]
    n_channels: usize,

    // Number of residual attention blocks (default: 2)
    #[arg(long, default_value = "2")]
    n_blocks: usize,

    // Use CUDA GPU for acceleration (if available)
    #[arg(long, conflicts_with = "metal")]
    cuda: bool,

    // Use Apple Metal GPU for acceleration (macOS only)
    #[arg(long, conflicts_with = "cuda")]
    metal: bool,

    // Also generate and save a bicubic interpolation comparison
    #[arg(long)]
    bicubic: bool,

    // Also generate and save a lanczos3 interpolation comparison
    #[arg(long)]
    lanczos: bool,
}

fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    let start_time = Instant::now();
    let args = Args::parse();


    info!("enhannce-rs - Neural Network Image Upscaling");
    info!("════════════════════════════════════════════════\n");

    let device = if args.cuda {
        Device::new_cuda(0).context("CUDA device not available")?
    } else if args.metal {
        Device::new_metal(0).context("Metal device not available")?
    } else {
        Device::Cpu
    };
    let device_init_time = start_time.elapsed().as_millis();
    info!(device = ?device, elapsed_ms = device_init_time, "Using Device");
    
    let mut available_devices = vec!["CPU"];
    if candle_core::utils::cuda_is_available() {
        available_devices.push("CUDA");
    }
    if candle_core::utils::metal_is_available() {
        available_devices.push("Metal");
    }
    debug!(devices = available_devices.join(", "), "Available devices");

    let model_load_start = Instant::now();
    let model = if let Some(model_path) = args.model {
        info!(path = %model_path.display(), "Loading model from file");
        SuperResolutionNet::load(
            args.upscale_factor,
            args.n_channels,
            args.n_blocks,
            false,
            model_path.to_str().context("Invalid model path")?,
            &device,
        )
        .context("Failed to load model")?
    } else {
        info!("Loading embedded default model");
        SuperResolutionNet::load_from_bytes(
            args.upscale_factor,
            args.n_channels,
            args.n_blocks,
            false,
            DEFAULT_MODEL,
            &device,
        )
        .context("Failed to load embedded model")?
    };
    let model_load_time = model_load_start.elapsed().as_millis();

    info!(elapsed_ms = model_load_time, "- Model loaded successfully");
    
    info!(upscale_factor = model.upscale_factor(), 
          elapsed_ms = model_load_time,
          "- Model configured");
    

    let image_load_start = Instant::now();
    info!(path = %args.input.display(), "Loading input image");
    let input_image = image::open(&args.input)
        .context(format!("Failed to open input image: {}", args.input.display()))?;
    let image_load_time = image_load_start.elapsed().as_millis();
    
    let (width, height) = input_image.dimensions();
    info!(width = width, height = height, elapsed_ms = image_load_time, "✓ Input image loaded");

    if args.bicubic {
        let bicubic_start = Instant::now();
        info!("🔄 Generating bicubic comparison...");
        let bicubic_image = input_image.resize_exact(
            width * args.upscale_factor as u32,
            height * args.upscale_factor as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let bicubic_time = bicubic_start.elapsed().as_millis();
        
        let bicubic_output = args.output.with_file_name(
            format!(
                "{}_bicubic.{}",
                args.output.file_stem().unwrap().to_str().unwrap(),
                args.output.extension().unwrap().to_str().unwrap()
            )
        );
        bicubic_image.save(&bicubic_output)
            .context(format!("Failed to save bicubic comparison: {}", bicubic_output.display()))?;
        info!(path = %bicubic_output.display(), elapsed_ms = bicubic_time, "✓ Bicubic comparison saved");
    }

    if args.lanczos {
        let lanczos_start = Instant::now();
        info!("Generating lanczos upscaled image...");
        let lanczos_image = input_image.resize_exact(
            width * args.upscale_factor as u32,
            height * args.upscale_factor as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let lanczos_time = lanczos_start.elapsed().as_millis();
        
        let lanczos_output = args.output.with_file_name(
            format!(
                "{}_lanczos.{}",
                args.output.file_stem().unwrap().to_str().unwrap(),
                args.output.extension().unwrap().to_str().unwrap()
            )
        );
        lanczos_image.save(&lanczos_output)
            .context(format!("Failed to save lanczos comparison: {}", lanczos_output.display()))?;
        info!(path = %lanczos_output.display(), elapsed_ms = lanczos_time, "✓ Lanczos comparison saved");
    }

   let processing_start = Instant::now();
info!("Processing image...");
let output_image = upscale_image(&model, input_image)?;
let processing_time = processing_start.elapsed().as_millis();

let (new_width, new_height) = output_image.dimensions();
info!(
    width = new_width,
    height = new_height,
    elapsed_ms = processing_time,
    "Image processed"
);

let save_start = Instant::now();
info!(path = %args.output.display(), "Saving output");
output_image
    .save(&args.output)
    .with_context(|| format!("Failed to save output image: {}", args.output.display()))?;
let save_time = save_start.elapsed().as_millis();
info!(elapsed_ms = save_time, "Image saved");

let total_time = start_time.elapsed().as_millis();
info!(total_elapsed_ms = total_time, "Done. Image successfully upscaled");

Ok(())

}
