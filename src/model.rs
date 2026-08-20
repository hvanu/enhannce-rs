use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder, VarMap};
use tracing::debug;

struct Conv2DResAttnBlock {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl Conv2DResAttnBlock {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        
        let conv1 = candle_nn::conv2d(channels, channels, 3, conv_config, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(channels, channels, 3, conv_config, vb.pp("conv2"))?;
        
        Ok(Self { conv1, conv2 })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // First conv + SiLU activation
        let out = self.conv1.forward(x)?;
        let out = candle_nn::ops::silu(&out)?;
        
        let out = self.conv2.forward(&out)?;
        
        // Attention mechanism: (out + x) * (sigmoid(out) - 0.5)
        let sigmoid_out = candle_nn::ops::sigmoid(&out)?;
        let sim_att = sigmoid_out.affine(1.0, -0.5)?;
        let residual = (&out + x)?;
        let result = residual.mul(&sim_att)?;
        
        Ok(result)
    }
}

// Super Resolution Network matching the PyTorch implementation
// This network performs image upscaling using sub-pixel convolution (PixelShuffle)
pub struct SuperResolutionNet {
    r: usize,
    #[allow(dead_code)]
    n_channels: usize,
    n_blocks: usize,
    activation: bool,
    head: Conv2d,
    feature_blocks: Vec<Conv2DResAttnBlock>,
    tail: Conv2d,
    last_layer: Conv2d,
    device: Device,
    #[allow(dead_code)]
    var_map: VarMap,
}

impl SuperResolutionNet {
    pub fn new(r: usize, n_channels: usize, n_blocks: usize, activation: bool, vb: VarBuilder, var_map: VarMap) -> Result<Self> {
        let device = vb.device().clone();

        // Head: Conv2d(3 -> n_channels, kernel=3, padding=1, no bias)
        let head_config = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let head = candle_nn::conv2d_no_bias(3, n_channels, 3, head_config, vb.pp("head"))?;

        // Feature extractor: n_blocks of Conv2DResAttnBlock
        let mut feature_blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let block = Conv2DResAttnBlock::new(
                n_channels,
                vb.pp(format!("feature_extractor.{}", i)),
            )?;
            feature_blocks.push(block);
        }

        // Tail: Conv2d(n_channels -> 32, kernel=3, padding=1)
        let tail_config = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let tail = candle_nn::conv2d(n_channels, 32, 3, tail_config, vb.pp("tail"))?;

        // Last layer: Conv2d(32 -> r²×3, kernel=3, padding=1)
        let last_config = Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let last_layer = candle_nn::conv2d(
            32,
            r * r * 3,
            3,
            last_config,
            vb.pp("last_layer"),
        )?;

        Ok(Self {
            r,
            n_channels,
            n_blocks,
            activation,
            head,
            feature_blocks,
            tail,
            last_layer,
            device,
            var_map,
        })
    }

    pub fn load(
        r: usize,
        n_channels: usize,
        n_blocks: usize,
        activation: bool,
        weights_path: &str,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };
        let var_map = VarMap::new();
        Self::new(r, n_channels, n_blocks, activation, vb, var_map)
    }

    // Create a new model and load weights from embedded bytes
    pub fn load_from_bytes(
        r: usize,
        n_channels: usize,
        n_blocks: usize,
        activation: bool,
        weights_bytes: &[u8],
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_slice_safetensors(weights_bytes, DType::F32, device)?;
        let var_map = VarMap::new();
        Self::new(r, n_channels, n_blocks, activation, vb, var_map)
    }

    // [batch, channels, height, width] - > Upscaled tensor with shape [batch, channels, height*r, width*r]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        debug!(input_shape = ?x.dims(), "Starting forward pass");
        
        // Head: Conv + optional LeakyReLU activation
        let mut x = self.head.forward(x)?;
        if self.activation {
            x = candle_nn::ops::leaky_relu(&x, 0.05)?;
        }
        debug!("Head layer complete");
        
        let res = x.clone();

        debug!(n_blocks = self.n_blocks, "Processing feature blocks");
        for (i, block) in self.feature_blocks.iter().enumerate() {
            x = block.forward(&x)?;
            debug!(block = i, "Feature block complete");
        }
        
        x = (&x + res)?;
        x = self.tail.forward(&x)?;
        
        x = self.last_layer.forward(&x)?;

        x = pixel_shuffle(&x, self.r)?;
        debug!(output_shape = ?x.dims(), "Pixel shuffle complete");

        Ok(x)
    }

    pub fn upscale_factor(&self) -> usize {
        self.r
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn var_map(&self) -> &VarMap {
        &self.var_map
    }
}


fn pixel_shuffle(x: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let shape = x.dims();
    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    let out_channels = channels / (upscale_factor * upscale_factor);

    // Reshape: [B, C*r^2, H, W] -> [B, C, r, r, H, W]
    let x = x.reshape((
        batch_size,
        out_channels,
        upscale_factor,
        upscale_factor,
        height,
        width,
    ))?;

    // Permute: [B, C, r, r, H, W] -> [B, C, H, r, W, r]
    let x = x.permute((0, 1, 4, 2, 5, 3))?;

    // Reshape to final output: [B, C, H*r, W*r]
    let x = x.reshape((
        batch_size,
        out_channels,
        height * upscale_factor,
        width * upscale_factor,
    ))?;

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_shuffle() {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 12, 4, 4), &device).unwrap();
        let result = pixel_shuffle(&x, 2);
        assert!(result.is_ok());
        let result = result.unwrap();
        let shape = result.dims();
        assert_eq!(shape, &[1, 3, 8, 8]);
    }
}
