import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Optional


class SuperResolutionNet(L.LightningModule):
    def __init__(
        self,
        r: int,
        activation: Optional[bool] = False,
        n_blocks: int = 2,
        n_channels: int = 48,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['activation'])
        self.r = r
        self.activation = activation
        self.deconvolution = nn.PixelShuffle(self.r)

        self.head = nn.Conv2d(3, n_channels, 3, padding=1, bias=False)

        body = [Conv2DResAttnBlock(n_channels, n_channels) for _ in range(n_blocks)]
        self.feature_extractor = nn.Sequential(*body)

        self.tail = nn.Conv2d(n_channels, 32, 3, 1, padding=1)

        self.last_layer = nn.Conv2d(32, self.r * self.r * 3, 3, padding=1)
        self.loss_fn = lambda pred, target: nn.functional.l1_loss(pred, target) + nn.functional.mse_loss(pred, target)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        if self.activation:
            x = nn.LeakyReLU(negative_slope=0.05, inplace=True)(x)
        res = x
        x = self.feature_extractor(x)
        x = x + res
        x = self.tail(x)
        x = self.last_layer(x)

        x = self.deconvolution(x)

        return x

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        lr_images, hr_images = batch
        sr_images = self(lr_images)
        loss = self.loss_fn(sr_images, hr_images)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        lr_images, hr_images = batch
        sr_images = self(lr_images)
        loss = self.loss_fn(sr_images, hr_images)

        mse = torch.mean((sr_images * 255.0 - hr_images * 255.0) ** 2)
        psnr = 10 * torch.log10(255.0**2 / (mse + 1e-10))

        # Calculate baseline PSNR using bicubic upsampling
        bicubic_upsampled = torch.nn.functional.interpolate(
            lr_images, scale_factor=self.r, mode='bicubic', align_corners=False
        )
        mse_bicubic = torch.mean((bicubic_upsampled * 255.0 - hr_images * 255.0) ** 2)
        psnr_bicubic = 10 * torch.log10(255.0**2 / (mse_bicubic + 1e-10))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_psnr_bicubic", psnr_bicubic, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self(batch)

    @torch.no_grad()
    def upscale_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Upscale a single image tensor
        
        Args:
            image: Tensor of shape (C, H, W) or (1, C, H, W)
        
        Returns:
            Upscaled tensor
        """
        was_3d = image.dim() == 3
        if was_3d:
            image = image.unsqueeze(0)
        
        self.eval()
        output = self(image)
        
        if was_3d:
            output = output.squeeze(0)
        
        return output

    @classmethod
    def load_for_inference(
        cls, checkpoint_path: str, device: str = "cuda", r: int = 4
    ) -> "SuperResolutionNet":
        model = cls.load_from_checkpoint(checkpoint_path, r=r)
        model.eval()
        model.to(device)
        return model


class Conv2DResAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.activation1 = (
            activation if activation is not None else nn.SiLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)
        # self.pointwise2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.activation1(out)
        out = self.conv2(out)

        sim_att = torch.sigmoid(out) - 0.5
        out = (out + x) * sim_att

        return out
