import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from lightning.pytorch.loggers import CSVLogger
from pathlib import Path
from srnet import SuperResolutionNet
from torchvision import transforms
from torchvision.transforms.v2 import JPEG
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torchvision


class SRAugmentation:
    """Data augmentation for super-resolution training"""
    def __init__(self, augmentation_prob: float = 0.3):
        self.augmentation_prob = augmentation_prob
    
    def __call__(self, lr_images, hr_images):
        """Apply common SR augmentations: horizontal flip, gaussian blur, gaussian noise, JPEG artifacts"""
        # Horizontal flip
        if torch.rand(1).item() < self.augmentation_prob:
            lr_images = torch.flip(lr_images, dims=[-1])
            hr_images = torch.flip(hr_images, dims=[-1])
        
        # # Gaussian blur (applied only to LR images)
        # if torch.rand(1).item() < self.augmentation_prob:
        #     kernel_size = torch.randint(3, 6, (1,)).item() | 1
        #     sigma = torch.rand(1).item() * 0.8 + 0.1  # Random sigma between 0.1 and 0.9
        #     self.gaussian_blur = transforms.GaussianBlur(
        #         kernel_size=(kernel_size, kernel_size),
        #         sigma=(sigma, sigma)
        #     )
        #     lr_images = self.gaussian_blur(lr_images)
        

        # # Gaussian noise (applied only to LR images)
        # # Very aggressive noise level: 0.05-0.15
        # if torch.rand(1).item() < self.augmentation_prob:
        #     noise_level = torch.rand(1).item() * 0.02 + 0.005  
        #     noise = torch.randn_like(lr_images) * noise_level
        #     lr_images = (lr_images + noise).clamp(0, 1)
        
 

        return lr_images, hr_images


class SuperResolutionDataset(Dataset):
    """Example dataset - adapt to your data structure"""
    def __init__(self, hr_image_dir: str, scale_factor: int, transform=None, max_images=None, augmentation=None):
        self.hr_image_dir = Path(hr_image_dir)
        self.scale_factor = scale_factor
        self.image_paths = list(self.hr_image_dir.glob('*.png')) + \
                          list(self.hr_image_dir.glob('*.jpg'))

        # Limit the number of images if specified
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        self.transform = transform or transforms.ToTensor()
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load HR image
        # todo, fix rgb, torchvision everywhere
        hr_image = Image.open(self.image_paths[idx]).convert('RGB')
        hr_tensor = self.transform(hr_image)
        
        # Create LR image by downsampling
        h, w = hr_tensor.shape[1:]
        lr_size = [h // self.scale_factor, w // self.scale_factor]
        lr_image = transforms.functional.resize(
            hr_tensor,
            lr_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            lr_image, hr_tensor = self.augmentation(lr_image, hr_tensor)
        
        return lr_image, hr_tensor


class SaveValidationImages(Callback):
    """Save SR and HR validation images every N epochs"""
    def __init__(self, save_dir: str = 'validation_images', save_every_n_epochs: int = 5, num_images: int = 4, augmentation=None):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.num_images = num_images
        self.augmentation = augmentation
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.save_every_n_epochs != 0:
            return
        
        # Get a few validation samples
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader))
        lr_images, hr_images = batch
        lr_images = lr_images[:self.num_images].to(pl_module.device)
        hr_images = hr_images[:self.num_images].to(pl_module.device)
        
        # Apply augmentation to show degradation effects
        lr_images_degraded = lr_images.clone()
        if self.augmentation is not None:
            for i in range(lr_images_degraded.shape[0]):
                lr_degraded, _ = self.augmentation(lr_images_degraded[i], hr_images[i])
                lr_images_degraded[i] = lr_degraded
        
        pl_module.eval()
        with torch.no_grad():
            sr_images = pl_module(lr_images_degraded)
        
        # Create comparison grid: [LR (clean upsampled), LR (degraded upsampled), SR, HR]
        comparison_images = []
        for i in range(self.num_images):
            # Upsample clean LR for visualization
            lr_upsampled = torch.nn.functional.interpolate(
                lr_images[i:i+1], 
                size=hr_images[i].shape[-2:], 
                mode='bicubic',
                align_corners=False
            )
            # Upsample degraded LR for visualization
            lr_degraded_upsampled = torch.nn.functional.interpolate(
                lr_images_degraded[i:i+1], 
                size=hr_images[i].shape[-2:], 
                mode='bicubic',
                align_corners=False
            )
            comparison_images.extend([lr_upsampled[0], lr_degraded_upsampled[0], sr_images[i], hr_images[i]])
        
        # Save grid with 4 columns: Clean LR, Degraded LR, SR, HR
        grid = torchvision.utils.make_grid(comparison_images, nrow=4, normalize=True, value_range=(0, 1))
        save_path = self.save_dir / f'epoch_{trainer.current_epoch + 1:03d}.png'
        torchvision.utils.save_image(grid, save_path)
        
        print(f"Saved validation images to: {save_path}")


def train(
    train_data_dir: str,
    val_data_dir: str,
    scale_factor: int = 4,
    batch_size: int = 16,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    accelerator: str = 'auto',
    devices: int = 1,
    checkpoint_path: str = None,
):
    """Train the super-resolution model"""
    

    augmentation = SRAugmentation(augmentation_prob=0.5)
    

    train_dataset = SuperResolutionDataset(train_data_dir, scale_factor, augmentation=augmentation) # todo
    val_dataset = SuperResolutionDataset(val_data_dir, scale_factor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    

    model = SuperResolutionNet(r=scale_factor)
    
    # Load only model weights from checkpoint if provided (not optimizer state or epoch)
    if checkpoint_path is not None:
        print(f"Loading model weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("Model weights loaded successfully - starting new training run from epoch 0")
    else:
        print("Initializing weights: starting from scratch")
        model._initialize_weights()
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='sr-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Add validation image saving callback
    save_images_callback = SaveValidationImages(
        save_dir='validation_images',
        save_every_n_epochs=5,
        num_images=4,
        augmentation=augmentation
    )
    
    logger = logger = CSVLogger('logs', name='super_resolution')
    

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, save_images_callback],
        logger=logger,
        precision='16-mixed', 
        gradient_clip_val=1.0,
        log_every_n_steps=5,
    )
    
    model.configure_optimizers = lambda: {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'monitor': 'val_loss',
        }
    }
    
    # Train (always start fresh)
    trainer.fit(model, train_loader, val_loader)
    
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Super Resolution Model')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Validation data directory')
    parser.add_argument('--scale_factor', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    best_model_path = train(
        train_data_dir=args.train_dir,
        val_data_dir=args.val_dir,
        scale_factor=args.scale_factor,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.workers,
        accelerator=args.accelerator,
        devices=args.devices,
        checkpoint_path=args.checkpoint,
    )