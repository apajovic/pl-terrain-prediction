# data/dataloader.py
# DataLoader and preprocessing utilities
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class RowAverageSplitTransform:
    """
    Custom transform that splits an image into two channels:
    - Channel 1: Original image minus row-averaged image
    - Channel 2: Row-averaged image (each column has the same pixel values, 
                 computed as the average of the corresponding row)
    """
    
    def __call__(self, img):
        # Convert PIL image to tensor if it isn't already
        if isinstance(img, Image.Image):
            img_tensor = transforms.ToTensor()(img)
        else:
            img_tensor = img
        
        # Ensure single channel (grayscale)
        if img_tensor.dim() == 3 and img_tensor.size(0) == 1:
            img_tensor = img_tensor.squeeze(0)  # Remove channel dimension
        elif img_tensor.dim() == 3:
            # If RGB, convert to grayscale
            img_tensor = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        
        # Calculate row averages
        row_averages = img_tensor.mean(dim=1, keepdim=True)  # Average across columns for each row
        
        # Create row-averaged image (same value across each row)
        row_avg_image = row_averages.expand_as(img_tensor)  # Broadcast to full image size
        
        # Create difference image (original - row averaged)
        diff_image = img_tensor - row_avg_image
        
        # Stack into 2-channel image: [diff_channel, row_avg_channel]
        two_channel_image = torch.stack([diff_image, row_avg_image], dim=0)
        
        return two_channel_image
    
class RowAverageSplitInverseTransform:
    def __call__(self, two_channel_tensor):
        """
        Reconstruct the original image from the two-channel representation.
        
        Args:
            two_channel_tensor: Tensor of shape (2, H, W) where:
                                - channel 0: difference (original - row_avg)
                                - channel 1: row average image
        
        Returns:
            reconstructed_image: Tensor of shape (H, W) representing the original image
        """
        if two_channel_tensor.dim() != 3 or two_channel_tensor.size(0) != 2:
            raise ValueError(f"Expected tensor of shape (2, H, W), got {two_channel_tensor.shape}")
        
        diff_channel = two_channel_tensor[0]  # Original - row average
        row_avg_channel = two_channel_tensor[1]  # Row averaged image
        
        # Reconstruct: original = diff + row_avg
        reconstructed = diff_channel + row_avg_channel
        
        return reconstructed




ROW_SPLIT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    RowAverageSplitTransform()  # This will output 2-channel tensors
])
BASIC_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class ImageToImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_filenames = sorted([
            f for f in os.listdir(input_dir) if f.endswith('.png')
        ])
        self.target_filenames = sorted([
            f for f in os.listdir(target_dir) if f.endswith('.png')
        ])

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_filenames[idx])
        target_path = os.path.join(self.target_dir, self.target_filenames[idx])
        input_img = Image.open(input_path).convert('L')
        target_img = Image.open(target_path).convert('L')
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        return input_img, target_img


def get_dataloaders(config, val_split=0.1, transform=BASIC_TRANSFORM):
    """
    Get train and validation dataloaders.
    
    Args:
        config: Configuration object
        val_split: Fraction of data to use for validation
        use_row_split: Whether to use the row average split transformation
    """
   

    input_dir = config.get('data.input_dir')
    target_dir = config.get('data.target_dir')
    dataset = ImageToImageDataset(input_dir, target_dir, transform=transform)
    batch_size = config.get('training.batch_size', 16)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.get('training.seed', 42)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

