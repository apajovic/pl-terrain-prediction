# data/dataloader.py
# DataLoader and preprocessing utilities
import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
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
        row_averages = img_tensor.min(dim=1, keepdim=True).values  # Max across columns for each row

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
    def __init__(self, input_dirs, target_dir, transform=None):
        """
        Args:
            input_dirs: List of input directories (one per channel) or single directory string
            target_dir: Target directory path
            transform: Transform to apply (used for resize and ToTensor operations)
        """
        # Convert single directory to list for uniform handling
        if isinstance(input_dirs, str):
            input_dirs = [input_dirs]

        self.input_dirs = input_dirs
        self.target_dir = target_dir
        self.transform = transform
        self.num_channels = len(input_dirs)

        # Load filenames from first directory as reference
        self.input_filenames = sorted([
            f for f in os.listdir(input_dirs[-1]) if f.endswith('.png')
        ])
        self.target_filenames = sorted([
            f for f in os.listdir(target_dir) if f.endswith('.png')
        ])

        # Verify filenames match across all input directories
        if self.num_channels > 1:
            for i, input_dir in enumerate(input_dirs[1:], start=1):
                filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
                if filenames != self.input_filenames:
                    print(f"Warning: Filenames in input_dirs[{i}] don't match input_dirs[0]")

        print(f"Loaded {len(self.input_filenames)} samples with {self.num_channels} input channel(s)")
        print(f"Sample filenames: {self.input_filenames[:5]}")

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        filename = self.input_filenames[idx]
        target_path = os.path.join(self.target_dir, self.target_filenames[idx])
        target_img = Image.open(target_path).convert('L')

        # Load all input channels
        input_tensors = []
        for input_dir in self.input_dirs:
            input_path = os.path.join(input_dir, filename)
            if not os.path.exists(input_path):
              # try finding the file without _
              alt_input_path = os.path.join(input_dir, filename.split('_')[0]+'.png')
              if os.path.exists(alt_input_path):
                input_path = alt_input_path
              else:
                raise FileNotFoundError(f"Input file not found: {input_path} or {alt_input_path}")
            input_img = Image.open(input_path).convert('L')
            # print("input image:", input_path)

            # Convert to tensor
            input_tensor = transforms.ToTensor()(input_img)

            # Apply resize if present in transform
            if self.transform:
                for t in (self.transform.transforms if hasattr(self.transform, 'transforms') else [self.transform]):
                    if isinstance(t, transforms.Resize):
                        input_tensor = t(input_tensor)
                        break

            input_tensors.append(input_tensor.type(torch.float32))  # Ensure float32 for model input

        # Stack all channels into single tensor (C, H, W)
        if len(input_tensors) > 1:
            input_img = torch.cat(input_tensors, dim=0)

        else:
            # Single channel: apply full transform if available
            if self.transform:
                input_img = Image.open(os.path.join(self.input_dirs[0], filename)).convert('L')
                input_img = self.transform(input_img)
            else:
                input_img = input_tensors[0].type(torch.float32)  # Ensure float32 for model input

        # Transform target (always single channel)
        if self.transform:
            target_img = self.transform(target_img).type(torch.float32)  # Ensure float32 for model output
        else:
            target_img = transforms.ToTensor()(target_img).type(torch.float32)  # Ensure float32 for model output
        # print("target image:", target_path)
        return input_img, target_img


def get_dataloaders(config, val_split=0.1, transform=BASIC_TRANSFORM, use_ddp=False):
    """
    Get train and validation dataloaders.

    Args:
        config: Configuration object
        val_split: Fraction of data to use for validation
        transform: Transform to apply to images
        use_ddp: Whether to use DistributedDataParallel (requires DDP initialization)
    """
    input_dirs = config.get('data.input_dirs', None)
    if input_dirs is None:
        input_dirs = config.get('data.input_dir')

    target_dir = config.get('data.target_dir')
    dataset = ImageToImageDataset(input_dirs, target_dir, transform=transform)
    batch_size = config.get('training.batch_size', 16)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('training.seed', 42))
    )

    if use_ddp:
        import torch.distributed as dist

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=config.get('training.seed', 42),
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            seed=config.get('training.seed', 42),
            drop_last=True,   # Both ranks must have equal batch count
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=0, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=config.get('training.num_workers', 4), pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=config.get('training.num_workers', 4), pin_memory=True,
        )

    return train_loader, val_loader

