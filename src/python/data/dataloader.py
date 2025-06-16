# data/dataloader.py
# DataLoader and preprocessing utilities
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os

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


def get_dataloaders(config, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_dir = config.get('data.input_dir')
    target_dir = config.get('data.target_dir')
    dataset = ImageToImageDataset(input_dir, target_dir, transform=transform)
    batch_size = config.get('training.batch_size', 16)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

