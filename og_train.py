import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(64, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = conv_block(128, 256)
        self.pool5 = nn.MaxPool2d(2)
        self.enc6 = conv_block(256, 512)
        self.pool6 = nn.MaxPool2d(2)
        self.enc7 = conv_block(512, 1024)
        self.pool7 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(1024, 2048)

        self.up7 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec7 = conv_block(2048, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec6 = conv_block(1024, 512)

        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec5 = conv_block(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = conv_block(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = conv_block(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)

        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc5 = self.enc5(self.pool4(enc4))
        enc6 = self.enc6(self.pool5(enc5))
        enc7 = self.enc7(self.pool6(enc6))

        bottleneck = self.bottleneck(self.pool7(enc7))

        dec7 = self.dec7(torch.cat((self.up7(bottleneck), enc7), dim=1))
        dec6 = self.dec6(torch.cat((self.up6(dec7), enc6), dim=1))
        dec5 = self.dec5(torch.cat((self.up5(dec6), enc5), dim=1))
        dec4 = self.dec4(torch.cat((self.up4(dec5), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))

        return self.final(dec1)
# Dataset klasa
class TerrainDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, normalizer=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.normalizer = normalizer

        self.input_filenames = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
        self.target_filenames = sorted([f for f in os.listdir(target_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        input_name = self.input_filenames[idx]
        index = input_name.split("_")[-1]
        target_name = f"PL_{index}"

        input_img = Image.open(os.path.join(self.input_dir, input_name)).convert("L")
        target_img = Image.open(os.path.join(self.target_dir, target_name)).convert("L")

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img

# Putanje
train_input_dir = './data/ter_unwrap'
train_target_dir = './data/PL_unwrap_orig'

# Transformacija
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor() #umesto ToTensorNoNorm
])

# Dataset i DataLoader
full_dataset = TerrainDataset(train_input_dir, train_target_dir, transform=transform)
train_size = int(0.95 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=4)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
#model = FlexibleUNet(in_channels=1, out_channels=1, init_features=32, depth=5).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=130, gamma=0.1)

# Trening
num_epoch = 150
for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    '''# Validacija
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")'''
    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}")
    for param_group in optimizer.param_groups:
        print(f"Learning rate = {param_group['lr']}")
    # Čuvanje modela i denormalizatora
    torch.save(model.state_dict(), "./out/my_model_py.pth")
