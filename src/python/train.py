# train.py
# Main training script (configurable, supports Optuna, local/AML)
import os
import torch

from data.dataloader import get_dataloader
import matplotlib.pyplot as plt
from config import get_config
from utils import set_seed
import optuna
from preprocess import unwrap_img
from postprocess import wrap_img
from models import unet


def preprocess_data(config):
    unwrap_img(
        config.get('data.unwrap_dir'),
        config.get('data.wrap_dir'),
        config.get('data.num_angles', 256),
        config.get('data.num_radii', 256),
        (256, 256),
        config.get('data.base_name', 'PL')
    )


def postprocess_and_plot(pred_tensor, config, save_dir=None, show=True):
    pred_np = pred_tensor.detach().cpu().numpy().squeeze(1)
    wrap_dir = save_dir or config.get('model.wrap_pred_dir', './wrap_pred')
    os.makedirs(wrap_dir, exist_ok=True)
    wrapped = wrap_img(
        wrap_dir,
        pred_np,
        (256, 256),
        config.get('data.base_name', 'PL_pred'),
        indikator=True
    )
    for i in range(min(4, wrapped.shape[-1])):
        plt.imshow(wrapped[:, :, i], cmap='gray')
        plt.title(f'Postprocessed {i+1}')
        if save_dir:
            plt.savefig(os.path.join(wrap_dir, f'postprocessed_{i+1}.png'))
        if show:
            plt.show()
        plt.close()
    return wrapped


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    preds = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            preds.append(outputs.cpu())
    preds = torch.cat(preds, dim=0)
    return val_loss / len(loader), preds


def get_model(config):
    model_name = config.get('model.name', 'unet').lower()
    if model_name == 'unet':
        return unet.UNet(config)
    # Add more models here as needed
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def objective(trial):
    config = get_config()
    set_seed(config.get('training.seed', 42))
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    config.config['batch_size'] = batch_size
    model = get_model(config).to(config.get('training.device', 'cpu'))
    train_loader = get_dataloader(config, train=True)
    val_loader = get_dataloader(config, train=False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_one_epoch(
        model, train_loader, criterion, optimizer, config.get('training.device', 'cpu'))
    val_loss, _ = validate(
        model, val_loader, criterion, config.get('training.device', 'cpu'))
    return val_loss


def main():
    config = get_config()
    print("Loaded config:", config.as_dict())
    set_seed(config.get('training.seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.config['device'] = device
    if not config.get('data.skip_preprocess', False):
        print("Starting preprocessing...")
        preprocess_data(config)
        print("Preprocessing done.")
    else:
        print("Skipping preprocessing.")
    train_loader = get_dataloader(config, train=True)
    val_loader = get_dataloader(config, train=False)
    print("Data loaders ready.")
    model = get_model(config).to(device)
    print(f"Model '{config.get('model.name', 'unet')}' initialized.")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('training.lr', 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.get('training.lr_step', 80),
        gamma=config.get('training.lr_gamma', 0.1))
    num_epochs = config.get('training.epochs', 150)
    best_val_loss = float('inf')
    best_preds = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} starting...")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, preds = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = preds
            torch.save(
                model.state_dict(),
                config.get('model.model_out', './best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")
    if best_preds is not None:
        print("Postprocessing and plotting best predictions...")
        postprocess_and_plot(
            best_preds, config, save_dir=config.get('model.wrap_pred_dir'), show=True)
        print("Postprocessing done.")


if __name__ == '__main__':
    config = get_config()
    if config.get('training.optuna', False):
        print("Optuna hyperparameter search enabled.")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=config.get('training.n_trials', 10))
        print('Best trial:', study.best_trial.params)
    else:
        main()
