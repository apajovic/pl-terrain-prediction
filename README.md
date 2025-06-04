# PathLoss Terrain Prediction

Repository for the research in potential model architectures for pathloss prediction from terrain images.

## Features
- Data preprocessing
- Model training
- Hyperparameter optimization with Optuna
- Evaluation and postprocessing utilities
- Modular, configurable pipeline

## Project Structure
```
RadioNet/
├── data/                # Input and output data folders
│   ├── PL_unwrap_orig/  # Unwrapped input images
│   ├── PL_wrap_orig/    # Wrapped target images
│   ├── PL_wrap_pred/    # Model predictions
│   └── ...
├── output/              # Model checkpoints and metrics
├── src/
│   └── python/          # Main source code
│       ├── config.py
│       ├── train.py
│       ├── evaluate.py
│       ├── preprocess.py
│       ├── postprocess.py
│       ├── data/
│       └── models/
├── requirements.txt     # Python dependencies
├── default_config.json  # Main configuration file
└── README.md            # Project documentation
```

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Configure the project**:
   - Edit `default_config.json` to set data paths, model parameters, and training options.

## Usage
### Preprocessing
Preprocessing is run automatically before training (unless skipped in config):
- Unwraps images
- Saves processed tensors and images

### Training
To train a model:
```bash
python src/python/train.py
```
- Model type is selected via `"model.name"` in `default_config.json` (default: `unet`).
- Training, validation, and output paths are all configurable.
- Hyperparameter search with Optuna can be enabled via config.

### Evaluation
To evaluate a trained model:
```bash
python src/python/evaluate.py
```
- Loads the best model and computes metrics on the validation set.

## Configuration
All settings are managed in `default_config.json` using subkeys for logical grouping. Example:
```json
{
  "data": {
    "train_input_dir": "./data/PL_unwrap_orig",
    "train_target_dir": "./data/PL_wrap_orig",
    ...
  },
  "model": {
    "name": "unet",
    "model_out": "./output/best_model.pth"
  },
  "training": {
    "batch_size": 16,
    "epochs": 150,
    ...
  },
  "output": {
    "metrics_out": "./output/metrics.txt"
  }
}
```

## Extending
- Add new models in `src/python/models/` and update `get_model` in `train.py`.
- Add new preprocessing or postprocessing steps as needed.


