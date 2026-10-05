# evaluate.py
# Model evaluation script
from data.dataloader import get_dataloaders, BASIC_TRANSFORM, ROW_SPLIT_TRANSFORM
from config import get_config
import torch
import os
import time
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from postprocess import postprocess
from metrics import rmse_to_dB
from models import get_model
from preprocess import apply_circular_mask
import mlflow
import tqdm
from skimage.io import imread
from skimage.transform import resize as sk_resize

def _map_to_original_filename(filename):
    return filename.replace('_unwrap.png', '.png')


def _read_gray(path):
    img = imread(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(float)


def _resize_like(img, ref_shape):
    if img.shape == ref_shape:
        return img
    return sk_resize(img, ref_shape, order=1, preserve_range=True)


def _apply_building_mask(masked_img, building_img):
    if building_img is None:
        return masked_img
    out = masked_img.copy()
    out[building_img > 0] = np.nan
    return out


def _compute_zone_rmse(error_maps, num_zones=8, zone_step=16):
    if not error_maps:
        return []

    stack = np.stack(error_maps, axis=0)
    h, w = stack.shape[1], stack.shape[2]
    y_c = h / 2.0
    x_c = w / 2.0
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.sqrt((xx - x_c) ** 2 + (yy - y_c) ** 2)

    results = []
    for i in range(num_zones):
        inner_r = i * zone_step
        outer_r = (i + 1) * zone_step
        ring_mask = (dist >= inner_r) & (dist < outer_r)
        ring_values = stack[:, ring_mask]
        valid = ~np.isnan(ring_values)
        if not np.any(valid):
            zone_rmse = None
            zone_rmse_db = None
            zone_rmse_norm = None
            zone_rmse_scaled_db = None
            count_valid = 0
        else:
            mse = np.mean(ring_values[valid] ** 2)
            zone_rmse = float(np.sqrt(mse))
            zone_rmse_db = zone_rmse
            zone_rmse_norm = float(zone_rmse / 255.0)
            zone_rmse_scaled_db = float(rmse_to_dB(zone_rmse_norm))
            count_valid = int(np.sum(valid))

        results.append({
            'zone': i + 1,
            'inner_radius_px': inner_r,
            'outer_radius_px': outer_r,
            'rmse': zone_rmse,
            'rmse_dB': zone_rmse_db,
            'rmse_norm': zone_rmse_norm,
            'rmse_scaled_dB': zone_rmse_scaled_db,
            'valid_points': count_valid,
        })
    return results


def _write_zone_csv(zone_rows, csv_path):
    if not zone_rows:
        return
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'zone',
                'inner_radius_px',
                'outer_radius_px',
                'rmse',
                'rmse_dB',
                'rmse_norm',
                'rmse_scaled_dB',
                'valid_points',
            ]
        )
        writer.writeheader()
        writer.writerows(zone_rows)


def _write_per_image_csv(per_image_rows, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['index', 'filename', 'rmse', 'rmse_dB', 'rmse_norm', 'rmse_scaled_dB', 'valid_points']
        )
        writer.writeheader()
        writer.writerows(per_image_rows)


def _plot_zone_rmse(zone_rows, plot_path):
    if not zone_rows:
        return
    xs = [row['zone'] for row in zone_rows]
    ys = [np.nan if row['rmse_dB'] is None else row['rmse_dB'] for row in zone_rows]
    labels = [f"{row['inner_radius_px']}-{row['outer_radius_px']}" for row in zone_rows]

    plt.figure(figsize=(10, 5), dpi=140)
    plt.plot(xs, ys, '-o', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(xs, labels, rotation=30)
    plt.ylabel('RMSE [dB]')
    plt.xlabel('Distance from transmitter [px]')
    plt.title('RoI Circular RMSE by Radial Zone')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def _compute_and_save_cdf(error_maps, cdf_csv_path, cdf_plot_path):
    if not error_maps:
        return {'p90_abs_error_db': None}

    all_errors = np.concatenate([m[~np.isnan(m)].reshape(-1) for m in error_maps])
    if all_errors.size == 0:
        return {'p90_abs_error_db': None}

    abs_err_db = np.sort(np.abs(all_errors))
    y_vals = (np.arange(1, abs_err_db.size + 1) / abs_err_db.size).astype(float)
    abs_err_norm = abs_err_db / 255.0
    abs_err_scaled_db = np.abs(abs_err_norm * (-75 - (-162)))
    p90_abs_error_db = float(np.percentile(abs_err_db, 90))

    with open(cdf_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['abs_error_dB', 'abs_error_norm', 'abs_error_scaled_dB', 'cdf'])
        writer.writerows(zip(abs_err_db.tolist(), abs_err_norm.tolist(), abs_err_scaled_db.tolist(), y_vals.tolist()))

    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(abs_err_db, y_vals, linewidth=2, color='#d35400')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel('Absolute error [dB]')
    plt.ylabel('Cumulative probability (CDF)')
    plt.title('CDF of Absolute Error')
    plt.xlim(0, max(0.1, np.max(abs_err_db) * 1.05))
    plt.ylim(0, 1.02)
    plt.axhline(0.9, linestyle='--', linewidth=1, color='black')
    plt.axvline(p90_abs_error_db, linestyle='--', linewidth=1, color='black')
    plt.text(p90_abs_error_db, 0.88, f'90% < {p90_abs_error_db:.2f} dB')
    plt.tight_layout()
    plt.savefig(cdf_plot_path)
    plt.close()

    return {'p90_abs_error_db': p90_abs_error_db}


def normalize_tensor(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def evaluate(config, evaluation_samples=10, skip_wrapping=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).to(device)
    # Load weights
    weights_path = config.get('evaluation.model_path', './best_model.pth')
    path_from_mlflow = config.get('evaluation.path_from_mlflow', False)
    if path_from_mlflow:
        print(f"Loading model weights from MLflow artifact: {weights_path}")
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=weights_path)
        model.load_state_dict(torch.load(local_path, map_location=device))
        print(f"Loaded weights from MLflow artifact at {weights_path}")
    elif os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}, using random init.")

    model.eval()

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    preds = []
    targets = []
    total_pred_time = 0.0
    num_batches = 0
    two_channel_split = config.get('data.split_channels', False)
    input_dirs = config.get('data.input_dirs', None)
    has_multiple_input_dirs = input_dirs is not None and isinstance(input_dirs, list) and len(input_dirs) > 1

    # Use ROW_SPLIT_TRANSFORM only if split_channels is enabled and we don't have multiple input dirs
    transform = ROW_SPLIT_TRANSFORM if (two_channel_split and not has_multiple_input_dirs) else BASIC_TRANSFORM
    _, test_loader = get_dataloaders(config, transform=transform, val_split=0.1)

    with torch.no_grad():
        for inputs, targs in tqdm.tqdm(test_loader, total=len(test_loader)):
            inputs = inputs.to(device)
            targs = targs.to(device)
            start_time = time.perf_counter()
            outputs = model(inputs)
            end_time = time.perf_counter()
            batch_time = end_time - start_time
            total_pred_time += batch_time
            num_batches += 1
            preds.append(outputs.cpu())
            targets.append(targs.cpu())
            if num_batches >= evaluation_samples:
                break
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    # Compute MSE
    mse = torch.nn.functional.mse_loss(preds, targets).item()
    print(f"Test RMSE: {np.sqrt(mse):.6f}")
    print(f"Test RMSE in dB: {rmse_to_dB(np.sqrt(mse)):.6f}")

    batch_size = config.get('training.batch_size', 16)
    avg_pred_time = total_pred_time / num_batches if num_batches > 0 else 0.0
    avg_pred_time_per_sample = avg_pred_time / batch_size if batch_size > 0 else 0.0
    print(f"Total prediction time: {total_pred_time:.4f} s")
    print(f"Average prediction time per batch: {avg_pred_time:.6f} s")
    print(f"Average prediction time per sample: {avg_pred_time_per_sample:.6f} s")

    started_mlflow_run = False
    if mlflow.active_run() is None:
        mlflow.start_run(run_name="evaluate")
        started_mlflow_run = True

    mlflow.log_params({
        'model': config.get('model.name', 'unet'),
        'channels': config.get('model.params.channels', 16),
        'layers': config.get('model.params.layers', 7)
    })
    mlflow.log_metrics({
        'test_mse': mse,
        'total_pred_time': total_pred_time,
        'avg_pred_time_per_batch': avg_pred_time
    })

    if two_channel_split:
        preds = preds[:, 0, :, :] + preds[:, 1, :, :]
        preds = normalize_tensor(preds).unsqueeze(1)
        targets = targets[:, 0, :, :] + targets[:, 1, :, :]
        targets = normalize_tensor(targets).unsqueeze(1)

    # Postprocess and plot/save predictions (timed separately for e2e reporting)
    pp_start = time.perf_counter()
    postprocessed_preds = postprocess(preds, config, save_dir=config.get('output.wrap_pred_dir'), show=True, is_tensor=True)
    pp_end = time.perf_counter()
    total_postprocess_time = pp_end - pp_start
    n_postprocessed = len(postprocessed_preds)
    avg_postprocess_per_sample = total_postprocess_time / n_postprocessed if n_postprocessed > 0 else 0.0
    avg_e2e_time_per_sample = avg_pred_time_per_sample + avg_postprocess_per_sample
    print(f"Total postprocessing time: {total_postprocess_time:.4f} s")
    print(f"Average postprocessing time per sample: {avg_postprocess_per_sample:.6f} s")
    print(f"Average end-to-end time per sample (inference + postprocess): {avg_e2e_time_per_sample:.6f} s")
    postprocessed_targets = postprocess(targets, config, save_dir=config.get('output.wrap_targets_dir'), show=True, is_tensor=True)

    metrics_path = config.get('output.metrics_out', './metrics.txt')
    metrics_dir = os.path.dirname(metrics_path) if os.path.dirname(metrics_path) else '.'
    analysis_dir = os.path.join(metrics_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    val_indices = test_loader.dataset.indices
    n_preds = len(postprocessed_preds)
    val_filenames = [
        test_loader.dataset.dataset.input_filenames[i] for i in val_indices[:n_preds]
    ]

    # MATLAB-equivalent masked error: circular mask + optional building mask.
    original_gain_dir = config.get('data.original_gain', None)
    original_buildings_dir = config.get('data.original_buildings', None)

    per_image_rows = []
    per_image_transform_rows = []
    error_maps = []
    transform_error_maps = []

    for i in range(min(n_preds, len(postprocessed_targets), len(val_filenames))):
        filename = val_filenames[i]
        mapped_name = _map_to_original_filename(filename)

        pred_img = postprocessed_preds[i].astype(float)
        fallback_target = postprocessed_targets[i].astype(float)
        target_img = fallback_target

        if original_gain_dir and os.path.isdir(original_gain_dir):
            gain_path = os.path.join(original_gain_dir, mapped_name)
            if os.path.exists(gain_path):
                target_img = _resize_like(_read_gray(gain_path), pred_img.shape)

        building_img = None
        if original_buildings_dir and os.path.isdir(original_buildings_dir):
            building_path = os.path.join(original_buildings_dir, mapped_name)
            if os.path.exists(building_path):
                building_img = _resize_like(_read_gray(building_path), pred_img.shape)

        masked_pred = apply_circular_mask(pred_img)
        masked_targ = apply_circular_mask(target_img)
        masked_pred = _apply_building_mask(masked_pred, building_img)
        masked_targ = _apply_building_mask(masked_targ, building_img)

        error_map = masked_targ - masked_pred
        valid = ~np.isnan(error_map)
        if not np.any(valid):
            continue

        mse_i = np.mean(error_map[valid] ** 2)
        rmse_i = float(np.sqrt(mse_i))
        rmse_i_norm = float(rmse_i / 255.0)
        rmse_i_db = float(rmse_to_dB(rmse_i_norm))

        per_image_rows.append({
            'index': i,
            'filename': mapped_name,
            'rmse': rmse_i,
            'rmse_dB': rmse_i_db,
            'rmse_norm': rmse_i_norm,
            'rmse_scaled_dB': rmse_i_db,
            'valid_points': int(np.sum(valid)),
        })

        # Diagnostic baseline: error introduced by postprocessing targets alone.
        masked_pp_targ = apply_circular_mask(fallback_target)
        masked_pp_targ = _apply_building_mask(masked_pp_targ, building_img)
        transform_error_map = masked_targ - masked_pp_targ
        valid_transform = ~np.isnan(transform_error_map)
        if np.any(valid_transform):
            transform_mse_i = np.mean(transform_error_map[valid_transform] ** 2)
            transform_rmse_i = float(np.sqrt(transform_mse_i))
            per_image_transform_rows.append(transform_rmse_i)

        error_maps.append(error_map)
        transform_error_maps.append(transform_error_map)

    original_rmse = float(np.mean([r['rmse'] for r in per_image_rows])) if per_image_rows else None
    original_rmse_norm = float(original_rmse / 255.0) if original_rmse is not None else None
    original_rmse_db = float(rmse_to_dB(original_rmse_norm)) if original_rmse_norm is not None else None
    transform_only_rmse = float(np.mean(per_image_transform_rows)) if per_image_transform_rows else None
    transform_only_rmse_norm = float(transform_only_rmse / 255.0) if transform_only_rmse is not None else None
    transform_only_rmse_db = (
        float(rmse_to_dB(transform_only_rmse_norm)) if transform_only_rmse_norm is not None else None
    )

    # Corrected full RMSE: subtract transform-only error in quadrature.
    # corrected = sqrt(max(0, full² - transform²))
    if original_rmse is not None and transform_only_rmse is not None:
        corrected_mse = max(0.0, original_rmse ** 2 - transform_only_rmse ** 2)
        corrected_rmse = float(np.sqrt(corrected_mse))
        corrected_rmse_norm = float(corrected_rmse / 255.0)
        corrected_rmse_db = float(rmse_to_dB(corrected_rmse_norm))
    else:
        corrected_rmse = original_rmse
        corrected_rmse_norm = original_rmse_norm
        corrected_rmse_db = original_rmse_db

    if original_rmse is not None:
        print("Full RMSE (postprocessed vs original with radial/building mask): "
              f"{original_rmse:.6f} px  ({original_rmse_db:.6f} dB)")
        print(f"Corrected Full RMSE (transform error subtracted in quadrature): "
              f"{corrected_rmse:.6f} px  ({corrected_rmse_db:.6f} dB)")
        mlflow.log_metrics({
            'full_rmse_vs_original': original_rmse,
            'full_rmse_vs_original_dB': original_rmse_db,
            'full_rmse_vs_original_norm': original_rmse_norm,
            'corrected_full_rmse': corrected_rmse,
            'corrected_full_rmse_dB': corrected_rmse_db,
            'corrected_full_rmse_norm': corrected_rmse_norm,
        })
    else:
        print("Skipping full RMSE: no valid masked pixels found.")

    if transform_only_rmse is not None:
        print(
            "Transform-only RMSE (postprocessed target vs original with radial/building mask): "
            f"{transform_only_rmse:.6f} px  ({transform_only_rmse_db:.6f} dB)"
        )
        mlflow.log_metrics({
            'transform_only_rmse': transform_only_rmse,
            'transform_only_rmse_dB': transform_only_rmse_db,
            'transform_only_rmse_norm': transform_only_rmse_norm,
        })

    zone_rows = _compute_zone_rmse(error_maps, num_zones=8, zone_step=16)

    # Corrected zone RMSE: subtract transform-only zone error in quadrature.
    transform_zone_rows = _compute_zone_rmse(transform_error_maps, num_zones=8, zone_step=16)
    corrected_zone_rows = []
    for zr, tzr in zip(zone_rows, transform_zone_rows):
        czr = dict(zr)
        if zr['rmse'] is not None and tzr['rmse'] is not None:
            corr_mse = max(0.0, zr['rmse'] ** 2 - tzr['rmse'] ** 2)
            corr_rmse = float(np.sqrt(corr_mse))
            czr['rmse'] = corr_rmse
            czr['rmse_norm'] = float(corr_rmse / 255.0)
            czr['rmse_dB'] = float(rmse_to_dB(corr_rmse / 255.0))
            czr['rmse_scaled_dB'] = czr['rmse_dB']
        corrected_zone_rows.append(czr)
    if zone_rows:
        print("Zone RMSE [dB]:")
        for row in zone_rows:
            print(
                f"  Zone {row['zone']} ({row['inner_radius_px']:03d}-{row['outer_radius_px']:03d} px): "
                f"{row['rmse_dB']:.6f} | valid_points={row['valid_points']}"
                if row['rmse_dB'] is not None
                else f"  Zone {row['zone']} ({row['inner_radius_px']:03d}-{row['outer_radius_px']:03d} px): N/A"
            )

    per_image_csv_path = os.path.join(analysis_dir, 'per_image_rmse.csv')
    zone_csv_path = os.path.join(analysis_dir, 'rmse_by_zone.csv')
    corrected_zone_csv_path = os.path.join(analysis_dir, 'rmse_by_zone_corrected.csv')
    zone_plot_path = os.path.join(analysis_dir, 'rmse_by_zone.png')
    cdf_csv_path = os.path.join(analysis_dir, 'error_cdf.csv')
    cdf_plot_path = os.path.join(analysis_dir, 'error_cdf.png')

    _write_per_image_csv(per_image_rows, per_image_csv_path)
    _write_zone_csv(zone_rows, zone_csv_path)
    _write_zone_csv(corrected_zone_rows, corrected_zone_csv_path)
    _plot_zone_rmse(zone_rows, zone_plot_path)
    cdf_summary = _compute_and_save_cdf(error_maps, cdf_csv_path, cdf_plot_path)

    top_k = min(5, len(per_image_rows))
    if top_k > 0:
        sorted_rows = sorted(per_image_rows, key=lambda x: x['rmse'], reverse=True)
        hardest = sorted_rows[:top_k]
        easiest = list(reversed(sorted_rows[-top_k:]))
        topk_fields = ['index', 'filename', 'rmse', 'rmse_dB', 'rmse_norm', 'rmse_scaled_dB', 'valid_points']
        with open(os.path.join(analysis_dir, 'top5_hardest_images.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=topk_fields)
            writer.writeheader()
            writer.writerows(hardest)
        with open(os.path.join(analysis_dir, 'top5_easiest_images.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=topk_fields)
            writer.writeheader()
            writer.writerows(easiest)

    # Optionally: save metrics
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {config.get('model.name', 'unet')}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Test RMSE: {np.sqrt(mse):.6f}\n")
        f.write(f"Test RMSE in dB: {rmse_to_dB(np.sqrt(mse)):.6f}\n")
        if original_rmse is not None:
            f.write(f"Full RMSE (vs original, radial+building mask): {original_rmse:.6f}\n")
            f.write(f"Full RMSE in dB (vs original, radial+building mask): {original_rmse_db:.6f}\n")
            f.write(f"Full RMSE normalized [0..1]: {original_rmse_norm:.6f}\n")
        if transform_only_rmse is not None:
            f.write(
                "Transform-only RMSE (postprocessed target vs original, radial+building mask): "
                f"{transform_only_rmse:.6f}\n"
            )
            f.write(
                "Transform-only RMSE in dB (postprocessed target vs original, radial+building mask): "
                f"{transform_only_rmse_db:.6f}\n"
            )
            f.write(f"Transform-only RMSE normalized [0..1]: {transform_only_rmse_norm:.6f}\n")
        if corrected_rmse is not None:
            f.write(f"Corrected Full RMSE (transform subtracted in quadrature): {corrected_rmse:.6f}\n")
            f.write(f"Corrected Full RMSE in dB: {corrected_rmse_db:.6f}\n")
            f.write(f"Corrected Full RMSE normalized [0..1]: {corrected_rmse_norm:.6f}\n")
        if cdf_summary.get('p90_abs_error_db') is not None:
            f.write(f"P90 absolute error [dB]: {cdf_summary['p90_abs_error_db']:.6f}\n")
        f.write(f"Total prediction time: {total_pred_time:.4f} s\n")
        f.write(f"Average prediction time per batch: {avg_pred_time:.6f} s\n")
        f.write(f"Average prediction time per sample (inference only): {avg_pred_time_per_sample:.6f} s\n")
        f.write(f"Average postprocessing time per sample: {avg_postprocess_per_sample:.6f} s\n")
        f.write(f"Average end-to-end time per sample (inference + postprocess): {avg_e2e_time_per_sample:.6f} s")

    eval_summary = {
        'model': config.get('model.name', 'unet'),
        'test_rmse': float(np.sqrt(mse)),
        'test_rmse_dB': float(rmse_to_dB(np.sqrt(mse))),
        'full_rmse_vs_original': original_rmse,
        'full_rmse_vs_original_dB': original_rmse_db,
        'full_rmse_vs_original_norm': original_rmse_norm,
        'transform_only_rmse': transform_only_rmse,
        'transform_only_rmse_dB': transform_only_rmse_db,
        'transform_only_rmse_norm': transform_only_rmse_norm,
        'corrected_full_rmse': corrected_rmse,
        'corrected_full_rmse_dB': corrected_rmse_db,
        'corrected_full_rmse_norm': corrected_rmse_norm,
        'p90_abs_error_db': cdf_summary.get('p90_abs_error_db'),
        'total_pred_time': float(total_pred_time),
        'avg_pred_time_per_batch': float(avg_pred_time),
        'avg_pred_time_per_sample': float(avg_pred_time_per_sample),
        'avg_postprocess_per_sample': float(avg_postprocess_per_sample),
        'avg_e2e_time_per_sample': float(avg_e2e_time_per_sample),
        'analysis_dir': analysis_dir,
        'metrics_path': metrics_path,
        'per_image_rmse_csv': per_image_csv_path,
        'zone_csv': zone_csv_path,
        'corrected_zone_csv': corrected_zone_csv_path,
        'zone_plot': zone_plot_path,
        'cdf_csv': cdf_csv_path,
        'cdf_plot': cdf_plot_path,
        'predictions_dir': config.get('output.wrap_pred_dir'),
        'targets_dir': config.get('output.wrap_targets_dir'),
    }
    with open(os.path.join(analysis_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(eval_summary, f, indent=2)

    print(f"Metrics saved to {metrics_path}")
    if os.path.exists(metrics_path):
        mlflow.log_artifact(metrics_path, artifact_path="metrics")
    if os.path.exists(per_image_csv_path):
        mlflow.log_artifact(per_image_csv_path, artifact_path="metrics/analysis")
    if os.path.exists(zone_csv_path):
        mlflow.log_artifact(zone_csv_path, artifact_path="metrics/analysis")
    if os.path.exists(cdf_csv_path):
        mlflow.log_artifact(cdf_csv_path, artifact_path="metrics/analysis")

    if started_mlflow_run:
        mlflow.end_run()

    return eval_summary

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='default_config.json', help='Path to configuration file')
    parser.add_argument('--skip_wrapping', action='store_true', help='Skip the wrapping step in postprocessing')
    parser.add_argument('--evaluation_samples', type=int, default=10, help='Number of batches to evaluate')
    args = parser.parse_args()
    config = get_config(args.config)
    evaluate(config, skip_wrapping=args.skip_wrapping, evaluation_samples=args.evaluation_samples)
