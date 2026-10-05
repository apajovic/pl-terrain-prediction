import os
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from config import get_config
from evaluate import evaluate


DEFAULT_CONFIGS = [
    './configs/radiounet.json',
    './configs/radiounet-no-unwrapping.json',
    './configs/mamba.json',
    './configs/mamba-no-unwrapping.json',
    './configs/pmnetv3.json',
    './configs/pmnetv3-baseline-centered.json',
    './configs/transunet-r50-rt.json',
    './configs/transunet-r50-rt-no-unwrapping.json',
]


def _slug_from_config_path(config_path):
    return os.path.splitext(os.path.basename(config_path))[0]


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _safe_float(value):
    if value is None:
        return np.nan
    return float(value)


def _save_summary_csv(results, out_csv):
    fields = [
        'config',
        'slug',
        'model',
        'test_rmse',
        'test_rmse_dB',
        'full_rmse_vs_original',
        'full_rmse_vs_original_dB',
        'corrected_full_rmse',
        'corrected_full_rmse_dB',
        'transform_only_rmse',
        'transform_only_rmse_dB',
        'p90_abs_error_db',
        'avg_pred_time_per_sample',
        'avg_postprocess_per_sample',
        'avg_e2e_time_per_sample',
        'analysis_dir',
    ]  # full_rmse_vs_original_dB is rmse_to_dB(rmse_px/255) — same scale as test_rmse_dB
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


def _plot_metric_bars(results, out_dir):
    labels = [r['slug'] for r in results]
    x = np.arange(len(labels))

    full_rmse_db = np.array([_safe_float(r.get('full_rmse_vs_original_dB')) for r in results])
    corrected_rmse_db = np.array([_safe_float(r.get('corrected_full_rmse_dB')) for r in results])
    test_rmse_db = np.array([_safe_float(r.get('test_rmse_dB')) for r in results])

    plt.figure(figsize=(14, 6), dpi=140)
    width = 0.25
    plt.bar(x - width, test_rmse_db, width=width, label='Test RMSE [dB]')
    plt.bar(x, corrected_rmse_db, width=width, label='Corrected Full RMSE [dB]')
    plt.bar(x + width, full_rmse_db, width=width, label='Full RMSE [dB] (raw)', alpha=0.5)
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('RMSE [dB]')
    plt.title('Model Comparison: RMSE Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmse_comparison_bars.png'))
    plt.close()

    infer_time = np.array([_safe_float(r.get('avg_pred_time_per_sample')) for r in results])
    pp_time = np.array([_safe_float(r.get('avg_postprocess_per_sample')) for r in results])
    width = 0.38
    plt.figure(figsize=(12, 5), dpi=140)
    plt.bar(x - width / 2, infer_time, width=width, label='Inference only', color='#2c7fb8')
    plt.bar(x + width / 2, pp_time, width=width, label='Postprocessing (wrap)', color='#fc8d59')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('Average time per sample [s]')
    plt.title('Inference + Postprocessing Speed Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'speed_comparison.png'))
    plt.close()


def _load_zone_rows(zone_csv):
    if not os.path.exists(zone_csv):
        return []
    rows = []
    with open(zone_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _plot_zone_overlay(results, out_dir):
    # Raw zone overlay
    plt.figure(figsize=(10, 6), dpi=140)
    any_curve = False

    for r in results:
        rows = _load_zone_rows(r.get('zone_csv', ''))
        if not rows:
            continue
        zones = [int(z['zone']) for z in rows]
        vals = []
        for z in rows:
            v = z.get('rmse_dB')
            vals.append(np.nan if v in (None, '', 'None') else float(v))
        plt.plot(zones, vals, '-o', linewidth=1.8, label=r['slug'])
        any_curve = True

    if any_curve:
        plt.xlabel('Radial zone')
        plt.ylabel('RMSE [dB]')
        plt.title('RMSE by Distance Zone (raw, includes transform error)')
        plt.xticks(np.arange(1, 9))
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'rmse_by_zone_overlay.png'))
    plt.close()

    # Corrected zone overlay (transform error subtracted in quadrature)
    plt.figure(figsize=(10, 6), dpi=140)
    any_curve = False

    for r in results:
        csv_path = r.get('corrected_zone_csv', '')
        rows = _load_zone_rows(csv_path)
        if not rows:
            continue
        zones = [int(z['zone']) for z in rows]
        vals = []
        for z in rows:
            v = z.get('rmse_dB')
            vals.append(np.nan if v in (None, '', 'None') else float(v))
        plt.plot(zones, vals, '-o', linewidth=1.8, label=r['slug'])
        any_curve = True

    if any_curve:
        plt.xlabel('Radial zone')
        plt.ylabel('RMSE [dB]')
        plt.title('RMSE by Distance Zone (corrected — transform error removed)')
        plt.xticks(np.arange(1, 9))
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'rmse_by_zone_corrected_overlay.png'))
    plt.close()


def _load_cdf_points(cdf_csv, max_points=2000):
    if not os.path.exists(cdf_csv):
        return None, None

    xs, ys = [], []
    with open(cdf_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row['abs_error_dB']))
            ys.append(float(row['cdf']))

    if not xs:
        return None, None

    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs = [xs[i] for i in idx]
        ys = [ys[i] for i in idx]

    return xs, ys


def _plot_cdf_overlay(results, out_dir):
    plt.figure(figsize=(10, 6), dpi=140)
    any_curve = False

    for r in results:
        xs, ys = _load_cdf_points(r.get('cdf_csv', ''))
        if xs is None:
            continue
        plt.plot(xs, ys, linewidth=1.7, label=r['slug'])
        any_curve = True

    if not any_curve:
        plt.close()
        return

    plt.xlabel('Absolute error [dB]')
    plt.ylabel('CDF')
    plt.title('CDF Overlay: Absolute Error Distribution')
    plt.ylim([0, 1.02])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cdf_overlay.png'))
    plt.close()


def _collect_sorted_images(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.lower().endswith('.png')])


def _save_sample_grids(results, out_dir, max_images=5):
    if not results:
        return

    targets_dir = results[0].get('targets_dir')
    if not targets_dir or not os.path.isdir(targets_dir):
        return

    target_files = _collect_sorted_images(targets_dir)
    if not target_files:
        return

    n_images = min(max_images, len(target_files))
    grid_dir = os.path.join(out_dir, 'sample_visuals')
    _ensure_dir(grid_dir)

    for idx in range(n_images):
        t_name = target_files[idx]
        t_path = os.path.join(targets_dir, t_name)
        target_img = imread(t_path)

        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 13), dpi=120)
        axes = axes.reshape(-1)

        axes[0].imshow(target_img, cmap='viridis')
        axes[0].set_title('Target')
        axes[0].axis('off')

        for ax_i, r in enumerate(results[:8], start=1):
            pred_dir = r.get('predictions_dir')
            pred_files = _collect_sorted_images(pred_dir)
            if idx < len(pred_files):
                pred_img = imread(os.path.join(pred_dir, pred_files[idx]))
                diff = np.abs(pred_img.astype(float) - target_img.astype(float))
                axes[ax_i].imshow(diff, cmap='magma')
                axes[ax_i].set_title(f"{r['slug']} | abs diff")
            else:
                axes[ax_i].text(0.5, 0.5, 'missing', ha='center', va='center')
            axes[ax_i].axis('off')

        fig.suptitle(f'Sample {idx + 1}: target + per-model absolute error', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(grid_dir, f'sample_{idx + 1:02d}_comparison.png'))
        plt.close(fig)


def _plot_benchmark_896(out_dir, benchmark_csv='./output/benchmark_896/benchmark.csv'):
    """Generate 256 vs 896 comparison charts from the benchmark CSV."""
    if not os.path.exists(benchmark_csv):
        print(f'Benchmark CSV not found at {benchmark_csv}, skipping 896 charts.')
        return

    rows = []
    with open(benchmark_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('status') != 'ok':
                continue
            rows.append(row)

    if not rows:
        return

    models = sorted(set(r['model'] for r in rows))
    sizes = [256, 896]
    lookup = {}
    for r in rows:
        lookup[(r['model'], int(r['img_size']))] = r

    # --- Inference time chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=140)
    x = np.arange(len(models))
    width = 0.35

    t256 = [float(lookup.get((m, 256), {}).get('inference_ms', 0)) for m in models]
    t896 = [float(lookup.get((m, 896), {}).get('inference_ms', 0)) for m in models]

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, t256, width, label='256×256', color='#2c7fb8')
    bars2 = ax.bar(x + width / 2, t896, width, label='896×896', color='#d95f02')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('Inference time [ms]')
    ax.set_title('Inference Time: 256×256 vs 896×896')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, val in zip(bars1, t256):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, t896):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # --- GPU memory chart ---
    m256 = [float(lookup.get((m, 256), {}).get('peak_gpu_mb', 0)) for m in models]
    m896 = [float(lookup.get((m, 896), {}).get('peak_gpu_mb', 0)) for m in models]

    ax = axes[1]
    bars1 = ax.bar(x - width / 2, m256, width, label='256×256', color='#2c7fb8')
    bars2 = ax.bar(x + width / 2, m896, width, label='896×896', color='#d95f02')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.set_ylabel('Peak GPU memory [MB]')
    ax.set_title('GPU Memory: 256×256 vs 896×896')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for bar, val in zip(bars1, m256):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, m896):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'benchmark_256_vs_896.png'))
    plt.close(fig)

    # --- Slowdown / memory factor table ---
    table_path = os.path.join(out_dir, 'benchmark_256_vs_896.csv')
    with open(table_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model', 'params',
            'inference_256_ms', 'inference_896_ms', 'slowdown_x',
            'gpu_256_mb', 'gpu_896_mb', 'mem_factor_x',
            'model_size_mb',
        ])
        for m in models:
            r256 = lookup.get((m, 256), {})
            r896 = lookup.get((m, 896), {})
            t2 = float(r256.get('inference_ms', 0))
            t8 = float(r896.get('inference_ms', 0))
            g2 = float(r256.get('peak_gpu_mb', 0))
            g8 = float(r896.get('peak_gpu_mb', 0))
            writer.writerow([
                m,
                r256.get('params_total', r896.get('params_total', '')),
                f'{t2:.2f}', f'{t8:.2f}',
                f'{t8 / t2:.1f}x' if t2 > 0 else 'n/a',
                f'{g2:.0f}', f'{g8:.0f}',
                f'{g8 / g2:.1f}x' if g2 > 0 else 'n/a',
                r256.get('model_size_mb', r896.get('model_size_mb', '')),
            ])

    print(f'Benchmark 256 vs 896 chart saved to {out_dir}/benchmark_256_vs_896.png')
    print(f'Benchmark 256 vs 896 CSV saved to {table_path}')


def run_comparison(config_paths, output_dir, evaluation_samples):
    _ensure_dir(output_dir)

    results = []
    for cfg_path in config_paths:
        slug = _slug_from_config_path(cfg_path)
        model_out_dir = os.path.join(output_dir, slug)
        pred_dir = os.path.join(model_out_dir, 'predictions')
        targ_dir = os.path.join(model_out_dir, 'targets')
        metrics_path = os.path.join(model_out_dir, 'metrics.txt')

        _ensure_dir(model_out_dir)
        _ensure_dir(pred_dir)
        _ensure_dir(targ_dir)

        cfg = get_config(cfg_path)
        cfg_dict = cfg.as_dict()
        cfg_dict.setdefault('output', {})
        cfg_dict['output']['wrap_pred_dir'] = pred_dir
        cfg_dict['output']['wrap_targets_dir'] = targ_dir
        cfg_dict['output']['metrics_out'] = metrics_path

        summary = evaluate(cfg, evaluation_samples=evaluation_samples)
        summary['config'] = cfg_path
        summary['slug'] = slug
        results.append(summary)

    summary_csv = os.path.join(output_dir, 'comparison_summary.csv')
    _save_summary_csv(results, summary_csv)
    _plot_metric_bars(results, output_dir)
    _plot_zone_overlay(results, output_dir)
    _plot_cdf_overlay(results, output_dir)
    _save_sample_grids(results, output_dir, max_images=5)
    _plot_benchmark_896(output_dir)

    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Comparison complete. Artifacts saved to: {output_dir}')
    print(f'Summary CSV: {summary_csv}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multiple configs and compare metrics in one directory.')
    parser.add_argument(
        '--configs',
        nargs='*',
        default=DEFAULT_CONFIGS,
        help='List of config file paths to evaluate.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/comparison_evaluation',
        help='Directory for consolidated comparison artifacts.'
    )
    parser.add_argument(
        '--evaluation_samples',
        type=int,
        default=10,
        help='Number of batches to evaluate for each config.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_comparison(args.configs, args.output_dir, args.evaluation_samples)
