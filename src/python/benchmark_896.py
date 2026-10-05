#!/usr/bin/env python3
"""
Quick benchmark: model size, GPU memory, and inference time at 256x256 vs 896x896.
No training data required — uses random tensors.
"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 3
REPEATS = 10

# ── model builders (bypass config objects) ──────────────────────────────

def build_radiounet(in_ch=1):
    from config import get_config
    cfg = get_config("configs/radiounet.json")
    from models.radiounet import RadioWNet
    return RadioWNet(cfg)

def build_mamba(in_ch=1):
    from models.mamba import RadioMambaNet
    return RadioMambaNet(in_channels=in_ch)

def build_pmnetv3(in_ch=1):
    from config import get_config
    cfg = get_config("configs/pmnetv3.json")
    from models.pmnet_v3 import PMNet
    return PMNet(cfg)

def build_transunet_r50(img_size=256):
    from models.transunet import VisionTransformer, CONFIGS
    return VisionTransformer(config=CONFIGS['R50-ViT-B_16'], img_size=img_size)

# ── helpers ─────────────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model):
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)

def measure_inference(model, dummy_input, warmup=WARMUP, repeats=REPEATS):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)

def measure_peak_memory(model, dummy_input):
    """Returns peak GPU memory in MB during a forward pass."""
    if DEVICE != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
        torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

# ── main ────────────────────────────────────────────────────────────────

MODELS = {
    "radiounet":       {"builder": build_radiounet,      "in_ch": 1},
    "mamba":           {"builder": build_mamba,           "in_ch": 1},
    "pmnetv3":         {"builder": build_pmnetv3,        "in_ch": 1},
    "transunet-r50":   {"builder": build_transunet_r50,  "in_ch": 1},
}

SIZES = [256, 896]

def run():
    results = []
    for name, spec in MODELS.items():
        for sz in SIZES:
            print(f"\n{'='*60}")
            print(f"  {name}  @  {sz}x{sz}")
            print(f"{'='*60}")
            torch.cuda.empty_cache() if DEVICE == "cuda" else None

            try:
                if name == "transunet-r50":
                    model = spec["builder"](img_size=sz)
                else:
                    model = spec["builder"](in_ch=spec["in_ch"])
                model = model.to(DEVICE)

                total_p, train_p = count_params(model)
                sz_mb = model_size_mb(model)
                print(f"  Params total:     {total_p:>12,}")
                print(f"  Params trainable: {train_p:>12,}")
                print(f"  Model size:       {sz_mb:>10.2f} MB")

                dummy = torch.randn(1, spec["in_ch"], sz, sz, device=DEVICE)
                peak_mem = measure_peak_memory(model, dummy)
                print(f"  Peak GPU mem:     {peak_mem:>10.1f} MB")

                avg_t, std_t = measure_inference(model, dummy)
                print(f"  Inference time:   {avg_t*1000:>10.2f} ms  (±{std_t*1000:.2f})")

                results.append({
                    "model": name,
                    "img_size": sz,
                    "params_total": total_p,
                    "params_trainable": train_p,
                    "model_size_mb": round(sz_mb, 2),
                    "peak_gpu_mb": round(peak_mem, 1),
                    "inference_ms": round(avg_t * 1000, 2),
                    "inference_std_ms": round(std_t * 1000, 2),
                    "status": "ok",
                })
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({
                    "model": name,
                    "img_size": sz,
                    "status": f"FAILED: {e}",
                })
            finally:
                # free GPU
                try:
                    del model
                except:
                    pass
                torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # ── save results ────────────────────────────────────────────────
    out_dir = "output/benchmark_896"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(out_dir, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "img_size", "params_total", "params_trainable",
            "model_size_mb", "peak_gpu_mb", "inference_ms", "inference_std_ms", "status"
        ])
        w.writeheader()
        w.writerows(results)

    # ── summary table ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Model':<20} {'Size':>5} {'Params':>12} {'MB':>8} {'GPU MB':>8} {'ms':>10} {'Status'}")
    print(f"{'-'*80}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['model']:<20} {r['img_size']:>5} {r['params_total']:>12,} "
                  f"{r['model_size_mb']:>8.2f} {r['peak_gpu_mb']:>8.1f} "
                  f"{r['inference_ms']:>10.2f} {r['status']}")
        else:
            print(f"{r['model']:<20} {r.get('img_size','?'):>5} {'':>12} "
                  f"{'':>8} {'':>8} {'':>10} {r['status']}")

    print(f"\nResults saved to {out_dir}/")

def run_transform_benchmark():
    """Benchmark radial_unwrap + radial_wrap round-trip at 256 vs 896 polar params."""
    from preprocess import radial_unwrap
    from postprocess import radial_wrap

    # Use a realistic 256x256 test image (smooth gradient + circle)
    img_size = (256, 256)
    center = (128, 128)
    YY, XX = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]), indexing='ij')
    dist = np.sqrt((XX - center[1])**2 + (YY - center[0])**2)
    test_img = np.clip(200 - dist, 0, 255).astype(np.float64)

    POLAR_SIZES = [256, 896]
    REPEATS_T = 5
    results = []

    for polar_sz in POLAR_SIZES:
        print(f"\n{'='*60}")
        print(f"  Radial transform  @  {polar_sz} angles x {polar_sz} radii")
        print(f"{'='*60}")

        # --- unwrap benchmark ---
        times_unwrap = []
        for _ in range(REPEATS_T):
            t0 = time.perf_counter()
            polar = radial_unwrap(test_img, num_angles=polar_sz, num_radii=polar_sz, center=center)
            times_unwrap.append(time.perf_counter() - t0)
        avg_unwrap = np.mean(times_unwrap) * 1000
        std_unwrap = np.std(times_unwrap) * 1000

        # --- wrap benchmark ---
        times_wrap = []
        for _ in range(REPEATS_T):
            t0 = time.perf_counter()
            recon = radial_wrap(polar, img_size, center)
            times_wrap.append(time.perf_counter() - t0)
        avg_wrap = np.mean(times_wrap) * 1000
        std_wrap = np.std(times_wrap) * 1000

        # --- reconstruction error ---
        max_r = min(center[0], center[1], img_size[0]-center[0], img_size[1]-center[1])
        circle_mask = dist <= max_r
        valid = circle_mask & ~np.isnan(recon)
        rmse = np.sqrt(np.mean((test_img[valid] - recon[valid])**2))

        print(f"  Unwrap time:      {avg_unwrap:>10.2f} ms  (±{std_unwrap:.2f})")
        print(f"  Wrap time:        {avg_wrap:>10.2f} ms  (±{std_wrap:.2f})")
        print(f"  Round-trip:       {avg_unwrap+avg_wrap:>10.2f} ms")
        print(f"  Recon RMSE:       {rmse:>10.4f} px  (on 0-255 scale)")

        results.append({
            "polar_size": polar_sz,
            "unwrap_ms": round(avg_unwrap, 2),
            "unwrap_std_ms": round(std_unwrap, 2),
            "wrap_ms": round(avg_wrap, 2),
            "wrap_std_ms": round(std_wrap, 2),
            "roundtrip_ms": round(avg_unwrap + avg_wrap, 2),
            "recon_rmse_px": round(rmse, 4),
        })

    # ── save ────────────────────────────────────────────────────────
    out_dir = "output/benchmark_896"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "transform_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "polar_size", "unwrap_ms", "unwrap_std_ms",
            "wrap_ms", "wrap_std_ms", "roundtrip_ms", "recon_rmse_px"
        ])
        w.writeheader()
        w.writerows(results)

    with open(os.path.join(out_dir, "transform_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Polar size':>12} {'Unwrap ms':>12} {'Wrap ms':>12} {'Total ms':>12} {'RMSE px':>10}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['polar_size']:>12} {r['unwrap_ms']:>12.2f} {r['wrap_ms']:>12.2f} "
              f"{r['roundtrip_ms']:>12.2f} {r['recon_rmse_px']:>10.4f}")

    if len(results) == 2:
        r256, r896 = results[0], results[1]
        print(f"\n  Unwrap slowdown:  {r896['unwrap_ms']/r256['unwrap_ms']:.1f}x")
        print(f"  Wrap slowdown:    {r896['wrap_ms']/r256['wrap_ms']:.1f}x")
        print(f"  RMSE improvement: {r256['recon_rmse_px']:.4f} → {r896['recon_rmse_px']:.4f} px "
              f"({r256['recon_rmse_px']/max(r896['recon_rmse_px'],1e-9):.1f}x better)")

    print(f"\nResults saved to {out_dir}/transform_benchmark.csv")
    return results


if __name__ == "__main__":
    run()
    run_transform_benchmark()
