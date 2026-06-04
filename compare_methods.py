#!/usr/bin/env python3
"""Compare FREE vs GEMM RCWA methods: timing and far-field image accuracy."""

import argparse, os, sys, time
import numpy as np
import torch

from Simulok import Simulok

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FREE vs GEMM comparison")
    parser.add_argument("--order", type=int, nargs=2, default=[15, 15])
    args_extra = parser.parse_args()

    methods = ['free', 'gemm']
    intensities = {}
    timings = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"  Running method: {method.upper()}")
        print(f"{'='*60}")

        # Build fresh args
        parser2 = argparse.ArgumentParser()
        parser2.add_argument("--device", type=str, default='cuda')
        parser2.add_argument("--pngpath", type=str, default='./SimPattern/DSW-c.png')
        parser2.add_argument("--order", type=int, nargs=2, default=args_extra.order)
        parser2.add_argument("--lamda", type=int, default=270)
        parser2.add_argument("--period", type=int, nargs='+', default=[20000])
        parser2.add_argument("--resolution", type=int, nargs='+', default=[128, 128])
        parser2.add_argument("--magnification", type=float, default=100.0)
        parser2.add_argument("--pixelsize", type=int, default=90)
        parser2.add_argument("--shape", type=str, default='annular')
        parser2.add_argument("--polarization", type=str, default='x_pol')
        parser2.add_argument("--sigmaOut", type=float, default=0.9)
        parser2.add_argument("--sigmaIn", type=float, default=0.15)
        parser2.add_argument("--openAngle", type=float, default=0.0)
        parser2.add_argument("--rotateAngle", type=float, default=0.0)
        parser2.add_argument("--RectX", type=float, default=0.0)
        parser2.add_argument("--RectY", type=float, default=0.0)
        parser2.add_argument("--imaging_pupil_shape", type=str, default='annular')
        parser2.add_argument("--imaging_pupil_sigmaOut", type=float, default=0.9)
        parser2.add_argument("--imaging_pupil_sigmaIn", type=float, default=0.15)
        parser2.add_argument("--imaging_pupil_openAngle", type=float, default=0.0)
        parser2.add_argument("--imaging_pupil_rotateAngle", type=float, default=0.0)
        parser2.add_argument("--imaging_pupil_RectX", type=float, default=0.0)
        parser2.add_argument("--imaging_pupil_RectY", type=float, default=0.0)
        parser2.add_argument("--imaging_pupil_polarization", type=str, default='None')
        parser2.add_argument("--ff_cal_type", type=str, default='abbe')
        parser2.add_argument("--rcwa_method", type=str, default=method)

        a = parser2.parse_args([])

        t0 = time.time()
        pipeline = Simulok(a)
        pipeline.save_name = f"compare_{method}"
        pipeline.Run()
        elapsed = time.time() - t0

        timings[method] = elapsed
        intensities[method] = pipeline.Intensity.cpu().squeeze().numpy()
        print(f"  Total time: {elapsed:.2f}s")

    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARISON RESULTS")
    print(f"{'='*60}")

    I_free = intensities['free']
    I_gemm = intensities['gemm']

    print(f"\n  Timing:")
    print(f"    FREE: {timings['free']:.2f}s")
    print(f"    GEMM: {timings['gemm']:.2f}s")
    print(f"    Speedup: {timings['gemm']/timings['free']:.2f}x")

    diff = np.abs(I_free - I_gemm)
    rel_diff = diff / (np.maximum(np.abs(I_free), np.abs(I_gemm)) + 1e-12)

    print(f"\n  Intensity difference:")
    print(f"    Max  |ΔI|:        {diff.max():.6e}")
    print(f"    Mean |ΔI|:        {diff.mean():.6e}")
    print(f"    Max  relative:    {rel_diff.max():.6e}")
    print(f"    Mean relative:    {rel_diff.mean():.6e}")

    # Correlation
    I_free_flat = I_free.flatten()
    I_gemm_flat = I_gemm.flatten()
    corr = np.corrcoef(I_free_flat, I_gemm_flat)[0,1]
    print(f"    Correlation:      {corr:.6f}")

    # Check if max diff is small relative to intensity range
    i_range = I_free.max() - I_free.min()
    print(f"    Max |ΔI| / range: {diff.max()/i_range:.6e}")

    if corr > 0.999 and diff.max()/i_range < 0.01:
        print(f"\n  ✓ FREE and GEMM produce nearly identical results")
    else:
        print(f"\n  ⚠ Differences may be significant — investigate")

    # Save comparison image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs('image_result/compare', exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(I_free, cmap='hot')
    axes[0].set_title('FREE')
    axes[1].imshow(I_gemm, cmap='hot')
    axes[1].set_title('GEMM')
    axes[2].imshow(diff, cmap='jet')
    axes[2].set_title(f'|ΔI| (max={diff.max():.2e})')
    axes[3].imshow(rel_diff, cmap='jet', vmin=0, vmax=0.01)
    axes[3].set_title('Relative diff')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('image_result/compare/free_vs_gemm.png', dpi=150)
    print(f"  Comparison plot saved to image_result/compare/free_vs_gemm.png")
