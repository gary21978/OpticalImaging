#!/usr/bin/env python3
"""Test script: run end-to-end simulation on DSW-c.png with low RCWA order."""

import argparse
import os

from Simulok import Simulok

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DSW-c test run")
    parser.add_argument("--device", type=str, default='cuda',
                        help="device: cuda or cpu")
    parser.add_argument("--pngpath", type=str, default='./SimPattern/DSW-c.png')
    parser.add_argument("--order", type=int, nargs=2, default=[15, 15])
    parser.add_argument("--lamda", type=int, default=270)
    parser.add_argument("--period", type=int, nargs='+', default=[20000])
    parser.add_argument("--resolution", type=int, nargs='+', default=[128, 128])
    parser.add_argument("--magnification", type=float, default=100.0)
    parser.add_argument("--pixelsize", type=int, default=90)
    parser.add_argument("--shape", type=str, default='annular')
    parser.add_argument("--polarization", type=str, default='x_pol')
    parser.add_argument("--sigmaOut", type=float, default=0.9)
    parser.add_argument("--sigmaIn", type=float, default=0.15)
    parser.add_argument("--openAngle", type=float, default=0.0)
    parser.add_argument("--rotateAngle", type=float, default=0.0)
    parser.add_argument("--RectX", type=float, default=0.0)
    parser.add_argument("--RectY", type=float, default=0.0)
    parser.add_argument("--imaging_pupil_shape", type=str, default='annular')
    parser.add_argument("--imaging_pupil_sigmaOut", type=float, default=0.9)
    parser.add_argument("--imaging_pupil_sigmaIn", type=float, default=0.15)
    parser.add_argument("--imaging_pupil_openAngle", type=float, default=0.0)
    parser.add_argument("--imaging_pupil_rotateAngle", type=float, default=0.0)
    parser.add_argument("--imaging_pupil_RectX", type=float, default=0.0)
    parser.add_argument("--imaging_pupil_RectY", type=float, default=0.0)
    parser.add_argument("--imaging_pupil_polarization", type=str, default='None')
    parser.add_argument("--ff_cal_type", type=str, default='abbe')
    parser.add_argument("--rcwa_method", type=str, default='free',
                        help="RCWA layer method: 'free' (default) or 'gemm'")

    args = parser.parse_args()

    print("=" * 50)
    print("DSW-c Test Simulation")
    print(f"  Device:      {args.device}")
    print(f"  PNG path:    {args.pngpath}")
    print(f"  RCWA order:  {args.order}")
    print(f"  Wavelength:  {args.lamda} nm")
    print(f"  Period:      {args.period} nm")
    print(f"  Resolution:  {args.resolution}")
    print(f"  Source:      {args.shape} (σ_in={args.sigmaIn}, σ_out={args.sigmaOut})")
    print(f"  Polarization:{args.polarization}")
    print(f"  RCWA method: {args.rcwa_method}")
    print(f"  Far-field:   {args.ff_cal_type}")
    print("=" * 50)

    if not os.path.exists(args.pngpath):
        print(f"ERROR: {args.pngpath} not found!")
        exit(1)

    pipeline = Simulok(args)
    pipeline.save_name = "DSW_test"
    pipeline.Run()

    print("\nDone. Check ./image_result/DSW_test/DSW-c/ for output.")
