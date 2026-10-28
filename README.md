# CUDA RCWA Optical Simulator

A CUDA C++ implementation of rigorous coupled-wave analysis (RCWA) for periodic optical structures. The project solves layered electromagnetic structures, samples near fields, and computes images using an RCWA-driven Abbe imaging model. Matrix operations can use GPU memory as well as host or file-backed storage for larger problems.

## Capabilities

- Define input and output media, homogeneous layers, and sampled patterned layers with Laurent or Lalanne factorization.
- Configure incidence and plane-wave polarization, assemble the global scattering matrix, sample electric fields, and calculate reflectance, transmittance, and absorption.
- Select different solver methods and their regular or high-order variants; repeat a layer stack and optionally select harmonic order automatically.
- Compute Abbe images with configurable illumination, polarization, aperture, focus, object planes, and Zernike aberrations.
- Use tiled matrix operations with GPU, pinned host-memory, or file-backed storage.

The public solver interface is in `include/rcwa.h`. Abbe imaging is exposed by `include/farfield_abbe.h` and the `AbbeImager` wrapper in `include/abbe_imager.h`. Implementations are in `src/`; executable examples are in `example/`.

## Requirements and build

- CMake 3.19 or later
- A CUDA-capable NVIDIA GPU, compatible driver, and CUDA Toolkit with cuBLAS, cuSOLVER, and cuFFT
- A C++17-capable host compiler supported by the installed CUDA Toolkit

From the repository root:

```bash
cmake -S . -B build
cmake --build build -j
```

The default CUDA architecture is `89`. Set an architecture appropriate for the target GPU when configuring, for example:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

CMake searches for `nvcc` on `PATH` and uses it when found. Check `nvcc --version` if multiple toolkits are installed. `USE_DOUBLE_PRECISION` defaults to `ON`; configure with `-DUSE_DOUBLE_PRECISION=OFF` to use single precision. The build enables aggressive host and CUDA floating-point optimization flags.

## Storage configuration

`RCWA_MEMORY_MODE` accepts `resident`, `ram` or `ram_only`, `hybrid`, and `out_of_core` or `ooc`. These select GPU-resident allocation or out-of-core policies using pinned host memory and/or files. Available memory and allocation thresholds also affect placement. For example:

```bash
RCWA_MEMORY_MODE=ram_only RCWA_RAM_MB=4096 ./build/TSV 2 output/tsv-test
```

Other controls include `RCWA_RESIDENT_MB`, `RCWA_GPU_BUDGET_MB`, `RCWA_TILE_SIZE`, `RCWA_LU_PANEL`, `RCWA_CACHE_DIR`, `RCWA_DIRECT_PINNED`, `RCWA_STAGING_POOL`, and `RCWA_GEMM_PIPELINE`. Their implementation and defaults are in `src/matrix_storage.cu`; tune them for the available hardware and workload.

## API and data layout

A typical workflow is to construct `RCWA(freq, order_x, order_y, Lx, Ly)`, add media and layers, set the incident angles, and solve the global S matrix. Set a plane-wave source before calling `FieldXY`, or provide axes and imaging configuration to `CalculateAbbeImageFromRCWA` or `AbbeImager::ComputeFromRCWA`. The examples show allocation, status checking, and cleanup for the corresponding operations. Use consistent length units: the examples set `freq` to the reciprocal of the wavelength.

Sampled material arrays in the examples use `[x][y]` indexing. `AbbeImageResult::intensity` is laid out as `[object plane][focus][y][x]`; image output is written in row-major order. Account for these layouts when preparing geometry or post-processing results.
