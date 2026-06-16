# OpticalImaging

This repository currently contains a CUDA/C++ near-field electromagnetic solver based on RCWA and one far-field imaging path based on the Abbe method.

## Current Scope

The project is split into:

- `include/common_utils.h`
  Common numeric types, CUDA helpers, status handling, complex helpers, and utility macros.
- `include/rcwa.h`
  Public interface for the RCWA solver.
- `src/rcwa.cu`
  RCWA implementation, including layer assembly, global S-matrix solve, field extraction, and energy balance.
- `include/farfield_abbe.h`
  Public Abbe imaging API.
- `include/abbe_imager.h`
  Thin C++ wrapper around the Abbe API.
- `src/farfield_abbe.cu`
  RCWA-driven Abbe imaging implementation.
- `src/complex_matrix.cu`
  Complex matrix operations used by the solver.
- `examples/`
  Small executables for rendering, consistency checks, and method benchmarking.

## Implemented Features

### RCWA solver

`RCWA` in `nearfield/include/rcwa.h` supports:

- input and output media definition
- homogeneous and patterned layers
- incident-angle setup
- plane-wave source injection
- global S-matrix solution
- near-field sampling on an `x/y` grid
- energy balance calculation
- repeated-stack acceleration through `RepeatCurrentStackSMatrix`

The RCWA class also exposes multiple matrix-exponential / S-matrix computation paths:

- `G` path
- `S` path
- `F` path
- regular and higher-order variants

These are mainly exercised by the benchmark-style example programs.

### Abbe imaging

The only active imaging path is Abbe imaging.

Public entry points:

- `CalculateAbbeImageFromRCWA(...)`
- `AbbeImager::ComputeFromRCWA(...)`

Supported Abbe configuration areas:

- source aperture shape:
  `kAnnular`, `kDipoleCirc`, `kQuasar`, `kRect`
- source polarization mode:
  `kJonesVector`, `kX`, `kY`, `kLine`, `kCircular`, `kAngleDegrees`
- projector focus sweep:
  `AbbeProjectorConfig::focus_range`
- object-plane sweep:
  `AbbeNumericsConfig::object_z_positions`
- projector aberration coefficients:
  `AbbeProjectorConfig::aberration_zernike`

Output format:

- `AbbeImageResult::intensity` stores `[z][focus][y][x]`
- `image_x` and `image_y` store the sampled image coordinates

## Build

Build system:

- CMake 3.19+
- CUDA toolkit
- C++17

Configured in `nearfield/CMakeLists.txt`:

- `USE_DOUBLE_PRECISION` defaults to `ON`
- CUDA architecture is currently hardcoded to `89`
- optimization flags are enabled for both host and device code

Build commands:

```bash
cmake -S nearfield -B nearfield/build
cmake --build nearfield/build -j
```

## Built Targets

The current CMake file builds these executables:

- `homogeneous_test`
- `method_compare_full`
- `farfield_abbe_render`

### `farfield_abbe_render`

Location:

- `nearfield/examples/farfield_abbe_render.cu`

Purpose:

- builds a small multilayer RCWA scene
- computes an Abbe far-field image from RCWA output
- computes a near-field image on the same sampling grid
- writes PNG outputs to an output directory

Default output files:

- `nearfield/output/mask.png`
- `nearfield/output/nearfield.png`
- `nearfield/output/farfield.png`

Run:

```bash
./nearfield/build/farfield_abbe_render
```

Optional output directory:

```bash
./nearfield/build/farfield_abbe_render my_output_dir
```

### `homogeneous_test`

Location:

- `nearfield/examples/homogeneous_test.cu`

Purpose:

- checks consistency between `AddLayer(...)` and `AddLayerHomogeneous(...)`
- checks explicit free-space input/output layer behavior
- checks repeated homogeneous stack behavior

Run:

```bash
./nearfield/build/homogeneous_test
```

### `method_compare_full`

Location:

- `nearfield/examples/method_compare_full.cu`

Purpose:

- benchmarks RCWA internal computation modes
- compares runtime and relative S-matrix differences across:
  `G12`, `G18`, `S12`, `S18`, `F12`, `F18`

This executable is a solver-method benchmark. It is not an imaging-method comparison anymore.

Run:

```bash
./nearfield/build/method_compare_full
```

## Data Layout Notes

Some arrays use solver-oriented layouts rather than image-oriented layouts.

- geometry masks in the examples are often filled as `[x][y]`
- Abbe output images are stored logically as `[z][focus][y][x]`
- several example helpers convert between solver storage order and raster image order before writing PNGs

When adding new post-processing code, check the existing indexing conventions before assuming row-major image layout.

## Minimal Usage Sketch

Typical flow for RCWA + Abbe imaging:

1. Construct `RCWA(freq, order_x, order_y, Lx, Ly)`.
2. Add input/output media and layers.
3. Call `SetIncidentAngles(...)`.
4. Call `SolveGlobalSMatrix()`.
5. Prepare image sampling axes.
6. Fill `AbbeSourceConfig`, `AbbeProjectorConfig`, and `AbbeNumericsConfig`.
7. Call `AbbeImager::ComputeFromRCWA(...)`.
8. Read `AbbeImageResult::intensity`.
