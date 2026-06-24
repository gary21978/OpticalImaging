# Simulok GPU Optical Workbench

The desktop workbench is a native C++/OpenGL frontend around the existing CUDA
RCWA and Abbe imaging pipeline.

## Build

Requirements: CUDA Toolkit, a working NVIDIA driver, CMake 3.19+, GLFW 3, and
OpenGL.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run from the repository root so the default JSON path resolves:

```bash
./build/simulok_gui
```

An alternate JSON file can be passed on the command line:

```bash
./build/simulok_gui path/to/config.json
```

## Controls

- Click a numeric field, type a value, then press Enter.
- Click the geometry button to cycle through cross, rectangle, and ellipse.
- Drag the 3D viewport to orbit; use the wheel to zoom.
- `Load JSON` and `Save JSON` use the active path.
- `Run Simulation` executes the CUDA pipeline in an isolated worker process and uploads
  the resulting far-field image to the OpenGL view.
- While running, the same button becomes `Force Stop Simulation`. Simulations
  execute in an isolated process, so stopping also terminates active CUDA work.
- The Source, Polarization, Zernike, and Solver tabs expose illumination shape,
  sigma, polarization direction, 37 aberration coefficients, 12th/18th-order
  matrix approximation,
  and single/double precision.
- Geometry sampling is derived from each RCWA order using
  `next_power_of_two(4 * order + 2)`, satisfying the Nyquist requirement for
  the retained positive and negative spatial harmonics.
- The Material tab reads the bundled dispersion cache. Layer JSON supports a
  `material` field; `custom` uses explicit `n_real`/`n_imag`, while named
  materials are interpolated at the active wavelength.

The complete refractiveindex.info source database is bundled under
`resources/material`, including its CC0 permission and attribution documents.
Regenerate the runtime cache from the bundled database:

```bash
python tools/build_material_cache.py \
  --database resources/material \
  --output resources/tztek_materials.csv
```

Pass `--materials Si SiO2 ...` only when a smaller cache is desired. Source
page paths are retained in the CSV cache for traceability.
- `Ctrl+S` saves the current configuration.

The JSON layer array controls layer thickness, complex refractive index,
patterned/solid behavior, and display color. The GUI displays the complete
stack; layer definitions can be changed in JSON and reloaded without restarting.
