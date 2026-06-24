#!/usr/bin/env python3
"""Build the runtime wavelength cache from the bundled refractiveindex.info database."""

import argparse
import csv
import importlib.util
import os
import sys
import yaml
import numpy as np


def load_parser():
    path = os.path.join(os.path.dirname(__file__), "refractiveindex_parser.py")
    spec = importlib.util.spec_from_file_location("refractiveindex_parser", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interpolate(series_list, wavelength, default):
    for series in series_list:
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) < 2:
            continue
        if values[0, 0] <= wavelength <= values[-1, 0]:
            return float(np.interp(wavelength, values[:, 0], values[:, 1]))
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="resources/material")
    parser.add_argument("--output", default="resources/tztek_materials.csv")
    parser.add_argument("--materials", nargs="*", default=[],
                        help="Material book IDs to export; empty exports the full catalog.")
    args = parser.parse_args()

    extra = load_parser()
    db = args.database
    with open(os.path.join(db, "catalog-nk.yml"), encoding="utf-8") as stream:
        catalog = yaml.safe_load(stream)
    materials = args.materials or [
        book["BOOK"]
        for shelf in catalog
        for book in shelf.get("content", [])
        if "BOOK" in book
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    wavelengths = np.geomspace(180.0, 25000.0, 320)
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["# material", "source", "wavelength_nm", "n", "k"])
        for name in materials:
            book = extra.find_book_info(name, catalog)
            if not book:
                print(f"missing material: {name}", file=sys.stderr)
                continue
            try:
                n_all, k_all = extra.get_nk_data(db, book)
            except Exception as exc:
                print(f"skipping malformed material {name}: {exc}", file=sys.stderr)
                continue
            n_all = [series for series in n_all if len(series) >= 2]
            k_all = [series for series in k_all if len(series) >= 2]
            source = next((item.get("data", "") for item in book.get("content", []) if "PAGE" in item), "")
            written = 0
            for wavelength_nm in wavelengths:
                n = interpolate(n_all, wavelength_nm / 1000.0, 1.0)
                k = interpolate(k_all, wavelength_nm / 1000.0, 0.0)
                if n == 1 and k == 0:
                    continue
                writer.writerow([name, source, f"{wavelength_nm:.9g}", f"{float(n):.12g}", f"{float(k):.12g}"])
                written += 1
            print(f"{name}: {written} samples")


if __name__ == "__main__":
    main()
