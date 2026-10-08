"""Matched original-space and projected-space FPS at budgets 64, 128 and 256.

No encoder forwards, labels, or search. Projection only selects IDs; scoring
uses their original band descriptors. Evaluate the frozen output separately.
"""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from wram.io import dump, read, check_run
from wram.pipeline import json_banks
from wram.reference import fps, distances, fit_statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    check_run(a.run)
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(
        seed=20260613,
        projection_dimensions=256,
        memory_sizes=[64, 128, 256],
        ratio_neighbors=[1],
    )
    groups = []
    banks_by_group = {}
    for model_file in sorted((a.run / "models").glob("*.json")):
        records = None
        # Group metadata is read from the normal cache only, before any query.
        for folder in sorted((a.run / "features").glob("*/*/normal")):
            candidate = read(folder / "RECORDS.json")
            if (
                f'{candidate[0]["split"]}_{candidate[0]["machine"]}'
                == model_file.stem
            ):
                records = candidate
                break
        if records is None:
            raise ValueError("Missing normal cache")
        split, machine = records[0]["split"], records[0]["machine"]
        path = folder / "wiener_rdp4.npy"
        normal = np.load(path)
        x = torch.tensor(normal, dtype=torch.float32).flatten(1)
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        projection = (
            torch.randn(x.shape[1], 256, generator=generator)
            / x.shape[1] ** 0.5
        )
        projected = x @ projection
        original_order = fps(normal, 256)
        projected_order = fps(projected.numpy()[:, None, :], 256)
        original_ids = original_order[:128]
        np.testing.assert_array_equal(
            original_ids, read(model_file)["wiener_rdp4"]["fps"]["indices"]
        )
        banks = {}
        policies = []
        for size in config["memory_sizes"]:
            policies.extend(
                [
                    (
                        "fps" if size == 128 else f"fps{size}",
                        original_order[:size],
                    ),
                    (f"projected256_fps{size}", projected_order[:size]),
                ]
            )
        for name, ids in policies:
            banks[name] = dict(
                indices=ids,
                statistics=fit_statistics(distances(normal[ids], normal[ids])),
            )
        banks_by_group[split, machine] = banks
        groups.append((split, machine, folder.parent))
        dump(out / "models" / model_file.name, json_banks(banks))
    for split, machine, root in groups:
        normal = np.load(root / "normal/wiener_rdp4.npy")
        query_path = root / "query/wiener_rdp4.npy"
        query = np.load(query_path)
        records = read(root / "query/RECORDS.json")
        scores = {}
        for name, bank in banks_by_group[split, machine].items():
            raw = distances(query, normal[bank["indices"]])
            for k in config["ratio_neighbors"]:
                values = (
                    raw
                    / np.maximum(bank["statistics"][f"ratio{k}"], 1e-12)[
                        :, None, :
                    ]
                )
                scores[f"wiener_rdp4|{name}|ratio|{k}|updated"] = (
                    values.min(-1).mean(0).tolist()
                )
        dump(
            out / "predictions" / f"{split}_{machine}.json",
            dict(
                split=split,
                machine=machine,
                ids=[r["id"] for r in records],
                scores=scores,
            ),
        )
    elapsed = time.monotonic() - started
    dump(out / "RUN.json", dict(status="completed", config=config, seconds=elapsed))
    print(
        "Completed matched reference-budget control in"
        f" {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
