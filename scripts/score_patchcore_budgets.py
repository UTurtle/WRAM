"""Evaluate official PatchCore approximate coreset selection on WRAM features.

The upstream sampler chooses normal recording IDs. WRAM scoring remains in the
original eight-band descriptor space. No encoder forwards or labels are used.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np
import torch

from wram.io import dump, read, check_run
from wram.pipeline import json_banks
from wram.reference import distances, fit_statistics

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/patchcore-inspection/sampler.py"
UPSTREAM_SHA256 = (
    "39612cd2b486865ece304348f740f4bc700a1c2e8684fbab3c2ed016c7801f2d"
)
UPSTREAM_COMMIT = "fcaa92f124fb1ad74a7acf56726decd4b27cbcad"


def official_sampler_class():
    if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Pinned PatchCore sampler source changed")
    spec = importlib.util.spec_from_file_location(
        "official_patchcore_sampler", UPSTREAM
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ApproximateGreedyCoresetSampler


def patchcore_order(normal, sampler_class, seed=20260613):
    """Call the pinned upstream projection and approximate greedy code."""
    x = torch.as_tensor(normal, dtype=torch.float32).flatten(1)
    if x.shape != (1000, 6144):
        raise ValueError(
            f"Unexpected normal descriptor shape: {tuple(x.shape)}"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = sampler_class(
        percentage=256 / len(x),
        device=torch.device("cpu"),
    )
    reduced = sampler._reduce_features(x)
    order = sampler._compute_greedy_coreset_indices(reduced)
    if len(order) != 256 or len(set(map(int, order))) != 256:
        raise AssertionError("Official sampler returned invalid recording IDs")
    return order.astype(np.int64)


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
    sampler_class = official_sampler_class()
    config = dict(
        seed=20260613,
        projection_dimensions=128,
        number_of_starting_points=10,
        memory_sizes=[64, 128, 256],
        ratio_neighbors=[1],
        upstream_commit=UPSTREAM_COMMIT,
        upstream_sampler_sha256=UPSTREAM_SHA256,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
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
        selected_order = patchcore_order(normal, sampler_class, config["seed"])
        banks = {}
        for size in config["memory_sizes"]:
            name = f"patchcore_approx{size}"
            ids = selected_order[:size]
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
