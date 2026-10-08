"""Prepare real, label-free reference-selection data for diagram tiles.

The projected policy is a descriptive comparison, not a scored ASD experiment.
It uses the historical local Gaussian-projection recipe (not PatchCore's
approximate sampler). Run with the same torch environment as WRAM.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wram.reference import fps
from wram.io import check_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", default="Dev")
    p.add_argument("--machine", default="fan")
    p.add_argument("--seed", type=int, default=20260613)
    a = p.parse_args()
    torch.set_num_threads(2)
    root = a.run / "features" / a.split / a.machine
    feature_path = root / "normal" / "wiener_rdp4.npy"
    model_path = a.run / "models" / f"{a.split}_{a.machine}.json"
    run_info = check_run(a.run)
    normal = np.load(feature_path)
    query = np.load(root / "query" / "wiener_rdp4.npy")[0]
    banks = json.loads(model_path.read_text())["wiener_rdp4"]
    records = json.loads((root / "normal" / "RECORDS.json").read_text())
    x = torch.tensor(normal, dtype=torch.float32).flatten(1)
    generator = torch.Generator(device="cpu").manual_seed(a.seed)
    projection = (
        torch.randn(x.shape[1], 256, generator=generator) / x.shape[1] ** 0.5
    )
    projected = x @ projection
    # Exact centroid-start greedy selection in the projected space.
    projected_ids = fps(projected.numpy()[:, None, :], 128)
    original_ids = fps(normal, 128)
    np.testing.assert_array_equal(original_ids, banks["fps"]["indices"])
    policies = {
        "all1000": np.arange(len(normal)),
        "fps64": np.array(banks["fps64"]["indices"]),
        "fps128": original_ids,
        "fps256": np.array(banks["fps256"]["indices"]),
        "projected_coreset128": projected_ids,
    }
    for name, bank in banks.items():
        if name.startswith("random_"):
            policies[name] = np.array(bank["indices"])
    common = set(original_ids) & set(projected_ids)
    a.output.mkdir(parents=True, exist_ok=False)
    arrays = dict(
        normal=normal,
        query=query,
        projected=projected.numpy(),
        projection=projection.numpy(),
        normal_ids=np.array([r["id"] for r in records]),
    )
    for name, ids in policies.items():
        arrays[f"ids_{name}"] = ids
    # Coverage distances use 6144-D, never the plotted 2-D view.
    xf = x.numpy().astype(np.float64)
    norms = np.einsum("ij,ij->i", xf, xf)
    distance = np.sqrt(
        np.maximum(norms[:, None] + norms[None, :] - 2 * xf @ xf.T, 0)
    )
    np.fill_diagonal(distance, 0)
    coverage = {}
    for name, ids in policies.items():
        values = distance[:, ids].min(1)
        arrays[f"coverage_{name}"] = values
        coverage[name] = dict(
            size=len(ids),
            mean=float(values.mean()),
            max=float(values.max()),
            target_count=sum(records[i]["domain"] == "target" for i in ids),
        )
    rng = np.random.default_rng(20260923)
    left, right = np.triu_indices(len(x), k=1)
    pick = rng.choice(len(left), min(5000, len(left)), replace=False)
    left, right = left[pick], right[pick]
    # Historical projection uses 1/sqrt(input_dim). Rescale this diagnostic
    # so its expected squared distances match original-space squared distances.
    pd = np.linalg.norm(
        projected.numpy()[left] - projected.numpy()[right], axis=1
    )
    arrays["distance_pairs"] = np.column_stack(
        [distance[left, right], pd * np.sqrt(x.shape[1] / 256)]
    )
    arrays["pair_indices"] = np.column_stack([left, right])
    selected = normal[original_ids]
    q = query / np.linalg.norm(query, axis=-1, keepdims=True)
    selected = selected / np.linalg.norm(selected, axis=-1, keepdims=True)
    raw = 0.5 * (1 - np.clip(np.einsum("bd,nbd->bn", q, selected), -1, 1))
    inherited = np.array(banks["all"]["statistics"]["ratio1"])[:, original_ids]
    refit = np.array(banks["fps"]["statistics"]["ratio1"])
    inherited_cost = raw / np.maximum(inherited, 1e-12)
    refit_cost = raw / np.maximum(refit, 1e-12)
    old = inherited_cost.argmin(1)
    new = refit_cost.argmin(1)
    arrays.update(
        raw_query_distances=raw,
        inherited_scales=inherited,
        refit_scales=refit,
        inherited_cost=inherited_cost,
        refit_cost=refit_cost,
        inherited_winner=old,
        rematch_winner=new,
        inherited_band_scores=inherited_cost[np.arange(8), old],
        fixed_band_scores=refit_cost[np.arange(8), old],
        rematch_band_scores=refit_cost[np.arange(8), new],
    )
    np.savez_compressed(a.output / "selection_arrays.npz", **arrays)
    report = dict(
        source_run=str(a.run.resolve()),
        split=a.split,
        machine=a.machine,
        view="wiener_rdp4",
        wiener_mode=run_info["config"]["wiener_mode"],
        coefficient_loading=0,
        numerical_epsilon=run_info["config"]["epsilon"],
        normal_count=len(normal),
        feature_shape=list(normal.shape),
        projected_shape=list(projected.shape),
        projection_seed=a.seed,
        projection_recipe=(
            "torch.randn(6144,256, seed)/sqrt(6144), float32; centroid-start"
            " exact FPS128"
        ),
        projection_status=(
            "Historical selection recipe reapplied to fresh floor-only"
            " descriptors; no projected-policy ASD metric computed"
        ),
        query_index=0,
        shared_selected_ids=len(common),
        union_selected_ids=len(set(original_ids) | set(projected_ids)),
        coverage=coverage,
        original_fps_ids_match_run=True,
        torch=torch.__version__,
        numpy=np.__version__,
    )
    (a.output / "PROVENANCE.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "shared_selected_ids",
                    "original_fps_ids_match_run",
                    "projected_shape",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
