"""Normal-reference selection and scoring. No labels or query fitting."""

import numpy as np
import torch


def validate(x):
    x = np.asarray(x, dtype=np.float64)
    if (
        x.ndim != 3
        or not np.isfinite(x).all()
        or np.any(np.linalg.norm(x, axis=-1) == 0)
    ):
        raise ValueError(
            "Expected finite nonzero [record,band,feature] descriptors"
        )
    return x


def unit(x):
    x = validate(x)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def fps(normal, size=128):
    """Unprojected greedy k-center: centroid-nearest start,
    CPU float32 distances.

    Returns real recording indices, shared by every band. This is an existing
    coreset construction, not learned or averaged prototypes.
    """
    x = torch.as_tensor(
        validate(normal), dtype=torch.float32, device="cpu"
    ).flatten(1)
    if not 1 <= size <= len(x):
        raise ValueError("Invalid bank size")
    current = int(((x - x.mean(0)) ** 2).sum(1).argmin())
    chosen = []
    nearest = torch.full((len(x),), float("inf"))
    for _ in range(size):
        chosen.append(current)
        nearest = torch.minimum(nearest, ((x - x[current]) ** 2).sum(1))
        nearest[chosen] = -float("inf")
        current = int(nearest.argmax())
    return np.asarray(chosen, dtype=np.int64)


def distances(left, right, device="cpu"):
    a = torch.as_tensor(validate(left), dtype=torch.float64, device=device)
    b = torch.as_tensor(validate(right), dtype=torch.float64, device=device)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError("Descriptor dimensions differ")
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (
        (0.5 * (1 - torch.einsum("nbd,mbd->bnm", a, b).clamp(-1, 1)))
        .cpu()
        .numpy()
    )


def fit_statistics(normal_distances):
    """Self-excluded k=1/4 scales and exact donor-deleted
    subtractive control.
    """
    d = np.array(normal_distances, dtype=np.float64, copy=True)
    if (
        d.ndim != 3
        or d.shape[1] != d.shape[2]
        or d.shape[1] < 6
        or not np.isfinite(d).all()
    ):
        raise ValueError("Need at least six finite normal references")
    bands, n, _ = d.shape
    d[:, np.arange(n), np.arange(n)] = np.inf
    order = np.argsort(d, axis=-1, kind="stable")[:, :, :5]
    neighbors = np.take_along_axis(d, order, axis=-1)
    rho = neighbors[:, :, :4].mean(-1)
    beta = []
    for band in range(bands):
        closest = order[band, :, 0]
        donors = rho[band, closest].copy()
        # Remove each calibration normal from its winning reference's donors.
        for query, rank in zip(
            *np.where(order[band, closest, :4] == np.arange(n)[:, None])
        ):
            ref = closest[query]
            donors[query] += (
                neighbors[band, ref, 4] - neighbors[band, ref, rank]
            ) / 4
        raw = neighbors[band, :, 0]
        variance = np.var(donors)
        beta.append(
            float(
                np.mean((raw - raw.mean()) * (donors - donors.mean()))
                / variance
            )
            if variance
            else 0.0
        )
    return {
        "ratio1": neighbors[:, :, 0],
        "ratio4": neighbors[:, :, :4].sum(-1),
        "rho": rho,
        "beta": np.asarray(beta),
    }


def matched_random(domains, selected, seed):
    domains = np.asarray(domains)
    rng = np.random.default_rng(seed)
    chosen = []
    for domain in ["source", "target"]:
        count = int(np.sum(domains[selected] == domain))
        pool = np.flatnonzero(domains == domain)
        chosen.extend(rng.choice(pool, count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def fit_banks(
    normal, domains, size=128, seeds=(), device="cpu", extra_sizes=()
):
    order = fps(normal, max([size, *extra_sizes]))
    selected = order[:size]
    indices = {"all": np.arange(len(normal)), "fps": selected}
    for k in extra_sizes:
        indices[f"fps{k}"] = order[:k]
    for seed in seeds:
        indices[f"random_{seed}"] = matched_random(domains, selected, seed)
    rr = distances(normal, normal, device)
    return {
        name: {
            "indices": ix,
            "statistics": fit_statistics(rr[:, ix][:, :, ix]),
        }
        for name, ix in indices.items()
    }


def gather(values, winner):
    return np.take_along_axis(values, winner[:, :, None], axis=-1)[:, :, 0]


def score_banks(normal, query, banks, device="cpu", controls=True):
    d = distances(query, normal, device)
    result = {}
    for name, bank in banks.items():
        ix = bank["indices"]
        stats = bank["statistics"]
        raw = d[:, :, ix]
        for k in ([1, 4] if controls else [1]):
            normalized = (
                raw / np.maximum(stats[f"ratio{k}"], 1e-12)[:, None, :]
            )
            winner = normalized.argmin(-1)
            key = f"{name}|ratio|{k}|updated"
            result[key] = {
                "scores": gather(normalized, winner).mean(0),
                "bands": gather(normalized, winner).T,
                "winners": ix[winner].T,
            }
            if controls and name.startswith("fps"):
                inherited = (
                    raw
                    / np.maximum(
                        banks["all"]["statistics"][f"ratio{k}"][:, ix], 1e-12
                    )[:, None, :]
                )
                old = inherited.argmin(-1)
                inherited_bands = gather(inherited, old)
                fixed_bands = gather(normalized, old)
                updated_bands = gather(normalized, winner)
                if np.any(updated_bands > fixed_bands + 1e-10) or np.any(
                    fixed_bands > inherited_bands + 1e-10
                ):
                    raise AssertionError(
                        "Reference factorization ordering failed"
                    )
                for state, values, win in [
                    ("inherited", inherited_bands, old),
                    ("fixed", fixed_bands, old),
                ]:
                    result[f"{name}|ratio|{k}|{state}"] = {
                        "scores": values.mean(0),
                        "bands": values.T,
                        "winners": ix[win].T,
                    }
        if controls and name in ["all", "fps"]:
            for scorer, k, values in [
                ("off", 0, raw),
                (
                    "subtractive",
                    4,
                    raw
                    - stats["rho"][:, None, :] * stats["beta"][:, None, None],
                ),
            ]:
                winner = values.argmin(-1)
                result[f"{name}|{scorer}|{k}|updated"] = {
                    "scores": gather(values, winner).mean(0),
                    "bands": gather(values, winner).T,
                    "winners": ix[winner].T,
                }
    return result


def fixed_id_score(normal, query, indices, device="cpu"):
    rr = distances(normal, normal, device)
    statistics = fit_statistics(rr[:, indices][:, :, indices])
    d = distances(query, normal[indices], device)
    value = d / np.maximum(statistics["ratio1"], 1e-12)[:, None, :]
    return value.min(-1).mean(0)
