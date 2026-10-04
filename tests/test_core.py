import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from wram.frontend import temporal_pool, wiener_residual
from wram.reference import (
    fps,
    distances,
    fit_statistics,
    fit_banks,
    score_banks,
)
from wram.metrics import components


def random_descriptors(n, seed=1):
    x = np.random.default_rng(seed).normal(size=(n, 3, 9))
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_rdp_constant_and_zero_exponent():
    x = torch.randn(2, 1, 3, 9, dtype=torch.float64).repeat(1, 12, 1, 1)
    assert torch.allclose(temporal_pool(x, 4), x[:, 0])
    y = torch.randn_like(x)
    assert torch.equal(temporal_pool(y, 0), y.mean(1))


def test_fps_unique_nested_and_representative():
    x = random_descriptors(20)
    a = fps(x, 8)
    b = fps(x, 12)
    assert np.array_equal(a, b[:8])
    assert len(set(b)) == 12
    t = torch.tensor(x, dtype=torch.float32).flatten(1)
    assert a[0] == int(((t - t.mean(0)) ** 2).sum(1).argmin())
    for j in range(1, len(a)):
        d = ((t[:, None] - t[a[:j]][None]) ** 2).sum(-1).min(1).values
        d[a[:j]] = -torch.inf
        assert a[j] == int(d.argmax())


def test_donor_deleted_scale_against_explicit_removal():
    x = random_descriptors(12)
    d = distances(x, x)
    fit = fit_statistics(d)
    for b in range(3):
        matrix = d[b].copy()
        np.fill_diagonal(matrix, np.inf)
        neighbor = matrix.argmin(1)
        raw = matrix[np.arange(12), neighbor]
        context = []
        for q, r in enumerate(neighbor):
            donors = [i for i in range(12) if i not in [q, r]]
            context.append(np.sort(matrix[r, donors])[:4].mean())
        context = np.array(context)
        beta = np.mean(
            (raw - raw.mean()) * (context - context.mean())
        ) / np.var(context)
        assert np.allclose(beta, fit["beta"][b], atol=1e-12)
        assert np.allclose(
            np.sort(matrix, axis=1)[:, :4].sum(1), fit["ratio4"][b]
        )


def test_fixed_matching_order_and_band_minimum():
    normal = random_descriptors(30)
    query = random_descriptors(7, 2)
    banks = fit_banks(
        normal, ["source"] * 25 + ["target"] * 5, 12, extra_sizes=[8, 16]
    )
    scores = score_banks(normal, query, banks)
    for k in [1, 4]:
        inherited = scores[f"fps|ratio|{k}|inherited"]["scores"]
        fixed = scores[f"fps|ratio|{k}|fixed"]["scores"]
        updated = scores[f"fps|ratio|{k}|updated"]["scores"]
        assert np.all(updated <= fixed + 1e-12)
        assert np.all(fixed <= inherited + 1e-12)
    raw = distances(query, normal)[:, :, banks["fps"]["indices"]]
    v = (
        raw
        / np.maximum(banks["fps"]["statistics"]["ratio1"], 1e-12)[:, None, :]
    )
    assert np.allclose(
        v.min(-1).mean(0), scores["fps|ratio|1|updated"]["scores"]
    )


def test_official_domains_include_all_anomalies_and_ties():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    domain = np.array(["source"] * 4 + ["target"] * 4)
    scores = np.array([0, 1, 1, 2, 0, 0, 1, 1], float)
    result, err = components(y, domain, scores)
    for i, d in enumerate(["source", "target"]):
        mask = (domain == d) | (y == 1)
        assert result[i] == roc_auc_score(y[mask], scores[mask])
    assert result[2] == pytest.approx(roc_auc_score(y, scores, max_fpr=0.1))
    assert err < 1e-12


def test_wiener_zero_far_preserves_near():
    torch.manual_seed(1)
    audio = torch.randn(1, 2, 160000)
    audio[:, 1] = 0
    result = wiener_residual(audio)
    assert torch.allclose(result, audio[:, 0], atol=1e-6, rtol=1e-6)


def test_zero_and_nonfinite_descriptors_rejected():
    with pytest.raises(ValueError):
        fps(np.zeros((10, 3, 4)), 8)
    x = random_descriptors(10)
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fps(x, 8)
