"""Per-record Wiener prediction and frozen original BEATs descriptors."""

from pathlib import Path
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import torch

from .io import sha256

VIEWS = (
    "near_mean",
    "near_rdp4",
    "wiener_mean",
    "wiener_rdp4",
    "residual_mean",
    "residual_rdp4",
)


def wiener_residual(audio, epsilon=1e-12):
    """Near - H Far, H=E[Near Far*]/(Pfar+load), per recording/frequency.

    The load is epsilon only; there is no power-dependent loading coefficient.
    Preserve the original h=0 MVDR arithmetic for bitwise waveform parity.
    """
    if audio.ndim != 3 or audio.shape[1:] != (2, 160000):
        raise ValueError("Expected [batch,2,160000] stereo audio")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    x = audio.double()
    window = torch.hann_window(1024, dtype=x.dtype, device=x.device)
    spectrum = torch.stft(
        x.flatten(0, 1),
        1024,
        256,
        1024,
        window,
        center=True,
        return_complex=True,
    ).reshape(len(x), 2, 513, -1)
    near, far = spectrum.unbind(1)
    pn, pf = near.abs().square().mean(-1), far.abs().square().mean(-1)
    cross = (near * far.conj()).mean(-1)
    load = torch.full_like(pf, epsilon)
    zero = torch.zeros_like(cross)
    numerator_near = pf + load - cross * zero
    numerator_far = -cross.conj() + (pn + load) * zero
    denominator = (numerator_near + zero.conj() * numerator_far).real
    wn, wf = numerator_near / denominator, numerator_far / denominator
    residual = wn.conj()[..., None] * near + wf.conj()[..., None] * far
    wave = torch.istft(
        residual, 1024, 256, 1024, window, center=True, length=160000
    ).float()
    if not torch.isfinite(wave).all():
        raise FloatingPointError("Nonfinite residual waveform")
    return wave


def temporal_pool(tokens, gamma=4.0):
    """Existing RDP: deviation from each band's temporal mean,
    then power weights.
    """
    v = tokens.double()
    if gamma == 0:
        return v.mean(1)
    deviation = (v - v.mean(1, keepdim=True)).norm(dim=-1)
    weights = (
        1 + deviation / deviation.amax(1, keepdim=True).clamp_min(1e-30)
    ).pow(gamma)
    return (
        weights[..., None] / weights.sum(1, keepdim=True)[..., None] * v
    ).sum(1)


def unit(v):
    if (v.norm(dim=-1) == 0).any():
        raise ValueError("Zero descriptor")
    return torch.nn.functional.normalize(v, dim=-1, eps=1e-8).cpu().numpy()


class AudioRecords(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = [{"id": r["id"], "path": r["path"]} for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x, sr = sf.read(row["path"], dtype="float32", always_2d=True)
        if sr != 16000 or x.shape[1] != 2:
            raise ValueError(
                "Expected 16kHz stereo WAV; no implicit resampling/channel"
                " substitution"
            )
        if len(x) < 160000:
            x = np.pad(x, ((0, 160000 - len(x)), (0, 0)))
        wave = torch.from_numpy(x[:160000].T.copy())
        return wave, row["id"]


def collate(items):
    return torch.stack([i[0] for i in items]), [i[1] for i in items]


class Encoder:
    def __init__(
        self, checkpoint, expected_hash, device="cuda", precision="bf16"
    ):
        if sha256(checkpoint) != expected_hash:
            raise ValueError("Checkpoint SHA256 mismatch")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; select CPU explicitly if intended"
            )
        from ._vendor.beats.BEATs import BEATs, BEATsConfig

        data = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=True, mmap=True
        )
        self.model = BEATs(BEATsConfig(data["cfg"]))
        self.model.load_state_dict(data["model"])
        if self.model.predictor is not None:
            raise ValueError(
                "Use the original pretrained iter3 checkpoint, not a finetuned"
                " classifier"
            )
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.autocast = precision == "bf16" and device == "cuda"
        if self.autocast and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 unsupported; do not silently change precision"
            )

    @torch.inference_mode()
    def batches(self, rows, config):
        loader = torch.utils.data.DataLoader(
            AudioRecords(rows),
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            shuffle=False,
            collate_fn=collate,
            pin_memory=self.device == "cuda",
        )
        for audio, metadata in loader:
            n = len(audio)
            if n < config["batch_size"]:
                audio = torch.cat(
                    [
                        audio,
                        audio[-1:].expand(config["batch_size"] - n, -1, -1),
                    ]
                )
            audio = audio.to(self.device)
            residual = wiener_residual(audio, config["epsilon"])
            signals = (
                [("wiener", residual)]
                if config["suite"] == "main"
                else [
                    ("near", audio[:, 0]),
                    ("wiener", residual),
                    ("far", audio[:, 1]),
                ]
            )
            pooled = {}
            values = {}
            for name, signal in signals:
                with (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.autocast
                    else nullcontext()
                ):
                    tokens, _ = self.model.extract_features(signal)
                if tokens.shape[1:] != (496, 768):
                    raise ValueError(f"Unexpected encoder grid {tokens.shape}")
                grid = tokens[:n].float().reshape(n, 62, 8, 768)
                pooled[name] = {
                    p: temporal_pool(grid, g)
                    for p, g in [("mean", 0), ("rdp4", 4)]
                }
                if name != "far":
                    for p, v in pooled[name].items():
                        values[f"{name}_{p}"] = unit(v)
            if config["suite"] == "ablations":
                for p in ["mean", "rdp4"]:
                    values[f"residual_{p}"] = unit(
                        pooled["near"][p] - 0.5 * pooled["far"][p]
                    )
            yield values, metadata
