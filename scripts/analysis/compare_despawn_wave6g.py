#!/usr/bin/env python3
"""
Comparação rápida DeSpaWN vs Wave6G (MERA) no traço 202406192000_bytes_1ms.

Gera:
- Curva PSNR vs retenção (MERA e DeSpaWN) e ΔPSNR vs retenção.
- Periodograma simples (FFT) do sinal original e da reconstrução DeSpaWN.
- Estimativa grosseira de Hurst (R/S) para original e reconstrução DeSpaWN.

Depende apenas de numpy/pandas/matplotlib.
"""
from __future__ import annotations
from pathlib import Path
import math
import json
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MERA_CSV = ROOT / "results/202406192000_bytes_1ms/L_5/train_202406192000_bytes_1ms.csv"
DESPAWN_DELTA = ROOT / "results/despawn/202406192000_bytes_1ms/delta_psnr_vs_retain.csv"
DESPAWN_RECON = ROOT / "results/despawn/202406192000_bytes_1ms/train_reconstruction.npy"
DESPAWN_ORIG = ROOT / "results/despawn/202406192000_bytes_1ms/train_original.npy"
OUTDIR = ROOT / "results/202406192000_bytes_1ms/L_5"


def load_mera_psnr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # média por retenção
    agg = df.groupby("retain").agg(psnr_mera=("psnr", "mean"))
    return agg.reset_index()


def load_despawn_psnr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "retain": df["retain"],
            "psnr_despawn": df["psnr_train"],
            "delta_psnr_despawn": df["delta_psnr_train"],
        }
    )
    return out


def align_curves(mera: pd.DataFrame, desp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # restringe ao range comum de retenção
    common = sorted(set(mera["retain"]).intersection(set(desp["retain"])))
    mera_c = mera[mera["retain"].isin(common)].sort_values("retain")
    desp_c = desp[desp["retain"].isin(common)].sort_values("retain")
    return mera_c, desp_c


def simple_periodogram(x: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """FFT periodogram; retorna (freq, power)."""
    n = len(x)
    x = x - np.mean(x)
    spec = np.fft.rfft(x)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    power = (np.abs(spec) ** 2) / n
    return freq[1:], power[1:]  # ignora DC


def hurst_rs(x: np.ndarray) -> float:
    """Estimativa simples de Hurst via R/S."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    scales = []
    rs_vals = []
    n = len(x)
    for s in [8, 16, 32, 64, 128, 256, 512, 1024]:
        if s * 8 > n:
            break
        m = n // s
        if m < 2:
            continue
        chunks = x[: m * s].reshape(m, s)
        rs_chunk = []
        for row in chunks:
            y = np.cumsum(row - row.mean())
            r = y.max() - y.min()
            s_dev = row.std(ddof=1)
            if s_dev == 0:
                continue
            rs_chunk.append(r / s_dev)
        if not rs_chunk:
            continue
        scales.append(math.log(s))
        rs_vals.append(math.log(np.mean(rs_chunk)))
    if len(scales) < 2:
        return float("nan")
    slope, _ = np.polyfit(scales, rs_vals, 1)
    return slope


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mera = load_mera_psnr(MERA_CSV)
    desp = load_despawn_psnr(DESPAWN_DELTA)
    mera_c, desp_c = align_curves(mera, desp)

    # referência comum: retain = 0.8 (maior retain do MERA e presente no grid DeSpaWN)
    ref_retain = 0.8
    ref_used = None
    if ref_retain in mera_c["retain"].values and ref_retain in desp_c["retain"].values:
        psnr_ref_mera = float(mera_c.loc[mera_c["retain"] == ref_retain, "psnr_mera"])
        psnr_ref_desp = float(desp_c.loc[desp_c["retain"] == ref_retain, "psnr_despawn"])
        mera_c = mera_c.assign(delta_psnr_mera=mera_c["psnr_mera"] - psnr_ref_mera)
        desp_c = desp_c.assign(delta_psnr_despawn=desp_c["psnr_despawn"] - psnr_ref_desp)
        ref_used = ref_retain
    else:
        # fallback: usar máximo de cada um (comportamento antigo)
        mera_c = mera_c.assign(delta_psnr_mera=mera_c["psnr_mera"] - mera_c["psnr_mera"].max())
        desp_c = desp_c.assign(delta_psnr_despawn=desp_c["psnr_despawn"] - desp_c["psnr_despawn"].max())
        ref_used = None

    # Plot PSNR
    plt.figure(figsize=(6, 4))
    plt.plot(mera["retain"], mera["psnr_mera"], "o-", label="MERA L5 χ2 (PSNR)")
    plt.plot(desp["retain"], desp["psnr_despawn"], "o-", label="DeSpaWN (PSNR)")
    plt.xscale("log")
    plt.xlabel("Retenção (fração de coeficientes mantidos)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "compare_psnr_vs_retain.png", dpi=150)
    plt.close()

    # Plot ΔPSNR
    plt.figure(figsize=(6, 4))
    plt.plot(mera_c["retain"], mera_c["delta_psnr_mera"], "o-", label="MERA ΔPSNR (ref 0.8)")
    plt.plot(desp_c["retain"], desp_c["delta_psnr_despawn"], "o-", label="DeSpaWN ΔPSNR (ref 1.0)")
    plt.xscale("log")
    plt.xlabel("Retenção (fração de coeficientes mantidos)")
    plt.ylabel("ΔPSNR (dB) vs retenção máxima do modelo")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "compare_delta_psnr_vs_retain.png", dpi=150)
    plt.close()

    # PSNR absoluto (nova figura redundante para clareza)
    plt.figure(figsize=(6, 4))
    plt.plot(mera["retain"], mera["psnr_mera"], "o-", label="MERA L5 χ2 (PSNR)")
    plt.plot(desp["retain"], desp["psnr_despawn"], "o-", label="DeSpaWN (PSNR)")
    plt.xscale("log")
    plt.xlabel("Retenção (fração de coeficientes mantidos)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "compare_psnr_abs_vs_retain.png", dpi=150)
    plt.close()

    # Periodograma DeSpaWN (recon x original)
    if DESPAWN_RECON.exists() and DESPAWN_ORIG.exists():
        recon = np.load(DESPAWN_RECON).flatten()
        orig = np.load(DESPAWN_ORIG).flatten()
        f_o, p_o = simple_periodogram(orig)
        f_r, p_r = simple_periodogram(recon)
        plt.figure(figsize=(6, 4))
        plt.loglog(f_o, p_o, label="Original")
        plt.loglog(f_r, p_r, label="DeSpaWN recon")
        plt.xlabel("Frequência (Hz, normalizada)")
        plt.ylabel("Potência")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTDIR / "despawn_periodogram.png", dpi=150)
        plt.close()
        # Hurst (R/S)
        h_orig = hurst_rs(orig)
        h_rec = hurst_rs(recon)
    else:
        h_orig = h_rec = float("nan")

    summary: Dict[str, float] = {
        "mera_psnr_max_retain": float(mera.loc[mera["retain"].idxmax(), "psnr_mera"]),
        "despawn_psnr_max_retain": float(desp.loc[desp["retain"].idxmax(), "psnr_despawn"]),
        "hurst_orig_despawn": h_orig,
        "hurst_recon_despawn": h_rec,
    }
    with (OUTDIR / "compare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
