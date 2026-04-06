#!/usr/bin/env python3
"""
Source Entropy Decomposition & KSG Continuous MI Estimator

Tests whether the MI gap between citizen and elite channels is explained
by differences in source signal structure (entropy) or by differential
channel treatment (bias).

Also verifies binned MI results using the Kraskov-Stögbauer-Grassberger
(KSG) continuous k-nearest-neighbor estimator, which requires no
discretization.

Requires: pandas, numpy, scipy, pyreadstat, scikit-learn
Data: DS1_v2.dta from Russell Sage Foundation (Gilens 2012 replication)
"""

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "DS1_v2.dta"  # Gilens replication dataset
N_BINS = 10
SEED = 42


def load_data(path=DATA_PATH):
    raw = pd.read_stata(path)
    raw = raw[raw['OUTCOME'] != 99].copy()
    df = pd.DataFrame({
        'pref_50': raw['pred50_sw'] * 100,
        'pref_90': raw['pred90_sw'] * 100,
        'interest_group': (raw['INTGRP_STFAV'] + raw['INTGRP_SWFAV']
                           - raw['INTGRP_STOPP'] - raw['INTGRP_SWOPP']),
        'outcome': (raw['OUTCOME'] >= 2).astype(int),
    })
    return df.dropna(subset=['pref_50', 'pref_90', 'outcome'])


def compute_entropy(x, n_bins=N_BINS):
    """Compute Shannon entropy H(X) of a discretized continuous variable."""
    x_binned = pd.cut(x, bins=n_bins, labels=False)
    mask = ~np.isnan(x_binned)
    x_b = x_binned[mask].astype(int)
    counts = np.bincount(x_b, minlength=n_bins)
    px = counts / counts.sum()
    return -sum(p * np.log2(p) for p in px if p > 0)


def compute_mi(x, y, n_bins=N_BINS):
    """Compute mutual information I(X;Y) via histogram estimation."""
    x_binned = pd.cut(x, bins=n_bins, labels=False)
    mask = ~np.isnan(x_binned)
    x_b = x_binned[mask].astype(int)
    y_b = y[mask].astype(int)
    joint = np.zeros((n_bins, 2))
    for xi, yi in zip(x_b, y_b):
        if 0 <= xi < n_bins:
            joint[xi, yi] += 1
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(2):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    hy = -sum(p * np.log2(p) for p in py if p > 0)
    return mi, hy


def run():
    df = load_data()
    print("=" * 70)
    print("SOURCE ENTROPY DECOMPOSITION")
    print("=" * 70)
    print(f"\nDataset: {len(df)} policy issues")

    for label, col in [('Citizen (50th %ile)', 'pref_50'),
                        ('Elite (90th %ile)', 'pref_90')]:
        hx = compute_entropy(df[col])
        mi, hy = compute_mi(df[col], df['outcome'])
        eta_output = mi / hy if hy > 0 else 0
        eta_input = mi / hx if hx > 0 else 0
        print(f"\n  {label}:")
        print(f"    H(X) = {hx:.4f} bits  (source entropy)")
        print(f"    H(Y) = {hy:.4f} bits  (output entropy)")
        print(f"    I(X;Y) = {mi:.4f} bits")
        print(f"    I(X;Y)/H(Y) = {eta_output:.4f}  ({eta_output:.1%})"
              f"  [output-normalized = channel efficiency]")
        print(f"    I(X;Y)/H(X) = {eta_input:.4f}  ({eta_input:.1%})"
              f"  [input-normalized = fraction of source that survives]")

    hx_cit = compute_entropy(df['pref_50'])
    hx_elite = compute_entropy(df['pref_90'])
    mi_cit, hy = compute_mi(df['pref_50'], df['outcome'])
    mi_elite, _ = compute_mi(df['pref_90'], df['outcome'])

    print(f"\n  KEY COMPARISON:")
    print(f"    H(X_citizen)  = {hx_cit:.4f} bits")
    print(f"    H(X_elite)    = {hx_elite:.4f} bits")
    print(f"    Ratio = {hx_cit/hx_elite:.3f}")
    print(f"\n    Input-normalized:  Citizen {mi_cit/hx_cit:.4f}"
          f"  Elite {mi_elite/hx_elite:.4f}"
          f"  Ratio = {(mi_elite/hx_elite)/(mi_cit/hx_cit):.2f}x")
    print(f"    Output-normalized: Citizen {mi_cit/hy:.4f}"
          f"  Elite {mi_elite/hy:.4f}"
          f"  Ratio = {(mi_elite/hy)/(mi_cit/hy):.2f}x")

    if abs(hx_cit - hx_elite) < 0.1:
        print(f"\n  CONCLUSION: Source entropies are nearly equal.")
        print(f"  -> MI gap is NOT explained by source structure.")
        print(f"  -> The channel selectively attenuates the citizen signal.")

    # KSG continuous estimator
    print(f"\n{'=' * 70}")
    print("KSG CONTINUOUS MI ESTIMATOR (no discretization)")
    print("=" * 70)

    try:
        from sklearn.feature_selection import mutual_info_classif
        X_cit = df['pref_50'].values.reshape(-1, 1)
        X_elite = df['pref_90'].values.reshape(-1, 1)
        y = df['outcome'].values

        mi_ksg_cit = mutual_info_classif(
            X_cit, y, discrete_features=False,
            random_state=SEED, n_neighbors=5)[0] / np.log(2)
        mi_ksg_elite = mutual_info_classif(
            X_elite, y, discrete_features=False,
            random_state=SEED, n_neighbors=5)[0] / np.log(2)

        print(f"\n  KSG (k=5 neighbors):")
        print(f"    I(Citizen; Policy) = {mi_ksg_cit:.4f} bits"
              f"  (η = {mi_ksg_cit/hy:.1%})")
        print(f"    I(Elite; Policy)   = {mi_ksg_elite:.4f} bits"
              f"  (η = {mi_ksg_elite/hy:.1%})")
        print(f"    Ratio = {mi_ksg_elite/mi_ksg_cit:.2f}x")
        print(f"\n  Comparison (binned K=10 vs KSG):")
        print(f"    Citizen: {mi_cit:.4f} (binned) vs {mi_ksg_cit:.4f} (KSG)")
        print(f"    Elite:   {mi_elite:.4f} (binned) vs {mi_ksg_elite:.4f} (KSG)")
    except ImportError:
        print("\n  scikit-learn not installed; skipping KSG estimator.")
        print("  Install with: pip install scikit-learn")


if __name__ == "__main__":
    run()
