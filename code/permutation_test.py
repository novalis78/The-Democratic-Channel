#!/usr/bin/env python3
"""
Permutation Test for MI Significance

Shuffles policy outcomes 10,000 times to construct a null distribution of MI
under the hypothesis that preferences and outcomes are independent.
Reports p-values and z-scores for all three channels.

Requires: pandas, numpy, pyreadstat
Data: DS1_v2.dta from Russell Sage Foundation (Gilens 2012 replication)
"""

import numpy as np
import pandas as pd

DATA_PATH = "DS1_v2.dta"
N_BINS = 10
N_PERM = 10000
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


def compute_mi(x, y, n_bins=N_BINS):
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
    return mi


def run():
    df = load_data()
    rng = np.random.RandomState(SEED)

    print("=" * 70)
    print(f"PERMUTATION TEST FOR MI SIGNIFICANCE ({N_PERM} permutations)")
    print("=" * 70)

    for label, col in [('Citizen (50th %ile)', 'pref_50'),
                        ('Elite (90th %ile)', 'pref_90'),
                        ('Interest Groups', 'interest_group')]:
        x = df[col].dropna()
        y = df.loc[x.index, 'outcome']
        observed = compute_mi(x, y)

        null = np.array([
            compute_mi(x, pd.Series(rng.permutation(y.values), index=y.index))
            for _ in range(N_PERM)
        ])

        p = (null >= observed).sum() / N_PERM
        z = (observed - null.mean()) / null.std()

        print(f"\n  {label}:")
        print(f"    Observed MI = {observed:.4f} bits")
        print(f"    Null: mean = {null.mean():.4f}, std = {null.std():.4f},"
              f" max = {null.max():.4f}")
        print(f"    p-value: {'p < 0.0001' if p == 0 else f'p = {p:.4f}'}")
        print(f"    z-score: {z:.1f}")


if __name__ == "__main__":
    run()
