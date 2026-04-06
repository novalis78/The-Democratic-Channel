#!/usr/bin/env python3
"""
Independent Validation 1: Stimson Policy Mood vs DW-NOMINATE

Computes mutual information between aggregate public mood (liberal/conservative)
and median congressional ideology, both measured on the same single dimension.

This tests the "low-pass filter" hypothesis: does the democratic channel transmit
broad ideological signals more efficiently than specific policy preferences?

Requires: pandas, numpy, openpyxl
Data:
  - Mood5224.xlsx from https://stimson.web.unc.edu/data/
  - HSall_members.csv from https://voteview.com/data
"""

import numpy as np
import pandas as pd


def compute_mi_continuous(x, y, n_bins=6):
    """Compute MI between two continuous variables via joint histogram."""
    x_binned = pd.cut(x, bins=n_bins, labels=False)
    y_binned = pd.cut(y, bins=n_bins, labels=False)
    mask = ~(np.isnan(x_binned) | np.isnan(y_binned))
    x_b = x_binned[mask].astype(int)
    y_b = y_binned[mask].astype(int)
    joint = np.zeros((n_bins, n_bins))
    for xi, yi in zip(x_b, y_b):
        if 0 <= xi < n_bins and 0 <= yi < n_bins:
            joint[xi, yi] += 1
    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    hy = -sum(p * np.log2(p) for p in py if p > 0)
    return mi, hy


def run():
    print("=" * 70)
    print("INDEPENDENT VALIDATION: Stimson Policy Mood vs DW-NOMINATE")
    print("=" * 70)

    # Load Stimson mood data
    mood_raw = pd.read_excel('Mood5224.xlsx')
    mood = mood_raw[['Year', 'Annual']].dropna()
    mood = mood[mood['Year'].apply(lambda x: str(x).replace('.', '').isdigit())]
    mood['Year'] = mood['Year'].astype(int)
    mood['mood'] = mood['Annual'].astype(float)
    mood = mood[['Year', 'mood']]

    # Load DW-NOMINATE
    nom = pd.read_csv('HSall_members.csv', low_memory=False)
    nom['year_start'] = 1787 + 2 * (nom['congress'] - 1)
    nom_dr = nom[nom['party_code'].isin([100, 200])].copy()
    annual_nom = nom_dr.groupby('year_start')['nominate_dim1'].median().reset_index()
    annual_nom.columns = ['Year', 'congress_ideology']

    # Merge
    merged = mood.merge(annual_nom, on='Year').dropna()
    print(f"\nMerged: {len(merged)} years ({merged.Year.min()}-{merged.Year.max()})")
    print(f"Correlation (mood, congress): {merged['mood'].corr(merged['congress_ideology']):.4f}")

    # MI across bin counts
    print(f"\n  {'Bins':>4} | {'I(Mood;Congress)':>17} | {'H(Congress)':>12} | {'η':>6}")
    print(f"  {'-'*4}-+-{'-'*17}-+-{'-'*12}-+-{'-'*6}")
    for nb in [4, 5, 6, 8, 10]:
        mi, hy = compute_mi_continuous(merged['mood'], merged['congress_ideology'], nb)
        eta = mi / hy if hy > 0 else 0
        print(f"  {nb:>4} | {mi:>17.4f} | {hy:>12.4f} | {eta:>5.1%}")

    # Main result
    mi_main, hy_main = compute_mi_continuous(
        merged['mood'], merged['congress_ideology'], 6)
    print(f"\n  HEADLINE: I(Public Mood; Congressional Ideology) = {mi_main:.4f} bits")
    print(f"  Channel efficiency η = {mi_main/hy_main:.1%}")
    print(f"  (Compare to Gilens citizen-policy: 2.7%)")

    # By era
    print(f"\n  Temporal breakdown:")
    for label, y1, y2 in [('1952-1980', 1952, 1980),
                           ('1981-2002', 1981, 2002),
                           ('2003-2024', 2003, 2024)]:
        sub = merged[(merged['Year'] >= y1) & (merged['Year'] <= y2)]
        if len(sub) < 10:
            continue
        mi_s, hy_s = compute_mi_continuous(sub['mood'], sub['congress_ideology'], 4)
        eta_s = mi_s / hy_s if hy_s > 0 else 0
        print(f"    {label} (n={len(sub)}): I = {mi_s:.4f}, η = {eta_s:.1%}")


if __name__ == "__main__":
    run()
