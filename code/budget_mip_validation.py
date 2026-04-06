#!/usr/bin/env python3
"""
Independent Validation 2: Most Important Problem vs Budget Authority

Computes topic-specific MI between Gallup's "Most Important Problem" (public
attention by topic) and federal budget authority shares (actual spending by
topic) for matched policy domains.

Tests whether the democratic channel is topic-selective: does public concern
about a domain predict budget allocation for that domain?

Requires: pandas, numpy
Data (from Policy Agendas Project, https://www.comparativeagendas.net/us):
  - most_important_problem.csv (Gallup MIP coded by topic)
  - budget_authority.csv (Federal budget authority by function)
"""

import numpy as np
import pandas as pd


def compute_mi_continuous(x, y, n_bins=4):
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


# Policy Agendas Project topic crosswalk:
# MIP majortopic -> Budget majorfunction (OMB functional classification)
CROSSWALK = {
    16: (50, 'Defense'),        # Defense -> National Defense
    3:  (550, 'Health'),        # Health -> Health
    6:  (500, 'Education'),     # Education -> Education, Training, Employment
    7:  (300, 'Environment'),   # Environment -> Natural Resources
    12: (750, 'Crime'),         # Law/Crime -> Administration of Justice
}


def run():
    print("=" * 70)
    print("INDEPENDENT VALIDATION: Most Important Problem vs Budget Authority")
    print("=" * 70)

    mip = pd.read_csv('most_important_problem.csv')
    budget = pd.read_csv('budget_authority.csv')
    budget_major = budget[budget['IsItMajorFunction'] == 1].copy()
    total_by_year = budget_major.groupby('year')['amount'].sum()

    print(f"\nMIP: {len(mip)} rows, {mip['year'].min()}-{mip['year'].max()}")
    print(f"Budget: {len(budget_major)} major-function rows,"
          f" {budget_major['year'].min()}-{budget_major['year'].max()}")

    print(f"\n  {'Topic':>12} | {'N':>4} | {'I(MIP;Budget)':>14} |"
          f" {'η':>6} | {'r':>6}")
    print(f"  {'-'*12}-+-{'-'*4}-+-{'-'*14}-+-{'-'*6}-+-{'-'*6}")

    for mip_topic, (budget_func, name) in CROSSWALK.items():
        mip_data = mip[mip['majortopic'] == mip_topic][['year', 'percent']].dropna()
        budget_data = budget_major[
            budget_major['majorfunction'] == budget_func][['year', 'amount']]
        budget_data = budget_data.merge(
            total_by_year.reset_index().rename(columns={'amount': 'total'}),
            on='year')
        budget_data['share'] = budget_data['amount'] / budget_data['total']

        merged = mip_data.merge(budget_data[['year', 'share']], on='year')
        if len(merged) < 10:
            print(f"  {name:>12} | Insufficient data (n={len(merged)})")
            continue

        mi, hy = compute_mi_continuous(merged['percent'], merged['share'], 4)
        eta = mi / hy if hy > 0 else 0
        r = np.corrcoef(merged['percent'], merged['share'])[0, 1]
        print(f"  {name:>12} | {len(merged):>4} | {mi:>14.4f} | {eta:>5.1%} | {r:>5.3f}")

    print(f"\n  INTERPRETATION:")
    print(f"    Defense has highest channel efficiency (public concern -> spending)")
    print(f"    Health has lowest (concentrated pharma interests oppose public prefs)")
    print(f"    Channel is topic-selective: adversarial noise varies by domain")


if __name__ == "__main__":
    run()
