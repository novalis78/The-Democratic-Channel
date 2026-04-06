# The Democratic Channel — Replication Code

Analysis code for "The Democratic Channel: An Information-Theoretic Measurement of Preference-Policy Transmission in the United States."

## Scripts

| Script | Description | Data Required |
|--------|-------------|---------------|
| `lossy_channel_MI_analysis.py` | **Core analysis.** Marginal MI, conditional MI, channel efficiency, temporal analysis, sensitivity analysis, bootstrap CIs on Gilens & Page data. | `DS1_v2.dta` |
| `source_entropy_decomposition.py` | Source entropy H(X) decomposition and KSG continuous MI estimator. Tests whether MI gap reflects source structure or channel bias. | `DS1_v2.dta` |
| `stimson_nominate_validation.py` | Independent validation: Stimson Policy Mood vs DW-NOMINATE congressional ideology. Tests the low-pass filter hypothesis. | `Mood5224.xlsx`, `HSall_members.csv` |
| `budget_mip_validation.py` | Independent validation: Gallup Most Important Problem vs federal budget authority by topic. Tests topic-selective channel hypothesis. | `most_important_problem.csv`, `budget_authority.csv` |

## Data Sources

| Dataset | Source | Access |
|---------|--------|--------|
| Gilens (2012) replication data (`DS1_v2.dta`, `DS2_v2.sav`, `DS3_v2.sav`) | [Russell Sage Foundation](https://www.russellsage.org/datasets/economic-inequality-and-political-representation) | Free download |
| Stimson Policy Mood (`Mood5224.xlsx`) | [stimson.web.unc.edu/data/](https://stimson.web.unc.edu/data/) | Direct download |
| DW-NOMINATE (`HSall_members.csv`) | [voteview.com/data](https://voteview.com/data) | Direct download |
| Policy Agendas: Most Important Problem | [comparativeagendas.net/us](https://www.comparativeagendas.net/us) | Direct download |
| Policy Agendas: Budget Authority | [comparativeagendas.net/us](https://www.comparativeagendas.net/us) | Direct download |

## Reproduction

```bash
pip install -r requirements.txt

# Place data files in this directory, then:
python lossy_channel_MI_analysis.py
python source_entropy_decomposition.py
python stimson_nominate_validation.py
python budget_mip_validation.py
```

## Key Results

| Analysis | Channel | Efficiency |
|----------|---------|------------|
| Gilens specific-policy | Citizen (50th %ile) → Policy | 2.7% |
| Gilens specific-policy | Elite (90th %ile) → Policy | 4.0% |
| Gilens specific-policy | Interest Groups → Policy | 2.1% |
| Stimson/NOMINATE macro-ideological | Public Mood → Congress | 28.9% |
| Policy Agendas topic-specific | MIP → Budget (defense) | 33.0% |
| Policy Agendas topic-specific | MIP → Budget (health) | 8.9% |
