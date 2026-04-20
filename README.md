# EV Battery Manufacturing Site Selection

**Python | Machine Learning | Michigan Ross Datathon 2026 — Runner-Up**

Data-driven site selection study for a hypothetical $5B EV battery manufacturing investment across all 50 US states. Built under competition conditions, scored Runner-Up at the Michigan Ross Datathon 2026.

---

## The Business Problem

Where should a company build a $5B EV battery plant in the United States? The decision involves tax environment, labor market, industrial base, and long-term policy stability — all of which vary dramatically by state.

This project builds a hybrid ML + domain expert scoring model to answer that question with 20 years of government data.

## Model Architecture

```
Final Score = (ML Score x 60%) + (Domain Score x 40%)

ML Score    = Pattern-discovered weights from Random Forest and Regression
Domain Score = Expert framework across 4 business pillars
```

## Four Pillars

| Pillar | Weight | Key Metrics |
|---|---|---|
| Tax Environment | 25% | Tax per capita, revenue volatility, CAGR |
| Labor Market | 25% | Workforce size, wage levels, participation rate |
| Industrial Base | 25% | Existing EV infrastructure, manufacturing density |
| Policy Stability | 25% | Regulatory consistency, incentive programs |

## Key Findings

- **Texas wins** with a score of 79.76/100, driven by low per-capita tax burden, large industrial labor pool, and policy stability
- **Gas tax CAGR** emerged as the strongest ML predictor at 14% feature importance
- Texas projects **$200M to $300M in annual savings** over Florida (72.9/100), with a $2B+ long-term cost advantage
- High-risk states eliminated through statistical screening on volatility coefficients

## Project Structure

```
ev-battery-site-selection/
├── analysis/
│   └── Final_Submission_Model.py     <- Full ML pipeline
├── report/
│   ├── FINAL_SUBMISSION_REPORT.md    <- Detailed methodology and results
│   ├── FINAL_SUBMISSION_SUMMARY.md   <- Executive summary
│   └── INDEX.md                      <- Navigation guide
├── data/
│   ├── final_submission_overall_rankings.csv   <- All 50 state scores
│   └── ML_Features_QuickRef.csv                <- Feature importance reference
└── visuals/
    └── final_submission_visualizations.png     <- Key charts
```

## Data Sources

- US Census Bureau State Tax Revenue (2004 to 2025, ~112K records)
- Current EV factory locations (34 sites, 15 states)
- BLS Economic Data: GDP, labor force, demographics (2020 to 2024 averages)

## Tools

`Python` `pandas` `scikit-learn` `Random Forest` `Regression` `matplotlib`

---

*Submitted to the 2026 Michigan Ross Datathon. Team 43.*
