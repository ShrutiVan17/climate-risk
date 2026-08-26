# Dataset contract

The application expects the prepared modeling dataset used by the original Climate Risk Copilot analysis.

Place it at `data/climate_model_no_leakage_dataset.csv` or upload it from the app sidebar.

## Required columns

```text
year
month
revenue
total_assets
market_cap
portfolio_weight
climate_event_count
hazard_primary_value
hazard_secondary_value
climate_avg_duration_days
base_climate_risk_score_0_100
month_sin
month_cos
asset_intensity
marketcap_to_assets
event_duration_interaction
hazard_gap
hazard_ratio
weighted_base_risk
weighted_primary_hazard
weighted_event_count
fragility_band
fragility_score
```

Do not commit confidential, licensed or personally identifiable data. The CSV path is ignored by Git.

