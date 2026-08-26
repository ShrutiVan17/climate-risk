# Climate Risk Copilot

An interactive portfolio risk workspace for detecting climate fragility, reviewing unusual assets and testing climate shock scenarios.

## What it does

- Classifies assets into climate fragility bands with a Random Forest model
- Groups exposures into four practical risk segments with K-Means clustering
- Flags unusual risk profiles with Isolation Forest
- Maps portfolio structure into two dimensions with PCA
- Simulates changes in event frequency, hazard severity and event duration
- Produces a prioritized review queue and concise asset-level explanations

## Project results

The original analysis covered **1,080 records**. Its regression benchmark achieved **R² 0.919**, **MAE 4.81** and **RMSE 7.26**. The portfolio was divided into four risk groups:

| Risk group | Records |
|---|---:|
| Low | 630 |
| Emerging | 234 |
| High | 90 |
| Critical | 126 |

## Application workflow

1. Load the prepared project dataset.
2. Standardize the model features.
3. identify four portfolio risk segments.
4. Detect anomalous assets and predict fragility bands.
5. Apply event, hazard and duration stress multipliers.
6. Rank assets for review and portfolio action.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Upload the prepared CSV in the sidebar, or save it as:

```text
data/climate_model_no_leakage_dataset.csv
```

The original dataset is not committed because it was not included with the recovered application source. The app validates the schema before training; see [`data/README.md`](data/README.md) for the required columns.

## Scenario score

The scenario engine combines baseline climate risk, event frequency, primary and secondary hazards, event duration and portfolio weight. Low, medium and high scenario bands are assigned using the 33rd and 66th percentile thresholds for the active portfolio.

## Technology

Python · pandas · NumPy · scikit-learn · Streamlit · Matplotlib

## Repository structure

```text
.
├── app.py              # Interactive Streamlit application
├── data/README.md      # Dataset contract
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Notes

- Model outputs support prioritization and scenario exploration; they are not financial advice.
- Segment names are analytical interpretations of the four learned clusters.
- Fixed random seeds make the model results reproducible for a given dataset.

