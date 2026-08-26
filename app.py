from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


st.set_page_config(
    page_title="Climate Risk Copilot",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #132b35;
        --muted: #58717a;
        --mist: #eef5f6;
        --cloud: #dce8eb;
        --rain: #7896a3;
        --storm: #233e4b;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 8%, rgba(167, 194, 201, .30), transparent 28rem),
            linear-gradient(180deg, #edf4f5 0%, #f8fbfb 42%, #eef5f3 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1320px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dce8eb 0%, #edf4f3 100%);
        border-right: 1px solid rgba(63, 91, 101, .12);
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, .82);
        border: 1px solid rgba(104, 139, 149, .22);
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 14px 40px rgba(42, 70, 80, .08);
        backdrop-filter: blur(12px);
    }
    [data-testid="stMetricLabel"] {color: var(--muted);}
    [data-testid="stMetricValue"] {color: var(--ink);}
    h1, h2, h3 {color: var(--ink); letter-spacing: -0.03em;}
    .stButton button {border-radius: 999px;}
    .weather-hero {
        min-height: 370px;
        display: flex;
        align-items: flex-end;
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        margin-bottom: 1.5rem;
        background-image:
            linear-gradient(90deg, rgba(8, 24, 34, .88) 0%, rgba(18, 38, 49, .60) 45%, rgba(24, 42, 51, .20) 100%),
            linear-gradient(180deg, rgba(21, 41, 50, .06) 0%, rgba(7, 22, 31, .70) 100%),
            url("https://images.unsplash.com/photo-1492011221367-f47e3ccd77a0?auto=format&fit=crop&fm=jpg&q=84&w=2200");
        background-position: center 48%;
        background-size: cover;
        box-shadow: 0 28px 70px rgba(22, 46, 58, .24);
    }
    .weather-hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 58%, rgba(202, 226, 231, .13));
        pointer-events: none;
    }
    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 720px;
        padding: clamp(2rem, 6vw, 4.5rem);
        color: #f5fafb;
    }
    .weather-kicker {
        display: inline-flex;
        align-items: center;
        gap: .55rem;
        margin-bottom: 1.1rem;
        padding: .5rem .8rem;
        border: 1px solid rgba(231, 244, 247, .35);
        border-radius: 999px;
        background: rgba(224, 239, 242, .11);
        color: #dcebef;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        backdrop-filter: blur(12px);
    }
    .hero-copy h1 {
        margin: 0;
        color: #ffffff;
        font-size: clamp(2.7rem, 6vw, 5.4rem);
        line-height: .94;
        letter-spacing: -.055em;
    }
    .hero-copy p {
        max-width: 600px;
        margin: 1.25rem 0 0;
        color: #e1edf0;
        font-size: clamp(1rem, 1.7vw, 1.24rem);
        line-height: 1.6;
    }
    .photo-credit {
        margin: -.8rem 0 1.6rem;
        color: #71878f;
        font-size: .72rem;
        text-align: right;
    }
    .photo-credit a {color: #5d7781 !important;}
    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
        padding: .4rem;
        border-radius: 18px;
        background: rgba(214, 228, 232, .60);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 13px;
        padding: .45rem .9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, .92);
        color: var(--storm);
    }
    .stDataFrame {
        border: 1px solid rgba(104, 139, 149, .18);
        border-radius: 18px;
        overflow: hidden;
    }
    @media (max-width: 700px) {
        .block-container {padding-top: .8rem;}
        .weather-hero {min-height: 440px; border-radius: 22px;}
        .hero-copy {padding: 2rem 1.4rem;}
        .photo-credit {text-align: left;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path("data/climate_model_no_leakage_dataset.csv")

FEATURE_COLUMNS = [
    "year",
    "month",
    "revenue",
    "total_assets",
    "market_cap",
    "portfolio_weight",
    "climate_event_count",
    "hazard_primary_value",
    "hazard_secondary_value",
    "climate_avg_duration_days",
    "base_climate_risk_score_0_100",
    "month_sin",
    "month_cos",
    "asset_intensity",
    "marketcap_to_assets",
    "event_duration_interaction",
    "hazard_gap",
    "hazard_ratio",
    "weighted_base_risk",
    "weighted_primary_hazard",
    "weighted_event_count",
]
TARGET_COLUMN = "fragility_band"
DISPLAY_COLUMNS = ["fragility_score"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN] + DISPLAY_COLUMNS


@st.cache_data
def load_data(uploaded_file):
    source = uploaded_file if uploaded_file is not None else DATA_PATH
    return pd.read_csv(source).drop_duplicates().reset_index(drop=True)


@st.cache_resource
def train_models(data: pd.DataFrame):
    features = data[FEATURE_COLUMNS].copy()
    target = data[TARGET_COLUMN].copy()

    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
    )
    anomaly_labels = isolation_forest.fit_predict(features_scaled)
    anomaly_scores = isolation_forest.decision_function(features_scaled)

    x_train, _, y_train, _ = train_test_split(
        features,
        target_encoded,
        test_size=0.25,
        random_state=42,
        stratify=target_encoded,
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
    )
    classifier.fit(x_train, y_train)

    return (
        scaler,
        label_encoder,
        classifier,
        clusters,
        anomaly_labels,
        anomaly_scores,
    )


def scenario_scores(
    data: pd.DataFrame,
    event_multiplier: float = 1.0,
    hazard_multiplier: float = 1.0,
    duration_multiplier: float = 1.0,
) -> pd.DataFrame:
    scenario = data.copy()
    scenario["climate_event_count_scn"] = (
        scenario["climate_event_count"] * event_multiplier
    )
    scenario["hazard_primary_value_scn"] = (
        scenario["hazard_primary_value"] * hazard_multiplier
    )
    scenario["hazard_secondary_value_scn"] = (
        scenario["hazard_secondary_value"] * hazard_multiplier
    )
    scenario["climate_avg_duration_days_scn"] = (
        scenario["climate_avg_duration_days"] * duration_multiplier
    )

    scenario["scenario_stress_score"] = (
        0.28 * scenario["base_climate_risk_score_0_100"]
        + 0.18 * scenario["climate_event_count_scn"]
        + 0.18 * np.log1p(scenario["hazard_primary_value_scn"])
        + 0.12 * np.log1p(scenario["hazard_secondary_value_scn"])
        + 0.12 * scenario["climate_avg_duration_days_scn"]
        + 0.12 * (scenario["portfolio_weight"] * 1000)
    )

    lower = scenario["scenario_stress_score"].quantile(0.33)
    upper = scenario["scenario_stress_score"].quantile(0.66)
    scenario["scenario_band"] = pd.cut(
        scenario["scenario_stress_score"],
        bins=[-np.inf, lower, upper, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return scenario


def asset_narrative(row: pd.Series, medians: pd.Series) -> str:
    drivers = []
    labels = {
        "climate_event_count": "frequent climate events",
        "hazard_primary_value": "high primary hazard severity",
        "hazard_secondary_value": "elevated secondary hazard pressure",
        "climate_avg_duration_days": "long-duration exposure",
        "portfolio_weight": "high portfolio concentration",
        "base_climate_risk_score_0_100": "high baseline climate risk",
    }
    for column, label in labels.items():
        if row[column] > medians[column]:
            drivers.append(label)

    driver_text = ", ".join(drivers) if drivers else "a balanced climate profile"
    anomaly_text = (
        " It is also flagged for peer review."
        if row["anomaly_label"] == "Anomalous"
        else ""
    )
    return (
        f"This asset is classified as {row['predicted_fragility_band']} with "
        f"{row['prediction_confidence'] * 100:.1f}% confidence. It belongs to the "
        f"{row['cluster_name']} segment.{anomaly_text} Main drivers: {driver_text}."
    )


st.markdown(
    """
    <section class="weather-hero">
        <div class="hero-copy">
            <div class="weather-kicker">Storm exposure intelligence</div>
            <h1>Climate Risk<br>Copilot</h1>
            <p>See where the portfolio is fragile before the next climate shock arrives.</p>
        </div>
    </section>
    <div class="photo-credit">
        Storm photograph by <a href="https://unsplash.com/@davidmoum" target="_blank">David Moum</a> on Unsplash
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader("Upload project CSV", type="csv")

if uploaded_file is None and not DATA_PATH.exists():
    st.info(
        "Upload the project dataset to begin. You can also place it at "
        "`data/climate_model_no_leakage_dataset.csv`."
    )
    with st.expander("Required columns"):
        st.code("\n".join(REQUIRED_COLUMNS))
    st.stop()

df = load_data(uploaded_file)
missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
if missing_columns:
    st.error("The dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

event_multiplier = st.sidebar.slider(
    "Climate event multiplier", 0.8, 2.0, 1.25, 0.05
)
hazard_multiplier = st.sidebar.slider(
    "Hazard severity multiplier", 0.8, 2.0, 1.20, 0.05
)
duration_multiplier = st.sidebar.slider(
    "Duration multiplier", 0.8, 2.0, 1.30, 0.05
)

(
    scaler,
    label_encoder,
    classifier,
    clusters,
    anomaly_labels,
    anomaly_scores,
) = train_models(df)

cluster_names = {
    0: "Resilient Core",
    1: "Chronic Exposure",
    2: "Shock Sensitive",
    3: "Portfolio Threat",
}
df["cluster"] = clusters
df["cluster_name"] = df["cluster"].map(cluster_names)
df["anomaly_label"] = np.where(anomaly_labels == -1, "Anomalous", "Normal")
df["anomaly_score"] = anomaly_scores

features = df[FEATURE_COLUMNS]
probabilities = classifier.predict_proba(features)
predictions = classifier.predict(features)
df["predicted_fragility_band"] = label_encoder.inverse_transform(predictions)
df["prediction_confidence"] = probabilities.max(axis=1)

pca = PCA(n_components=2, random_state=42)
pca_values = pca.fit_transform(scaler.transform(features))
df["pca_1"] = pca_values[:, 0]
df["pca_2"] = pca_values[:, 1]

median_columns = [
    "climate_event_count",
    "hazard_primary_value",
    "hazard_secondary_value",
    "climate_avg_duration_days",
    "portfolio_weight",
    "base_climate_risk_score_0_100",
]
medians = df[median_columns].median()
df["asset_summary"] = df.apply(asset_narrative, axis=1, medians=medians)

base_df = scenario_scores(df)
scenario_df = scenario_scores(
    df,
    event_multiplier=event_multiplier,
    hazard_multiplier=hazard_multiplier,
    duration_multiplier=duration_multiplier,
)
scenario_df["stress_change"] = (
    scenario_df["scenario_stress_score"] - base_df["scenario_stress_score"]
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Assets analyzed", f"{len(df):,}")
metric_2.metric("Average fragility", f"{df['fragility_score'].mean():.1f}")
metric_3.metric(
    "Flagged for review",
    f"{(df['anomaly_label'] == 'Anomalous').mean() * 100:.1f}%",
)
metric_4.metric(
    "Scenario stress",
    f"{scenario_df['scenario_stress_score'].mean():.1f}",
    f"{scenario_df['scenario_stress_score'].mean() - base_df['scenario_stress_score'].mean():.1f}",
)

overview_tab, segments_tab, asset_tab, scenario_tab, actions_tab = st.tabs(
    ["Overview", "Risk segments", "Asset review", "Scenario lab", "Actions"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Fragility distribution")
        counts = df[TARGET_COLUMN].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index, counts.values, color="#6f9f91")
        ax.set_ylabel("Assets")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
    with right:
        st.subheader("Review status")
        counts = df["anomaly_label"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index, counts.values, color=["#8eb8ad", "#d9a58b"])
        ax.set_ylabel("Assets")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    st.subheader("Highest scenario impact")
    st.dataframe(
        scenario_df.sort_values("stress_change", ascending=False)[
            [
                "predicted_fragility_band",
                "cluster_name",
                "anomaly_label",
                "scenario_band",
                "stress_change",
                "prediction_confidence",
            ]
        ].head(10),
        use_container_width=True,
        hide_index=True,
    )

with segments_tab:
    st.subheader("Portfolio risk segments")
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        ax.scatter(
            subset["pca_1"],
            subset["pca_2"],
            alpha=0.68,
            label=cluster_names.get(cluster_id, f"Segment {cluster_id}"),
        )
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    st.pyplot(fig, use_container_width=True)

    segment_profile = df.groupby("cluster_name")[
        [
            "fragility_score",
            "climate_event_count",
            "hazard_primary_value",
            "hazard_secondary_value",
            "climate_avg_duration_days",
            "portfolio_weight",
            "base_climate_risk_score_0_100",
        ]
    ].mean().round(2)
    st.dataframe(segment_profile, use_container_width=True)

with asset_tab:
    st.subheader("Single asset review")
    row_index = st.selectbox("Choose an asset row", df.index.tolist())
    row = scenario_df.loc[row_index]
    st.info(row["asset_summary"])

    current, impact = st.columns(2)
    with current:
        st.markdown("#### Current profile")
        st.json(
            {
                "Actual fragility band": row["fragility_band"],
                "Predicted fragility band": row["predicted_fragility_band"],
                "Risk segment": row["cluster_name"],
                "Review status": row["anomaly_label"],
                "Prediction confidence": round(row["prediction_confidence"], 3),
            }
        )
    with impact:
        st.markdown("#### Scenario impact")
        st.json(
            {
                "Scenario stress": round(row["scenario_stress_score"], 2),
                "Scenario band": str(row["scenario_band"]),
                "Stress change": round(row["stress_change"], 2),
                "Scenario event count": round(row["climate_event_count_scn"], 2),
                "Scenario duration": round(row["climate_avg_duration_days_scn"], 2),
            }
        )

    st.markdown("#### Strongest model drivers")
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": classifier.feature_importances_}
    ).nlargest(10, "importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = importance.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"], color="#719e91")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

with scenario_tab:
    st.subheader("Climate shock comparison")
    base_high = int((base_df["scenario_band"] == "High").sum())
    shock_high = int((scenario_df["scenario_band"] == "High").sum())
    left, right = st.columns(2)
    left.metric("Base average stress", f"{base_df['scenario_stress_score'].mean():.1f}")
    left.metric("Base high-risk assets", base_high)
    right.metric("Shock average stress", f"{scenario_df['scenario_stress_score'].mean():.1f}")
    right.metric("Shock high-risk assets", shock_high, shock_high - base_high)

with actions_tab:
    st.subheader("Prioritized actions")
    segment_stress = scenario_df.groupby("cluster_name")[
        "scenario_stress_score"
    ].mean().sort_values(ascending=False)
    top_segment = segment_stress.index[0]
    actions = [
        f"Review the {top_segment} segment first; it has the highest average stress.",
        "Prioritize assets combining high portfolio weight and high scenario stress.",
        "Test rebalancing toward lower-duration and lower-hazard assets.",
    ]
    if (scenario_df["anomaly_label"] == "Anomalous").mean() > 0.05:
        actions.insert(1, "Investigate flagged assets for hidden tail-risk exposure.")
    for action in actions:
        st.success(action)

    st.subheader("Review queue")
    review_queue = scenario_df.sort_values(
        ["scenario_stress_score", "anomaly_score", "prediction_confidence"],
        ascending=[False, True, False],
    ).head(15)
    st.dataframe(
        review_queue[
            [
                "predicted_fragility_band",
                "cluster_name",
                "anomaly_label",
                "scenario_band",
                "scenario_stress_score",
                "portfolio_weight",
                "asset_summary",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
