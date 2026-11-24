import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="Agro Mitra | Smart Crop Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- STYLING ---------------------------
st.markdown("""
<style>
    .stApp { background: #f7faf7; }
    [data-testid="stSidebar"] { background: #102f25; }
    [data-testid="stSidebar"] * { color: #f3fff7 !important; }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #123d2d, #28734c);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.25rem; }
    .hero p { color: #e5f5e9; margin: .4rem 0 0 0; }
    .metric-card {
        background: white; border: 1px solid #e3eee5;
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .crop-result {
        background: white; border: 1px solid #dcebdd;
        border-left: 5px solid #27965a; border-radius: 12px;
        padding: 1rem; margin: .5rem 0;
    }
    .small-note { color: #64756b; font-size: .88rem; }
    div.stButton > button[kind="primary"] {
        background: #23764a; border-color: #23764a; color: white;
        border-radius: 10px; font-weight: 650;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DISPLAY_NAMES = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}
DATA_PATH = os.path.join(os.path.dirname(__file__), "crop_recommendation.csv")


# --------------------------- DATA + MODEL ---------------------------
@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + ["label"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data[FEATURES + ["label"]].copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data.dropna(subset=FEATURES + ["label"])
    data = data[data["label"] != ""]
    if data.empty:
        raise ValueError("No valid rows were found in the dataset.")
    return data


@st.cache_resource
def train_model():
    data = load_dataset()
    X = data[FEATURES]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    return model, accuracy, report, cm, model.classes_, len(X_train), len(X_test)


try:
    data = load_dataset()
    model, accuracy, report, cm, classes, train_count, test_count = train_model()
except Exception as exc:
    st.error(f"Unable to start the application: {exc}")
    st.info("Check that crop_recommendation.csv is present beside app.py and has the required columns.")
    st.stop()


# --------------------------- HELPERS ---------------------------
def crop_title(name):
    return str(name).replace("_", " ").title()


def get_top_predictions(values):
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    order = np.argsort(probabilities)[::-1][:3]
    results = [
        {"crop": classes[i], "probability": float(probabilities[i])}
        for i in order
    ]
    return results


def fmt_range(column, digits=1):
    return f"{data[column].min():.{digits}f} – {data[column].max():.{digits}f}"


# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown("## 🌱 Agro Mitra")
    st.caption("Smart Crop Recommendation")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🌾 Crop Recommendation", "📊 Dataset Explorer", "🧠 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
        key="navigation",
    )
    st.markdown("---")
    st.markdown("**Model status**")
    st.success("Model trained successfully")
    st.caption(f"{len(data):,} valid dataset rows · {data['label'].nunique()} crops")
    st.markdown("---")
    st.caption("Prototype built with Streamlit + Scikit-learn")


# --------------------------- HEADER ---------------------------
st.markdown("""
<div class="hero">
  <h1>🌱 Agro Mitra</h1>
  <p>Your AI-powered crop recommendation companion</p>
</div>
""", unsafe_allow_html=True)


# --------------------------- DASHBOARD ---------------------------
if page == "🏠 Dashboard":
    st.subheader("Welcome to your smart farming assistant")
    st.write("Explore crop patterns in the dataset or enter soil and climate values to receive crop suggestions.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(data):,}")
    c2.metric("Crop categories", f"{data['label'].nunique()}")
    c3.metric("Model accuracy", f"{accuracy * 100:.2f}%")
    c4.metric("Test records", f"{test_count:,}")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Crop distribution")
        counts = data["label"].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        counts.plot(kind="barh", ax=ax, color="#3a9562")
        ax.set_xlabel("Number of records")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("### How it works")
        st.markdown("""
        1. **Enter soil values** – N, P, K and pH.
        2. **Enter climate values** – temperature, humidity and rainfall.
        3. **Get recommendations** – the model ranks the three most likely crops.
        4. **Review the result** – compare the suggested crops and their model scores.
        """)
        st.info("The recommendation is based on patterns learned from the supplied dataset. It is a decision-support prototype, not a substitute for local agronomic advice.")

    st.markdown("### Start here")
    if st.button("🌾 Open crop recommendation", type="primary"):
        st.session_state["navigation"] = "🌾 Crop Recommendation"
        st.rerun()


# --------------------------- RECOMMENDATION ---------------------------
elif page == "🌾 Crop Recommendation":
    st.subheader("🌾 Find a suitable crop")
    st.write("Enter values from your soil test and local climate information.")

    # Dataset-derived input ranges. N/P/K units are not specified in the source file.
    defaults = {col: float(data[col].median()) for col in FEATURES}
    ranges = {
        col: (float(data[col].min()), float(data[col].max()))
        for col in FEATURES
    }

    with st.form("crop_form"):
        st.markdown("#### Soil nutrient values")
        a, b, c = st.columns(3)
        inputs = {}
        for container, col in zip([a, b, c], ["N", "P", "K"]):
            with container:
                inputs[col] = st.number_input(
                    DISPLAY_NAMES[col],
                    min_value=float(max(0, ranges[col][0])),
                    max_value=float(ranges[col][1]),
                    value=float(defaults[col]),
                    step=1.0,
                    help="Use the same measurement scale as the dataset or your soil-test report.",
                )

        st.markdown("#### Climate and soil conditions")
        a, b, c, d = st.columns(4)
        for container, col in zip([a, b, c, d], ["temperature", "humidity", "ph", "rainfall"]):
            with container:
                inputs[col] = st.number_input(
                    DISPLAY_NAMES[col],
                    min_value=float(ranges[col][0]),
                    max_value=float(ranges[col][1]),
                    value=float(defaults[col]),
                    step=0.1,
                    format="%.2f",
                )

        submitted = st.form_submit_button("🌱 Recommend crops", type="primary", use_container_width=True)

    st.caption("Input limits are based on the minimum and maximum values in your uploaded dataset.")
    if submitted:
        results = get_top_predictions(inputs)
        st.markdown("---")
        st.markdown("### Recommended crops")
        best = results[0]
        st.success(f"Top model suggestion: **{crop_title(best['crop'])}**")
        cols = st.columns(3)
        for idx, (container, item) in enumerate(zip(cols, results), start=1):
            with container:
                st.markdown(f"""
                <div class="crop-result">
                    <div style="color:#617268;font-size:.85rem">RANK #{idx}</div>
                    <div style="font-size:1.35rem;font-weight:750;color:#174b32">{crop_title(item['crop'])}</div>
                    <div style="font-size:1.5rem;font-weight:750;color:#258653">{item['probability']*100:.1f}%</div>
                    <div class="small-note">Model probability</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(item["probability"])

        st.markdown("#### Your submitted values")
        st.dataframe(
            pd.DataFrame([inputs]).rename(columns=DISPLAY_NAMES).T.rename(columns={0: "Entered value"}),
            use_container_width=True,
        )
        st.warning("These percentages are model probabilities, not guaranteed crop success rates or yield predictions. Confirm suitability with local soil, season, irrigation and agricultural guidance.")


# --------------------------- DATA EXPLORER ---------------------------
elif page == "📊 Dataset Explorer":
    st.subheader("📊 Explore the crop dataset")
    st.write("The charts and tables below are generated from the uploaded dataset.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing values", int(data.isna().sum().sum()))

    selected_crop = st.selectbox("Filter by crop", ["All crops"] + sorted(data["label"].unique()))
    filtered = data if selected_crop == "All crops" else data[data["label"] == selected_crop]

    st.markdown("### Feature summary")
    st.dataframe(filtered[FEATURES].describe().T.round(2), use_container_width=True)

    st.markdown("### Compare feature distributions")
    feature = st.selectbox("Choose a feature", FEATURES, format_func=lambda x: DISPLAY_NAMES[x])
    fig, ax = plt.subplots(figsize=(10, 4))
    if selected_crop == "All crops":
        filtered.boxplot(column=feature, by="label", ax=ax, rot=70, grid=False)
        fig.suptitle("")
        ax.set_xlabel("Crop")
    else:
        ax.hist(filtered[feature], bins=18, color="#3a9562", edgecolor="white")
        ax.set_xlabel(DISPLAY_NAMES[feature])
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Dataset preview")
    st.dataframe(filtered.head(100), use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv, "agromitra_dataset.csv", "text/csv")


# --------------------------- MODEL INSIGHTS ---------------------------
elif page == "🧠 Model Insights":
    st.subheader("🧠 Model performance and explainability")
    st.write("A Random Forest classifier is trained when the app starts, using an 80/20 stratified train-test split.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{accuracy*100:.2f}%")
    c2.metric("Training rows", f"{train_count:,}")
    c3.metric("Testing rows", f"{test_count:,}")

    st.markdown("### Feature importance")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    importance.plot(kind="barh", ax=ax, color="#3a9562")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Per-crop evaluation")
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    st.dataframe(report_df.round(3), use_container_width=True)

    with st.expander("Show confusion matrix"):
        st.caption("Rows are actual crops; columns are predicted crops.")
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(cm, interpolation="nearest", cmap="Greens")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predicted crop")
        ax.set_ylabel("Actual crop")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels([crop_title(x) for x in classes], rotation=90, fontsize=7)
        ax.set_yticklabels([crop_title(x) for x in classes], fontsize=7)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# --------------------------- ABOUT ---------------------------
elif page == "ℹ️ About":
    st.subheader("About Agro Mitra")
    st.write("""
    Agro Mitra is a prototype crop recommendation application built around the supplied crop recommendation dataset.
    It uses soil nutrient values and climate measurements to estimate which crop label best matches the input pattern.
    """)
    st.markdown("### Dataset columns")
    st.dataframe(pd.DataFrame({
        "Column": FEATURES + ["label"],
        "Meaning": [
            "Nitrogen value", "Phosphorus value", "Potassium value",
            "Temperature", "Relative humidity", "Soil pH", "Rainfall", "Crop class / target"
        ],
        "Role": ["Input"] * 7 + ["Target"],
    }), use_container_width=True)

    st.markdown("### Important limitations")
    st.markdown("""
    - The dataset does not specify the units for N, P and K. Use values on a consistent scale.
    - The dataset contains crop labels, not yield, profit, disease or fertilizer recommendations.
    - This version does not fetch live weather, mandi prices or government scheme information.
    - Predictions reflect the dataset and model; they should be validated with local agricultural experts.
    """)
    st.markdown("### Technology stack")
    st.write("Python · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib")
    st.markdown("### Dataset integrity")
    st.write(f"Loaded **{len(data):,}** valid records across **{data['label'].nunique()}** crop classes.")
