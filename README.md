# Agro Mitra — Smart Crop Recommendation (Streamlit)

A deployable Streamlit prototype trained on the supplied crop recommendation dataset.

## Features
- Random Forest crop classifier trained at app startup
- Top-3 crop recommendations with model probabilities
- Dataset dashboard and crop distribution
- Dataset explorer with feature summaries, filters and CSV download
- Model evaluation: accuracy, per-class precision/recall/F1 and confusion matrix
- Feature importance chart
- Input validation using observed dataset ranges

## Dataset
`crop_recommendation.csv` contains 2,200 rows and these columns:

`N, P, K, temperature, humidity, ph, rainfall, label`

The source dataset does not specify N/P/K units. Keep those values on the same scale as the training data. Temperature is interpreted as °C, humidity as %, pH as pH, and rainfall as mm based on the conventional dataset schema.

## Run locally (Windows)
1. Install Python 3.10 or 3.11.
2. Extract this ZIP file.
3. Open a terminal in the extracted `agromitra_streamlit` folder.
4. Create and activate a virtual environment:

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

5. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the app:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy free on Streamlit Community Cloud
1. Sign in to GitHub and create a new public repository, for example `agromitra-streamlit`.
2. Upload **all three required files** from this folder:
   - `app.py`
   - `crop_recommendation.csv`
   - `requirements.txt`
   Also upload `README.md` if desired.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**.
5. Select your repository, branch (usually `main`) and main file path `app.py`.
6. Select **Deploy**.
7. Wait for the build to finish. Streamlit will provide a public app URL.

If the app fails to deploy, open **Manage app → Logs** and check the first Python error. Confirm the CSV is committed in the same repository directory as `app.py`.

## Project structure

```text
agromitra_streamlit/
├── app.py
├── crop_recommendation.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Model details
- Algorithm: `RandomForestClassifier`
- Trees: 200
- Maximum depth: 10
- Random state: 42
- Split: 80% train / 20% test
- Stratified split to preserve crop class proportions
- Top-3 recommendations use `predict_proba`

The model is trained from the CSV when the Streamlit app starts and cached for subsequent reruns.

## Important scope and limitations
This prototype only predicts the crop class from the seven dataset features. It does **not** currently provide live weather, soil laboratory integration, yield estimates, fertilizer doses, MSP/mandi prices, disease diagnosis or government scheme data. Do not treat a model probability as a guaranteed chance of crop success. Validate decisions with local agricultural experts and current local conditions.
