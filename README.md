from pathlib import Path

readme = r'''<div align="center">

# 🎬 Netflix Customer Churn Prediction & Engagement Intelligence

### Turning customer activity into actionable churn-risk insights with Machine Learning

<p>
  <img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-2E8B57" alt="XGBoost">
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JS-E34F26?logo=html5&logoColor=white" alt="Frontend">
  <img src="https://img.shields.io/badge/Deployment-Render-5A3FFF" alt="Render">
</p>

**[🚀 Live Backend](https://netflix-customer-churn-engagement-y069.onrender.com) · [👨‍💻 GitHub Profile](https://github.com/Rishu6262)**

> **Project type:** End-to-end Machine Learning + REST API + web frontend  
> **Problem type:** Binary classification · **Target:** `churned`

</div>

---

## 📑 Table of Contents

- [01 — Project Overview](#-01--project-overview)
- [02 — Business Problem](#-02--business-problem)
- [03 — Product Vision](#-03--product-vision)
- [04 — Project Highlights](#-04--project-highlights)
- [05 — System Architecture](#-05--system-architecture)
- [06 — Dataset & Feature Dictionary](#-06--dataset--feature-dictionary)
- [07 — Data Quality & Outlier Analysis](#-07--data-quality--outlier-analysis)
- [08 — Exploratory Data Analysis](#-08--exploratory-data-analysis)
- [09 — Feature Engineering & Preprocessing](#-09--feature-engineering--preprocessing)
- [10 — Machine Learning Experiments](#-10--machine-learning-experiments)
- [11 — Evaluation Metrics & Results](#-11--evaluation-metrics--results)
- [12 — Model Serialization](#-12--model-serialization)
- [13 — FastAPI Service](#-13--fastapi-service)
- [14 — Frontend Experience](#-14--frontend-experience)
- [15 — Repository Structure](#-15--repository-structure)
- [16 — Run Locally](#-16--run-locally)
- [17 — Deployment Guide](#-17--deployment-guide)
- [18 — Troubleshooting](#-18--troubleshooting)
- [19 — Limitations & Responsible Use](#-19--limitations--responsible-use)
- [20 — Future Roadmap](#-20--future-roadmap)
- [21 — Skills Demonstrated](#-21--skills-demonstrated)
- [22 — Author](#-22--author)

---

# 🎯 01 — Project Overview

The **Netflix Customer Churn Prediction & Engagement Intelligence System** is a Machine Learning portfolio project built to estimate whether a streaming-service customer is likely to churn based on customer profile, subscription, viewing activity, login recency, device usage, payment method, and genre preference.

The goal is to take a trained classification model beyond a notebook and make it usable through a web API and an interactive frontend. A user can enter customer details, submit them to the backend, and receive a predicted churn class along with the model's estimated churn probability.

The project brings together the major stages of an applied ML workflow:

1. Inspect and understand customer data.
2. Perform data-quality checks and exploratory analysis.
3. Prepare numerical and categorical features.
4. Train and compare classification models.
5. Evaluate the models on a held-out test set.
6. serialize the selected preprocessing/model pipeline.
7. expose predictions through FastAPI.
8. connect the API to a responsive HTML/CSS/JavaScript interface.
9. deploy the API to the cloud.

**Important:** This is an independent educational project inspired by a streaming-service use case. It is not an official Netflix product, does not represent Netflix internal systems, and should not be presented as affiliated with or endorsed by Netflix.

---

# 💼 02 — Business Problem

Subscription-based services depend on customers continuing to use and pay for their service. When a customer stops using a service or cancels their subscription, the business may lose recurring revenue and the opportunity to maintain a long-term relationship.

A churn analysis system attempts to identify patterns associated with customers who have churned. For example, a dataset may contain information about viewing activity, how recently a customer logged in, subscription type, or device usage. A classification model can learn statistical relationships between these inputs and a historical churn label.

## Why this problem matters

- **Retention planning:** Identify customer groups that may deserve further investigation.
- **Engagement analysis:** Explore whether usage signals are associated with churn.
- **Prioritization:** Use model scores to rank records for additional analysis.
- **Decision support:** Give a business team a consistent prediction output rather than relying only on manual inspection.
- **Practical ML delivery:** Demonstrate how a model can be served through an API and accessed by a user interface.

## Problem formulation

| Component | Definition |
|---|---|
| ML task | Supervised binary classification |
| Input | Customer profile, subscription, and engagement features |
| Target | `churned` |
| Class `0` | Not churned |
| Class `1` | Churned |
| Output | Predicted class, result label, churn probability |
| Intended use | Educational demonstration and decision-support prototype |

A model prediction is not a guarantee about future behavior. It reflects patterns learned from the dataset and depends on the quality and representativeness of that data.

---

# 🧭 03 — Product Vision

The project is organized around three connected layers.

### 1. Data and ML layer

Prepare customer data, inspect behavior patterns, transform the input columns consistently, and train classifiers to predict the target label.

### 2. API layer

Load the serialized model pipeline and expose a predictable JSON endpoint. FastAPI validates incoming customer data and returns a structured prediction response.

### 3. User experience layer

Provide a dark, cinematic, streaming-inspired web interface where a user can enter customer attributes and view the API result without directly interacting with Python code.

```text
Customer / Analyst
       |
       v
Frontend Form
HTML + CSS + JavaScript
       |
       | JSON POST request
       v
FastAPI: /predict
       |
       v
Input Validation
       |
       v
Saved Preprocessing + ML Pipeline
       |
       v
Prediction + Probability
       |
       v
JSON Response
       |
       v
Result shown in the UI
```

---

# ✨ 04 — Project Highlights

## 🧹 Data understanding

- Inspected dataset structure, feature types, and summary statistics.
- Checked missing values and duplicate records.
- Reviewed the distributions of numerical features.
- Investigated possible outliers in viewing-related variables.
- Separated the target label from the model inputs.
- Excluded the customer identifier from the feature matrix.

## ⚙️ Preprocessing

- Used `ColumnTransformer` to apply transformations by feature type.
- Used `StandardScaler` for numerical columns.
- Used `OneHotEncoder(handle_unknown="ignore")` for categorical columns.
- Used a stratified train-test split.
- Recommended a single saved pipeline to keep training and inference preprocessing consistent.

## 🤖 Model development

- Logistic Regression as a baseline.
- Decision Tree for tree-based classification.
- Random Forest for ensemble learning.
- XGBoost for gradient-boosted classification.
- Compared accuracy, precision, recall, F1-score, and ROC-AUC where recorded.

## 🌐 Application delivery

- FastAPI REST service.
- Pydantic input schema.
- JSON-based `/predict` endpoint.
- JavaScript `fetch()` integration.
- Churn prediction and probability display.
- Render-hosted API.

---

# 🏗️ 05 — System Architecture

## End-to-end training workflow

```text
Raw Customer Dataset
        |
        v
Dataset Inspection
        |
        v
Data Quality Checks
        |
        v
EDA + Outlier Investigation
        |
        v
Separate Features (X) and Target (y)
        |
        v
Train-Test Split (80/20)
        |
        v
Fit Preprocessor on Training Data
        |
        v
Transform Train and Test Data
        |
        v
Train Candidate Classifiers
        |
        v
Evaluate on Held-Out Test Data
        |
        v
Check for Leakage / Validate Results
        |
        v
Save Final Preprocessing + Model Pipeline
```

## Live prediction workflow

```text
User enters customer details
             |
             v
JavaScript builds a JSON payload
             |
             v
POST /predict
             |
             v
FastAPI validates the request
             |
             v
Pandas DataFrame created from input
             |
             v
Saved pipeline transforms raw columns
             |
             v
Classifier predicts class and probability
             |
             v
FastAPI returns JSON
             |
             v
Frontend renders prediction result
```

## Deployment architecture

```text
Browser
  |
  | opens the static frontend URL
  v
Frontend host (Vercel / Netlify / GitHub Pages)
  |
  | HTTPS request to /predict
  v
Render-hosted FastAPI service
  |
  v
Saved ML pipeline (.pkl)
```

The frontend host and backend host are separate services unless explicitly configured to serve both from the same deployment.

---

# 🧾 06 — Dataset & Feature Dictionary

The project uses a customer-level dataset containing subscription and engagement information. The feature names below correspond to the fields used by the prediction form and model input schema.

## Customer and subscription features

| Feature | Data type | Description |
|---|---|---|
| `age` | Numerical | Customer age |
| `gender` | Categorical | Gender category in the dataset |
| `subscription_type` | Categorical | Subscription plan category |
| `monthly_fee` | Numerical | Monthly subscription fee |
| `number_of_profiles` | Numerical | Number of profiles on the account |

## Engagement features

| Feature | Data type | Description |
|---|---|---|
| `watch_hours` | Numerical | Viewing hours represented in the dataset |
| `last_login_days` | Numerical | Days since the last login |
| `avg_watch_time_per_day` | Numerical | Average daily viewing time |
| `favorite_genre` | Categorical | Preferred content genre |

## Usage and payment features

| Feature | Data type | Description |
|---|---|---|
| `region` | Categorical | Customer region |
| `device` | Categorical | Device category |
| `payment_method` | Categorical | Payment method category |

## Target and identifier

| Column | Role | Handling |
|---|---|---|
| `churned` | Target | Used as `y` during training |
| `customer_id` | Identifier | Excluded from the model input features |

### Target label

| Value | Interpretation |
|---:|---|
| `0` | Not churned |
| `1` | Churned |

### Dataset transparency

This README intentionally does not invent a row count, collection source, or dataset license. Add those details after checking the exact dataset file and its source. If the dataset is synthetic, state that only when its origin confirms it.

---

# 🔍 07 — Data Quality & Outlier Analysis

Before training, a dataset should be checked for issues that can affect model quality and reliability.

## Data quality checklist

- Inspect the number of rows and columns.
- Review data types and unexpected category values.
- Check missing values.
- Check duplicate records.
- Validate the target label and class distribution.
- Identify identifier columns that should not be treated as behavioral predictors.
- Confirm that features available at prediction time are also available during training.

## Outlier investigation

The IQR method flagged the following values during the project's earlier analysis:

| Feature | Reported outlier count |
|---|---:|
| `watch_hours` | 238 |
| `avg_watch_time_per_day` | 549 |

The previously calculated IQR limits were:

| Feature | Lower IQR limit | Upper IQR limit | Observed minimum | Observed maximum |
|---|---:|---:|---:|---:|
| `watch_hours` | -15.70125 | 35.06875 | 0.01 | 110.4 |
| `avg_watch_time_per_day` | -0.805 | 1.635 | 0.0 | 98.42 |

### How to interpret these flags

An IQR flag means a value lies outside the interval calculated from the middle 50% of the data. It does **not** automatically mean that the record is invalid.

For a streaming service, unusually high watch time could represent genuine heavy engagement, a unit/measurement issue, or a data-quality problem. Investigate flagged rows before deciding whether to keep, cap, transform, or remove them. The counts above describe the analysis; they do not assert that every flagged row was removed.

---

# 📊 08 — Exploratory Data Analysis

EDA is used to understand the dataset before selecting a model. The project explored the distribution and range of numerical features and considered how customer and subscription categories could relate to the churn label.

## Suggested EDA views

| Analysis | Question it can help answer |
|---|---|
| Target count plot | Is the target class distribution balanced? |
| Age distribution | What age ranges appear in the dataset? |
| Subscription type vs churn | Does churn frequency differ across plan categories? |
| Watch hours vs churn | Do viewing patterns differ between target classes? |
| Last login vs churn | Is recency associated with churn in this dataset? |
| Device vs churn | Does the target distribution vary by device category? |
| Payment method vs churn | Are there visible differences across payment groups? |
| Genre vs churn | Do churn rates vary across preferred genres? |
| Numerical correlation heatmap | Which numerical variables have linear associations? |
| Box plots | Which numerical features contain extreme values? |

### Interpretation principles

- Compare **rates**, not only raw counts, when group sizes differ.
- Correlation does not establish causation.
- A pattern in a synthetic or limited dataset may not generalize to actual customers.
- Use only information that would be available when the prediction is made.
- Avoid target leakage when creating features or preparing visualizations.

> Add your actual EDA plots or screenshots to the repository and link them here once they are available. This README does not claim that every suggested chart is already included in the committed notebook.

---

# ⚙️ 09 — Feature Engineering & Preprocessing

A machine learning model needs a consistent representation of numerical and categorical data. This project uses scikit-learn preprocessing components to prepare each feature group.

## 9.1 Feature and target separation

```python
X = df.drop(columns=["churned"])
y = df["churned"]

# Exclude the identifier if it exists in the feature matrix.
if "customer_id" in X.columns:
    X = X.drop(columns=["customer_id"])
```

## 9.2 Train-test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The test set should remain held out from model fitting and preprocessing fitting. `stratify=y` helps maintain the target class proportions in both splits.

## 9.3 Numerical and categorical columns

The following column lists should match the dataset exactly:

```python
numeric_cols = [
    "age",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day",
]

categorical_cols = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
]
```

## 9.4 Build the preprocessor

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)
```

### Why `StandardScaler`?

`StandardScaler` transforms each numerical feature using statistics learned from the training data. It is especially useful for models sensitive to feature scale, such as Logistic Regression.

### Why `OneHotEncoder`?

Many classifiers require numeric inputs. One-hot encoding represents a category using binary indicator columns. `handle_unknown="ignore"` prevents a newly encountered category from crashing the transformation; unseen categories do not gain a learned category-specific effect.

### Why `ColumnTransformer`?

It applies the numerical and categorical transformations to their correct columns and keeps the preprocessing logic together.

## 9.5 Fit only on training data

For a manual preprocessing workflow:

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

Use `fit_transform()` on training data and only `transform()` on test data. Fitting the preprocessor on all data before the split can leak information from the test set.

## 9.6 Recommended deployment approach: a single Pipeline

For deployment, it is safer to package the preprocessor and classifier together so the API can accept raw input columns.

```python
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

final_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(random_state=42)),
    ]
)

final_model.fit(X_train, y_train)
```

This example shows the recommended structure. Use the model configuration and selected estimator that were actually evaluated and approved for the project.

---

# 🤖 10 — Machine Learning Experiments

The project compared several supervised classification algorithms. Each model has different strengths and trade-offs.

| Model | Why evaluate it? | Practical consideration |
|---|---|---|
| Logistic Regression | Useful baseline and relatively interpretable | Benefits from sensible scaling and encoding |
| Decision Tree | Captures nonlinear decision rules | Can overfit without suitable constraints |
| Random Forest | Combines multiple decision trees | Can be more robust than one tree, but requires tuning |
| XGBoost | Learns boosted tree ensembles | Requires careful validation and tuning |

## Training examples

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

lg_model = LogisticRegression(max_iter=100, random_state=42)
lg_model.fit(X_train_processed, y_train)
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_processed, y_train)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)
```

### XGBoost

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_processed, y_train)
```

These snippets demonstrate the experiment setup. In a final reproducible training script, also record library versions, model hyperparameters, split strategy, and the exact evaluation outputs.

---

# 📈 11 — Evaluation Metrics & Results

A classification model should not be selected using accuracy alone. Precision, recall, F1-score, and ROC-AUC provide additional information about different types of errors and ranking performance.

## Results recorded during development

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.7% | 88.3% | 91.6% | 89.9% | 0.9663 |
| Decision Tree | 98.0% | 97.6% | 98.4% | 98.0% | 0.9800 |
| Random Forest | 97.8% | 98.4% | 97.2% | 97.8% | Not confirmed |
| XGBoost | 99.6% | 100.0% | 99.2% | 99.6% | 0.9999 |

**These are recorded development results, not independently verified production results.** Re-run the evaluation against the current model and test split before treating them as final portfolio claims.

## What the metrics mean

- **Accuracy:** How many predictions were correct overall.
- **Precision:** Of the customers predicted as churned, how many were labelled churned.
- **Recall:** Of all customers labelled churned, how many the model identified.
- **F1-score:** A combined measure of precision and recall.
- **ROC-AUC:** How well the model separates the two classes across classification thresholds.

## Validation warning: investigate unusually high scores

The recorded XGBoost accuracy and ROC-AUC are exceptionally high. Before concluding that XGBoost is the best model, verify:

1. The target `churned` or a direct derivative is not present in `X`.
2. No features were created using the target label.
3. Duplicate or near-duplicate records do not appear across train and test sets.
4. Preprocessing is fitted only on the training split.
5. The same test split is not repeatedly used to tune decisions until it becomes effectively part of model selection.
6. The feature values used by the deployed API have the same meaning and units as those used during training.

### Calculate ROC-AUC from the correct estimator

```python
from sklearn.metrics import roc_auc_score

y_probability = xgb_model.predict_proba(X_test_processed)[:, 1]
xgb_auc = roc_auc_score(y_test, y_probability)

print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
```

Use the probabilities from `rf_model` when evaluating Random Forest, `dt_model` for Decision Tree, and so on. Do not accidentally calculate one model's ROC-AUC using another model's predictions.

## Confusion matrix

A confusion matrix helps show correct and incorrect predictions for both classes.

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not Churned", "Churned"]
)

plt.title("Customer Churn — Confusion Matrix")
plt.tight_layout()
plt.show()
```

Make sure `y_pred` was produced by the same model named in the chart title.

---

# 💾 12 — Model Serialization

The backend loads a saved model file when the service starts. The most deployment-friendly artifact is a pipeline containing both preprocessing and the trained estimator.

```python
import joblib

joblib.dump(final_model, "best_churn_model.pkl")
```

Load it in the API:

```python
import joblib

model = joblib.load("best_churn_model.pkl")
```

### Model artifact checklist

- The pipeline includes the fitted preprocessing steps.
- The input column names match the API schema.
- The target column is not included in prediction input.
- The model file path matches the deployment environment.
- The Python, scikit-learn, XGBoost, and Joblib versions are compatible.
- The artifact is generated from trusted training code.

**Version compatibility matters:** serialized scikit-learn estimators are not guaranteed to load across arbitrary library versions. If the training environment and Render environment differ, pin compatible dependencies and retrain/re-save the pipeline if necessary.

---

# 🔌 13 — FastAPI Service

FastAPI exposes the trained model through an HTTP interface so that the frontend does not need to load Python or the model itself.

## API routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Health-check message |
| `GET` | `/docs` | Interactive Swagger documentation |
| `POST` | `/predict` | Accept customer attributes and return a churn prediction |

### Deployed backend

- **Base URL:** https://netflix-customer-churn-engagement-y069.onrender.com
- **API documentation:** https://netflix-customer-churn-engagement-y069.onrender.com/docs
- **Prediction endpoint:** https://netflix-customer-churn-engagement-y069.onrender.com/predict

### Example request body

The following payload illustrates the expected field structure. Values and category labels must match the conventions of the dataset used to train the model.

```json
{
  "age": 29,
  "gender": "Female",
  "subscription_type": "Premium",
  "watch_hours": 18.5,
  "last_login_days": 2,
  "region": "North",
  "device": "Smart TV",
  "monthly_fee": 15.99,
  "payment_method": "Credit Card",
  "number_of_profiles": 3,
  "avg_watch_time_per_day": 1.8,
  "favorite_genre": "Drama"
}
```

### Example response structure

```json
{
  "prediction": 1,
  "result": "Customer is likely to churn",
  "churn_probability": 0.874
}
```

This is an illustrative response, not a live result for the example input.

### API behavior

1. FastAPI validates incoming values against the Pydantic schema.
2. The request is converted into a one-row DataFrame.
3. The loaded pipeline performs preprocessing and prediction.
4. `predict_proba()` returns the probability for class `1`, assuming the estimator's class order is the standard `[0, 1]`.
5. The response includes the predicted class, a readable result, and the churn probability.

### API input contract

The API should receive the original feature columns—not already scaled or one-hot-encoded columns—when the saved artifact contains the full preprocessing pipeline.

---

# 🎨 14 — Frontend Experience

The frontend is built with vanilla web technologies and communicates with FastAPI over HTTP.

| Technology | Responsibility |
|---|---|
| HTML | Page structure, form fields, and result containers |
| CSS | Layout, typography, dark theme, responsiveness, and animations |
| JavaScript | Form handling, payload creation, API requests, loading state, and result rendering |
| FastAPI | Request validation and model inference |
| ML pipeline | Preprocessing and churn classification |

## Frontend-to-API connection

The JavaScript API URL must include the backend host and `/predict` path:

```javascript
const API_URL =
  "https://netflix-customer-churn-engagement-y069.onrender.com/predict";
```

If you are testing the local API, use the local endpoint instead:

```javascript
const API_URL = "http://127.0.0.1:8000/predict";
```

Use the deployed URL for the hosted frontend and the local URL only while testing against a locally running backend.

## Why the backend URL can show JSON

Opening the backend root URL in a browser requests `GET /`. If the code defines the root route to return a JSON health-check message, that is the expected response.

Changing `API_URL` inside `script.js` only changes where the frontend sends its prediction request. It does **not** turn the backend's root route into the frontend website.

If frontend and backend are deployed separately:

- Open the **frontend host URL** to see the UI.
- The UI calls the **Render `/predict` endpoint** to get predictions.
- Open `/docs` on Render to inspect and test the API directly.

## CORS

A browser may block a frontend request if the API has not allowed the frontend's origin. Configure FastAPI CORS with the exact deployed frontend origin (and any local development origin needed). Avoid leaving broad wildcard access in production when a specific origin is available.

---

# 📁 15 — Repository Structure

A recommended structure is shown below. Update it to match the exact folders and files that are actually committed to your GitHub repository.

```text
netflix-customer-churn/
│
├── app.py
├── best_churn_model.pkl
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
└── netflix_churn_ui/
    ├── index.html
    ├── style.css
    ├── script.js
    └── README.md
```

### Important repository notes

- `app.py` — FastAPI backend entry point.
- `best_churn_model.pkl` — serialized fitted pipeline, if stored in the repository or provided through a deployment artifact.
- `requirements.txt` — Python dependencies and compatible versions.
- `notebooks/` — EDA, preprocessing experiments, and model evaluation.
- `netflix_churn_ui/` — static frontend assets.

Do not include secrets or credentials in Git. Large datasets and model artifacts should be committed only when repository size and dataset/model licensing permit. If the `.pkl` file is excluded from Git, document how it is generated or supplied to the deployment environment.

---

# 🛠️ 16 — Run Locally

## Prerequisites

- Python installed
- Git installed
- The project source code
- A compatible saved model pipeline (`best_churn_model.pkl`) or the code and data needed to generate it

## Step 1 — Clone the repository

Replace `REPOSITORY-NAME` with the actual repository name. The supplied GitHub link points to the profile, not a confirmed repository URL.

```bash
git clone https://github.com/Rishu6262/REPOSITORY-NAME.git
cd REPOSITORY-NAME
```

## Step 2 — Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

The exact requirements should reflect the packages used by your code and the versions compatible with the saved model, including packages such as:

- `fastapi`
- `uvicorn`
- `pandas`
- `joblib`
- `scikit-learn`
- `xgboost`

Use pinned, tested versions for reproducible deployment.

## Step 4 — Start FastAPI

If the FastAPI object in `app.py` is named `app`:

```bash
uvicorn app:app --reload
```

Then check:

| Purpose | Local URL |
|---|---|
| Health check | http://127.0.0.1:8000/ |
| API docs | http://127.0.0.1:8000/docs |
| Prediction endpoint | http://127.0.0.1:8000/predict |

The `/predict` route is a POST endpoint, so opening it directly in the browser may not execute a prediction. Use Swagger docs or send a POST request with a valid JSON body.

## Step 5 — Serve the frontend

Open a second terminal in the directory containing `index.html`:

```bash
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500
```

Serving the page over HTTP is preferable to opening it directly through a `file:///` URL, particularly when debugging browser-origin and CORS behavior.

For local frontend testing, configure JavaScript to use `http://127.0.0.1:8000/predict`. For hosted frontend testing, configure it to use the deployed Render `/predict` URL.

---

# ☁️ 17 — Deployment Guide

## A. Deploy the FastAPI backend on Render

1. Push the backend code and deployment files to the repository.
2. Create a Render Web Service connected to the backend repository.
3. Configure the build command:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure the start command:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

5. Make sure the saved pipeline exists at the path used by `joblib.load()`.
6. Ensure the dependency versions are compatible with the serialized artifact.
7. Deploy and check the health endpoint.
8. Open `/docs` and test `POST /predict` with a valid payload.

## B. Deploy the static frontend

The frontend contains plain HTML, CSS, and JavaScript, so a separate build system is not normally required.

1. Push `index.html`, `style.css`, and `script.js` to GitHub.
2. Import the repository into a static hosting provider such as Vercel or Netlify.
3. Select the folder that directly contains `index.html` as the root/publish directory.
4. If the provider asks for a framework preset, choose **Other** for a plain static site.
5. Leave the build command empty if no build process is configured.
6. Ensure the published directory points to the actual static files, not a parent folder that does not contain `index.html`.
7. Verify the `API_URL` in `script.js` points to the deployed Render `/predict` endpoint.
8. Deploy the frontend and open the frontend URL—not the backend root URL.

## C. Final end-to-end checklist

- [ ] Render service starts without model-loading errors.
- [ ] `GET /` returns the expected health-check response.
- [ ] `/docs` opens.
- [ ] `POST /predict` returns a valid JSON response.
- [ ] Frontend URL displays the website interface.
- [ ] The deployed JavaScript uses the correct API URL.
- [ ] FastAPI CORS allows the frontend origin.
- [ ] The browser Network tab shows a successful `/predict` request.
- [ ] The returned prediction is rendered correctly by the frontend.
- [ ] The model's evaluation has been checked for leakage and reproducibility.

---

# 🧯 18 — Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Root URL displays JSON | The root endpoint is a health-check route | Open the separate frontend URL to view the UI |
| Frontend loads but prediction fails | Wrong API URL, CORS, backend error, or network issue | Browser Console and Network tabs |
| `404 Not Found` on prediction | Incorrect route or URL | Confirm the URL ends with `/predict` |
| `405 Method Not Allowed` | Request method does not match the route | Send `POST` to `/predict` |
| `422 Unprocessable Entity` | Missing fields or wrong data types | Compare payload with the Pydantic schema |
| `500 Internal Server Error` | Model inference or backend exception | Inspect Render logs and the error detail |
| `Failed to fetch` | Could be CORS, service availability, or an incorrect URL | Inspect Console and Network request details |
| Model fails to load after deployment | Incompatible serialized model/library versions or missing file | Check `requirements.txt`, artifact path, and Render logs |
| Categories behave unexpectedly | Input labels differ from training categories | Compare UI options with training data values |
| Local page works but deployed page fails | Production API URL or CORS configuration differs | Inspect the deployed `script.js` and allowed origins |

### Useful browser debugging steps

1. Open the frontend website.
2. Press `F12` to open Developer Tools.
3. Select **Console** and submit a prediction.
4. Select **Network**, then filter by `Fetch/XHR`.
5. Open the `/predict` request.
6. Inspect the Request URL, Status Code, request payload, and response body.

Use the actual browser error to decide what to change instead of repeatedly changing working backend settings without evidence.

---

# ⚠️ 19 — Limitations & Responsible Use

- Model quality depends on the dataset's completeness, correctness, and representativeness.
- A probability estimate is not a guarantee that a customer will churn.
- Results from a single train-test split may not generalize to future customers.
- Very high scores should be investigated for target leakage and duplicated records.
- The project does not establish causal reasons why a customer churns.
- Customer information should be handled with appropriate authorization and privacy safeguards.
- This is an educational prototype, not a validated production retention system.
- Do not use predictions as the sole basis for consequential customer decisions.

---

# 🔮 20 — Future Roadmap

## Model quality
- Perform cross-validation and a documented leakage audit.
- Tune model hyperparameters using a validation strategy.
- Evaluate calibration of predicted probabilities.
- Choose a decision threshold based on the cost of false positives and false negatives.
- Compare performance against a simple baseline.

## Explainability
- Add feature importance or SHAP explanations.
- Show the strongest signals associated with each prediction where technically supported.
- Clearly distinguish model associations from causal explanations.

## Backend and MLOps
- Add automated tests for the preprocessing pipeline and API schema.
- Add structured logging and robust error responses.
- Version model artifacts and training datasets.
- Add a reproducible training script.
- Monitor data drift and prediction drift.
- Add batch inference when required.

## Frontend
- Improve accessibility and form validation.
- Add clearer input guidance and API loading/error states.
- Add a downloadable prediction summary if useful.
- Add an analytics page only when supported by the available dataset and implementation.

---

# 🧠 21 — Skills Demonstrated

| Area | Skills |
|---|---|
| Python | Data handling, reusable code, model integration |
| Data analysis | Pandas, descriptive statistics, data quality checks |
| Visualization | EDA, distribution analysis, confusion matrix |
| Preprocessing | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` |
| Machine Learning | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| Evaluation | Accuracy, precision, recall, F1-score, ROC-AUC |
| Model deployment | Joblib serialization, dependency compatibility |
| Backend development | FastAPI, Pydantic, REST endpoints, JSON |
| Frontend development | HTML, CSS, JavaScript, `fetch()` |
| Deployment | Render backend and static frontend hosting |
| Debugging | API status codes, CORS, browser Console and Network tools |

---

# 👨‍💻 22 — Author

<div align="center">

## Rishu Gurjar

**B.Tech Computer Science Student**  
Python · Machine Learning · Data Science · AI Engineering

[![GitHub](https://img.shields.io/badge/GitHub-Rishu6262-181717?logo=github)](https://github.com/Rishu6262)

</div>

---

## 📜 License

Add a `LICENSE` file to the repository before declaring a license. If you choose the MIT License, include the complete MIT license text in that file and confirm that the project assets and dataset can be distributed under the chosen terms.

---

<div align="center">

## ⭐ Final Note

This project demonstrates the journey from customer data and model experimentation to an API-backed prediction experience. The next step is to validate the reported model performance, keep preprocessing consistent between training and deployment, and make the live frontend-to-API flow reliable and reproducible.

**If you find the project interesting, consider starring the repository and sharing constructive feedback.**

</div>
'''

path = Path("/mnt/data/README_Netflix_Customer_Churn_Deep_Dive.md")
path.write_text(readme, encoding="utf-8")
print(f"Created {path} ({path.stat().st_size:,} bytes)")
