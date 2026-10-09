🎬 Netflix Customer Churn Prediction

A machine-learning web application that predicts whether a Netflix-style customer is likely to churn based on their profile, subscription, viewing activity, and preferences. The project combines a trained classification model with a FastAPI backend and a web-based frontend.

> **Live Demo:** https://netflix-customer-churn-engagement-y069.onrender.com  
> **GitHub Profile:** https://github.com/Rishu6262

---

## 📌 Project Overview

Customer churn happens when a customer stops using or renewing a service. For subscription-based platforms, identifying customers who may churn can help teams investigate engagement drops and plan retention strategies.

This project accepts customer information through a web form and returns a churn prediction with a churn probability.

### 🎯 Objectives

- Predict whether a customer is likely to churn.
- Connect a machine-learning model to a web application.
- Provide a simple interface for entering customer details and viewing predictions.
- Demonstrate an end-to-end ML workflow: preprocessing, model training, API development, and deployment.

## ✨ Features

- Dark, cinematic Netflix-inspired interface.
- Customer input form for profile, subscription, device, payment, and viewing details.
- Prediction through a FastAPI REST API.
- Churn probability displayed in the frontend.
- Loading and error handling during API requests.
- Preprocessing pipeline designed to handle numerical and categorical features.
- Deployed live demo.

## 🧰 Tech Stack

| Area | Technologies |
|---|---|
| Programming language | Python, JavaScript |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost (if used by the final saved model) |
| Preprocessing | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` |
| Backend/API | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Model persistence | Joblib |
| Deployment | Render; frontend hosting depends on the current deployment setup |

## 🧾 Input Features

The API expects the following customer features. The feature names and types must match the model's training pipeline.

| Feature | Description | Expected type |
|---|---|---|
| `age` | Customer age | Integer |
| `gender` | Customer gender category | String |
| `subscription_type` | Subscription plan | String |
| `watch_hours` | Total viewing hours represented by the dataset | Number |
| `last_login_days` | Days since the last login | Integer |
| `region` | Customer region | String |
| `device` | Device used to watch content | String |
| `monthly_fee` | Monthly subscription fee | Number |
| `payment_method` | Payment method category | String |
| `number_of_profiles` | Number of profiles on the account | Integer |
| `avg_watch_time_per_day` | Average daily viewing time | Number |
| `favorite_genre` | Preferred content genre | String |

**Note:** These fields describe the expected API input schema. Use the definitions and units from the actual dataset when interpreting predictions.

## 🔄 How It Works

1. A user enters customer details in the frontend.
2. JavaScript converts the form values into a JSON payload.
3. The frontend sends a `POST` request to the FastAPI `/predict` endpoint.
4. The backend validates the input using Pydantic.
5. The saved model pipeline preprocesses the data and generates a prediction.
6. The API returns the predicted class and churn probability.
7. The frontend displays the result.

```text
User
  ↓
HTML / CSS / JavaScript frontend
  ↓  POST /predict (JSON)
FastAPI backend
  ↓
Preprocessing + trained ML model
  ↓
Prediction + churn probability
  ↓
Result displayed in the UI
```

## 📊 Machine Learning Workflow

The project workflow includes:

1. Load and inspect the dataset.
2. Perform exploratory data analysis (EDA).
3. Check missing values and investigate outliers.
4. Separate features (`X`) from the target (`y`, `churned`).
5. Split the data into training and test sets.
6. Scale numerical features and one-hot encode categorical features.
7. Train and compare classification models.
8. Evaluate models using suitable classification metrics.
9. Save the final model pipeline for API inference.

### Evaluation metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Of the customers predicted to churn, the proportion who actually churned.
- **Recall:** Of the customers who actually churned, the proportion identified by the model.
- **F1-score:** Balance between precision and recall.
- **ROC-AUC:** Ability to rank churn cases above non-churn cases across thresholds.

Exact model scores are intentionally not listed here because they should be copied from the final verified evaluation run. Very high scores should be checked for target leakage, duplicate records, or other data leakage before being presented as final results.

## 🗂️ Suggested Project Structure

Your repository may look similar to this. Keep only files that actually exist in your repository.

```text
Netflix-Customer-Churn/
├── app.py                    # FastAPI application
├── best_churn_model.pkl      # Saved preprocessing + model pipeline
├── requirements.txt          # Python dependencies
├── .python-version            # Optional Python runtime pin
├── netflix_churn_ui/
│   ├── index.html             # Frontend page
│   ├── style.css              # Styling and responsive layout
│   ├── script.js              # Form handling and API requests
│   └── README.md              # Optional frontend-specific notes
└── README.md
```


## 🌐 Deployment Notes

The frontend and backend can be deployed separately.

- **Frontend:** serves `index.html`, `style.css`, and `script.js`.
- **Backend:** runs FastAPI and loads the saved model pipeline.
- **API URL:** the frontend's `API_URL` must point to the deployed backend's `/predict` endpoint.
- **CORS:** configure FastAPI to allow the exact deployed frontend origin. Avoid leaving unrestricted origins in production.
- **Model file:** ensure the deployed backend has access to `best_churn_model.pkl`.
- **Build settings:** for a plain HTML/CSS/JavaScript frontend with no build tool, use the hosting provider's static/Other configuration and set the publish directory to the directory containing `index.html` (often `.` when that directory is the project root). A publish-directory setting is unrelated to the API URL.
- **Render cold starts:** depending on the hosting plan, the first request after inactivity may take longer.

### Troubleshooting

| Symptom | What to check |
|---|---|
| The backend root URL shows JSON instead of the website | This is normal when the root endpoint returns JSON. Open the frontend's own deployed URL to see the UI. |
| `Failed to fetch` or a CORS error | Confirm the API URL, frontend origin in CORS settings, and browser Console/Network errors. |
| `404 Not Found` on prediction | Confirm the URL ends in `/predict` and the route exists in `/docs`. |
| `422 Unprocessable Entity` | Check JSON field names, types, and required values against the Pydantic schema. |
| `500 Internal Server Error` | Check the backend logs, model file, preprocessing pipeline, and installed package versions. |
| Works locally but not after deployment | Check that the deployed JavaScript contains the current API URL and that the frontend has been redeployed after changes. |
| Model fails to load | Align the deployed scikit-learn and other library versions with the model's training environment. |

## 🔒 Limitations and Responsible Use

- This is an educational portfolio project, not an official Netflix system.
- Predictions depend on the dataset and training process and may not generalize to real customers.
- A churn probability is a model estimate, not a guarantee that a customer will leave.
- Review model quality, data leakage, class balance, and performance on genuinely unseen data before any real-world use.
- Avoid entering real personal or payment information into a demo application.

## 👨‍💻 Author

**Rishu Gurjar**

- GitHub: https://github.com/Rishu6262

