# Online Shopper Purchase Prediction: ML Assignment 2

**Course:** Machine Learning (AIMLZG565), M.Tech AIML, BITS Pilani WILP
**Author:** <Nandini Maheshwari> (<BITSID: 2026AK05287>)
**Live app:** <Streamlit app URL>

## a. Problem statement

Predict whether an online shopping session will end in a purchase, using only how the visitor browsed the website (pages visited, time spent, bounce/exit rates, page values, visit timing and visitor type). This is a **binary classification** problem: the target `Revenue` is 1 if the session ended in a purchase and 0 otherwise.

5 classifiers are trained on the same data, compared on 6 eval. metrics, and
deployed as a Streamlit app that evaluates the saved models on uploded test data.

## b. Dataset description

- **Name:** Online Shoppers Purchasing Intention
- **Source:** Kaggle: [https://www.kaggle.com/datasets/sairupeshl/online-shoppers-purchase-intention](https://www.kaggle.com/datasets/sairupeshl/online-shoppers-purchase-intention)
- **Size:** 12,330 sessions × 18 columns (17 features + target), no missing values.
  125 exact duplicate rows were removed → **12,205 sessions** used.
- **Target:** `Revenue` (True/False → 1/0). Imbalanced: only **~15.6%** of sessions end in a purchase.
- **Features (17):**
  - _Numeric (10):_ `Administrative`, `Administrative_Duration`, `Informational`,
    `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`,
    `ExitRates`, `PageValues`, `SpecialDay`
  - _Categorical (7):_ `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`,
    `VisitorType`, `Weekend` (the integer-coded columns are IDs, so they are treated as categories)
- **Split:** stratified 80/20 train/test (`random_state=42`) → 9,764 training and
  2,441 test sessions. The test split is provided as `test_data.csv`.
- **Preprocessing (inside each saved pipeline):** `StandardScaler` on numeric features,
  `OneHotEncoder(handle_unknown="ignore")` on categorical features → 74 model inputs.

## c. GitHub repository

<GitHub repository URL>

```
├── app.py                    # Streamlit app (loads saved models, no retraining)
├── ml_utils.py               # shared metric computation used by notebook and app
├── model_training.ipynb      # EDA, preprocessing, training, evaluation, saving models
├── model/                    # 5 trained pipelines (.joblib), model_info.json, metrics_comparison.csv
├── data/                     # original dataset CSV
├── test_data.csv             # held-out 20% test split (features + Revenue)
├── requirements.txt
└── README.md
```

**Run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

## d. Models used

All 5 models use the same reusble preprocessing pipeline & the same train/test split:

| Model                  | Settings                                                                    |
| ---------------------- | --------------------------------------------------------------------------- |
| Logistic Regression    | `max_iter=1000`                                                             |
| Decision Tree          | `max_depth=6` (limits overfitting)                                          |
| KNN                    | `n_neighbors=7`                                                             |
| Naive Bayes            | `GaussianNB(var_smoothing=0.1)`, chosen by 5-fold CV F1 on the training set |
| Support Vector Machine | RBF kernel, probabilities via`CalibratedClassifierCV`                       |

### Comparison of evaluation metrics (test set, n = 2,441)

Positive class = purchase. Precision, Recall and F1 refer to the purchase class; AUC uses predicted probabilitis.

| ML Model Name          | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.8869     | 0.8996     | 0.7524     | 0.4136     | 0.5338     | 0.5032     |
| Decision Tree          | **0.8992** | **0.9133** | 0.7086     | 0.6047     | **0.6525** | **0.5966** |
| KNN                    | 0.8763     | 0.8221     | 0.6869     | 0.3848     | 0.4933     | 0.4526     |
| Naive Bayes            | 0.8161     | 0.8418     | 0.4419     | **0.6675** | 0.5318     | 0.4371     |
| Support Vector Machine | 0.8939     | 0.8729     | **0.7450** | 0.4895     | 0.5908     | 0.5484     |

_Baseline: always predicting "no purchase" scores 0.8435 accuracy on the test set._

### Observations

| ML Model Name          | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Highest precision (0.75): when it predicts a purchase it is usually right, but it is too cautious, missing 59% of buyers (224 of 382; recall 0.41). Train and test accuracy are almost equal (0.883 vs 0.887), so it does not overfit. Its AUC (0.90) is second best; it ranks buyers well even though the default threshold is conservative.                                                                                                                                                                                                      |
| Decision Tree          | Best on Accuracy, AUC, F1 and MCC. It catches 60% of buyers (231 of 382) with a moderate 95 false alarms. Limiting depth to 6 keeps the train–test gap small (0.916 vs 0.899), and its threshold-based splits suit the skewed numeric features.                                                                                                                                                                                                                                                                                                    |
| KNN                    | Weakest on AUC (0.82) and recall (0.38, catching only 147 of 382 buyers); the train–test gap (0.892 vs 0.876) shows some overfitting.                                                                                                                                                                                                                                                                                                                                                                                                              |
| Naive Bayes            | Highst recall (0.67, catching 255 of 382 buyers), but more than half of its purchase predictions are false alarms (322; precision 0.44), which pulls accuracy below the 0.844 baseline. With default settings it collapsed to 0.275 accuracy because near-constant one-hot columns distorted the likelihoods;`var_smoothing=0.1` fixed this. Highly correlted features (e.g. BounceRates–ExitRates, ProductRelated-ProductRelated_Duration, etc.) also break its independence assumption. Useful if missing a buyer costs more than a false alarm. |
| Support Vector Machine | Second best on F1 (0.59) and MCC (0.55), with high precision (0.745). However, its AUC (0.87) is lower than Logistic Regression's, so it does well at the default threshold but ranks buyers less well overall.                                                                                                                                                                                                                                                                                                                                    |
| **Overall winner**     | **Decision Tree**: best on 4 of 6 metrics, including MCC (0.597), the most reliable single metric for this imbalaced data. Note that accuracy alone is misleading here: all models score 0.82–0.90 against a 0.84 baseline, while MCC separates them much more clearly (0.44–0.60).                                                                                                                                                                                                                                                                |

## Streamlit app

<Streamlit app URL>

Features: CSV upload (test data), model selection dropdown, all six evaluation metrics, confusion matrix and classification report, a predictions download, & a side-by-side comparison of all 5 models on the uploaded file. The app only loads the saved pipelines from `model/`; it never retrains on uploaded data.

> Upload only `test_data.csv` (or other unseen data). Uploading the full original dataset
> gives inflated scores, because ~80% of those rows were used for training.
