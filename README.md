# Customer Churn Prediction

MAT311 final project. The goal is to predict whether a subscription customer will churn (cancel), using their usage, support and billing history. The project compares a k-nearest neighbors (KNN) model with a decision tree, picks the better one by ROC AUC, and uses it to predict churn probabilities for an unlabeled test set.

## Data

The data lives in `data/raw/`:

- `churn_train.csv`: about 303,000 customers with a `Churn` label (1 = churned, 0 = stayed). Roughly 30% of customers churned.
- `churn_test.csv`: customers without labels. The pipeline predicts churn for these.

Each row has `CustomerID`, `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, `Last Interaction`, `Customer Status`, `Last Due Date` and `Last Payment Date`.

## Pipeline

`main.py` runs every step from start to finish:

1. Load and clean the training data (fill in missing values, drop unused and leaking columns).
2. Build model features (encode categories).
3. Plot exploratory charts.
4. Split the data into training, validation and test sets.
5. Train a KNN model and a decision tree.
6. Compare them on the validation set and check the better one on the test set.
7. Predict churn probabilities for `data/raw/churn_test.csv` and save them to `output/submission.csv`.

## Project layout

```
.
├── main.py                  # Runs the whole pipeline
├── requirements.txt         # Package versions used
├── data/
│   ├── processed/           # Cleaned data and splits, written by src/data modules
│   └── raw/
│       ├── churn_train.csv
│       └── churn_test.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb   # Exploration and prototyping
├── output/
│   └── submission.csv       # Created by main.py (not tracked by git)
└── src/
    ├── data/
    │   ├── load_data.py
    │   ├── preprocess.py    # Cleaning and imputing missing values
    │   └── split_data.py    # Train/validation/test split
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── train_model.py   # ROC curve helper
    │   ├── knn_model.py
    │   └── dt_model.py
    └── visualization/
        ├── eda.py
        └── performance.py
```

The notebook is only for exploration. `main.py` is the entry point that reproduces the results.

## Running it

Set up the environment (Python 3.11):

```bash
conda create -n data-science python=3.11
conda activate data-science
pip install -r requirements.txt
```

Run the pipeline from the project root:

```bash
python main.py
```

Plots open one at a time. Close each window to continue. The predictions are saved to `output/submission.csv`.

---

*🤖✨ README written with the help of Claude (Anthropic).*
