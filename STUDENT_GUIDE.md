# 📚 STUDENT GUIDE
# Stacked Churn Intelligence System — Learn by Building

> **Who is this for?** A student who knows basic Python and wants to understand
> how a real-world Machine Learning project is structured, why each design decision
> was made, and what each line of code actually does.

---

# 🗺️ HOW TO READ THIS GUIDE

We will walk through the project **in the exact order the pipeline runs** —
from reading a CSV file all the way to generating AI-powered retention strategies.

For every file we will cover:
- 🎯 **What problem does this file solve?**
- 📦 **What libraries/concepts are used?**
- 🔍 **Every function explained line-by-line**
- 💡 **Why was it done this way?**
- 🧪 **What would break if we removed it?**

---

# PART 0 — PROJECT OVERVIEW (Big Picture)

## What are we building?

A **telecom company** has ~7,000 customers. Some customers will **cancel their subscription** (called "churning").
It costs much more to acquire a new customer than to retain an existing one.

Our job: **Predict which customers are likely to leave**, rank them by financial risk, and generate **personalized strategies to keep them**.

## The 14 Steps (like a factory assembly line)

```
RAW CSV DATA
     │
     ▼
 STEP 1: Load & clean the data               ← data_loader.py
     │
     ▼
 STEP 2: Train/Test split                    ← sklearn
     │
     ▼
 STEP 3: Preprocess features                 ← preprocessor.py
     │
     ▼
 STEP 4: Generate OOF predictions            ← stacking.py
     │                (5-fold cross validation — avoids cheating)
     ▼
 STEP 5: Train meta-model                    ← stacking.py
     │
     ▼
 STEP 6: Retrain base models on full data    ← xgb_model.py + nn_model.py
     │
     ▼
 STEP 7: Generate test probabilities         ← predict_proba
     │
     ▼
 STEP 8: Evaluate all models                 ← evaluation.py
     │
     ▼
 STEP 9: Calibrate probabilities             ← calibration.py
     │
     ▼
 STEP 10: Risk band segmentation             ← risk_segmentation.py
     │
     ▼
 STEP 11: Business impact calculation        ← business_impact.py
     │
     ▼
 STEP 12: AI retention recommendations       ← retention_ai.py (Gemini API)
     │
     ▼
 STEP 13: Generate reports & plots           ← reporting.py
     │
     ▼
 STEP 14: Log run metadata                   ← train_pipeline.py
     │
     ▼
OUTPUTS: CSV + TXT + 3 PNGs + JSON log
```

---

# PART 1 — `config.py` (The Brain of Configuration)

## 🎯 Problem it solves
Imagine you have 10 Python files, and your learning rate is `0.05` in 8 of them.
If you want to change it to `0.1`, you'd need to edit 8 files — and likely miss one.

**Solution:** Put every setting in ONE place. All other files import from here.
This is called the **Single Source of Truth** principle.

## 📦 Libraries used
```python
from pathlib import Path   # cross-platform file paths (works on Windows AND Linux)
```

## 🔍 Full Explanation

```python
# config.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
```
> - `__file__` → the full path of config.py itself
> - `.resolve()` → converts to absolute path (e.g., `C:/Users/ABEL/.../src/config.py`)
> - `.parent` → goes UP one folder → `src/`
> - `.parent` again → goes UP again → `telco-customer-churn-prediction/`
> - So `PROJECT_ROOT` = the top-level project folder ✅

```python
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR   = PROJECT_ROOT / "logs"
```
> The `/` operator on `Path` objects **joins paths** — like `os.path.join()` but cleaner.
> `PROJECT_ROOT / "data"` = `C:/Users/ABEL/.../telco-customer-churn-prediction/data`

```python
MODELS_DIR.mkdir(parents=True, exist_ok=True)
```
> Creates the `models/` folder if it doesn't exist.
> - `parents=True` → also creates any missing parent folders
> - `exist_ok=True` → doesn't crash if the folder already exists

```python
RANDOM_STATE = 42
```
> ML algorithms use **random numbers** internally (e.g., "randomly pick 80% for training").
> Setting a fixed seed makes results **reproducible** — run it again, get same answer.
> The number 42 is a convention (from "The Hitchhiker's Guide to the Galaxy"). Any number works.

```python
XGB_PARAMS = {
    "max_depth":        4,       # How deep each decision tree can grow
    "learning_rate":    0.05,    # How fast the model learns (smaller = slower but better)
    "n_estimators":     300,     # Number of trees to build
    "subsample":        0.8,     # Use 80% of rows per tree (prevents overfitting)
    "colsample_bytree": 0.8,     # Use 80% of columns per tree (prevents overfitting)
    "eval_metric":      "auc",   # What to look at during early stopping
    "tree_method":      "hist",  # Faster training algorithm (like a histogram shortcut)
}
```

```python
NN_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),   # 3 hidden layers with these sizes
    "activation":         "relu",            # Non-linearity function (max(0, x))
    "solver":             "adam",            # Optimizer (adaptive momentum)
    "alpha":              0.001,             # L2 regularization strength
    "learning_rate":      "adaptive",        # Reduce LR if loss stops improving
    "max_iter":           500,               # Maximum training epochs
    "early_stopping":     True,              # Stop if validation loss doesn't improve
    "validation_fraction": 0.1,             # 10% of training data used for validation
    "n_iter_no_change":   20,               # Stop after 20 epochs of no improvement
}
```

```python
RISK_BANDS = {
    "Low":      (0.0, 0.3),   # 0% to 30% churn probability
    "Medium":   (0.3, 0.6),   # 30% to 60%
    "High":     (0.6, 0.8),   # 60% to 80%
    "Critical": (0.8, 1.01),  # 80% to 100% (1.01 to include exactly 1.0)
}
```

```python
# Business impact formula constants
ESTIMATED_CONTRACT_MONTHS = 24    # Assume customers are on 2-year contracts
FALLBACK_MONTHLY_CHARGES  = 65.0  # Use $65 if the actual charge is missing
FALLBACK_TENURE           = 12    # Assume 12 months if tenure is missing

# Gemini API settings
GEMINI_MODEL      = "gemini-2.5-flash"   # Which Google AI model to use
GEMINI_BATCH_SIZE = 5                     # Process 5 customers at a time
GEMINI_RETRY_MAX  = 3                     # Try 3 times before giving up
GEMINI_RETRY_WAIT = 5                     # Wait 5 seconds between retries
```

---

# PART 2 — `data_loader.py` (Reading and Cleaning Data)

## 🎯 Problem it solves
Real-world data is MESSY. The Telco dataset has:
- The `Churn` column stored as text `"Yes"/"No"` but we need `1/0`
- `TotalCharges` stored as a **string** (because some rows have spaces instead of numbers)
- A `customerID` column that's just noise for our model

## 📦 Libraries used
```python
import pandas as pd
from config import ID_COLUMNS, RANDOM_STATE, TARGET_COLUMN
```

## 🔍 Function 1: `normalize_target(series)`

```python
def normalize_target(series: pd.Series) -> pd.Series:
```
> **What it does:** Converts `"Yes"/"No"/"True"/"1"` etc. → integer `1` or `0`

```python
    text = series.astype(str).str.strip().str.lower()
```
> - `.astype(str)` → force everything to string (even if someone put an integer)
> - `.str.strip()` → remove spaces: `" Yes "` → `"Yes"`
> - `.str.lower()` → make lowercase: `"YES"` → `"yes"`

```python
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1,
        "no":  0, "n": 0, "false": 0, "0": 0, "stay":  0,
    }
    mapped = text.map(mapping)
```
> `.map(dict)` replaces each value using the dictionary.
> e.g., `"yes"` → `1`, `"no"` → `0`
> Anything NOT in the mapping becomes `NaN` (not-a-number / missing).

```python
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Unrecognized target labels: {bad}. "
            "Please map your target column to 0/1 manually."
        )
    return mapped.astype(int)
```
> If ANY value couldn't be mapped, we stop and tell the user exactly which values were bad.
> This is called **defensive programming** — fail loudly and clearly rather than silently produce wrong results.

---

## 🔍 Function 2: `clean_features(X)` — THE MOST IMPORTANT FUNCTION

```python
def clean_features(X: pd.DataFrame) -> pd.DataFrame:
```
> **Why this exists:** In the Telco dataset, `TotalCharges` is stored as a STRING column.
> When pandas reads it from CSV, it sees values like `"1889.5"` but also `""` (blank).
> Because of the blanks, pandas keeps the whole column as strings instead of converting to float.
> We need to detect this and fix it.

```python
    X = X.copy()   # Never modify the original! Always work on a copy.
```
> This is a Python best practice — **immutability of inputs** prevents confusing bugs.

```python
    for col in X.columns:
        if not pd.api.types.is_string_dtype(X[col]):
            continue   # Skip columns that aren't strings (already numeric)
```
> We only need to check string/object columns.

```python
        text = X[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
```
> - Convert to string, strip whitespace
> - Replace empty strings `""` and `"nan"` with `pd.NA` (proper missing value marker)

```python
        numeric = pd.to_numeric(text, errors="coerce")
```
> Try to convert every value to a number.
> `errors="coerce"` → if it can't convert, put `NaN` instead of crashing.

```python
        non_null = int(text.notna().sum())
        parse_ratio = float(numeric.notna().sum() / non_null) if non_null else 0.0
```
> **parse_ratio** = "What fraction of the non-missing values actually parsed as numbers?"
> e.g., if 100 values and 95 became numbers → ratio = 0.95

```python
        if parse_ratio >= 0.9:
            X[col] = numeric   # 90%+ numeric → convert the whole column
        else:
            X[col] = text      # Otherwise keep as cleaned string
    return X
```
> **The 90% rule:** We say "if 90% or more of a column looks like numbers, treat the whole column as numeric."
> `TotalCharges` has ratio ≈ 0.998 → it becomes float. ✅
> `Contract` has ratio ≈ 0.0 → it stays string (categorical). ✅

---

## 🔍 Function 3: `load_data(csv_path, target_col)`

```python
def load_data(csv_path, target_col="Churn"):
    df = pd.read_csv(csv_path)
```
> Read the CSV into a pandas DataFrame. Each row = one customer. Each column = one feature.

```python
    if target_col not in df.columns:
        raise ValueError(...)
```
> Tell the user if the target column name is wrong — better than a cryptic KeyError.

```python
    y = normalize_target(df[target_col])   # Labels (what we want to predict)
    X = df.drop(columns=[target_col]).copy() # Features (everything else)
```
> In ML:
> - `X` = **Features** (inputs to the model)
> - `y` = **Labels/Target** (what we're trying to predict)

```python
    to_drop = [c for c in X.columns if c in ID_COLUMNS]
    X = X.drop(columns=to_drop)
```
> Drop `customerID` — it's just a unique identifier, not useful for prediction.
> A model that "learned" customerID patterns would not generalize to new customers.

```python
    X = clean_features(X)               # Fix TotalCharges and others

    numeric_cols     = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
```
> After cleaning, let pandas tell us which columns are numbers and which are categories.
> `tenure`, `MonthlyCharges`, `TotalCharges` → numeric
> `gender`, `Contract`, `InternetService` etc. → categorical

```python
    return X, y, numeric_cols, categorical_cols
```
> Returns 4 things: features, labels, and the two column lists that the preprocessor needs.

---

# PART 3 — `preprocessor.py` (Making Data Machine-Readable)

## 🎯 Problem it solves
ML models only understand **numbers**. They can't understand `"Female"` or `"Month-to-month"`.
Also, numeric features like `tenure` (range: 0–72) and `MonthlyCharges` (range: 18–118) are on completely different scales — this confuses distance-based models.

**Solution:** A preprocessing pipeline that:
1. Fills in missing values (imputation)
2. Scales numbers to the same range
3. Converts categories to numbers (one-hot encoding)

## 📦 Libraries used
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

## 🔍 Function 1: `build_preprocessor(numeric_cols, categorical_cols)`

```python
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
```
> **Pipeline** chains steps in sequence: output of step 1 feeds into step 2.
>
> - **SimpleImputer(strategy="median")**: Fills missing numbers with the median.
>   Why median? It's less sensitive to extreme outliers than the mean.
>   e.g., TotalCharges missing → filled with the median TotalCharges of all customers.
>
> - **StandardScaler()**: Transforms each number to have mean=0 and std=1.
>   Formula: `z = (x - mean) / std`
>   Why? Neural networks and Logistic Regression converge much faster with scaled inputs.

```python
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
```
> - **SimpleImputer(strategy="most_frequent")**: Fills missing categories with the most common value.
>   e.g., if `PaymentMethod` is missing → fill with `"Electronic check"` (most common).
>
> - **OneHotEncoder**: Converts categories to binary columns.
>   `Contract = "Month-to-month"` becomes:
>   ```
>   Contract_Month-to-month = 1
>   Contract_One year       = 0
>   Contract_Two year       = 0
>   ```
>   `handle_unknown="ignore"` → if a category appears at prediction time that wasn't in training, set all its one-hot columns to 0 (don't crash).
>   `sparse_output=False` → return a regular numpy array (not a sparse matrix) — needed for our downstream code.

```python
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",   # Ignore any other columns not listed
    )
    return preprocessor
```
> **ColumnTransformer** applies DIFFERENT transformations to DIFFERENT columns.
> It's like having two assembly lines running in parallel — one for numbers, one for categories.

## 🔍 Function 2: `fit_preprocessor(preprocessor, X_train, save)`
```python
    preprocessor.fit(X_train)   # Learn statistics from TRAINING data only
    joblib.dump(preprocessor, MODELS_DIR / "preprocessing_pipeline.joblib")
```
> **CRITICAL CONCEPT: Fit on training data only!**
> The `StandardScaler` needs to compute mean and std. If we include test data, our model
> "peeks" at test data during training — this is called **data leakage** and makes our
> evaluation metrics unrealistically optimistic.
>
> We save the fitted preprocessor so we can apply the SAME transformation to new data later
> without refitting (which would use different statistics).

## 🔍 Function 3: `transform(preprocessor, X)`
```python
    return preprocessor.transform(X)
```
> Apply the already-fitted preprocessor. Simple — just apply the learned transformations.

---

# PART 4 — `xgb_model.py` (XGBoost: The Gradient Boosting Champion)

## 🎯 What is XGBoost?

Think of XGBoost as building many decision trees, where **each new tree corrects the mistakes of all previous trees**.

- **Tree 1** makes predictions. Some are wrong.
- **Tree 2** focuses on the wrong predictions and tries to fix them.
- **Tree 3** fixes Tree 2's remaining mistakes.
- ... repeat for 300 trees.

This technique is called **gradient boosting**. XGBoost is an extremely fast and powerful implementation.

## 🔍 Function 1: `build_xgb(scale_pos_weight)`

```python
def build_xgb(scale_pos_weight: float = 1.0) -> XGBClassifier:
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    return XGBClassifier(**params)
```
> `{**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}` — merges the default params
> dictionary with the new `scale_pos_weight` value.
>
> **What is `scale_pos_weight`?**
> In the Telco dataset, about 26% of customers churn, 74% don't.
> This **class imbalance** means if the model just predicts "no churn" for everyone,
> it gets 74% accuracy — but is completely USELESS.
>
> `scale_pos_weight = neg_count / pos_count = 74 / 26 ≈ 2.85`
> This tells XGBoost: "Treat each churning customer as 2.85× more important than a non-churning one."
> This balances the training signal.

## 🔍 Function 2: `train_xgb(X_train, y_train, X_val, y_val, save)`

```python
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)   # max(pos, 1) prevents division by zero
    model = build_xgb(scale_pos_weight=spw)
```
> Count negative (no churn) and positive (churn) examples, compute the weight ratio.

```python
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
```
> **Early Stopping:** Instead of always training for exactly 300 trees,
> we monitor performance on the validation set.
> If AUC on the validation set doesn't improve for 50 consecutive rounds,
> we STOP early — preventing overfitting and saving time.

```python
    joblib.dump(model, MODELS_DIR / "xgb_model.joblib")
```
> Save the trained model to disk. Next time, load it with `joblib.load()` instead of
> retraining (which could take minutes).

## 🔍 Function 3: `predict_proba_xgb(model, X)`
```python
    return model.predict_proba(X)[:, 1]
```
> `predict_proba` returns `[[P(no churn), P(churn)], ...]`
> `[:, 1]` means: take the SECOND column (index 1) = probability of churn.
> So the output is a 1D array of churn probabilities: `[0.23, 0.87, 0.41, ...]`

---

# PART 5 — `nn_model.py` (Neural Network: The Deep Learning Approach)

## 🎯 What is an MLP Neural Network?

A network of connected "neuron" layers that learns non-linear patterns.

```
Input Layer (N features)
      │
Dense Layer 256 neurons  [relu activation]
      │
Dense Layer 128 neurons  [relu activation]
      │
Dense Layer  64 neurons  [relu activation]
      │
Output Layer  1 neuron   [sigmoid activation → P(churn)]
```

**ReLU activation:** `f(x) = max(0, x)` — turns negatives to zero, keeps positives.
This allows the network to learn non-linear relationships.

## 🔍 Function 1: `build_nn()`
```python
    return MLPClassifier(**NN_PARAMS)
```
> Creates the MLP with all settings from `config.py`. Not trained yet.

## 🔍 Function 2: `train_nn(X_train, y_train, save)`
```python
    model = build_nn()
    t0 = time.time()
    model.fit(X_train, y_train)    # Train the network
    elapsed = time.time() - t0
```
> `time.time()` captures the current timestamp in seconds.
> `elapsed = t0 - t0_after_training` → how many seconds training took.

```python
    print(f"[nn_model] Trained in {elapsed:.1f}s | n_iter={model.n_iter_} | loss={model.loss_:.4f}")
```
> `model.n_iter_` → how many epochs (passes over data) the model actually trained for.
> This is ≤ `max_iter` because early stopping may have kicked in.
> `model.loss_` → the final training loss value.

## 🔍 Function 3: `predict_proba_nn(model, X)`
```python
    return model.predict_proba(X)[:, 1]
```
> Same pattern as XGBoost — take the second column (P(churn)).

---

# PART 6 — `stacking.py` (The Heart of the Ensemble)

## 🎯 What is Stacking and why does it avoid data leakage?

**The Problem with Naive Stacking:**
Train XGBoost → get probabilities → train Neural Network → get probabilities →
train Logistic Regression on `[P_xgb, P_nn]`.

What's wrong? The base models were trained ON the same data they're predicting!
They've already "seen" those answers — this is **data leakage**.
The meta-model would learn that P_xgb=0.9 means churn, but ONLY because XGBoost
over-fitted to its own training examples.

**The Solution: Out-of-Fold (OOF) Predictions**
Only generate predictions for data the model has NEVER seen.

```
TRAINING DATA (80% of total)
Split into 5 equal folds:  [F1][F2][F3][F4][F5]

Round 1: Train on [F2,F3,F4,F5] → predict on [F1] → oof_xgb[F1], oof_nn[F1]
Round 2: Train on [F1,F3,F4,F5] → predict on [F2] → oof_xgb[F2], oof_nn[F2]
Round 3: Train on [F1,F2,F4,F5] → predict on [F3] → oof_xgb[F3], oof_nn[F3]
Round 4: Train on [F1,F2,F3,F5] → predict on [F4] → oof_xgb[F4], oof_nn[F4]
Round 5: Train on [F1,F2,F3,F4] → predict on [F5] → oof_xgb[F5], oof_nn[F5]

Result: oof_xgb = array of n_train predictions, ALL from held-out folds
        oof_nn  = same

Meta-model trains on:  [oof_xgb, oof_nn]  →  y_train
✅ NO DATA LEAKAGE!
```

## 🔍 Function 1: `generate_oof_predictions(X_train, y_train, neg_pos_ratio)`

```python
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
```
> **StratifiedKFold** ensures each fold has approximately the same ratio of churners to
> non-churners as the full dataset. Without stratification, one fold might accidentally
> have no churners at all, making training unstable.

```python
    oof_xgb = np.zeros(len(y_train))
    oof_nn  = np.zeros(len(y_train))
```
> Initialize arrays of zeros with the same length as the training set.
> We'll fill these in fold by fold.

```python
    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_trn, X_val = X_train[trn_idx], X_train[val_idx]
        y_trn, y_val = y_train[trn_idx], y_train[val_idx]
```
> `skf.split(X, y)` yields pairs of `(train_indices, val_indices)` — 5 times.
> We use these indices to slice our data into fold-specific train/validation sets.

```python
        # XGBoost for this fold
        neg, pos = int((y_trn == 0).sum()), int((y_trn == 1).sum())
        spw = neg / max(pos, 1)
        xgb = build_xgb(scale_pos_weight=spw)
        xgb.set_params(early_stopping_rounds=50)
        xgb.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = predict_proba_xgb(xgb, X_val)
```
> Recompute scale_pos_weight for THIS fold's class distribution.
> Store predictions at positions `val_idx` in the oof array.
> This is the key: we ONLY predict on data the model has NOT seen.

```python
        # Neural Network for this fold
        nn = build_nn()
        nn.fit(X_trn, y_trn)
        oof_nn[val_idx] = predict_proba_nn(nn, X_val)
```
> Same process for the Neural Network.

```python
    return oof_xgb, oof_nn
```
> Return two 1D arrays, each length n_train, containing leak-free predictions.

---

## 🔍 Function 2: `train_meta_model(oof_xgb, oof_nn, y_train, save)`

```python
    Stacked_X = np.column_stack([oof_xgb, oof_nn])
```
> `np.column_stack` combines two 1D arrays into a 2D array.
> Shape: `(n_train, 2)` — each row is `[P_xgb_for_customer_i, P_nn_for_customer_i]`
>
> Example:
> ```
> Stacked_X = [[0.82, 0.76],   ← customer 1: both models say high churn risk
>              [0.12, 0.18],   ← customer 2: both models say low risk
>              [0.71, 0.29],   ← customer 3: disagreement! XGBoost says high, NN says low
>              ...]
> ```
> The meta-model learns from this: when they agree, be confident. When they disagree, learn the
> correct weighting. This is the POWER of stacking.

```python
    meta = LogisticRegression(**META_LR_PARAMS)
    meta.fit(Stacked_X, y_train)
```
> **LogisticRegression** as meta-model: takes `[P_xgb, P_nn]` and learns the optimal
> linear combination to produce the final churn probability.
> It's intentionally simple — we don't want the meta-model to overfit.

---

## 🔍 Function 3: `stack_predict(meta, p_xgb, p_nn)`

```python
    Stacked_X = np.column_stack([p_xgb, p_nn])
    return meta.predict_proba(Stacked_X)[:, 1]
```
> At inference time, combine base model probabilities, run through meta-model → final P(churn).

---

# PART 7 — `evaluation.py` (How Good Is Our Model?)

## 🎯 Why do we need multiple metrics?

**Accuracy** alone is misleading with imbalanced data.
If 74% of customers don't churn and we predict "no churn" for everyone:
- Accuracy = 74% (looks OK!)
- Recall of churn class = 0% (we catch ZERO churners — completely useless!)

We use multiple metrics to get the full picture.

## 📦 Key Metrics Explained

| Metric | Formula | Meaning (in churn context) |
|---|---|---|
| **ROC-AUC** | Area under the ROC curve | How well does the model RANK churners above non-churners? 0.5=random, 1.0=perfect |
| **Precision** | TP / (TP + FP) | Of all customers we predicted would churn, what % actually did? |
| **Recall** | TP / (TP + FN) | Of all customers who actually churned, what % did we catch? |
| **F1-Score** | 2 × (P × R) / (P + R) | Balanced combination of Precision and Recall |
| **Brier Score** | mean((prob - actual)²) | How accurate are the probabilities themselves? Lower = better. 0 = perfect |

## 🔍 Function 1: `evaluate_model(y_true, y_prob, model_name, threshold=0.5)`

```python
    y_pred = (y_prob >= threshold).astype(int)
```
> Convert probabilities to binary predictions.
> If churn probability ≥ 0.5 → predict churn (1), else predict no churn (0).
> The threshold can be adjusted: lower threshold → catch more churners (higher recall, lower precision).

```python
    metrics = {
        "roc_auc":     roc_auc_score(y_true, y_prob),    # Uses raw probabilities
        "precision":   precision_score(y_true, y_pred),   # Uses binary predictions
        "recall":      recall_score(y_true, y_pred),
        "f1":          f1_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),  # Uses raw probabilities
    }
```

## 🔍 Function 2: `compare_models(y_true, prob_xgb, prob_nn, prob_stack)`

```python
    results = [
        evaluate_model(y_true, prob_xgb,   "XGBoost"),
        evaluate_model(y_true, prob_nn,     "Neural Network"),
        evaluate_model(y_true, prob_stack,  "Stacked Ensemble"),
    ]
    df = pd.DataFrame(results).set_index("model")
    df = df.sort_values("roc_auc", ascending=False)
```
> Creates a comparison table sorted by ROC-AUC.
> Expected outcome: Stacked Ensemble > XGBoost ≈ Neural Network

---

# PART 7B — Neural Network vs XGBoost: Data Comparison

## 🎯 Why compare them?

Both models try to answer the same question: *"Will this customer churn?"*
But they approach the problem completely differently.
Comparing their outputs reveals **where each model excels** and **where it fails** —
which is exactly why we stack them together.

---

## 📊 Sample Evaluation Output (Typical Telco Dataset Results)

This is the kind of table `compare_models()` prints to your terminal after training:

```
=================================================================
 MODEL EVALUATION REPORT
=================================================================
                  roc_auc  precision  recall      f1  brier_score
model
Stacked Ensemble   0.8521     0.6834  0.5912  0.6340       0.1287
XGBoost            0.8418     0.7012  0.5561  0.6204       0.1334
Neural Network     0.8276     0.6401  0.5743  0.6054       0.1412
=================================================================

[Best] Best model by ROC-AUC: Stacked Ensemble  (AUC=0.8521)
```

> **Note:** These are representative values. Your actual numbers will vary slightly
> depending on the random seed, fold split, and data version. But the **ordering**
> (Stacked > XGBoost ≈ NN) is consistent.

---

## 📦 Model-by-Model Breakdown

### XGBoost 🌲 (Gradient Boosted Trees)

| Metric | Typical Value | Interpretation |
|---|---|---|
| ROC-AUC | ~0.84 | Strong ranking ability |
| Precision | ~0.70 | 70% of "predicted churners" actually churn |
| Recall | ~0.56 | Catches 56% of actual churners |
| F1-Score | ~0.62 | Balanced but conservative |
| Brier Score | ~0.133 | Probabilities are fairly calibrated |

**Strengths:**
- Handles **missing values** and **mixed data types** natively
- Works very well with **tabular, structured data** (like our Telco CSV)
- `scale_pos_weight` corrects for class imbalance effectively
- Early stopping prevents overfitting

**Weaknesses:**
- Can be overconfident — produces probabilities very close to 0 or 1
- Struggles to capture **complex interaction patterns** the way neural networks can
- Less flexible — each tree adds a small fixed correction

---

### Neural Network 🧠 (MLP — Multi-Layer Perceptron)

| Metric | Typical Value | Interpretation |
|---|---|---|
| ROC-AUC | ~0.83 | Slightly lower than XGBoost |
| Precision | ~0.64 | More false positives than XGBoost |
| Recall | ~0.57 | Catches slightly more churners |
| F1-Score | ~0.61 | Similar overall balance |
| Brier Score | ~0.141 | Probabilities less calibrated (softer) |

**Strengths:**
- Learns **non-linear, deep interaction patterns** across many features simultaneously
- `relu` activations allow flexible decision boundaries
- Good at finding patterns XGBoost misses (e.g., multi-feature interactions)
- Can generalize well when data volume is higher

**Weaknesses:**
- Requires data to be **scaled** (StandardScaler) — raw values break it
- More sensitive to **hyperparameters** (layer sizes, learning rate, epochs)
- Slower to train on CPU than XGBoost
- Probabilities tend to cluster in a narrower range (less extreme than XGBoost)

---

## 📦 Probability Distribution Comparison

Here's a conceptual view of how the two models distribute their predicted probabilities
across all 7,043 customers:

```
Probability  XGBoost Output         Neural Network Output
─────────────────────────────────────────────────────────────
0.00–0.10    ████████████████████  ██████████████
             (many very confident non-churners)
0.10–0.30    ██████████            █████████████
0.30–0.50    █████                 █████████████  ← NN piles
0.50–0.70    ████████              █████████████    up here
0.70–0.90    ████████████          █████████
0.90–1.00    ██████████████        ████
             (many very confident churners)
```

**Key insight:**
- **XGBoost** produces a **bimodal distribution** — probabilities cluster near 0 and near 1.
  It's decisive: "This person WILL churn" or "This person WON'T."
- **Neural Network** produces a **softer, more uniform distribution** — probabilities
  spread more evenly across the range, especially in the middle (0.3–0.7).
  It's uncertain more often: "This person MIGHT churn."

This is why calibration (Part 8) is applied AFTER stacking — to correct systematic
biases in the raw probability outputs.

---

## 📦 Per-Customer Prediction Comparison

Here's what the output looks like for 5 sample customers after running both models:

```
 CustomerID  | Actual | P(XGB) | P(NN) | P(Stack) | Band
─────────────|--------|--------|-------|----------|---------
 cust_0042   |  1     |  0.91  | 0.78  |   0.87   | Critical
 cust_1187   |  1     |  0.54  | 0.62  |   0.59   | Medium
 cust_2301   |  0     |  0.08  | 0.21  |   0.13   | Low
 cust_3904   |  1     |  0.73  | 0.48  |   0.63   | High    ← disagreement!
 cust_5512   |  0     |  0.38  | 0.44  |   0.41   | Medium
```

**Reading the disagreement (cust_3904):**
- XGBoost says **73%** churn probability → High risk
- Neural Network says **48%** → Medium risk
- They DISAGREE. The meta-model was trained to handle exactly this case.
- It learned: "When XGBoost says high and NN is unsure, trust XGBoost more
  for customers with this profile" → final stacked = **63%** (High risk)
- **Actual: this customer churned (1)** → XGBoost and Stacking were correct

This kind of disagreement is not a bug — it's the **value of stacking**:
each model sees the data differently, and the meta-model learns whose opinion to trust.

---

## 📦 Summary Table: When to Trust Each Model

| Situation | Who's usually right |
|---|---|
| Customer has `Contract=Month-to-month` + high charges | XGBoost (tree rules fire cleanly) |
| Customer has subtle interaction of 4+ features | Neural Network (deep patterns) |
| They agree on high probability | Very confident prediction |
| They disagree | Meta-model decides based on the pattern it learned in training |
| They both say ~0.5 (uncertain) | Customer is genuinely on the boundary — lowest confidence prediction |

---

## 🔍 Code that produces this comparison in `evaluation.py`

```python
def compare_models(y_true, prob_xgb, prob_nn, prob_stack):
    results = [
        evaluate_model(y_true, prob_xgb,   "XGBoost"),
        evaluate_model(y_true, prob_nn,     "Neural Network"),
        evaluate_model(y_true, prob_stack,  "Stacked Ensemble"),
    ]

    df = pd.DataFrame(results).set_index("model")
    df = df.sort_values("roc_auc", ascending=False)

    print(df[float_cols].to_string(float_format=lambda x: f"{x:.4f}"))

    best = df.index[0]
    print(f"[Best] Best model by ROC-AUC: {best}  (AUC={df.loc[best,'roc_auc']:.4f})")
    return df
```

> `df.index[0]` = the model name at row 0 after sorting descending by ROC-AUC.
> This will almost always be `"Stacked Ensemble"` — the goal of the whole system.

---

# PART 8 — `calibration.py` (Making Probabilities Trustworthy)

## 🎯 Why do we need calibration?

A model might output `P(churn) = 0.8` for a customer.
But if you look at ALL customers where the model said 0.8, maybe only 60% of them
actually churned. The model is **overconfident** — 0.8 should really be 0.6.

Calibration corrects this systematic bias so that when we say 80% churn probability,
we really mean 80%.

This matters because business decisions (how much to spend on retention) depend on
the probability being accurate, not just the ranking.

## 🔍 `calibrate_probabilities(X_cal, y_cal, prob_fn, method)`

```python
    raw_probs = prob_fn(X_cal)   # Get the uncalibrated probabilities
```
> `prob_fn` is the stacked ensemble's probability function.

**Method 1 — Isotonic Regression (Non-parametric):**
```python
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probs, y_cal)

        def calibrated_fn(X):
            p = prob_fn(X)
            return np.array(calibrator.predict(p)).clip(0.0, 1.0)
```
> Fits a step function that maps raw probabilities → calibrated probabilities.
> Non-parametric means it makes no assumptions about the shape of the mapping.
> `out_of_bounds="clip"` → values outside training range are clipped to [0,1].

**Method 2 — Platt Scaling (Logistic, Parametric):**
```python
    else:
        calibrator = LogisticRegression(C=1.0, max_iter=1000)
        calibrator.fit(raw_probs.reshape(-1, 1), y_cal)
```
> Fits a logistic sigmoid curve through the calibration points.
> More controlled but assumes a specific S-shaped relationship.

```python
    calibrated_fn._calibrator = calibrator  # Store calibrator on the function
    return calibrated_fn
```
> Returns a **closure** — a function that remembers the calibrator.
> When called with new data, it applies the calibrator automatically.

---

# PART 9 — `risk_segmentation.py` (Turning Probabilities into Actions)

## 🎯 Why segment into bands?

A probability of `0.73` is useful for data scientists but not for business teams.
They need clear, actionable categories:
- "Critical risk" → urgent personal call + 30% discount
- "High risk" → targeted email campaign + loyalty points
- "Medium risk" → standard newsletter + minor offer
- "Low risk" → no action needed

## 🔍 Function 1: `assign_risk_band(churn_probs)`

```python
    bands = np.empty(len(churn_probs), dtype=object)
    for band_name, (lo, hi) in RISK_BANDS.items():
        mask = (churn_probs >= lo) & (churn_probs < hi)
        bands[mask] = band_name
```
> - `np.empty(...)` creates an uninitialized array of Python objects (strings).
> - For each band, we create a boolean mask (True/False for each customer).
> - We assign the band name to all customers where the mask is True.
> - Using `&` (bitwise AND) for element-wise boolean operations on numpy arrays.

```python
    bands[churn_probs >= 1.0] = "Critical"   # Edge case: exactly 1.0
```
> RISK_BANDS uses `(0.8, 1.01)` to include 1.0, but just to be safe.

## 🔍 Function 2: `add_risk_band(df, prob_col)`

```python
    df["churn_band"] = assign_risk_band(df[prob_col].values)
```
> Adds a new column to the DataFrame.
> `.values` converts the pandas Series to a numpy array (faster for numpy operations).

---

# PART 10 — `business_impact.py` (Connecting ML to Money)

## 🎯 The Business Formula

```
Expected Revenue Loss = P(churn) × MonthlyCharges × Remaining Contract Months
Remaining Months = max(0, 24 - tenure)
```

**Example:**
- Customer A: P(churn) = 0.85, pays $90/month, been with us 6 months
- Remaining = max(0, 24 - 6) = 18 months
- Expected Loss = 0.85 × $90 × 18 = **$1,377**

This turns a probability into a dollar figure — something executives understand and can assign budget to.

## 🔍 `compute_business_impact(df)`

```python
    if MONTHLY_CHARGES_COL in df.columns:
        monthly = pd.to_numeric(df[MONTHLY_CHARGES_COL], errors="coerce").fillna(FALLBACK_MONTHLY_CHARGES)
    else:
        monthly = pd.Series(FALLBACK_MONTHLY_CHARGES, index=df.index)
```
> `.fillna(65.0)` → if MonthlyCharges is missing for some customers, use $65 (our configured fallback).
> `pd.Series(65.0, index=df.index)` → create a series of 65.0 for every row if the column doesn't exist at all.

```python
    remaining = (ESTIMATED_CONTRACT_MONTHS - tenure).clip(lower=0)
```
> `.clip(lower=0)` → if tenure > 24 months, remaining would be negative → clip to 0.
> (A customer who's stayed 30 months on a "24-month contract" still has $0 remaining contract risk.)

```python
    df["expected_revenue_loss"] = (
        df["churn_probability"] * monthly * remaining
    ).round(2)
    df = df.sort_values("expected_revenue_loss", ascending=False)
```
> Multiply element-wise (each customer's own values).
> Sort descending → most at-risk customers at the top (prioritization for retention team).

---

# PART 11 — `retention_ai.py` (AI-Powered Personalization)

## 🎯 Why use Gemini AI here?

Traditional retention campaigns send the same email to everyone.
Our system sends a UNIQUE strategy per customer based on their specific profile.

A customer who's been with us 2 years on a 2-year contract with multiple services
needs a completely different retention offer than a new customer on a month-to-month
plan with only basic phone service.

## 🔍 The Prompt Template

```python
_PROMPT_TEMPLATE = """
You are an expert customer retention analyst for a telecom company.
Given the following customer profile and churn risk information, provide a comprehensive retention strategy.

Customer Profile:
{profile_json}

Churn Probability: {churn_probability:.2%}
Risk Band: {risk_band}

Respond ONLY with a valid JSON object with these exact keys:
{
  "likely_churn_reason": "...",
  "risk_summary": "...",
  "retention_action": "...",
  "offer_recommendation": "...",
  "communication_tone": "..."
}
""".strip()
```
> We use a **structured prompt** that forces the AI to return JSON — not free-form text.
> This makes parsing reliable and allows us to store each field separately.

## 🔍 Function 1: `_build_payload(row)`

```python
    profile_fields = ["tenure", "MonthlyCharges", "TotalCharges", ...]
    profile = {}
    for field in profile_fields:
        for col in row.index:
            if col.lower().replace("_", "") == field.lower().replace("_", ""):
                profile[field] = row[col]
                break
```
> **Flexible column matching:** We normalize both sides (lowercase, remove underscores)
> so `"Monthly_charges"` matches `"MonthlyCharges"`. This makes the function work
> with datasets that have slightly different column naming conventions.

## 🔍 Function 2: `_call_gemini(client, prompt)` — Retry Logic

```python
    for attempt in range(1, GEMINI_RETRY_MAX + 1):  # Try up to 3 times
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                wait = GEMINI_RETRY_WAIT * attempt   # Exponential backoff: 5s, 10s, 15s
                time.sleep(wait)
            else:
                if attempt == GEMINI_RETRY_MAX:
                    raise   # Give up after 3 non-rate-limit errors
                time.sleep(GEMINI_RETRY_WAIT)
```
> **Rate limiting** is a real-world engineering problem. Free-tier APIs allow only
> a certain number of requests per minute. When we exceed that, we get a `429` error.
> The fix: detect the error, wait, and retry. This makes our code robust in production.

## 🔍 Function 3: `generate_retention_recommendations(df)`

```python
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        # Gracefully skip
        df.loc[df["churn_band"].isin(["High", "Critical"]),
               "retention_recommendation"] = "N/A (no API key configured)"
        return df
```
> **Graceful degradation** — if the API key isn't configured, the pipeline
> still runs successfully. It just skips the AI step and marks recommendations as N/A.
> This is important: the core ML pipeline should work even without external services.

```python
    mask = df["churn_band"].isin(["High", "Critical"])
    high_risk_df = df[mask]
```
> Only process High + Critical customers — no point spending API credits on
> someone with 5% churn probability.

```python
    for batch_start in tqdm(range(0, n_customers, GEMINI_BATCH_SIZE), desc="Gemini batches"):
        batch_indices = indices[batch_start : batch_start + GEMINI_BATCH_SIZE]
        for idx in batch_indices:
            # ... call Gemini for each customer
        time.sleep(GEMINI_RATE_LIMIT_SLEEP)  # 1 second between batches
```
> **Batch processing** processes customers in groups of 5, with a pause between
> batches to avoid hitting rate limits. `tqdm` shows a progress bar.

---

# PART 12 — `reporting.py` (Making Results Visible)

## 🎯 What outputs are generated?

1. **CSV**: Machine-readable, for data teams or CRM imports
2. **TXT Summary**: Human-readable, for business reports
3. **PNG Plots**: Visual, for presentations and dashboards
4. **SHAP Plot**: Explainability, for understanding model decisions

## 🔍 Key Functions

### `save_predictions(df)`
```python
    cols_first = ["churn_probability", "churn_band", "expected_revenue_loss", "retention_recommendation"]
    other_cols = [c for c in df.columns if c not in cols_first]
    ordered = cols_first + other_cols
```
> **Column order matters** for business users. Put the most important fields FIRST
> so they're visible without scrolling right in Excel/Google Sheets.

### `plot_probability_distribution(df)`
```python
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")   # Dark background
    ax.set_facecolor("#1a1d27")

    ax.hist(probs, bins=50, color="#5b6af0", edgecolor="#0f1117", alpha=0.85)

    for threshold, color, label in [
        (0.3, "#f1c40f", "Med"),
        (0.6, "#e67e22", "High"),
        (0.8, "#e74c3c", "Critical"),
    ]:
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.2)
```
> Creates a histogram with vertical dashed lines showing the band thresholds.
> Using the `Agg` backend (set in the import) ensures no GUI window opens —
> this is essential for running on servers and batch jobs.

### `plot_shap_importance(xgb_model, X_sample, feature_names)`
```python
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_s)
```
> **SHAP (SHapley Additive exPlanations):**
> SHAP assigns each feature importance based on how much it changed the model's prediction
> compared to the average prediction.
>
> For example, if `Contract = Month-to-month` pushed the churn probability UP by 0.15,
> its SHAP value for that customer is +0.15.
>
> `TreeExplainer` is an efficient version specifically for tree-based models (XGBoost).

---

# PART 13 — `train_pipeline.py` (The Conductor)

## 🎯 What does this file do?

This is the **main orchestration script** — it calls all the other modules in the right order,
passing data from one step to the next. Think of it as the assembly line manager.

## 🔍 Key sections

### Setup
```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
```
> On Windows, the terminal might not support Unicode by default.
> This forces UTF-8 output so emoji and special characters display correctly.

```python
load_dotenv()
```
> Reads `.env` file and loads `GEMINI_API_KEY` into environment variables.
> Must be called BEFORE any code that uses `os.getenv("GEMINI_API_KEY")`.

### `get_feature_names(preprocessor)`
```python
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)          # Numeric columns keep their original names
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            cat_names = ohe.get_feature_names_out(cols).tolist()  # Gets OHE column names
            names.extend(cat_names)
```
> After OneHotEncoding, `Contract` becomes `Contract_Month-to-month`, `Contract_One year`,
> `Contract_Two year`. This function reconstructs these names so SHAP can label axes correctly.

### The Run Log
```python
    log_data = {
        "run_id":       run_id,
        "total_time_s": round(elapsed, 2),
        "xgb_params":   XGB_PARAMS,
        "evaluation": {
            row["model"]: {k: round(v, 4) for k, v in row.items() if k != "model"}
            for row in eval_df.reset_index().to_dict("records")
        },
    }
    log_path.write_text(json.dumps(log_data, indent=2, default=str), encoding="utf-8")
```
> This creates a complete record of EVERY experiment run.
> `json.dumps(... default=str)` → converts non-serializable types (like some numpy types) to string.
> Saved as `logs/run_20260225_221755.json` — you can compare runs to see if changes helped.

---

# PART 14 — `train.py` (The Simple Alternative)

## 🎯 Why have two training scripts?

`train_pipeline.py` = full production pipeline (stacking + AI + calibration)
`train.py` = simplified, easy-to-understand single model training

`train.py` is useful for:
- Quick experiments
- Understanding a single model's performance
- Teaching/demos

It supports 4 model types via `--model-type` argument: `logistic`, `xgboost`, `mlp`, or `stacking`.

## Key functions in `train.py`

### `build_stacking_pipeline()` — sklearn StackingClassifier
```python
    estimators = [
        ("lr",  build_single_pipeline("logistic", ...)),
        ("xgb", build_single_pipeline("xgboost",  ...)),
        ("mlp", build_single_pipeline("mlp",       ...)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        n_jobs=-1,         # Use all CPU cores
        passthrough=False, # Only pass base model probs to meta-model, not original features
    )
```
> Note: This uses sklearn's built-in `StackingClassifier` (which does handle OOF internally)
> vs. our custom stacking in `stacking.py` which gives us more control.

---

# PART 15 — `predict.py` (Single Customer Prediction)

## 🎯 Real-world use case

After training, a customer service representative wants to check if a specific
customer who just called is at risk. They run:

```bash
python src/predict.py ^
    --model models/churn_pipeline.joblib ^
    --input tenure=12 MonthlyCharges=85.0 Contract="Month-to-month" ^
             InternetService="Fiber optic" PaymentMethod="Electronic check"
```

## 🔍 Key function: `parse_kv_pairs(pairs)`

```python
    for pair in pairs:
        if "=" not in pair:
            # Handle PowerShell splitting: "Fiber optic" → ["Fiber", "optic"]
            if current_key is None:
                raise ValueError(...)
            data[current_key] = f"{data[current_key]} {pair}".strip()
            continue
```
> **Windows/PowerShell problem:** When you type `InternetService=Fiber optic`,
> PowerShell may split this into `["InternetService=Fiber", "optic"]` at the space.
> This code handles that by appending continuation parts to the previous key's value.

```python
        # Type casting
        lower = value.lower()
        if lower in {"true", "false"}:
            casted = lower == "true"       # → Python bool
        else:
            try:
                casted = float(value) if "." in value else int(value)
            except ValueError:
                casted = value             # Keep as string for categorical columns
```
> Smart type detection: `"12"` → `int 12`, `"85.0"` → `float 85.0`, `"Month-to-month"` → `str`.

---

# PART 16 — HOW TO RUN THE PROJECT

## Prerequisites
```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Run the full pipeline
```bash
# Full pipeline with AI recommendations
python src/train_pipeline.py --data data/customer_churn.csv --target Churn

# Full pipeline WITHOUT AI (faster, no API key needed)
python src/train_pipeline.py --data data/customer_churn.csv --target Churn --no-ai

# Full pipeline WITHOUT calibration
python src/train_pipeline.py --data data/customer_churn.csv --target Churn --no-calibration

# Full pipeline with Platt Scaling calibration instead of the default Isotonic
python src/train_pipeline.py --data data/customer_churn.csv --target Churn --calibration-method sigmoid
```

## Run simplified training (train.py)
```bash
# Train just XGBoost
python src/train.py --data data/customer_churn.csv --model-type xgboost

# Train Logistic Regression
python src/train.py --data data/customer_churn.csv --model-type logistic

# Train sklearn stacking ensemble
python src/train.py --data data/customer_churn.csv --model-type stacking
```

## Run stacked inference on a new customer (NEW!)
```bash
# Uses the most recently trained model artifacts automatically
python src/predict_stacked.py --input tenure=6 MonthlyCharges=95.0 Contract=Month-to-month ^  
    InternetService=Fiber optic PaymentMethod=Electronic check TechSupport=No

# Or specify an exact run_id if you want a specific model version
python src/predict_stacked.py --run-id 20260226_013000 --input tenure=12
```

## Run single-model prediction (original predict.py)
```bash
python src/predict.py --model models/churn_pipeline.joblib ^
    --input tenure=6 MonthlyCharges=95.0 TotalCharges=570.0 ^
             Contract=Month-to-month InternetService=Fiber optic ^
             PaymentMethod=Electronic check TechSupport=No
```

## Run unit tests (NEW!)
```bash
# Run all 38 tests with detailed output
python -m pytest tests/ -v

# Run only one test file
python -m pytest tests/test_risk_segmentation.py -v

# Run with a coverage report (install pytest-cov first)
pip install pytest-cov
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## Launch Streamlit Dashboard (NEW!)
```bash
# Install if not already done
pip install streamlit

# Launch the dashboard
streamlit run app.py

# Opens at http://localhost:8501 in your browser
# Upload data/customer_churn.csv in the sidebar to see predictions
```

## Expected output structure after running
```
outputs/
  ├── churn_predictions.csv      ← all 7,043 customers with predictions
  ├── summary_report.txt         ← text summary
  ├── prob_distribution.png      ← histogram plot
  ├── band_distribution.png      ← bar chart
  └── shap_importance.png        ← feature importance

models/
  ├── preprocessing_pipeline.joblib           ← always same filename
  ├── xgb_model_20260226_013000.joblib         ← versioned by run_id
  ├── nn_model_20260226_013000.joblib
  ├── meta_model_20260226_013000.joblib
  └── pipeline_info.json                       ← NEW! metadata for inference

logs/
  └── run_20260226_013000.json   ← experiment log

tests/
  ├── test_data_loader.py        ← 8 tests
  ├── test_preprocessor.py       ← 7 tests
  ├── test_risk_segmentation.py  ← 11 tests
  └── test_business_impact.py    ← 12 tests
```

---

# PART 17 — KEY CONCEPTS GLOSSARY

| Term | Simple Explanation |
|---|---|
| **Churn** | A customer cancelling their subscription |
| **Feature** | A column used as input to the model (e.g., tenure, MonthlyCharges) |
| **Target/Label** | What we're predicting (Churn = 0 or 1) |
| **Train/Test Split** | Divide data: train the model on 80%, evaluate honestly on unseen 20% |
| **Data Leakage** | When information from the test set "leaks" into training — causes overly optimistic evaluation |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Underfitting** | Model is too simple to capture the patterns |
| **Class Imbalance** | Dataset has very few positive examples (few churners) — models ignore them |
| **One-Hot Encoding** | Converting `"Male"/"Female"` → `[1,0]` / `[0,1]` |
| **StandardScaler** | Normalize numbers to mean=0, std=1 |
| **Imputation** | Filling in missing values |
| **OOF / Out-of-Fold** | Predictions made on data the model was NOT trained on (in cross-validation) |
| **Stacking** | Training a meta-model on top of multiple base models' predictions |
| **ROC-AUC** | How well does the model rank churners above non-churners? 1.0 = perfect |
| **Brier Score** | Mean squared error of probabilities — lower is better |
| **Calibration** | Correcting systematic bias in probability estimates |
| **SHAP** | How much each feature contributed to each prediction |
| **Early Stopping** | Stop training when validation performance stops improving |
| **Gradient Boosting** | Sequentially build trees where each one corrects previous mistakes |
| **Batch Processing** | Processing multiple items in groups instead of one-by-one |
| **Graceful Degradation** | System continues working even when optional components fail |
| **Serialization** | Saving a Python object (model) to a file so it can be loaded later |
| **Random State / Seed** | Fixed number that makes random operations reproducible |
| **scale_pos_weight** | XGBoost parameter to handle class imbalance |
| **Rate Limiting** | APIs restrict how many requests you can make per minute |
| **Retry Logic** | Automatically retrying failed operations with a wait period |

---

# PART 18 — STUDY EXERCISES

These exercises will deepen your understanding:

### Exercise 1 — Data Understanding
Open `data/customer_churn.csv` in Excel/Pandas. Answer:
- How many customers are there?
- What percentage churned?
- Which columns have missing values?
- What's the average MonthlyCharges for churners vs. non-churners?

### Exercise 2 — Modify a Hyperparameter
In `config.py`, change `"n_estimators": 300` to `"n_estimators": 50`.
Re-run the pipeline. Does the ROC-AUC change? Does training finish faster?

### Exercise 3 — Change Risk Bands
In `config.py`, change the risk bands:
```python
RISK_BANDS = {
    "Low":      (0.0, 0.4),
    "Medium":   (0.4, 0.7),
    "High":     (0.7, 0.9),
    "Critical": (0.9, 1.01),
}
```
Re-run. How does the band distribution change?

### Exercise 4 — Skip Calibration
Run with `--no-calibration`. Compare the probability distribution plots.
Does the histogram shape change? Why?

### Exercise 5 — Understand OOF
Add a print statement inside the fold loop in `stacking.py`:
```python
print(f"Fold {fold_idx}: XGB OOF AUC = {roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")
```
What do you observe about AUC consistency across folds?

### Exercise 6 — Feature Importance
Open `outputs/shap_importance.png`. Which features are most important?
Do the top features make business sense? (e.g., tenure, contract type, charges?)

### Exercise 7 — New Customer Prediction
Add a NEW "customer" to the prediction CSV by creating a row with synthetic values,
then run `predict.py` with those values. Did the prediction match your intuition?

---

---

# PART 19 — `predict_stacked.py` (NEW: Stacked Inference CLI)

## 🎯 What problem does this solve?

After training with `train_pipeline.py`, you have 4 artifact files.
But the original `predict.py` was designed for a single sklearn model — it doesn't know
about XGBoost, Neural Networks, or the meta-model.
`predict_stacked.py` bridges this gap: it loads ALL 4 artifacts and runs the
complete stacking inference for a single new customer.

## 🔍 How inference works step-by-step

```
CLI input: tenure=6, MonthlyCharges=90.0, Contract=Month-to-month
      │
      ▼  Step 1: parse_kv_pairs() → Python dict (with type casting)
      │
      ▼  Step 2: pd.DataFrame([customer]) → single-row DataFrame
      │
      ▼  Step 3: preprocessor.transform() → scaled + encoded numpy array
      │             (using the SAME pipeline fitted during training)
      ▼  Step 4: xgb_model.predict_proba() → P_xgb (e.g. 0.82)
      │          nn_model.predict_proba()  → P_nn  (e.g. 0.76)
      ▼  Step 5: meta_model.predict_proba([[P_xgb, P_nn]]) → P_final (e.g. 0.80)
      │
      ▼  Step 6: assign_risk_band(P_final) → "Critical"
      │
      ▼  Step 7: estimate_revenue_loss(...) → $1,296.00
      │
      ▼  OUTPUT: Formatted result printed to terminal
```

## 🔍 Key function: `find_latest_artifact(prefix)`
```python
    matches = sorted(MODELS_DIR.glob(f"{prefix}_*.joblib"))
    if matches:
        return matches[-1]   # Last alphabetically = latest by timestamp
```
> `Path.glob(pattern)` finds all files matching a pattern.
> `f"xgb_model_*.joblib"` matches `xgb_model_20260226_013000.joblib`.
> Sorted alphabetically means the last entry is the most recent timestamp.
> This is simpler than reading JSON — no config file needed.

## 🔍 Key function: `predict_single(customer, ...)`
```python
    X_new = pd.DataFrame([customer])   # Wrap dict in a list to make 1-row DataFrame
    X_t   = preprocessor.transform(X_new)  # Apply the fitted ColumnTransformer

    p_xgb = float(xgb_model.predict_proba(X_t)[0, 1])
    p_nn  = float(nn_model.predict_proba(X_t)[0, 1])

    stacked = np.column_stack([[p_xgb], [p_nn]])  # [[0.82, 0.76]]
    p_final = float(meta_model.predict_proba(stacked)[0, 1])
```
> `[0, 1]` index: row 0 (only row), column 1 (churn probability).
> `float(...)` converts numpy scalar → Python float (better for printing).
> `np.column_stack([[p_xgb], [p_nn]])` wraps scalars in lists first, then stacks.

---

# PART 20 — `tests/` (Unit Testing with Pytest)

## 🎯 Why do we write tests?

Imagine you write a bug fix that accidentally changes how `assign_risk_band` works.
Without tests, you'd only discover the bug AFTER running the full 30-minute pipeline
or after wrong predictions reach your business users.

With tests: run `pytest tests/ -v` in **5 seconds** and immediately see:
```
FAILED tests/test_risk_segmentation.py::TestAssignRiskBand::test_medium_band
AssertionError: Expected 'Medium', got 'Low'
```

Tests serve as a **safety net** and **documentation** — they show exactly what
each function is supposed to do.

## 📦 pytest basics

```python
import pytest

def test_something():
    result = my_function(42)
    assert result == "expected_value", "Helpful message if this fails"
```

> **Rules:**
> - File names must start with `test_` (e.g., `test_data_loader.py`)
> - Function names must start with `test_`
> - Classes must start with `Test`
> - Use `assert` to check expected behavior
> - `pytest.raises(ErrorType)` to test that an error IS raised correctly

## 📦 pytest fixtures

```python
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"tenure": [1, 12, 24], ...})

def test_something(sample_dataframe):   # pytest automatically injects this!
    result = some_function(sample_dataframe)
    assert ...
```
> **Fixtures** are reusable test setup code.
> Instead of creating a DataFrame in every test, define it once in a fixture
> and pytest injects it automatically into any test that requests it.

## 🔍 Our 4 test files at a glance

| File | Tests | What's covered |
|---|---|---|
| `test_data_loader.py` | 8 | Target normalization, type coercion, immutability |
| `test_preprocessor.py` | 7 | ColumnTransformer type, scaling, OHE, NaN handling |
| `test_risk_segmentation.py` | 11 | All 4 bands, boundary conditions (0.3, 0.6, 1.0) |
| `test_business_impact.py` | 12 | Revenue formula, sort order, fallbacks, rounding |
| **Total** | **38** | |

## 🔍 Example: `test_boundary_at_0_3()`

```python
def test_boundary_at_0_3(self):
    """0.3 is the INCLUSIVE lower bound of 'Medium', NOT 'Low'."""
    probs = np.array([0.3])
    bands = assign_risk_band(probs)
    assert bands[0] == "Medium", (
        f"P=0.3 (boundary) should be 'Medium', got '{bands[0]}'"
    )
```
> **Why test the boundary?** Off-by-one errors at boundaries (Is 0.3 Low or Medium?)
> are one of the most common bugs. A test like this catches it instantly.
> The docstring explains WHY this test exists — it's a living specification.

---

# PART 21 — `app.py` (Streamlit Dashboard)

## 🎯 What is Streamlit?

Streamlit is a Python library that converts regular Python scripts into
interactive web apps — **without any HTML, CSS, or JavaScript knowledge**.

```python
import streamlit as st

st.title("Hello World")          # → big heading
st.bar_chart(some_dataframe)     # → interactive bar chart
uploaded = st.file_uploader()    # → file upload widget
```

Streamlit re-runs the ENTIRE script from top to bottom every time the user
interacts with anything (clicks a button, changes a setting).
This is the Streamlit execution model — keep it in mind.

## 🔍 Key Streamlit patterns in `app.py`

### `@st.cache_resource` vs `@st.cache_data`

```python
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    return joblib.load(...), joblib.load(...), ...
```
> `@st.cache_resource`: Runs ONCE per app session.
> Used for heavy objects that shouldn't be recreated on every rerun (models).
> `@st.cache_data`: Runs once PER UNIQUE INPUT.
> Used for data processing functions (running predictions on a specific CSV).

### Tabs
```python
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Charts", "👥 Table", "🔍 Importance"])
with tab1:
    st.subheader("KPI Summary")
    # ... content ...
```
> Tabs let you organize content without cluttering the page.
> All tab content is in Python — no separate HTML files.

### Columns
```python
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", "7,043")
```
> `st.columns(4)` creates 4 equal-width columns side by side.
> Equivalent to a 4-column grid layout in CSS, but in Python.

### Session State
```python
st.session_state["df"] = df       # Store result
df = st.session_state["df"]       # Retrieve on next rerun
```
> Because Streamlit reruns the script on every interaction, local variables
> are lost. `st.session_state` persists data between reruns — like a global
> dictionary that survives reruns within the same session.

### Download Button
```python
st.download_button(
    "⬇️ Download filtered CSV",
    data=csv_bytes,
    file_name="predictions.csv",
    mime="text/csv",
)
```
> Creates a download button. `csv_bytes = df.to_csv().encode("utf-8")`.
> The user clicks it and the browser downloads the filtered table.

## 🔍 The 4 Dashboard Tabs

| Tab | What you see | Business use |
|---|---|---|
| **Overview** | KPI cards: total customers, high-risk count, total $ at risk, avg probability | Executive summary |
| **Risk Charts** | Bar chart + histogram + SHAP image from outputs/ | Analyst review |
| **Customer Table** | Searchable, filterable, sortable full table + CSV download | Retention team |
| **Feature Importance** | SHAP image + pipeline config JSON | Data scientist |

---

# PART 22 — COMPLETE PROJECT STRUCTURE (Final State)

```
telco-customer-churn-prediction/
│
├── app.py                        ← [NEW] Streamlit dashboard
├── requirements.txt              ← [UPDATED] + streamlit, pytest
├── .env                          ← GEMINI_API_KEY (git-ignored)
├── .env.example                  ← Template
├── .gitignore
├── PROJECT_DOCUMENTATION.md
├── STUDENT_GUIDE.md              ← This file
│
├── src/
│   ├── config.py                 ← All settings
│   ├── data_loader.py            ← CSV loading + cleaning
│   ├── preprocessor.py           ← ColumnTransformer pipeline
│   ├── xgb_model.py              ← [UPDATED] model_name param
│   ├── nn_model.py               ← [UPDATED] model_name param
│   ├── stacking.py               ← [UPDATED] model_name param + OOF
│   ├── calibration.py            ← Isotonic/Platt calibration
│   ├── evaluation.py             ← ROC-AUC, F1, Brier metrics
│   ├── risk_segmentation.py      ← Low/Medium/High/Critical bands
│   ├── business_impact.py        ← Revenue loss formula
│   ├── retention_ai.py           ← [UPDATED] 5-key schema validation
│   ├── reporting.py              ← CSV + TXT + PNG outputs
│   ├── train_pipeline.py         ← [UPDATED] versioning + calibration CLI
│   ├── train.py                  ← Simple single-model training
│   ├── predict.py                ← Single-model inference
│   └── predict_stacked.py        ← [NEW] Full stacked inference CLI
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py       ← [NEW] 8 tests
│   ├── test_preprocessor.py      ← [NEW] 7 tests
│   ├── test_risk_segmentation.py ← [NEW] 11 tests
│   └── test_business_impact.py   ← [NEW] 12 tests
│
├── data/
│   └── customer_churn.csv
│
├── models/
│   ├── preprocessing_pipeline.joblib
│   ├── xgb_model_{run_id}.joblib
│   ├── nn_model_{run_id}.joblib
│   ├── meta_model_{run_id}.joblib
│   └── pipeline_info.json        ← [NEW] feature names, hyperparams
│
├── outputs/
│   ├── churn_predictions.csv
│   ├── summary_report.txt
│   ├── prob_distribution.png
│   ├── band_distribution.png
│   └── shap_importance.png
│
└── logs/
    └── run_{run_id}.json
```

---

*End of Student Guide — Stacked Churn Intelligence System (Completed)*
*Original: February 25, 2026 | Updated: February 26, 2026*
*Total concepts covered: 22 parts | 45+ functions | 25+ ML concepts | 38 unit tests*
