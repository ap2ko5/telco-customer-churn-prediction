import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_target(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "churn": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
        "stay": 0,
    }
    mapped = text.map(mapping)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(
            "Unrecognized target labels found. Please map your target manually. "
            f"Unknown labels: {bad}"
        )
    return mapped.astype(int)


def build_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train customer churn model")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="Churn", help="Target column name")
    parser.add_argument(
        "--model-out",
        default="models/churn_pipeline.joblib",
        help="Output path for trained pipeline",
    )
    parser.add_argument(
        "--info-out",
        default="models/model_info.json",
        help="Output path for model metadata",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Available columns: {df.columns.tolist()}"
        )

    y = normalize_target(df[args.target])
    X = df.drop(columns=[args.target]).copy()

    # Common ID column in churn datasets; not useful as feature.
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype(str).str.strip()

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    info = {
        "target_column": args.target,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "test_size": 0.2,
        "random_state": 42,
    }

    info_out = Path(args.info_out)
    info_out.parent.mkdir(parents=True, exist_ok=True)
    info_out.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved model to: {model_out}")
    print(f"Saved metadata to: {info_out}")


if __name__ == "__main__":
    main()
