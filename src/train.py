import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency handled at runtime
    XGBClassifier = None


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


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_base_estimator(model_type: str, random_state: int):
    if model_type == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced")

    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=600,
            random_state=random_state,
        )

    if model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_type '{model_type}'")


def build_single_pipeline(
    model_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", build_base_estimator(model_type, random_state)),
        ]
    )


def build_stacking_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int,
) -> StackingClassifier:
    estimators = [
        (
            "lr",
            build_single_pipeline(
                model_type="logistic",
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                random_state=random_state,
            ),
        ),
        (
            "xgb",
            build_single_pipeline(
                model_type="xgboost",
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                random_state=random_state,
            ),
        ),
        (
            "mlp",
            build_single_pipeline(
                model_type="mlp",
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                random_state=random_state,
            ),
        ),
    ]

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def build_pipeline(
    model_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int,
):
    if model_type == "stacking":
        return build_stacking_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            random_state=random_state,
        )
    return build_single_pipeline(
        model_type=model_type,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=random_state,
    )


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    for col in X.columns:
        if X[col].dtype != "object":
            continue

        text = X[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        numeric = pd.to_numeric(text, errors="coerce")
        non_null_count = int(text.notna().sum())
        parse_ratio = float(numeric.notna().sum() / non_null_count) if non_null_count else 0.0

        # Convert columns that are mostly numeric strings (e.g., TotalCharges).
        if parse_ratio >= 0.9:
            X[col] = numeric
        else:
            X[col] = text
    return X


def main():
    parser = argparse.ArgumentParser(description="Train customer churn model")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="Churn", help="Target column name")
    parser.add_argument(
        "--model-type",
        default="logistic",
        choices=["logistic", "xgboost", "mlp", "stacking"],
        help="Model type to train",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
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

    X = clean_features(X)

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(
        model_type=args.model_type,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=args.random_state,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"Model type: {args.model_type}")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_out)

    info: Dict[str, object] = {
        "target_column": args.target,
        "model_type": args.model_type,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    info_out = Path(args.info_out)
    info_out.parent.mkdir(parents=True, exist_ok=True)
    info_out.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved model to: {model_out}")
    print(f"Saved metadata to: {info_out}")


if __name__ == "__main__":
    main()
