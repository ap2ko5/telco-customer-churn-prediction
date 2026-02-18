import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def parse_kv_pairs(pairs):
    data = {}
    current_key = None

    for pair in pairs:
        if "=" not in pair:
            # PowerShell commonly splits unquoted values with spaces into multiple args.
            # Example: InternetService=Fiber optic -> ["InternetService=Fiber", "optic"]
            if current_key is None:
                raise ValueError(f"Invalid input '{pair}'. Use key=value format.")
            data[current_key] = f"{data[current_key]} {pair}".strip()
            continue

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        current_key = key

        # Lightweight type casting.
        lower = value.lower()
        if lower in {"true", "false"}:
            casted = lower == "true"
        else:
            try:
                if "." in value:
                    casted = float(value)
                else:
                    casted = int(value)
            except ValueError:
                casted = value

        data[key] = casted
    return data


def main():
    parser = argparse.ArgumentParser(description="Predict churn for a new customer")
    parser.add_argument(
        "--model",
        default="models/churn_pipeline.joblib",
        help="Path to trained model pipeline",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Feature values as key=value pairs",
    )
    parser.add_argument(
        "--labels",
        default='{"0":"No Churn","1":"Churn"}',
        help="JSON dict mapping predicted class to label",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    feature_map = parse_kv_pairs(args.input)

    X_new = pd.DataFrame([feature_map])
    pred = int(pipeline.predict(X_new)[0])
    prob = float(pipeline.predict_proba(X_new)[0, 1])

    labels = json.loads(args.labels)
    pred_label = labels.get(str(pred), str(pred))

    print(f"Predicted class: {pred} ({pred_label})")
    print(f"Churn probability: {prob:.4f}")


if __name__ == "__main__":
    main()
