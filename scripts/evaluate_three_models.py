from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config import MODELS_DIR
from src.data_loader import load_data
from src.nn_model import predict_proba_nn
from src.predict_stacked import load_artifacts
from src.stacking import stack_predict
from src.xgb_model import predict_proba_xgb


def main() -> int:
    data_path = Path("data/test_dataset.csv")
    out_pred = Path("outputs/three_model_predictions.csv")
    out_acc = Path("outputs/three_model_accuracy.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {data_path}")

    X, y, _num, _cat = load_data(str(data_path), target_col="Churn")

    run_id = None
    info_path = MODELS_DIR / "pipeline_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            run_id = info.get("run_id")
        except Exception:
            run_id = None

    preprocessor, xgb_model, nn_model, meta_model = load_artifacts(run_id=run_id, models_dir=MODELS_DIR)

    X_t = preprocessor.transform(X)
    p_xgb = predict_proba_xgb(xgb_model, X_t)
    p_nn = predict_proba_nn(nn_model, X_t)
    p_stack = stack_predict(meta_model, p_xgb, p_nn)

    y_true = y.astype(int).to_numpy()
    pred_xgb = (p_xgb >= 0.5).astype(int)
    pred_nn = (p_nn >= 0.5).astype(int)
    pred_stack = (p_stack >= 0.5).astype(int)

    def metrics_row(name: str, pred: pd.Series | list | tuple) -> dict:
        acc = accuracy_score(y_true, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        return {
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }

    metrics = [
        metrics_row("xgb", pred_xgb),
        metrics_row("nn", pred_nn),
        metrics_row("stacked", pred_stack),
    ]

    pred_df = X.copy().reset_index(drop=True)
    pred_df["actual_churn"] = y_true
    pred_df["xgb_probability"] = p_xgb.round(6)
    pred_df["nn_probability"] = p_nn.round(6)
    pred_df["stacked_probability"] = p_stack.round(6)
    pred_df["xgb_prediction"] = pred_xgb
    pred_df["nn_prediction"] = pred_nn
    pred_df["stacked_prediction"] = pred_stack

    out_pred.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_pred, index=False)

    acc_df = pd.DataFrame(metrics)
    acc_df.to_csv(out_acc, index=False)

    print(f"Saved: {out_pred} ({len(pred_df)} rows)")
    print(f"Saved: {out_acc}")
    print(acc_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
