"""Utility functions for churn prediction project.

Provides helper functions for data loading, preprocessing,
model evaluation, and result visualization.
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.metrics import confusion_matrix, classification_report
import logging


logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate data for churn prediction."""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load CSV file with error handling."""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load JSON configuration file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {filepath}")
            raise
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
        """Validate that required columns exist in dataframe."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True


class ModelEvaluator:
    """Evaluate model performance with various metrics."""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    @staticmethod
    def evaluate_probabilistic(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Evaluate probabilistic predictions."""
        from sklearn.metrics import roc_auc_score, log_loss
        
        return {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'brier_score': np.mean((y_proba - y_true) ** 2)
        }


class FileManager:
    """Manage file operations for models and data."""
    
    @staticmethod
    def save_model(model: Any, filepath: str) -> bool:
        """Save trained model using pickle."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load trained model from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
    
    @staticmethod
    def save_predictions(predictions: Dict, filepath: str) -> bool:
        """Save predictions to JSON file."""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Predictions saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            return False


def create_feature_importance_report(model: Any, feature_names: list) -> Dict:
    """Create feature importance report from trained model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.to_dict('records')
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return []


def setup_logging(log_file: Optional[str] = None) -> None:
    """Configure logging for the project."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            filename=log_file
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )
