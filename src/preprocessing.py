"""Data preprocessing module for customer churn prediction."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Load customer churn dataset from CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df, target_col='Churn'):
    """Complete preprocessing pipeline."""
    df = df.dropna()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
