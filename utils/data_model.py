from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


COST_COLS = [
    "Tuition_USD",
    "Living_Cost_Index",
    "Rent_USD",
    "Visa_Fee_USD",
    "Insurance_USD",
]

FEATURE_COLS = [
    "Country",
    "Level",
    "Program",
    "Duration_Years",
    "Living_Cost_Index",
    "Exchange_Rate",
]

TARGET_COL = "Estimated_Annual_Cost"


def estimate_annual_cost_row(row: pd.Series) -> float:
    tuition = row["Tuition_USD"]
    living = row["Living_Cost_Index"] / 100.0 * 12000.0
    rent = row["Rent_USD"] * 12.0
    visa = row["Visa_Fee_USD"]
    insurance = row["Insurance_USD"]
    return float(tuition + living + rent + visa + insurance)


def load_raw_data(csv_path: str = "International_Education_Costs.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        # Try relative to current file
        alt_path = os.path.join(os.path.dirname(__file__), os.pardir, csv_path)
        alt_path = os.path.abspath(alt_path)
        if not os.path.exists(alt_path):
            raise FileNotFoundError(f"Dataset not found at '{csv_path}' or '{alt_path}'.")
        csv_path = alt_path
    df = pd.read_csv(csv_path)
    return df


def clean_enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing cost columns and remove zeros for cost integrity
    df = df.dropna(subset=COST_COLS).copy()
    mask_nonzero = (df[COST_COLS] != 0).all(axis=1)
    df = df[mask_nonzero].copy()

    # Compute target consistent with training notebook
    df[TARGET_COL] = df.apply(estimate_annual_cost_row, axis=1)
    return df


def get_feature_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    options = {}
    for col in ["Country", "Level", "Program"]:
        if col in df.columns:
            options[col] = sorted([x for x in df[col].dropna().astype(str).unique() if x != ""])
        else:
            options[col] = []
    return options


@dataclass
class ModelMetrics:
    mae: float
    r2: float
    test_size: int

    def to_dict(self) -> Dict[str, float]:
        return {"MAE": self.mae, "R2": self.r2, "Test_Size": self.test_size}


def build_training_preprocessor(categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_features),
            ("num", num_pipe, numerical_features),
        ]
    )
    return preprocessor


def recompute_metrics_with_loaded_pipeline(pipeline: Pipeline, df: pd.DataFrame, random_state: int = 42) -> ModelMetrics:
    # Mirror the training split to report metrics for the loaded pipeline
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    # The pipeline already contains preprocessing + model; fit is not needed for loaded model
    # However, to evaluate fairly on the same test data distribution, do not re-fit.
    # Just predict on X_test.
    y_pred = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return ModelMetrics(mae=mae, r2=r2, test_size=len(y_test))


def validate_user_inputs(
    country: str,
    level: str,
    program: str,
    duration_years: float,
    living_cost_index: float,
    exchange_rate: float,
) -> Optional[str]:
    # Return an error message string if invalid, else None
    if not country:
        return "Country is required."
    if not level:
        return "Level is required."
    if not program:
        return "Program is required."

    try:
        dy = float(duration_years)
        if dy <= 0 or dy > 10:
            return "Duration_Years must be between 0 and 10."
    except Exception:
        return "Duration_Years must be a number."

    try:
        lci = float(living_cost_index)
        if lci <= 0 or lci > 300:
            return "Living_Cost_Index must be between 0 and 300."
    except Exception:
        return "Living_Cost_Index must be a number."

    try:
        er = float(exchange_rate)
        if er <= 0 or er > 1000:
            return "Exchange_Rate must be between 0 and 1000."
    except Exception:
        return "Exchange_Rate must be a number."

    return None


def make_input_dataframe(
    country: str,
    level: str,
    program: str,
    duration_years: float,
    living_cost_index: float,
    exchange_rate: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country": [country],
            "Level": [level],
            "Program": [program],
            "Duration_Years": [float(duration_years)],
            "Living_Cost_Index": [float(living_cost_index)],
            "Exchange_Rate": [float(exchange_rate)],
        }
    )


def load_trained_pipeline(model_path: str = "model.pkl") -> Pipeline:
    # Use pickle to load the full sklearn Pipeline (preprocessor + model)
    import pickle
    import warnings

    if not os.path.exists(model_path):
        # Try relative to repository root
        alt = os.path.join(os.path.dirname(__file__), os.pardir, model_path)
        alt = os.path.abspath(alt)
        if not os.path.exists(alt):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}' or '{alt}'. Ensure 'model.pkl' exists."
            )
        model_path = alt

    # Suppress version warnings when loading the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(model_path, "rb") as f:
            pipeline: Pipeline = pickle.load(f)
    
    return pipeline
