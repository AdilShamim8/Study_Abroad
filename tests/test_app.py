import os
import math

import numpy as np
import pandas as pd

from utils.data_model import (
    clean_enrich_data,
    load_raw_data,
    load_trained_pipeline,
    make_input_dataframe,
    recompute_metrics_with_loaded_pipeline,
)


def test_data_load_and_clean():
    df = load_raw_data("International_Education_Costs.csv")
    assert not df.empty
    df2 = clean_enrich_data(df)
    assert "Estimated_Annual_Cost" in df2.columns
    assert len(df2) <= len(df)


def test_model_load_and_predict():
    pipe = load_trained_pipeline("model.pkl")
    # build a minimal input frame
    X = make_input_dataframe(
        country="USA",
        level="Master",
        program="Computer Science",
        duration_years=2.0,
        living_cost_index=100.0,
        exchange_rate=1.0,
    )
    y_pred = pipe.predict(X)
    assert y_pred.shape == (1,)
    assert math.isfinite(float(y_pred[0]))


def test_metrics_recompute():
    df = clean_enrich_data(load_raw_data("International_Education_Costs.csv"))
    pipe = load_trained_pipeline("model.pkl")
    metrics = recompute_metrics_with_loaded_pipeline(pipe, df)
    assert 0.0 <= metrics.r2 <= 1.0 or math.isnan(metrics.r2)
    assert metrics.mae >= 0.0
