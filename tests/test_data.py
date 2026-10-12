import numpy as np
import pandas as pd
import pytest

from app.data.schema import FEATURES, DataValidationError, chronological_split, load_dataset, validate_raw
from app.data.synthetic import make_synthetic_creditcard
from app.features.preprocess import build_preprocessor


def test_missing_columns_rejected():
    with pytest.raises(DataValidationError):
        validate_raw(pd.DataFrame({"Time": [1], "Class": [0]}))


def test_cleaning_counts_invalid_rows():
    df = make_synthetic_creditcard(500, 0.1, seed=1)
    bad = df.iloc[:1].copy(); bad["V1"] = np.nan
    neg = df.iloc[1:2].copy(); neg["Amount"] = -5
    lab = df.iloc[2:3].copy(); lab["Class"] = 7
    clean, rep = validate_raw(pd.concat([df, bad, neg, lab, df.iloc[10:12]]))
    assert rep["dropped_missing_or_invalid"] == 1 and rep["dropped_negative_amount"] == 1
    assert rep["dropped_bad_label"] == 1 and rep["dropped_duplicates"] == 2
    assert clean[FEATURES].notna().all().all()


def test_missing_file_gives_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_dataset(tmp_path / "nope.csv")


def test_sample_mode_keeps_all_fraud_and_is_labelled(tmp_path):
    df = make_synthetic_creditcard(3000, 0.05, seed=2)
    p = tmp_path / "d.csv"; df.to_csv(p, index=False)
    full, _ = load_dataset(p)
    small, rep = load_dataset(p, sample_frac=0.2)
    assert small["Class"].sum() == full["Class"].sum() and len(small) < len(full) and "DEV SAMPLE" in rep["mode"]


def test_chronological_split_order_and_min_fraud():
    df = make_synthetic_creditcard(4000, 0.05, seed=3)
    sp = chronological_split(df)
    assert sp.train["Time"].max() <= sp.val["Time"].min() and sp.val["Time"].max() <= sp.test["Time"].min()
    with pytest.raises(DataValidationError):
        chronological_split(df, min_fraud_per_split=10_000)


def test_preprocessor_fit_on_train_only_and_consistent():
    df = make_synthetic_creditcard(2000, 0.05, seed=4)
    a, b = df.iloc[:1000][FEATURES], df.iloc[1000:][FEATURES]
    pre = build_preprocessor().fit(a)
    mean_before = pre.named_steps["scale"].mean_.copy()
    z1 = pre.transform(b)
    assert np.allclose(pre.named_steps["scale"].mean_, mean_before)  # transform never refits
    assert np.allclose(z1, pre.transform(b.copy()))
