"""Modeling utilities: preprocessing, pipelines, evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class EvalResult:
    name: str
    roc_auc: float
    pr_auc: float
    proba: np.ndarray


def build_preprocessor(cat_cols: list[str], num_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def build_models(preprocess: ColumnTransformer, random_state: int = 42) -> Dict[str, Pipeline]:
    pipe_lr = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe_hgb = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", HistGradientBoostingClassifier(random_state=random_state)),
        ]
    )
    return {"Logistic Regression": pipe_lr, "HistGradientBoosting": pipe_hgb}


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold: float = 0.5,
    verbose: bool = True,
) -> EvalResult:
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    if verbose:
        preds = (proba >= threshold).astype(int)
        print(f"\n=== {name} ===")
        print(f"ROC-AUC: {roc:.4f}")
        print(f"PR-AUC : {pr:.4f}")
        print(f"Confusion matrix (thr={threshold:.2f}):\n{confusion_matrix(y_test, preds)}")
        print("Classification report:")
        print(classification_report(y_test, preds, digits=4))

    return EvalResult(name=name, roc_auc=roc, pr_auc=pr, proba=proba)
