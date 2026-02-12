"""Financial simulation utilities for cost-based intervention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    scenario: str
    r: float
    cost_action: float
    baseline_cost: float
    new_cost: float
    savings: float
    savings_pct: float
    n_actions: int
    action_rate: float


def simulate_policy(
    proba: np.ndarray,
    X_features: pd.DataFrame,
    r: float,
    cost_action: float,
    exposure_col: str = "OUTSTANDING_SUM",
) -> ScenarioResult:
    exposure = X_features[exposure_col].to_numpy(dtype=float)

    baseline_cost = float(np.sum(proba * exposure))

    act = (r * proba * exposure) > cost_action
    proba_after = proba.copy()
    proba_after[act] = proba_after[act] * (1.0 - r)

    new_cost = float(np.sum(proba_after * exposure) + act.sum() * cost_action)

    savings = baseline_cost - new_cost
    savings_pct = savings / baseline_cost if baseline_cost > 0 else 0.0

    return ScenarioResult(
        scenario=f"r={r:.2f}, cost={cost_action:.2f}",
        r=r,
        cost_action=cost_action,
        baseline_cost=baseline_cost,
        new_cost=new_cost,
        savings=savings,
        savings_pct=savings_pct,
        n_actions=int(act.sum()),
        action_rate=float(act.mean()),
    )


def run_scenarios(
    proba: np.ndarray,
    X_features: pd.DataFrame,
    scenarios: List[Dict[str, float]],
    exposure_col: str = "OUTSTANDING_SUM",
) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        res = simulate_policy(
            proba=proba,
            X_features=X_features,
            r=float(sc["r"]),
            cost_action=float(sc["cost_action"]),
            exposure_col=exposure_col,
        )
        rows.append(res.__dict__)
    return pd.DataFrame(rows)
