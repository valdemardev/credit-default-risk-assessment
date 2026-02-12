# Credit Risk Intervention Model

## Project Overview

This project develops a machine learning model to predict the probability of credit card default in the next month and proposes a cost-based intervention strategy to minimize expected financial losses.

The objective is not only to predict default, but to translate model outputs into financially optimal decisions.

Dataset: UCI Credit Card Default Dataset (loaded via KaggleHub).

---

## Business Objective

- Predict customer default propensity.
- Estimate financial exposure.
- Define an intervention policy that minimizes expected provision-related losses.

---

## Feature Engineering

Following the business rule:

> The difference between the monthly bill amount and the previous month's payment represents the outstanding balance.

For each month:

OUTSTANDING_m = max(BILL_AMT_m − PAY_AMT_(m+1), 0)

Engineered features include:

- Outstanding exposure (monthly and aggregated)
- Payment ratios
- Billing and payment trends
- Delinquency dynamics

The aggregated outstanding balance is used as an estimate of financial exposure at risk.

---

## Modeling Approach

Two models were trained:

- Logistic Regression (baseline linear model)
- HistGradientBoostingClassifier (non-linear model)

Evaluation Metrics:

- ROC-AUC
- PR-AUC

The non-linear model achieved superior ranking performance and was selected for financial simulation.

---

## Financial Decision Strategy

An intervention is triggered when:

r × p × exposure > cost_action

Where:
- p = predicted probability of default
- exposure = estimated financial exposure
- r = expected probability reduction due to intervention
- cost_action = cost of intervention

Multiple scenarios were simulated to assess expected cost savings.

---

## Key Insights

- Non-linear models improve risk ranking.
- Probability-based decisions outperform fixed thresholds.
- Financial optimization requires aligning ML outputs with economic cost structure.

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
