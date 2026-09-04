"""
Evaluation Metrics Module for Epidemic Time Series Forecasting.
Computes MAE, RMSE, MAPE, SMAPE, WAPE, Directional Accuracy, Peak Timing/Value Error, and CI Coverage.
"""

from typing import Dict, Any, Optional
import numpy as np


def compute_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    last_historical_val: Optional[float] = None,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    eps: float = 1e-6
) -> Dict[str, float]:
    """
    Calculates comprehensive forecasting and epidemic accuracy metrics.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if len(actual) == 0 or len(predicted) == 0:
        return {}

    min_len = min(len(actual), len(predicted))
    y_true = actual[:min_len]
    y_pred = predicted[:min_len]

    # 1. Standard Error Metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # MAPE
    denom_mape = np.where(np.abs(y_true) < 1.0, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom_mape)) * 100.0)
    
    # SMAPE
    denom_smape = np.abs(y_true) + np.abs(y_pred) + eps
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / denom_smape) * 100.0)
    
    # WAPE
    sum_true = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (sum_true + eps) * 100.0)

    # 2. Directional Accuracy (趨勢方向吻合度)
    if last_historical_val is not None:
        y_true_seq = np.insert(y_true, 0, last_historical_val)
        y_pred_seq = np.insert(y_pred, 0, last_historical_val)
        diff_true = np.diff(y_true_seq)
        diff_pred = np.diff(y_pred_seq)
        dir_matches = (diff_true * diff_pred >= 0).astype(float)
        dir_acc = float(np.mean(dir_matches) * 100.0)
    else:
        if len(y_true) > 1:
            diff_true = np.diff(y_true)
            diff_pred = np.diff(y_pred)
            dir_matches = (diff_true * diff_pred >= 0).astype(float)
            dir_acc = float(np.mean(dir_matches) * 100.0)
        else:
            dir_acc = 100.0

    # 3. Epidemic Peak Metrics (疫情達峰預測評估)
    actual_peak_idx = int(np.argmax(y_true))
    actual_peak_val = float(y_true[actual_peak_idx])
    
    pred_peak_idx = int(np.argmax(y_pred))
    pred_peak_val = float(y_pred[pred_peak_idx])
    
    peak_val_err_pct = float(abs(pred_peak_val - actual_peak_val) / (actual_peak_val + eps) * 100.0)
    peak_timing_err_steps = int(abs(pred_peak_idx - actual_peak_idx))

    # 4. Uncertainty Interval Coverage
    ci_coverage = None
    if ci_lower is not None and ci_upper is not None:
        low = np.asarray(ci_lower)[:min_len]
        high = np.asarray(ci_upper)[:min_len]
        in_ci = (y_true >= low) & (y_true <= high)
        ci_coverage = float(np.mean(in_ci) * 100.0)

    res = {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'SMAPE (%)': round(smape, 2),
        'WAPE (%)': round(wape, 2),
        'Directional Accuracy (%)': round(dir_acc, 1),
        'Actual Peak Value': round(actual_peak_val, 1),
        'Predicted Peak Value': round(pred_peak_val, 1),
        'Peak Value Error (%)': round(peak_val_err_pct, 2),
        'Peak Timing Error (Steps)': peak_timing_err_steps
    }
    if ci_coverage is not None:
        res['CI Coverage (%)'] = round(ci_coverage, 1)

    return res
