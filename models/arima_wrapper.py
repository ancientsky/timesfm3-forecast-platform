"""
Auto ARIMA Model Wrapper for Epidemic Time Series Forecasting.
Uses pmdarima for automated order selection (p, d, q) based on AIC/BIC,
with log1p transformation for non-negative epidemic disease count guarantees.
"""

from typing import Dict, Any, Optional, List
import time
import pandas as pd
import numpy as np
import pmdarima as pm
import warnings

warnings.filterwarnings('ignore')


class AutoARIMAForecasterWrapper:
    """Wrapper class for Automated ARIMA / SARIMA epidemic forecasting."""

    def __init__(
        self,
        interval_width: float = 0.8,
        use_log_transform: bool = True,
        max_p: int = 4,
        max_q: int = 4,
        max_d: int = 2,
        seasonal: bool = False,
        m: int = 1
    ):
        self.interval_width = interval_width
        self.use_log_transform = use_log_transform
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.seasonal = seasonal
        self.m = m
        self.model = None

    def forecast(
        self,
        df_history: pd.DataFrame,
        future_dates: List[str],
        horizon: int = 8,
        freq: str = 'W'
    ) -> Dict[str, Any]:
        """
        Fits Auto ARIMA on historical epidemic data and predicts the next `horizon` periods.
        Input df_history must have column 'y'.
        """
        start_time = time.time()

        y_vals = pd.to_numeric(df_history['y'], errors='coerce').fillna(0.0).values.astype(float)
        y_vals = np.maximum(y_vals, 0.0)

        # Apply log(y + 1) transformation for epidemic count data
        if self.use_log_transform:
            y_train = np.log1p(y_vals)
        else:
            y_train = y_vals.copy()

        # Decide seasonal m based on frequency if seasonal is True
        m_val = self.m
        if self.seasonal and m_val <= 1:
            if freq == 'D':
                m_val = 7
            elif freq == 'M':
                m_val = 12
            else:
                m_val = 1 # weekly m=52 is often too slow and prone to singular matrices

        alpha = max(0.01, min(0.50, 1.0 - self.interval_width))

        self.model = pm.auto_arima(
            y_train,
            seasonal=self.seasonal if m_val > 1 else False,
            m=m_val if self.seasonal and m_val > 1 else 1,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        fc_raw, conf_int_raw = self.model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=alpha
        )

        # Invert log-transform
        if self.use_log_transform:
            yhat = np.expm1(fc_raw)
            yhat_lower = np.expm1(conf_int_raw[:, 0])
            yhat_upper = np.expm1(conf_int_raw[:, 1])
        else:
            yhat = fc_raw
            yhat_lower = conf_int_raw[:, 0]
            yhat_upper = conf_int_raw[:, 1]

        # Guarantee non-negative epidemic values
        yhat = np.maximum(yhat, 0.0)
        yhat_lower = np.maximum(yhat_lower, 0.0)
        yhat_upper = np.maximum(yhat_upper, yhat)

        elapsed = time.time() - start_time

        order_str = str(self.model.order)
        if hasattr(self.model, 'seasonal_order') and self.model.seasonal_order is not None and self.model.seasonal_order != (0, 0, 0, 0):
            order_str += f" x {self.model.seasonal_order}"

        return {
            'model_name': 'Auto ARIMA',
            'dates': future_dates[:horizon],
            'forecast': yhat.astype(float),
            'ci_lower': yhat_lower.astype(float),
            'ci_upper': yhat_upper.astype(float),
            'elapsed_time_sec': round(elapsed, 3),
            'order': self.model.order,
            'order_str': order_str,
            'aic': round(float(self.model.aic()), 1) if hasattr(self.model, 'aic') else None,
            'bic': round(float(self.model.bic()), 1) if hasattr(self.model, 'bic') else None,
            'use_log_transform': self.use_log_transform,
            'horizon': horizon
        }
