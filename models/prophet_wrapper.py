"""
Meta Prophet Wrapper for Epidemic Time Series Forecasting (Optimized for Infectious Diseases).
Key improvements:
1. Log1p Transformation (log(y+1) / expm1) to model multiplicative exponential transmission
   and strictly prevent negative values.
2. Expanded changepoint_range (0.95) to detect recent epidemic wave turns.
3. Adjusted changepoint_prior_scale (0.15) to avoid Laplace prior collapse.
4. Intelligent seasonality adaptation (requires >=2 years for yearly, reduced Fourier order).
"""

from typing import Dict, Any, Optional
import time
import pandas as pd
import numpy as np
from prophet import Prophet
import logging

# Suppress cmdstanpy verbose logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)


class ProphetForecasterWrapper:
    """Optimized wrapper class for Meta Prophet epidemic forecasting."""

    def __init__(
        self,
        interval_width: float = 0.8,
        use_log_transform: bool = True,
        changepoint_prior_scale: float = 0.15,
        changepoint_range: float = 0.95,
        seasonality_prior_scale: float = 10.0
    ):
        self.interval_width = interval_width
        self.use_log_transform = use_log_transform
        self.changepoint_prior_scale = changepoint_prior_scale
        self.changepoint_range = changepoint_range
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None

    def forecast(
        self,
        df_history: pd.DataFrame,
        horizon: int = 8,
        freq: str = 'W',
        growth: str = 'linear'
    ) -> Dict[str, Any]:
        """
        Fits Prophet on historical epidemic data and predicts the next `horizon` periods.
        Input df_history must have columns 'ds' and 'y'.
        """
        start_time = time.time()

        df = df_history[['ds', 'y']].copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0.0)
        df['y'] = np.maximum(df['y'].values, 0.0)

        n_points = len(df)

        # Apply log(y + 1) transformation for epidemic transmission modeling
        if self.use_log_transform:
            df_train = df.copy()
            df_train['y'] = np.log1p(df_train['y'].values)
        else:
            df_train = df.copy()

        # Adaptive seasonality rules
        # Yearly seasonality requires at least 2 full cycles (104 weeks or 730 days)
        # to avoid under-identified Fourier explosion
        if freq == 'W':
            enable_yearly = (n_points >= 104)
            enable_weekly = False
            enable_daily = False
        elif freq == 'D':
            enable_yearly = (n_points >= 730)
            enable_weekly = (n_points >= 14)
            enable_daily = False
        else: # Monthly
            enable_yearly = (n_points >= 24)
            enable_weekly = False
            enable_daily = False

        self.model = Prophet(
            growth=growth,
            interval_width=self.interval_width,
            yearly_seasonality=False, # We configure manually with restrained fourier order if needed
            weekly_seasonality=enable_weekly,
            daily_seasonality=enable_daily,
            changepoint_prior_scale=self.changepoint_prior_scale,
            changepoint_range=self.changepoint_range,
            seasonality_prior_scale=self.seasonality_prior_scale
        )

        # Add restrained yearly seasonality if eligible
        if enable_yearly:
            if freq == 'W':
                self.model.add_seasonality(name='yearly', period=52.18, fourier_order=3, prior_scale=5.0)
            elif freq == 'D':
                self.model.add_seasonality(name='yearly', period=365.25, fourier_order=4, prior_scale=5.0)
            elif freq == 'M':
                self.model.add_seasonality(name='yearly', period=12.0, fourier_order=2, prior_scale=5.0)

        self.model.fit(df_train)

        # Map frequency to pandas date offset
        freq_map = {'D': 'D', 'W': 'W-SUN', 'M': 'MS'}
        pd_freq = freq_map.get(freq, 'W-SUN' if freq == 'W' else 'D')

        future = self.model.make_future_dataframe(periods=horizon, freq=pd_freq, include_history=False)
        forecast_df = self.model.predict(future)

        # Invert log transformation
        if self.use_log_transform:
            yhat = np.expm1(forecast_df['yhat'].values)
            yhat_lower = np.expm1(forecast_df['yhat_lower'].values)
            yhat_upper = np.expm1(forecast_df['yhat_upper'].values)
        else:
            yhat = forecast_df['yhat'].values
            yhat_lower = forecast_df['yhat_lower'].values
            yhat_upper = forecast_df['yhat_upper'].values

        # Ensure strictly non-negative epidemic disease counts
        yhat = np.maximum(yhat, 0.0)
        yhat_lower = np.maximum(yhat_lower, 0.0)
        yhat_upper = np.maximum(yhat_upper, yhat)

        elapsed = time.time() - start_time

        return {
            'model_name': 'Meta Prophet (Optimized)',
            'dates': forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'forecast': yhat.astype(float),
            'ci_lower': yhat_lower.astype(float),
            'ci_upper': yhat_upper.astype(float),
            'elapsed_time_sec': round(elapsed, 3),
            'n_history_points': n_points,
            'interval_width': self.interval_width,
            'use_log_transform': self.use_log_transform,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'changepoint_range': self.changepoint_range,
            'seasonality_used': {
                'yearly': enable_yearly,
                'weekly': enable_weekly
            }
        }
