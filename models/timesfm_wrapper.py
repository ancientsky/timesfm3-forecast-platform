"""
Google TimesFM 3.0 Model Wrapper for Epidemic Time Series Forecasting.
Loads google/timesfm-3.0-pytorch foundation model, handles tensor transformations,
quantile interval extraction, and zero-shot inference.
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress minor PyTorch / HF warnings
warnings.filterwarnings('ignore')

_GLOBAL_TIMESFM_FORECASTER = None


class TimesFMWrapper:
    """Wrapper class for Google TimesFM 3.0 (330M parameters) Foundation Model."""

    def __init__(self, pretrained_path: str = "google/timesfm-3.0-pytorch", device: Optional[str] = None):
        self.pretrained_path = pretrained_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.forecaster = None
        self._load_model()

    def _load_model(self):
        global _GLOBAL_TIMESFM_FORECASTER
        if _GLOBAL_TIMESFM_FORECASTER is not None:
            self.forecaster = _GLOBAL_TIMESFM_FORECASTER
            return

        from timesfm import TimesFM3Forecaster

        # Try CUDA first, fall back to CPU if unsupported compute capability
        try:
            if self.device == 'cuda':
                # Test dummy tensor on cuda
                t = torch.zeros(1, device='cuda')
                _ = t + 1
            self.forecaster = TimesFM3Forecaster.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_path,
                device=self.device
            )
        except Exception as e:
            print(f"[TimesFM] Failed loading on device={self.device} ({e}). Falling back to CPU.")
            self.device = 'cpu'
            self.forecaster = TimesFM3Forecaster.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_path,
                device='cpu'
            )

        _GLOBAL_TIMESFM_FORECASTER = self.forecaster

    def forecast(
        self,
        df_history: pd.DataFrame,
        future_dates: List[str],
        horizon: int = 8,
        freq: str = 'W',
        confidence_level: float = 0.80
    ) -> Dict[str, Any]:
        """
        Runs zero-shot inference with Google TimesFM 3.0.
        Inputs:
            df_history: DataFrame with historical 'ds' and 'y'
            future_dates: List of string dates for future steps
            horizon: forecast horizon (e.g. 8)
            confidence_level: e.g. 0.80 (uses 10th and 90th quantiles)
        """
        start_time = time.time()
        
        context_values = df_history['y'].values.astype(float)
        
        # Run TimesFM 3.0 prediction
        # return_quantiles=True returns 9 quantiles [0.1, 0.2, ..., 0.9]
        res = self.forecaster.predict(
            context=context_values,
            horizon=horizon,
            return_quantiles=True,
            make_positive=True,
            use_znorm=True
        )

        forecast_median = np.maximum(res.forecast[:horizon], 0.0)
        
        # Quantiles extraction
        if res.quantiles is not None and res.quantiles.shape[-1] >= 9:
            # 9 quantiles: [q10, q20, q30, q40, q50, q60, q70, q80, q90]
            if confidence_level >= 0.80:
                q_low = res.quantiles[:horizon, 0]  # 10%
                q_high = res.quantiles[:horizon, 8] # 90%
            else:
                q_low = res.quantiles[:horizon, 1]  # 20%
                q_high = res.quantiles[:horizon, 7] # 80%
            
            ci_lower = np.maximum(q_low, 0.0)
            ci_upper = np.maximum(q_high, forecast_median)
        else:
            # Fallback heuristic uncertainty if quantiles are missing
            std_est = np.std(context_values[-min(len(context_values), 12):])
            ci_lower = np.maximum(forecast_median - 1.28 * std_est, 0.0)
            ci_upper = forecast_median + 1.28 * std_est

        elapsed = time.time() - start_time

        return {
            'model_name': 'Google TimesFM 3.0',
            'dates': future_dates[:horizon],
            'forecast': forecast_median.astype(float),
            'ci_lower': ci_lower.astype(float),
            'ci_upper': ci_upper.astype(float),
            'quantiles': res.quantiles[:horizon, :] if res.quantiles is not None else None,
            'elapsed_time_sec': round(elapsed, 3),
            'device_used': self.device,
            'context_length': len(context_values),
            'horizon': horizon
        }
