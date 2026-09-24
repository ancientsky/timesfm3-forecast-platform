"""
Google TimesFM 3.0 Model Wrapper for Epidemic Time Series Forecasting.
Loads google/timesfm-3.0-pytorch foundation model, handles tensor transformations,
quantile interval extraction, and zero-shot inference with memory optimization and cloud resilience.
"""

from typing import Dict, Any, Optional, List
import os
import sys
import time
import json
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import scipy.stats as stats

# Suppress minor PyTorch / HF warnings
warnings.filterwarnings('ignore')

_GLOBAL_TIMESFM_FORECASTER = None


def get_hf_token() -> Optional[str]:
    """Retrieves Hugging Face Access Token from environment or Streamlit secrets."""
    # 1. Environment variables
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()

    # 2. Streamlit Cloud secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "HF_TOKEN" in st.secrets and st.secrets["HF_TOKEN"]:
                token = str(st.secrets["HF_TOKEN"]).strip()
            elif "HUGGING_FACE_HUB_TOKEN" in st.secrets and st.secrets["HUGGING_FACE_HUB_TOKEN"]:
                token = str(st.secrets["HUGGING_FACE_HUB_TOKEN"]).strip()
    except Exception:
        pass

    if token and token.strip():
        # Export to environment for all downstream huggingface_hub tools
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        return token

    return None


def _load_timesfm_memory_efficient(pretrained_path: str = "google/timesfm-3.0-pytorch", device: str = "cpu", token: Optional[str] = None):
    """
    Loads TimesFM 3.0 with minimal peak memory footprint (~1.9 GB vs ~3.1 GB standard),
    preventing Out-Of-Memory (OOM) kills on cloud platforms like Streamlit Community Cloud (2.7GB limit).
    """
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from timesfm3 import model as torch_model_lib
    from timesfm3.timesfm3_forecaster import TimesFM3Forecaster, _ModelConfig

    cfg_path = hf_hub_download(pretrained_path, "config.json", token=token)
    weights_path = hf_hub_download(pretrained_path, "model.safetensors", token=token)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. Instantiate on meta device (0 MB RAM overhead)
    with torch.device("meta"):
        m = torch_model_lib.TimesFM3Torch(**cfg)

    # 2. Fix the 20 meta timescale buffers in RotaryPositionEmbedding
    for mod in m.modules():
        if hasattr(mod, "timescale") and hasattr(mod.timescale, "is_meta") and mod.timescale.is_meta:
            fraction = 2.0 * torch.arange(mod.timescale.shape[0], dtype=torch.float32) / (mod.timescale.shape[0] * 2)
            mod.timescale = mod.min_timescale * (mod.max_timescale / mod.min_timescale) ** fraction

    # 3. Load state dict directly to CPU
    sd = safetensors.torch.load_file(weights_path, device="cpu")

    # 4. Directly assign weights without duplication
    m.load_state_dict(sd, assign=True)
    del sd
    gc.collect()

    m.eval()
    if device != "cpu" and torch.cuda.is_available():
        m.to(device)

    # 5. Build forecaster instance
    fc = TimesFM3Forecaster.__new__(TimesFM3Forecaster)
    median_q_idx = len(m.quantiles) // 2
    fc.config = _ModelConfig(
        checkpoint_path=pretrained_path,
        device=device,
        input_patch_length=m.input_patch_len,
        output_patch_length=m.output_patch_len,
        quantiles=list(m.quantiles),
        median_quantile_index=median_q_idx,
        residual_block_config=m.residual_block_config,
        transformer_config=m.transformer_config,
        use_variate_attention=m.use_variate_attention,
        value_clip=m.value_clip,
        use_stitching=m.use_stitching,
        use_linear_detrending=m.use_linear_detrending,
        linear_detrending_threshold=m.linear_detrending_threshold,
        use_iterative_cpm_revin=m.use_iterative_cpm_revin,
        use_frozen_running_stats=m.use_frozen_running_stats,
        input_transform=m.input_transform,
        token=token,
    )
    fc.device = torch.device(device)
    fc.model = m
    return fc


class TimesFMWrapper:
    """Wrapper class for Google TimesFM 3.0 (330M parameters) Foundation Model."""

    def __init__(self, pretrained_path: str = "google/timesfm-3.0-pytorch", device: Optional[str] = None):
        self.pretrained_path = pretrained_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.token = get_hf_token()
        self.forecaster = None
        self.is_fallback = False
        self.load_error = None
        self._load_model()

    def _load_model(self):
        global _GLOBAL_TIMESFM_FORECASTER
        if _GLOBAL_TIMESFM_FORECASTER is not None:
            self.forecaster = _GLOBAL_TIMESFM_FORECASTER
            return

        # Attempt 1: Memory-efficient loading (keeps peak RAM < 2.0 GB)
        try:
            target_device = self.device
            if target_device == 'cuda':
                # Test dummy tensor on cuda
                t = torch.zeros(1, device='cuda')
                _ = t + 1
            self.forecaster = _load_timesfm_memory_efficient(
                pretrained_path=self.pretrained_path,
                device=target_device,
                token=self.token
            )
            _GLOBAL_TIMESFM_FORECASTER = self.forecaster
            return
        except Exception as e1:
            print(f"[TimesFM] Memory-efficient loading failed ({e1}). Attempting standard loader...")

        # Attempt 2: Standard loader from timesfm library
        try:
            from timesfm import TimesFM3Forecaster
            self.forecaster = TimesFM3Forecaster.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_path,
                device='cpu',
                token=self.token
            )
            self.device = 'cpu'
            _GLOBAL_TIMESFM_FORECASTER = self.forecaster
            return
        except Exception as e2:
            print(f"[TimesFM] All TimesFM 3.0 loading methods failed ({e2}). Entering graceful fallback mode.")
            self.load_error = f"{type(e2).__name__}: {str(e2)}"
            self.is_fallback = True
            self.forecaster = None

    def _run_heuristic_fallback(
        self,
        df_history: pd.DataFrame,
        future_dates: List[str],
        horizon: int = 8,
        confidence_level: float = 0.80
    ) -> Dict[str, Any]:
        """
        High-fidelity heuristic fallback when model weights cannot be loaded in constrained cloud environments.
        Uses damped Holt-Winters trend modeling with empirical quantile extrapolation.
        """
        start_time = time.time()
        context_values = df_history['y'].values.astype(float)
        n = len(context_values)

        # Baseline forecast using Exponential Smoothing with trend dampening
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            # Use last up to 52 points for epidemic stability
            fit_series = context_values[-min(n, 52):]
            model = ExponentialSmoothing(fit_series, trend='add', seasonal=None, damped_trend=True).fit()
            forecast_median = np.maximum(model.forecast(horizon), 0.0)
            residuals = fit_series - model.fittedvalues
            std_err = float(np.std(residuals)) if len(residuals) > 1 else max(float(np.std(context_values)), 1.0)
        except Exception:
            # Fallback to rolling linear trend
            window = min(n, 8)
            y_recent = context_values[-window:]
            x = np.arange(window)
            slope, intercept = np.polyfit(x, y_recent, 1)
            # Dampen slope over future horizon
            damp_factors = 0.85 ** np.arange(1, horizon + 1)
            last_val = context_values[-1]
            forecast_median = np.maximum(last_val + slope * np.arange(1, horizon + 1) * damp_factors, 0.0)
            std_err = max(float(np.std(context_values[-min(n, 12):])), 1.0)

        # Standard normal quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        q_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        z_scores = stats.norm.ppf(q_levels)

        # Horizon widening uncertainty factor: uncertainty grows with sqrt(h)
        horizon_factors = np.sqrt(np.arange(1, horizon + 1))
        quantiles_matrix = np.zeros((horizon, 9), dtype=float)
        for h_idx in range(horizon):
            h_std = std_err * horizon_factors[h_idx]
            for q_idx, z in enumerate(z_scores):
                quantiles_matrix[h_idx, q_idx] = max(0.0, forecast_median[h_idx] + z * h_std)

        # Extract CI lower and upper matching requested confidence level
        if confidence_level >= 0.80:
            ci_lower = quantiles_matrix[:, 0]  # 10%
            ci_upper = quantiles_matrix[:, 8]  # 90%
        else:
            ci_lower = quantiles_matrix[:, 1]  # 20%
            ci_upper = quantiles_matrix[:, 7]  # 80%

        elapsed = time.time() - start_time

        return {
            'model_name': 'Google TimesFM 3.0 (雲端備援 / Cloud Fallback)',
            'dates': future_dates[:horizon],
            'forecast': forecast_median.astype(float),
            'ci_lower': ci_lower.astype(float),
            'ci_upper': ci_upper.astype(float),
            'quantiles': quantiles_matrix,
            'elapsed_time_sec': round(elapsed, 3),
            'device_used': 'cpu (heuristic fallback)',
            'context_length': n,
            'horizon': horizon,
            'is_fallback': True,
            'fallback_reason': self.load_error or 'Cloud memory/rate limit constraint'
        }

    def forecast(
        self,
        df_history: pd.DataFrame,
        future_dates: List[str],
        horizon: int = 8,
        freq: str = 'W',
        confidence_level: float = 0.80
    ) -> Dict[str, Any]:
        """
        Runs zero-shot inference with Google TimesFM 3.0, or gracefully runs fallback.
        """
        if self.forecaster is None:
            return self._run_heuristic_fallback(
                df_history=df_history,
                future_dates=future_dates,
                horizon=horizon,
                confidence_level=confidence_level
            )

        start_time = time.time()
        context_values = df_history['y'].values.astype(float)

        try:
            # Run TimesFM 3.0 prediction
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
                if confidence_level >= 0.80:
                    q_low = res.quantiles[:horizon, 0]  # 10%
                    q_high = res.quantiles[:horizon, 8] # 90%
                else:
                    q_low = res.quantiles[:horizon, 1]  # 20%
                    q_high = res.quantiles[:horizon, 7] # 80%

                ci_lower = np.maximum(q_low, 0.0)
                ci_upper = np.maximum(q_high, forecast_median)
            else:
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
                'horizon': horizon,
                'is_fallback': False
            }
        except Exception as e:
            print(f"[TimesFM] Inference exception ({e}). Falling back to heuristic baseline.")
            self.load_error = f"InferenceError: {str(e)}"
            return self._run_heuristic_fallback(
                df_history=df_history,
                future_dates=future_dates,
                horizon=horizon,
                confidence_level=confidence_level
            )
