from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    period_label: str
    revenue: float
    revenue_low: float
    revenue_high: float
    units: int
    units_low: int
    units_high: int
    method: str = "weighted_trend"

    def as_dict(self) -> Dict[str, object]:
        return {
            "period": self.period_label,
            "forecast_revenue": round(self.revenue, 2),
            "forecast_revenue_low": round(self.revenue_low, 2),
            "forecast_revenue_high": round(self.revenue_high, 2),
            "forecast_units": int(self.units),
            "forecast_units_low": int(self.units_low),
            "forecast_units_high": int(self.units_high),
            "method": self.method,
        }


def _make_confidence_interval(series: Iterable[float], prediction: float) -> float:
    arr = np.asarray(list(series), dtype=float)
    if arr.size < 2:
        return prediction * 0.1 or 1.0
    residuals = arr - arr.mean()
    std = residuals.std()
    return max(std, prediction * 0.1)


def forecast_next_period(monthly: pd.DataFrame) -> ForecastResult:
    """
    Predict the next month's revenue and units with a lightweight weighted trend.

    The approach combines a linear trend (degree 1 polyfit) with a rolling mean on
    the last three months. This balances responsiveness and stability for demo data.
    """
    if monthly.empty:
        raise ValueError("No data provided for forecasting.")

    monthly = monthly.sort_values("month_start").reset_index(drop=True)
    periods = monthly["month_start"]
    revenue = monthly["revenue"].to_numpy(dtype=float)
    units = monthly["units"].to_numpy(dtype=float)

    x = np.arange(len(monthly), dtype=float)

    if len(monthly) >= 2:
        revenue_trend = np.poly1d(np.polyfit(x, revenue, deg=1))
        units_trend = np.poly1d(np.polyfit(x, units, deg=1))
    else:
        revenue_trend = lambda idx: revenue[-1]  # type: ignore[assignment]
        units_trend = lambda idx: units[-1]  # type: ignore[assignment]

    trend_weight = 0.65
    mean_weight = 0.35

    revenue_mean = revenue[-3:].mean() if len(revenue) >= 3 else revenue.mean()
    units_mean = units[-3:].mean() if len(units) >= 3 else units.mean()

    next_index = len(monthly)
    revenue_pred = trend_weight * float(revenue_trend(next_index)) + mean_weight * revenue_mean
    units_pred = trend_weight * float(units_trend(next_index)) + mean_weight * units_mean

    revenue_pred = max(revenue_pred, 0.0)
    units_pred = max(units_pred, 0.0)

    revenue_ci = _make_confidence_interval(revenue[-6:], revenue_pred)
    units_ci = _make_confidence_interval(units[-6:], units_pred)

    last_period = pd.Period(periods.iloc[-1], freq="M")
    next_period = (last_period + 1).strftime("%Y-%m")

    return ForecastResult(
        period_label=next_period,
        revenue=revenue_pred,
        revenue_low=max(revenue_pred - revenue_ci, 0.0),
        revenue_high=revenue_pred + revenue_ci,
        units=round(units_pred),
        units_low=max(round(units_pred - units_ci), 0),
        units_high=round(units_pred + units_ci),
    )
