from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List

import pandas as pd

from .forecasting import ForecastResult, forecast_next_period


def compute_time_series(transactions: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Aggregate sales over time for charting."""
    ts = (
        transactions.set_index("transaction_date")
        .resample(freq, label="left", closed="left")
        .agg(units=("units", "sum"), revenue=("gross_revenue", "sum"))
        .reset_index()
    )
    ts.rename(columns={"transaction_date": "period_start"}, inplace=True)
    ts["revenue"] = ts["revenue"].round(2)
    return ts


def compute_monthly_rollup(transactions: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        transactions.set_index("transaction_date")
        .resample("M")
        .agg(units=("units", "sum"), revenue=("gross_revenue", "sum"))
        .reset_index()
    )
    monthly["month_start"] = monthly["transaction_date"].dt.to_period("M")
    return monthly[["month_start", "units", "revenue"]]


def _growth(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return round((current - previous) / previous * 100, 2)


def compute_summary_metrics(transactions: pd.DataFrame) -> Dict[str, float]:
    total_revenue = transactions["gross_revenue"].sum()
    total_units = transactions["units"].sum()
    avg_unit_price = transactions["unit_price"].mean()
    gross_margin_pct = 38.0  # mocked constant for demo purposes

    end_date = transactions["transaction_date"].max()
    recent_start = end_date - pd.Timedelta(days=30)
    prev_start = recent_start - pd.Timedelta(days=30)

    recent = transactions[transactions["transaction_date"] > recent_start]
    prev = transactions[
        (transactions["transaction_date"] > prev_start)
        & (transactions["transaction_date"] <= recent_start)
    ]

    recent_revenue = recent["gross_revenue"].sum()
    prev_revenue = prev["gross_revenue"].sum()
    recent_units = recent["units"].sum()
    prev_units = prev["units"].sum()

    return {
        "total_revenue": round(total_revenue, 2),
        "total_units": int(total_units),
        "avg_unit_price": round(avg_unit_price, 2),
        "gross_margin_pct": gross_margin_pct,
        "revenue_30d": round(recent_revenue, 2),
        "revenue_30d_growth": _growth(recent_revenue, prev_revenue),
        "units_30d": int(recent_units),
        "units_30d_growth": _growth(recent_units, prev_units),
    }


def compute_mix(transactions: pd.DataFrame, column: str) -> List[Dict[str, object]]:
    grouped = (
        transactions.groupby(column)["gross_revenue"]
        .sum()
        .reset_index(name="revenue")
        .sort_values("revenue", ascending=False)
    )
    total = grouped["revenue"].sum() or 1.0
    grouped["share"] = (grouped["revenue"] / total * 100).round(2)
    return grouped.to_dict(orient="records")


def channel_mix(transactions: pd.DataFrame) -> List[Dict[str, object]]:
    return compute_mix(transactions, "channel")


def region_mix(transactions: pd.DataFrame) -> List[Dict[str, object]]:
    return compute_mix(transactions, "region")


def compute_forecast(transactions: pd.DataFrame) -> ForecastResult:
    monthly = compute_monthly_rollup(transactions)
    return forecast_next_period(monthly)


def build_mock_additional_insights(
    product_name: str, metrics: Dict[str, float], forecast: ForecastResult
) -> List[Dict[str, str]]:
    insights = [
        {
            "title": "Cross-Sell Opportunity",
            "body": (
                f"Merchandising teams can bundle {product_name} with complementary accessories to grow average order value. "
                "Position curated looks on product detail pages and at checkout to save shoppers time."
            ),
        },
        {
            "title": "Inventory Planning",
            "body": (
                "Supply planners should maintain roughly two weeks of forward cover in regional hubs to stay ahead of demand spikes. "
                "Prioritise store-to-store transfers before triggering new buys to protect margin."
            ),
        },
        {
            "title": "Marketing Insight",
            "body": (
                "Growth marketing can re-engage lapsed purchasers with a limited-time offer and refreshed creative. "
                "Highlight comfort and versatility across channels to reinforce value."
            ),
        },
    ]
    return insights


def _hash_product(product_name: str) -> int:
    return int(hashlib.sha1(product_name.encode("utf-8")).hexdigest(), 16)


def generate_mock_forecast(product_name: str) -> ForecastResult:
    seed = _hash_product(product_name)
    revenue = 40000 + (seed % 25000)
    revenue_ci = max(revenue * 0.12, 3500)
    units = 900 + (seed % 350)
    units_ci = max(int(units * 0.15), 45)

    return ForecastResult(
        period_label="Next 30 Days",
        revenue=revenue,
        revenue_low=revenue - revenue_ci,
        revenue_high=revenue + revenue_ci,
        units=units,
        units_low=max(units - units_ci, 0),
        units_high=units + units_ci,
        method="mock_projection",
    )


def generate_mock_segments(product_name: str) -> List[Dict[str, str]]:
    seed = _hash_product(product_name)
    base_shares = [
        45 + (seed % 10),
        30 + ((seed // 3) % 10),
        100,
    ]
    total = sum(base_shares)
    shares = [int(round(value / total * 100)) for value in base_shares]
    adjustment = 100 - sum(shares)
    shares[0] += adjustment

    return [
        {
            "segment": "Digital Loyalists",
            "share": f"{shares[0]}%",
            "traits": "Frequent shoppers who engage with push alerts and in-app exclusives.",
        },
        {
            "segment": "Store Stylists",
            "share": f"{shares[1]}%",
            "traits": "Prefer in-store styling sessions and tactile experiences before purchasing.",
        },
        {
            "segment": "Seasonal Gifters",
            "share": f"{shares[2]}%",
            "traits": "Buy around key holidays; respond well to curated gift guides and bundles.",
        },
    ]
