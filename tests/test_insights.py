from app.data_loader import filter_transactions_by_product, load_product_catalog
from app.insights import (
    build_mock_additional_insights,
    channel_mix,
    compute_summary_metrics,
    compute_time_series,
    generate_mock_forecast,
)


def _pick_product_id() -> str:
    catalog = load_product_catalog()
    return catalog.iloc[0]["product_id"]


def test_summary_metrics_keys():
    product_id = _pick_product_id()
    tx = filter_transactions_by_product(product_id)
    metrics = compute_summary_metrics(tx)
    expected = {
        "total_revenue",
        "total_units",
        "avg_unit_price",
        "gross_margin_pct",
        "revenue_30d",
        "revenue_30d_growth",
        "units_30d",
        "units_30d_growth",
    }
    assert expected.issubset(metrics.keys())


def test_time_series_not_empty():
    product_id = _pick_product_id()
    tx = filter_transactions_by_product(product_id)
    ts = compute_time_series(tx)
    assert not ts.empty
    assert {"period_start", "units", "revenue"}.issubset(ts.columns)


def test_forecast_result_structure():
    product_id = _pick_product_id()
    forecast = generate_mock_forecast(product_id).as_dict()
    required = {
        "period",
        "forecast_revenue",
        "forecast_revenue_low",
        "forecast_revenue_high",
        "forecast_units",
        "forecast_units_low",
        "forecast_units_high",
    }
    assert required.issubset(forecast.keys())


def test_mock_insights_generate_three_cards():
    product_id = _pick_product_id()
    tx = filter_transactions_by_product(product_id)
    metrics = compute_summary_metrics(tx)
    forecast = generate_mock_forecast("Example Product")
    cards = build_mock_additional_insights("Example Product", metrics, forecast)
    assert len(cards) >= 3


def test_channel_mix_has_share():
    product_id = _pick_product_id()
    tx = filter_transactions_by_product(product_id)
    mix = channel_mix(tx)
    assert all("share" in entry for entry in mix)
