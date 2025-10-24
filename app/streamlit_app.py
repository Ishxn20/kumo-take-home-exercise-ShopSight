from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

if __package__ is None:  # Allow `streamlit run app/streamlit_app.py`
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.data_loader import (  # noqa: E402
    filter_transactions_by_product,
    get_product_details,
    load_product_catalog,
    search_products,
)
from app.insights import (  # noqa: E402
    build_mock_additional_insights,
    channel_mix,
    compute_summary_metrics,
    compute_time_series,
    generate_mock_forecast,
    generate_mock_segments,
    region_mix,
)
from app.llm import (  # noqa: E402
    InsightBundle,
    answer_question,
    summarise_insights,
    summarise_trend,
)


def _prepare_product_selection() -> str:
    catalog = load_product_catalog()
    default_product_id = catalog.iloc[0]["product_id"]

    with st.sidebar:
        st.header("🔎 Product Lookup")
        query = st.text_input(
            "Search by name or ID",
            value="",
            placeholder="e.g. Jade denim, 706016001",
        )
        matches = search_products(query, limit=6)

        if not matches:
            st.info("No matches found — showing the default product instead.")
            return default_product_id

        labels = {
            f"{match['product_name']} · {match.get('descriptor', match['category'])} · #{match['product_id']}": match[
                "product_id"
            ]
            for match in matches
        }
        selected_label = st.selectbox("Select a product", list(labels.keys()))
        st.markdown(
            "Tip: try searches like `Ultraboost`, `Hoka`, or a product ID such as `P-3050`."
        )
        return labels[selected_label]


def _render_kpis(metrics: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lifetime Revenue", f"${metrics['total_revenue']:,.0f}")
    col2.metric("Lifetime Units", f"{metrics['total_units']:,}")
    col3.metric(
        "Last 30 Days Revenue",
        f"${metrics['revenue_30d']:,.0f}",
        delta=f"{metrics['revenue_30d_growth']:.2f}%",
    )
    col4.metric(
        "Last 30 Days Units",
        f"{metrics['units_30d']:,}",
        delta=f"{metrics['units_30d_growth']:.2f}%",
    )


def _render_sales_chart(time_series: pd.DataFrame) -> None:
    ts_df = time_series.copy()
    ts_df["period_start"] = pd.to_datetime(ts_df["period_start"])

    base = alt.Chart(ts_df).encode(
        x=alt.X("period_start:T", title="Week Starting"),
        tooltip=[
            alt.Tooltip("period_start:T", title="Week"),
            alt.Tooltip("revenue:Q", title="Revenue ($)", format=",.0f"),
            alt.Tooltip("units:Q", title="Units"),
        ],
    )

    revenue_line = base.mark_line(color="#2f7de1", strokeWidth=2).encode(
        y=alt.Y("revenue:Q", axis=alt.Axis(title="Revenue ($)", titleColor="#2f7de1"))
    )
    units_area = base.mark_area(opacity=0.25, color="#f39c12").encode(
        y=alt.Y(
            "units:Q",
            axis=alt.Axis(title="Units Sold", titleColor="#f39c12"),
        )
    )

    chart = alt.layer(revenue_line, units_area).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)


def _render_mix_section(channel: list, region: list) -> None:
    mix_col1, mix_col2 = st.columns(2)
    if channel:
        channel_df = pd.DataFrame(channel)
        chart = (
            alt.Chart(channel_df)
            .mark_bar(color="#7e62d9")
            .encode(
                x=alt.X("share:Q", title="Revenue Share (%)"),
                y=alt.Y("channel:N", title="Channel", sort="-x"),
                tooltip=[
                    alt.Tooltip("channel:N", title="Channel"),
                    alt.Tooltip("revenue:Q", title="Revenue ($)", format=",.0f"),
                    alt.Tooltip("share:Q", title="Share (%)"),
                ],
            )
        )
        mix_col1.subheader("Channel Mix")
        mix_col1.altair_chart(chart, use_container_width=True)
    else:
        mix_col1.info("No channel data available for this selection.")

    if region:
        region_df = pd.DataFrame(region)
        chart = (
            alt.Chart(region_df)
            .mark_bar(color="#1abc9c")
            .encode(
                x=alt.X("share:Q", title="Revenue Share (%)"),
                y=alt.Y("region:N", title="Region", sort="-x"),
                tooltip=[
                    alt.Tooltip("region:N", title="Region"),
                    alt.Tooltip("revenue:Q", title="Revenue ($)", format=",.0f"),
                    alt.Tooltip("share:Q", title="Share (%)"),
                ],
            )
        )
        mix_col2.subheader("Regional Mix")
        mix_col2.altair_chart(chart, use_container_width=True)
    else:
        mix_col2.info("No regional data available for this selection.")


def main() -> None:
    st.set_page_config(
        page_title="ShopSight Prototype",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Report a bug": "mailto:demo-feedback@example.com",
        },
    )
    st.markdown(
        """
        <style>
            [data-testid="stDeployButton"] {display: none !important;}
            div[data-testid="stToolbar"] button[data-testid="baseButton-secondary"] {display: none !important;}
            div[data-testid="stToolbar"] a[title="Deploy"] {display: none !important;}
            div[data-testid="stToolbar"] button[title="Deploy"] {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("ShopSight")

    product_id = _prepare_product_selection()
    product = get_product_details(product_id)
    if product is None:
        st.error("Unable to load product details.")
        st.stop()

    transactions = filter_transactions_by_product(product_id)
    if transactions.empty:
        st.warning("No transactions available for this product in the sample data.")
        st.stop()

    segments = generate_mock_segments(product["product_name"])
    metrics = compute_summary_metrics(transactions)
    time_series = compute_time_series(transactions)
    weekly_points = time_series.to_dict(orient="records")
    forecast = generate_mock_forecast(product["product_name"])
    forecast_dict = forecast.as_dict()
    channel = channel_mix(transactions)
    region = region_mix(transactions)
    fallback_cards = build_mock_additional_insights(
        product["product_name"], metrics, forecast
    )
    trend_commentary = summarise_trend(
        product["product_name"], weekly_points, metrics
    )
    insight_bundle: InsightBundle = summarise_insights(
        product["product_name"], metrics, forecast_dict, segments, fallback_cards
    )
    narrative = insight_bundle.summary
    recommended_cards = insight_bundle.actions or fallback_cards

    st.subheader(f"{product['product_name']} — {product['brand']}")
    st.write(
        f"Category: **{product['category']}** · Average selling price "
        f"(sample period): **${product['avg_price']:.2f}** · "
        f"Sales observed between {product['first_sale']} and {product['last_sale']}."
    )

    _render_kpis(metrics)

    st.markdown("### Sales Momentum")
    _render_sales_chart(time_series)
    st.caption(trend_commentary)

    mix_section = st.container()
    with mix_section:
        _render_mix_section(channel, region)

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("### Customer Segments")
        if segments:
            seg_df = pd.DataFrame(segments)
            st.dataframe(seg_df, use_container_width=True, hide_index=True)
        else:
            st.info("Segments are mocked for this prototype and not available here.")

        st.markdown("### Forecast")
        st.metric(
            f"Next Month Revenue ({forecast_dict['period']})",
            f"${forecast_dict['forecast_revenue']:,.0f}",
            delta=f"{forecast_dict['forecast_units']} units expected",
        )
        st.caption(
            f"68% interval: ${forecast_dict['forecast_revenue_low']:,.0f} "
            f"– ${forecast_dict['forecast_revenue_high']:,.0f} revenue."
        )

    with right_col:
        st.markdown("### Narrative Summary")
        st.write(narrative)

        st.markdown("### Recommended Actions")
        for card in recommended_cards:
            st.success(f"**{card['title']}** — {card['body']}")

    st.markdown("### Ask ShopSight Assistant")
    chat_state_key = f"chat_history_{product_id}"
    if chat_state_key not in st.session_state:
        st.session_state[chat_state_key] = []
    chat_history = st.session_state[chat_state_key]

    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask anything about this product's performance")
    if user_prompt:
        st.session_state[chat_state_key].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        assistant_reply = answer_question(
            product["product_name"],
            user_prompt,
            metrics,
            forecast_dict,
            segments,
            weekly_points,
            channel,
            region,
        )
        st.session_state[chat_state_key].append(
            {"role": "assistant", "content": assistant_reply}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
