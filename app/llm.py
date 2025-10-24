from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


DEFAULT_MODEL = os.getenv("SHOP_SIGHT_OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class InsightBundle:
    summary: str
    actions: List[Dict[str, str]]


def _call_openai(prompt: str, *, max_tokens: int = 400, temperature: float = 0.4) -> Optional[str]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=DEFAULT_MODEL,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return response.output_text.strip()
    except Exception:
        return None


def _fallback_summary(
    product_name: str,
    metrics: Dict[str, float],
    forecast: Dict[str, object],
    segments: Iterable[Dict[str, str]],
) -> str:
    top_segment = next(iter(segments), {"segment": "Core Buyers", "share": "n/a"})
    return (
        f"{product_name} generated ${metrics['total_revenue']:,} in lifetime revenue, "
        f"with ${metrics['revenue_30d']:,} in the last 30 days "
        f"({metrics['revenue_30d_growth']}% vs prior period). "
        f"Average selling price holds at ${metrics['avg_unit_price']:.2f}. "
        f"Next month we expect roughly ${forecast['forecast_revenue']:,} in revenue "
        f"({forecast['forecast_units']} units). Focus engagement on the "
        f"{top_segment['segment']} segment ({top_segment.get('share', 'n/a')} of buyers) "
        "to maintain momentum."
    )


def summarise_insights(
    product_name: str,
    metrics: Dict[str, float],
    forecast: Dict[str, object],
    segments: List[Dict[str, str]],
    fallback_actions: List[Dict[str, str]],
) -> InsightBundle:
    prompt = json.dumps(
        {
            "instruction": (
                "You are an executive retail analytics assistant. Analyse the product performance "
                "data and produce a concise summary plus up to three recommended actions. "
                "Return a JSON object with keys 'summary' (string) and 'actions' (list of "
                "objects with 'title' and 'body'). Ground your response in the numbers provided."
            ),
            "context": {
                "product": product_name,
                "metrics": metrics,
                "forecast": forecast,
                "segments": segments,
            },
            "style": {
                "summary_length": "2 paragraphs max",
                "actions_expectation": "actionable, specific, prioritised",
            },
        },
        indent=2,
    )
    actions = fallback_actions
    summary = _fallback_summary(product_name, metrics, forecast, segments)

    raw_response = _call_openai(prompt)
    if raw_response:
        try:
            payload = json.loads(raw_response)
            summary = payload.get("summary", summary)
            parsed_actions = [
                {"title": action.get("title", "Recommendation"), "body": action.get("body", "")}
                for action in payload.get("actions", [])[:3]
                if isinstance(action, dict)
            ]
            if parsed_actions:
                actions = parsed_actions
        except json.JSONDecodeError:
            summary = raw_response

    return InsightBundle(summary=summary.strip(), actions=actions)


def summarise_trend(
    product_name: str,
    weekly_points: Sequence[Dict[str, object]],
    metrics: Dict[str, float],
) -> str:
    """
    Produce a short commentary on the recent weekly sales momentum.
    weekly_points should contain dicts with 'period_start', 'revenue', and 'units'.
    """
    if not weekly_points:
        return "Sales momentum commentary unavailable."

    serialisable_points = []
    for point in weekly_points[-12:]:
        serialisable_points.append(
            {
                "period_start": str(point.get("period_start")),
                "revenue": point.get("revenue"),
                "units": point.get("units"),
            }
        )

    prompt = json.dumps(
        {
            "instruction": (
                "You are analysing weekly sales data. Highlight trend changes, spikes, or slowdowns "
                "in 2 sentences. Use the data provided and avoid generic statements."
            ),
            "product": product_name,
            "recent_weeks": serialisable_points,
            "latest_metrics": {
                "revenue_30d": metrics.get("revenue_30d"),
                "revenue_30d_growth": metrics.get("revenue_30d_growth"),
                "units_30d": metrics.get("units_30d"),
            },
        },
        indent=2,
    )

    commentary = _call_openai(prompt, max_tokens=200, temperature=0.3)
    if commentary:
        return commentary.strip()

    growth = metrics.get("revenue_30d_growth", 0.0)
    return (
        f"Revenue over the past 30 days is tracking {growth:.1f}% versus the prior month; "
        "momentum looks steady with minor weekly fluctuations."
    )


def answer_question(
    product_name: str,
    question: str,
    metrics: Dict[str, float],
    forecast: Dict[str, object],
    segments: List[Dict[str, str]],
    recent_weeks: Sequence[Dict[str, object]],
    channel_mix: List[Dict[str, object]],
    region_mix: List[Dict[str, object]],
) -> str:
    serialisable_weeks = []
    for point in recent_weeks[-12:]:
        serialisable_weeks.append(
            {
                "period_start": str(point.get("period_start")),
                "revenue": point.get("revenue"),
                "units": point.get("units"),
            }
        )

    prompt = json.dumps(
        {
            "instruction": (
                "You are an analytics copilot inside a commerce dashboard. "
                "Answer the user's question using the context provided. "
                "If data is insufficient, acknowledge the limitation."
            ),
            "context": {
                "product": product_name,
                "metrics": metrics,
                "forecast": forecast,
                "segments": segments,
                "recent_weeks": serialisable_weeks,
                "channel_mix": channel_mix,
                "region_mix": region_mix,
            },
            "question": question,
        },
        indent=2,
    )
    response = _call_openai(prompt, max_tokens=450, temperature=0.4)
    if response:
        return response.strip()
    return _fallback_answer(
        product_name,
        question,
        metrics,
        forecast,
        segments,
        recent_weeks,
        channel_mix,
        region_mix,
    )


def _fallback_answer(
    product_name: str,
    question: str,
    metrics: Dict[str, float],
    forecast: Dict[str, object],
    segments: Sequence[Dict[str, str]],
    recent_weeks: Sequence[Dict[str, object]],
    channel_mix: Sequence[Dict[str, object]],
    region_mix: Sequence[Dict[str, object]],
) -> str:
    question_lower = (question or "").lower()
    forecast_rev = forecast.get("forecast_revenue")
    forecast_units = forecast.get("forecast_units")

    def last_week_delta() -> str:
        if len(recent_weeks) < 2:
            return ""
        try:
            latest = float(recent_weeks[-1]["revenue"])
            prev = float(recent_weeks[-2]["revenue"])
        except (KeyError, TypeError, ValueError):
            return ""
        delta = latest - prev
        pct = (delta / prev * 100) if prev else 0.0
        direction = "up" if delta >= 0 else "down"
        return (
            f"Last week revenue was ${latest:,.0f}, {direction} {abs(pct):.1f}% versus the prior week."
        )

    def top_segment() -> str:
        if not segments:
            return ""
        seg = segments[0]
        return (
            f"The leading buyer cohort is {seg.get('segment')} at {seg.get('share')} share."
        )

    def top_channel() -> str:
        if not channel_mix:
            return ""
        sorted_mix = sorted(
            channel_mix, key=lambda entry: entry.get("revenue", 0.0), reverse=True
        )
        top = sorted_mix[0]
        share = top.get("revenue", 0.0) / max(
            sum(entry.get("revenue", 0.0) for entry in channel_mix), 1.0
        ) * 100
        return (
            f"{top.get('channel')} contributes about {share:.0f}% of revenue."
        )

    revenue_growth = metrics.get("revenue_30d_growth", 0.0)
    base_summary = (
        f"{product_name} generated ${metrics.get('revenue_30d', 0.0):,.0f} in the last 30 days "
        f"({revenue_growth:.1f}% vs. the prior month). "
    )

    if any(keyword in question_lower for keyword in ("trend", "momentum", "trajectory")):
        return (
            base_summary
            + (last_week_delta() or "")
            + (
                f" Next month is forecasted at ${forecast_rev:,.0f} and {forecast_units:,} units."
                if forecast_rev and forecast_units
                else ""
            )
        ).strip()

    if "forecast" in question_lower or "next" in question_lower:
        details = (
            f"We expect around ${forecast_rev:,.0f} revenue and {forecast_units:,} units next month."
            if forecast_rev and forecast_units
            else "Forecast data is available once enough historical sales accumulate."
        )
        return f"{base_summary}{details}"

    if "segment" in question_lower or "customer" in question_lower:
        return f"{base_summary}{top_segment()}"

    if "channel" in question_lower or "region" in question_lower:
        return f"{base_summary}{top_channel()}"

    if "what can you do" in question_lower or "help" in question_lower:
        return (
            "I can summarise performance, call out momentum shifts, highlight key buyer cohorts, "
            "and forecast next month’s sales. Try asking about the revenue trend, top segments, "
            "or how the forecast looks."
        )

    return (
        base_summary
        + (top_segment() or top_channel())
        + (
            f" Forecast: ${forecast_rev:,.0f} / {forecast_units:,} units next month."
            if forecast_rev and forecast_units
            else ""
        )
    ).strip()
