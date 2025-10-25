#!/usr/bin/env python3
"""
Ingest the official H&M demo dataset into a compact SQLite database for the
ShopSight prototype. This replaces the synthetic CSV workflow with data derived
from the parquet files hosted at s3://kumo-public-datasets/hm_with_images/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import ARTICLES_PATH, DB_PATH, TOP_N_ARTICLES, TRANSACTIONS_DIR

DEFAULT_TOP_N_ARTICLES = TOP_N_ARTICLES

CHANNEL_MAP = {1: "Online", 2: "Retail Store"}
REGIONS = ["US-West", "US-East", "US-South", "US-Midwest", "Canada", "Europe"]


def assign_region(customer_id: str) -> str:
    digest = hashlib.sha1(customer_id.encode("utf-8")).digest()
    return REGIONS[digest[0] % len(REGIONS)]


def ensure_inputs() -> None:
    missing = []
    if not ARTICLES_PATH.exists():
        missing.append(str(ARTICLES_PATH))
    if not TRANSACTIONS_DIR.exists():
        missing.append(str(TRANSACTIONS_DIR))
    if missing:
        raise FileNotFoundError(
            "Required parquet files not found. Make sure you downloaded them:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def determine_top_articles(dataset: ds.Dataset, top_n: int) -> List[int]:
    revenue_by_article: Dict[int, float] = defaultdict(float)
    units_by_article: Dict[int, int] = defaultdict(int)

    for batch in dataset.to_batches(columns=["article_id", "price"], batch_size=100_000):
        df = batch.to_pandas()
        grouped = df.groupby("article_id")["price"]
        for article_id, prices in grouped:
            revenue_by_article[int(article_id)] += float(prices.sum())
            units_by_article[int(article_id)] += int(prices.count())

    if not revenue_by_article:
        raise RuntimeError("No transactions detected in the parquet dataset.")

    ranked = sorted(
        revenue_by_article.keys(), key=lambda aid: revenue_by_article[aid], reverse=True
    )
    return ranked[:top_n]


def aggregate_metrics(
    dataset: ds.Dataset, target_articles: Iterable[int]
) -> Tuple[
    Dict[Tuple[int, datetime, str, str], Dict[str, float]],
    Dict[Tuple[int, str], Dict[str, float]],
    Dict[Tuple[int, str], Dict[str, float]],
    Dict[int, Dict[str, object]],
    Dict[int, Counter],
]:
    granular: Dict[Tuple[int, datetime, str, str], Dict[str, float]] = defaultdict(
        lambda: {"units": 0, "revenue": 0.0}
    )
    channel_mix: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(
        lambda: {"units": 0, "revenue": 0.0}
    )
    region_mix: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(
        lambda: {"units": 0, "revenue": 0.0}
    )
    summary: Dict[int, Dict[str, object]] = defaultdict(
        lambda: {
            "total_units": 0,
            "total_revenue": 0.0,
            "first_sale": None,
            "last_sale": None,
            "online_units": 0,
            "store_units": 0,
        }
    )
    customer_counts: Dict[int, Counter] = defaultdict(Counter)

    target_set = set(target_articles)

    for batch in dataset.to_batches(
        columns=[
            "article_id",
            "customer_id",
            "t_dat",
            "price",
            "sales_channel_id",
        ],
        batch_size=75_000,
    ):
        df = batch.to_pandas()
        df["article_id"] = df["article_id"].astype("int64")
        df = df[df["article_id"].isin(target_set)]
        if df.empty:
            continue

        df["t_dat"] = pd.to_datetime(df["t_dat"])
        df["date"] = df["t_dat"].dt.date
        df["channel"] = df["sales_channel_id"].map(CHANNEL_MAP).fillna("Other")
        df["region"] = df["customer_id"].astype(str).apply(assign_region)

        for row in df.itertuples(index=False):
            article_id = int(row.article_id)
            date = row.date
            channel = row.channel
            region = row.region
            price = float(row.price)

            key = (article_id, date, channel, region)
            granular[key]["units"] += 1
            granular[key]["revenue"] += price

            channel_key = (article_id, channel)
            channel_mix[channel_key]["units"] += 1
            channel_mix[channel_key]["revenue"] += price

            region_key = (article_id, region)
            region_mix[region_key]["units"] += 1
            region_mix[region_key]["revenue"] += price

            summary_entry = summary[article_id]
            summary_entry["total_units"] += 1
            summary_entry["total_revenue"] += price
            summary_entry["first_sale"] = (
                date
                if summary_entry["first_sale"] is None
                else min(summary_entry["first_sale"], date)
            )
            summary_entry["last_sale"] = (
                date
                if summary_entry["last_sale"] is None
                else max(summary_entry["last_sale"], date)
            )
            if channel == "Online":
                summary_entry["online_units"] += 1
            elif channel == "Retail Store":
                summary_entry["store_units"] += 1

            customer_counts[article_id][row.customer_id] += 1

    return granular, channel_mix, region_mix, summary, customer_counts


def build_segments(
    article_id: int,
    summary_entry: Dict[str, object],
    channel_records: Dict[Tuple[int, str], Dict[str, float]],
    customer_counter: Counter,
) -> List[Dict[str, str]]:
    total_units = summary_entry["total_units"] or 1
    online_units = channel_records.get((article_id, "Online"), {}).get("units", 0)
    store_units = channel_records.get((article_id, "Retail Store"), {}).get("units", 0)

    online_share = int(round(online_units / total_units * 100))
    store_share = int(round(store_units / total_units * 100))
    remainder = max(0, 100 - online_share - store_share)

    repeat_buyers = sum(1 for count in customer_counter.values() if count > 1)
    repeat_rate = repeat_buyers / max(len(customer_counter), 1) * 100

    segments = [
        {
            "segment": "Digital Loyalists",
            "share": f"{online_share}%",
            "traits": "Prefers app and web experiences; responds to push promos.",
        },
        {
            "segment": "Store Stylists",
            "share": f"{store_share}%",
            "traits": "Visits flagships for fit/feel; influenced by in-store visuals.",
        },
        {
            "segment": "Hybrid Regulars",
            "share": f"{remainder}%",
            "traits": f"{repeat_rate:.0f}% made repeat purchases; split between channels.",
        },
    ]
    return segments


def write_database(
    conn: sqlite3.Connection,
    granular,
    channel_mix,
    region_mix,
    summary,
    customer_counts,
    articles_lookup: pd.DataFrame,
) -> List[Dict[str, object]]:
    conn.execute("PRAGMA foreign_keys = OFF;")
    tables = [
        "article_daily_metrics",
        "article_channel_mix",
        "article_region_mix",
        "article_summary",
        "articles",
    ]
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    daily_rows = [
        {
            "article_id": article_id,
            "transaction_date": date.isoformat(),
            "channel": channel,
            "region": region,
            "units": metrics["units"],
            "gross_revenue": round(metrics["revenue"], 2),
            "unit_price": round(
                metrics["revenue"] / metrics["units"], 2
            )
            if metrics["units"]
            else 0.0,
        }
        for (article_id, date, channel, region), metrics in granular.items()
    ]
    pd.DataFrame(daily_rows).to_sql(
        "article_daily_metrics", conn, index=False, if_exists="replace"
    )
    conn.execute(
        "CREATE INDEX idx_daily_article_date ON article_daily_metrics(article_id, transaction_date)"
    )

    channel_rows = [
        {
            "article_id": article_id,
            "channel": channel,
            "units": metrics["units"],
            "revenue": round(metrics["revenue"], 2),
        }
        for (article_id, channel), metrics in channel_mix.items()
    ]
    pd.DataFrame(channel_rows).to_sql(
        "article_channel_mix", conn, index=False, if_exists="replace"
    )

    region_rows = [
        {
            "article_id": article_id,
            "region": region,
            "units": metrics["units"],
            "revenue": round(metrics["revenue"], 2),
        }
        for (article_id, region), metrics in region_mix.items()
    ]
    pd.DataFrame(region_rows).to_sql(
        "article_region_mix", conn, index=False, if_exists="replace"
    )

    segment_records: List[Dict[str, object]] = []
    summary_rows = []

    for _, article_row in articles_lookup.iterrows():
        article_id = int(article_row["article_id"])
        if article_id not in summary:
            continue
        entry = summary[article_id]
        total_units = entry["total_units"]
        total_revenue = entry["total_revenue"]
        avg_price = total_revenue / total_units if total_units else 0.0
        first_sale = entry["first_sale"].isoformat()
        last_sale = entry["last_sale"].isoformat()

        # Recent windows
        last_date = entry["last_sale"]
        recent_start = last_date - timedelta(days=30)
        prev_start = recent_start - timedelta(days=30)

        recent = conn.execute(
            """
            SELECT
                SUM(units) AS units,
                SUM(gross_revenue) AS revenue
            FROM article_daily_metrics
            WHERE article_id = ? AND transaction_date > ?
            """,
            (article_id, recent_start.isoformat()),
        ).fetchone()

        previous = conn.execute(
            """
            SELECT
                SUM(units) AS units,
                SUM(gross_revenue) AS revenue
            FROM article_daily_metrics
            WHERE article_id = ? AND transaction_date > ? AND transaction_date <= ?
            """,
            (article_id, prev_start.isoformat(), recent_start.isoformat()),
        ).fetchone()

        recent_units = recent[0] or 0
        recent_revenue = recent[1] or 0.0
        prev_units = previous[0] or 0
        prev_revenue = previous[1] or 0.0

        summary_rows.append(
            {
                "article_id": article_id,
                "product_name": article_row["product_name"],
                "product_type_name": article_row.get("product_type_name"),
                "product_group_name": article_row.get("product_group_name"),
                "department_name": article_row.get("department_name"),
                "garment_group_name": article_row.get("garment_group_name"),
                "index_name": article_row.get("index_name"),
                "first_sale": first_sale,
                "last_sale": last_sale,
                "total_units": total_units,
                "total_revenue": round(total_revenue, 2),
                "avg_price": round(avg_price, 2),
                "recent_units": recent_units,
                "recent_revenue": round(recent_revenue, 2),
                "prev_units": prev_units,
                "prev_revenue": round(prev_revenue, 2),
                "online_units": entry["online_units"],
                "store_units": entry["store_units"],
                "unique_customers": len(customer_counts[article_id]),
            }
        )

        segments = build_segments(
            article_id, entry, channel_mix, customer_counts[article_id]
        )
        for segment in segments:
            segment_records.append(
                {
                    "article_id": article_id,
                    "segment": segment["segment"],
                    "share": segment["share"],
                    "traits": segment["traits"],
                }
            )

    pd.DataFrame(summary_rows).to_sql(
        "article_summary", conn, index=False, if_exists="replace"
    )
    conn.execute("CREATE INDEX idx_summary_article ON article_summary(article_id)")

    pd.DataFrame(segment_records).to_sql(
        "article_segments", conn, index=False, if_exists="replace"
    )

    articles_lookup.to_sql("articles", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX idx_articles_name ON articles(product_name)")

    return segment_records


def update_segments_json(segments: List[Dict[str, object]]) -> None:
    output = defaultdict(lambda: {"segments": []})
    for record in segments:
        entry = output[str(record["article_id"])]
        entry["segments"].append(
            {
                "segment": record["segment"],
                "share": record["share"],
                "traits": record["traits"],
            }
        )
    serialised = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "products": [
            {"product_id": product_id, "segments": data["segments"]}
            for product_id, data in output.items()
        ],
    }
    (DATA_DIR / "product_segments.json").write_text(json.dumps(serialised, indent=2))


def main(top_n: int) -> None:
    ensure_inputs()

    dataset = ds.dataset(str(TRANSACTIONS_DIR), format="parquet")
    print(f"Scanning parquet dataset from {TRANSACTIONS_DIR} …")
    top_articles = determine_top_articles(dataset, top_n)
    print(f"Top {len(top_articles)} articles selected.")

    print("Aggregating metrics for selected articles …")
    granular, channel_mix, region_mix, summary, customer_counts = aggregate_metrics(
        dataset, top_articles
    )

    articles_table = pq.read_table(ARTICLES_PATH)
    articles_df = articles_table.to_pandas()
    articles_df = articles_df[articles_df["article_id"].isin(top_articles)].copy()
    articles_df.rename(columns={"prod_name": "product_name"}, inplace=True)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        segments = write_database(
            conn,
            granular,
            channel_mix,
            region_mix,
            summary,
            customer_counts,
            articles_df,
        )
        conn.commit()

    update_segments_json(segments)
    print(f"SQLite database written to {DB_PATH}")
    print("Updated product_segments.json with generated segments.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load HM parquet data into SQLite.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N_ARTICLES,
        help="Number of top-selling articles to include (default: 60).",
    )
    args = parser.parse_args()
    main(args.top_n)
