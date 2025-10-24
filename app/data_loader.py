from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "shopsight.db"


class DataSourceError(RuntimeError):
    """Raised when the prepared SQLite database is missing."""


def _ensure_database() -> None:
    if not DB_PATH.exists():
        raise DataSourceError(
            "SQLite database not found. Run scripts/load_hm_data.py to build it."
        )


def _connect() -> sqlite3.Connection:
    _ensure_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _normalise_product_id(product_id: str) -> int:
    try:
        return int(product_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid product identifier: {product_id}") from exc


@lru_cache(maxsize=1)
def load_product_catalog() -> pd.DataFrame:
    """Return curated product metadata sourced from the article_summary table."""
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                article_id AS product_id,
                product_name,
                COALESCE(product_group_name, index_name, 'Assortment') AS category,
                COALESCE(department_name, 'H&M Originals') AS department,
                avg_price,
                total_units,
                total_revenue,
                first_sale,
                last_sale
            FROM article_summary
            ORDER BY total_revenue DESC
            """,
            conn,
            parse_dates=["first_sale", "last_sale"],
        )
    df["product_id"] = df["product_id"].astype(str)
    df["brand"] = df["department"]
    return df.drop(columns=["department"]).reset_index(drop=True)


def get_product_details(product_id: str) -> Optional[Dict[str, object]]:
    product_id_int = _normalise_product_id(product_id)
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM article_summary
            WHERE article_id = ?
            """,
            (product_id_int,),
        ).fetchone()
    if row is None:
        return None

    record = dict(row)
    record["product_id"] = str(record.pop("article_id"))
    record["category"] = record.pop("product_group_name") or record.get("index_name")
    record["brand"] = record.get("department_name") or "H&M"
    for field in ["first_sale", "last_sale"]:
        value = record.get(field)
        if value:
            record[field] = value[:10]
    record["avg_price"] = round(record["avg_price"], 2)
    record["total_revenue"] = round(record["total_revenue"], 2)
    return record


def search_products(query: str, limit: int = 5) -> List[Dict[str, str]]:
    query = (query or "").strip()
    with _connect() as conn:
        base_query = """
            SELECT summary.article_id AS product_id,
                   summary.product_name,
                   COALESCE(summary.department_name, 'H&M') AS brand,
                   COALESCE(summary.product_group_name, summary.index_name, 'Assortment') AS category,
                   COALESCE(summary.product_type_name, summary.product_group_name, summary.index_name, 'Assortment') AS descriptor,
                   articles.colour_group_name AS colour,
                   summary.total_revenue
            FROM article_summary AS summary
            LEFT JOIN articles ON articles.article_id = summary.article_id
        """
        if not query:
            df = pd.read_sql_query(
                base_query + " ORDER BY summary.total_revenue DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        else:
            like_pattern = f"%{query}%"
            df = pd.read_sql_query(
                base_query
                + """
                WHERE summary.product_name LIKE ? OR CAST(summary.article_id AS TEXT) LIKE ?
                ORDER BY summary.total_revenue DESC
                LIMIT ?
                """,
                conn,
                params=(like_pattern, like_pattern, limit),
            )

    records = df.to_dict(orient="records")
    grouped: Dict[str, Dict[str, object]] = {}
    ordered_keys: List[str] = []
    query_id = query.strip()

    for record in records:
        record["product_id"] = str(record["product_id"])
        key = record["product_name"].lower()
        if key not in grouped:
            grouped[key] = {
                "base": record,
                "colours": set(filter(None, [record.get("colour")])),
                "variants": [record["product_id"]],
            }
            ordered_keys.append(key)
        else:
            entry = grouped[key]
            entry["variants"].append(record["product_id"])
            if record.get("colour"):
                entry["colours"].add(record["colour"])
            base = entry["base"]
            if (
                record["total_revenue"] > base["total_revenue"]
                and record["product_id"] != query_id
            ):
                entry["base"] = record

        if query_id and record["product_id"] == query_id:
            grouped[key]["base"] = record

    results = []
    for key in ordered_keys:
        entry = grouped[key]
        base = entry["base"]
        colours = sorted(entry["colours"])
        if colours:
            colour_label = ", ".join(colours[:3])
            if len(colours) > 3:
                colour_label += " +"
            descriptor = f"{base['category']} · Colours: {colour_label}"
        else:
            descriptor = base["descriptor"]

        results.append(
            {
                "product_id": base["product_id"],
                "product_name": base["product_name"],
                "brand": base["brand"],
                "category": base["category"],
                "descriptor": descriptor,
            }
        )
        if len(results) >= limit:
            break

    return results


def get_customer_segments(product_id: str) -> List[Dict[str, str]]:
    product_id_int = _normalise_product_id(product_id)
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT segment, share, traits
            FROM article_segments
            WHERE article_id = ?
            ORDER BY rowid
            """,
            conn,
            params=(product_id_int,),
        )
    return df.to_dict(orient="records")


def filter_transactions_by_product(product_id: str) -> pd.DataFrame:
    product_id_int = _normalise_product_id(product_id)
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                transaction_date,
                channel,
                region,
                units,
                gross_revenue,
                unit_price
            FROM article_daily_metrics
            WHERE article_id = ?
            ORDER BY transaction_date
            """,
            conn,
            params=(product_id_int,),
            parse_dates=["transaction_date"],
        )
    return df
