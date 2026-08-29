"""
storage/financial_actuals.py
----------------------------
Persistence repository and canonical metric mapper for financial actuals scraped from Screener.in.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import get_connection, DB_PATH
from ingestion.screener_scraper import scrape_screener

# Canonical mapping: Screener row metric name → ContraGuard prediction metric
SCREENER_METRIC_MAP = {
    "sales": "revenue_growth",
    "revenue": "revenue_growth",
    "operating profit": "ebitda_margin",
    "opm %": "operating_margin",
    "net profit": "net_profit",
    "eps in rs": "eps",
    "eps": "eps",
}


def upsert_financial_actual(
    company_id: int,
    quarter: str,
    metric: str,
    value: float,
    unit_currency: str = "INR Crores",
    source_url: str = "",
    provenance: str = "Screener.in",
    db_path: Path = DB_PATH,
) -> int:
    """
    Upsert a financial actual entry in financial_actuals table.
    """
    conn = get_connection(db_path)
    conn.execute(
        """
        INSERT INTO financial_actuals
            (company_id, quarter, metric, value, unit_currency, source_url, provenance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(company_id, quarter, metric) DO UPDATE SET
            value = excluded.value,
            unit_currency = excluded.unit_currency,
            source_url = excluded.source_url,
            fetched_at = datetime('now')
        """,
        (company_id, quarter, metric, value, unit_currency, source_url, provenance),
    )
    conn.commit()

    row = conn.execute(
        "SELECT id FROM financial_actuals WHERE company_id=? AND quarter=? AND metric=?",
        (company_id, quarter, metric),
    ).fetchone()
    conn.close()

    return row["id"]


def get_financial_actuals(
    company_id: int,
    quarter: Optional[str] = None,
    metric: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Retrieve stored financial actuals.
    """
    conn = get_connection(db_path)
    query = "SELECT * FROM financial_actuals WHERE company_id = ?"
    params: List[Any] = [company_id]

    if quarter:
        query += " AND quarter = ?"
        params.append(quarter)

    if metric:
        query += " AND metric = ?"
        params.append(metric)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def map_screener_metric(screener_row_name: str) -> Optional[str]:
    """
    Map Screener row label to canonical prediction metric name.
    """
    cleaned = screener_row_name.strip().lower()
    for kw, canonical in SCREENER_METRIC_MAP.items():
        if kw in cleaned:
            return canonical
    return None


def fetch_and_store_screener_actuals(
    ticker: str,
    company_id: int,
    db_path: Path = DB_PATH,
) -> int:
    """
    Scrape quarterly financials from Screener.in for ticker and persist normalized actuals into DB.
    Returns count of actuals inserted/updated.
    """
    logger.info(f"Fetching Screener.in actuals for ticker '{ticker}' (company_id={company_id})...")
    data = scrape_screener(ticker)
    q_results = data.get("quarterly_results")

    if not isinstance(q_results, pd.DataFrame) or q_results.empty:
        logger.warning(f"No quarterly results returned for {ticker}.")
        return 0

    count = 0
    source_url = f"https://www.screener.in/company/{ticker.upper()}/"

    for row_metric, row_series in q_results.iterrows():
        canonical_metric = map_screener_metric(str(row_metric))
        if not canonical_metric:
            continue

        for quarter_label, val in row_series.items():
            if pd.isna(val) or val is None:
                continue

            try:
                numeric_val = float(val)
                upsert_financial_actual(
                    company_id=company_id,
                    quarter=str(quarter_label),
                    metric=canonical_metric,
                    value=numeric_val,
                    unit_currency="INR Crores" if canonical_metric != "eps" else "INR",
                    source_url=source_url,
                    provenance="Screener.in",
                    db_path=db_path,
                )
                count += 1
            except Exception as e:
                logger.debug(f"Could not parse actual value '{val}' for {quarter_label}: {e}")

    logger.info(f"Persisted {count} financial actual entries for {ticker}.")
    return count
