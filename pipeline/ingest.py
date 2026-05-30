"""Ingest USAspending contract CSVs from data/raw/ and emit a cleaned parquet.

Normalizes vendor names (uppercases, strips whitespace, removes common corporate
suffixes for dedup) and NAICS codes (6-digit zero-padded strings). Drops rows
with null or zero award amounts.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/contracts.parquet")

KEEP_COLS = [
    "recipient_name",
    "awarding_agency_name",
    "naics_code",
    "naics_description",
    "award_amount",
    "period_of_performance_start_date",
    "potential_total_value_of_award",
]

CORP_SUFFIX_RE = re.compile(
    r"[,\.]?\s*\b(LLC|L\.L\.C\.|INC|INC\.|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|"
    r"LP|LLP|PLC|GMBH|HOLDINGS|GROUP|THE)\b\.?",
    re.IGNORECASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ingest")


def normalize_vendor(name: str) -> str:
    """Uppercase, strip whitespace, drop trailing corporate suffixes."""
    if not isinstance(name, str):
        return ""
    s = name.strip().upper()
    s = CORP_SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.")
    return s


def normalize_naics(code) -> str:
    """Cast NAICS code to a 6-digit zero-padded string."""
    if pd.isna(code):
        return ""
    try:
        return str(int(float(code))).zfill(6)
    except (ValueError, TypeError):
        return str(code).strip().zfill(6)


def load_raw() -> pd.DataFrame:
    """Concatenate every CSV in data/raw/, keeping only known columns."""
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        log.error("No CSVs found in %s. Drop USAspending downloads there.", RAW_DIR)
        sys.exit(1)
    log.info("Found %d CSV files in %s", len(csvs), RAW_DIR)

    frames: list[pd.DataFrame] = []
    for path in csvs:
        log.info("Reading %s", path.name)
        # USAspending CSVs are wide; only read columns that exist
        head = pd.read_csv(path, nrows=0)
        cols = [c for c in KEEP_COLS if c in head.columns]
        if not cols:
            log.warning("No expected columns in %s; skipping.", path.name)
            continue
        df = pd.read_csv(path, usecols=cols, low_memory=False)
        # Ensure all keep-cols exist downstream
        for c in KEEP_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        frames.append(df[KEEP_COLS])
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and drop rows missing the fields we need downstream."""
    n_in = len(df)
    df = df.copy()
    df["recipient_name"] = df["recipient_name"].map(normalize_vendor)
    df["naics_code"] = df["naics_code"].map(normalize_naics)
    df["awarding_agency_name"] = (
        df["awarding_agency_name"].astype("string").str.strip().str.upper()
    )
    df["award_amount"] = pd.to_numeric(df["award_amount"], errors="coerce")

    null_award = df["award_amount"].isna() | (df["award_amount"] <= 0)
    null_vendor = df["recipient_name"].eq("") | df["recipient_name"].isna()
    null_naics = df["naics_code"].eq("") | df["naics_code"].isna()
    null_agency = df["awarding_agency_name"].isna() | df["awarding_agency_name"].eq("")

    drop_mask = null_award | null_vendor | null_naics | null_agency
    df = df.loc[~drop_mask].reset_index(drop=True)

    log.info(
        "Cleaned rows: in=%d out=%d dropped=%d (null_award=%d null_vendor=%d "
        "null_naics=%d null_agency=%d)",
        n_in,
        len(df),
        int(drop_mask.sum()),
        int(null_award.sum()),
        int(null_vendor.sum()),
        int(null_naics.sum()),
        int(null_agency.sum()),
    )
    return df


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    log.info("Loaded %d raw rows total", len(raw))
    clean_df = clean(raw)
    clean_df.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s (%d rows)", OUT_PATH, len(clean_df))


if __name__ == "__main__":
    main()
