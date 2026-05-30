"""Stream the giant USAspending archive CSVs into a defense-only slice.

Reads every CSV under data/raw/ (recursively) in chunks, keeps only rows
whose awarding agency looks like a DoD component, projects down to the
columns we care about, and writes data/processed/contracts.parquet directly.

Run this INSTEAD OF `make ingest` when working from the FYxxxx_All_Contracts
bulk archives — those files are too big to load whole.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/contracts.parquet")

CHUNK_SIZE = 200_000

SRC_COLS = [
    "recipient_name",
    "awarding_agency_name",
    "awarding_sub_agency_name",
    "naics_code",
    "naics_description",
    "federal_action_obligation",
    "period_of_performance_start_date",
    "potential_total_value_of_award",
]

DEFENSE_KEYWORDS = (
    "DEFENSE",
    "ARMY",
    "NAVY",
    "AIR FORCE",
    "MARINE CORPS",
    "SPACE FORCE",
    "DLA",
    "DARPA",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("prefilter")


def is_defense(agency: str) -> bool:
    if not isinstance(agency, str):
        return False
    a = agency.upper()
    return any(k in a for k in DEFENSE_KEYWORDS)


def stream_filter(path: Path) -> pd.DataFrame:
    """Stream one CSV, keep defense rows + needed columns."""
    head = pd.read_csv(path, nrows=0)
    cols = [c for c in SRC_COLS if c in head.columns]
    log.info("  using cols: %s", cols)

    kept: list[pd.DataFrame] = []
    total_in = 0
    total_kept = 0
    for chunk in pd.read_csv(
        path, usecols=cols, chunksize=CHUNK_SIZE, low_memory=False
    ):
        total_in += len(chunk)
        mask = chunk["awarding_agency_name"].map(is_defense)
        sub = chunk.loc[mask].copy()
        total_kept += len(sub)
        if not sub.empty:
            kept.append(sub)
        if total_in % (CHUNK_SIZE * 5) == 0:
            log.info(
                "    %s scanned=%d kept=%d", path.name, total_in, total_kept
            )
    log.info("  %s: scanned=%d kept=%d", path.name, total_in, total_kept)
    if not kept:
        return pd.DataFrame(columns=cols)
    return pd.concat(kept, ignore_index=True)


def main() -> None:
    csvs = sorted(RAW_DIR.rglob("*.csv"))
    if not csvs:
        log.error("No CSVs under %s", RAW_DIR)
        return
    log.info("Prefiltering %d CSVs", len(csvs))

    frames = []
    for p in csvs:
        log.info("Reading %s (%.0f MB)", p, p.stat().st_size / 1e6)
        frames.append(stream_filter(p))
    df = pd.concat(frames, ignore_index=True)
    log.info("Defense rows total: %d", len(df))

    # Rename federal_action_obligation -> award_amount to match ingest schema.
    if "federal_action_obligation" in df.columns:
        df = df.rename(columns={"federal_action_obligation": "award_amount"})

    # In the FYxxxx archives, awarding_agency_name is almost always
    # "DEPARTMENT OF DEFENSE". The actual differentiation (Army/Navy/AF/DLA)
    # lives in awarding_sub_agency_name. Treat sub-agency as the effective
    # agency for the graph so AGENCY nodes are meaningful.
    if "awarding_sub_agency_name" in df.columns:
        sub = df["awarding_sub_agency_name"].astype("string").str.strip()
        df["awarding_agency_name"] = sub.where(
            sub.notna() & sub.ne(""), df["awarding_agency_name"]
        )

    # Reuse normalization from ingest module.
    from pipeline.ingest import clean  # local import to avoid cycles

    # Ensure all KEEP_COLS exist with the names ingest.clean expects.
    expected = [
        "recipient_name",
        "awarding_agency_name",
        "naics_code",
        "naics_description",
        "award_amount",
        "period_of_performance_start_date",
        "potential_total_value_of_award",
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[expected]

    cleaned = clean(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s (%d rows)", OUT_PATH, len(cleaned))


if __name__ == "__main__":
    main()
