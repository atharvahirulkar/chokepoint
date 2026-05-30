"""Stream the giant USAspending archive CSVs into a defense-only slice.

Recurses through every directory under data/raw/ matching FY-pattern,
streams each CSV in chunks, keeps only rows whose awarding agency looks
like a DoD component, attaches `fiscal_year` (parsed from the parent dir
name like FY2024_All_Contracts_Full_*), and writes
data/processed/contracts.parquet.

Run via `make prefilter`. Configure the years included via the YEARS env
var (default: 2024,2025,2026).
"""
from __future__ import annotations

import logging
import os
import re
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

# Which fiscal years to include. Override with `YEARS=2021,2022,...` env var.
DEFAULT_YEARS = (2024, 2025, 2026)
FY_DIR_PATTERN = re.compile(r"FY(\d{4})_All_Contracts_Full", re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("prefilter")


def configured_years() -> set[int]:
    raw = os.environ.get("YEARS")
    if not raw:
        return set(DEFAULT_YEARS)
    out: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            log.warning("Skipping unparsable YEARS token: %r", token)
    return out or set(DEFAULT_YEARS)


def is_defense(agency: str) -> bool:
    if not isinstance(agency, str):
        return False
    a = agency.upper()
    return any(k in a for k in DEFENSE_KEYWORDS)


def fiscal_year_for_path(path: Path) -> int | None:
    """Extract FY from a path like data/raw/FY2024_All_Contracts_Full_*/foo.csv."""
    for part in path.parts:
        m = FY_DIR_PATTERN.search(part)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def stream_filter(path: Path, fy: int) -> pd.DataFrame:
    """Stream one CSV, keep defense rows + needed columns. Attach fiscal_year."""
    head = pd.read_csv(path, nrows=0)
    cols = [c for c in SRC_COLS if c in head.columns]

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
    log.info("  %s [FY%d]: scanned=%d kept=%d", path.name, fy, total_in, total_kept)
    if not kept:
        return pd.DataFrame(columns=cols)
    out = pd.concat(kept, ignore_index=True)
    out["fiscal_year"] = fy
    return out


def main() -> None:
    years = configured_years()
    log.info("Configured fiscal years: %s", sorted(years))

    # Discover CSVs under the FY directories matching the configured years.
    csvs: list[tuple[Path, int]] = []
    for sub in sorted(RAW_DIR.iterdir()):
        if not sub.is_dir():
            continue
        m = FY_DIR_PATTERN.search(sub.name)
        if not m:
            continue
        fy = int(m.group(1))
        if fy not in years:
            log.info("Skipping %s (FY%d not in scope)", sub.name, fy)
            continue
        for csv in sorted(sub.glob("*.csv")):
            csvs.append((csv, fy))

    if not csvs:
        log.error(
            "No CSVs found under %s for years %s. Drop USAspending archives "
            "into data/raw/FY{year}_All_Contracts_Full_*/.",
            RAW_DIR,
            sorted(years),
        )
        return

    total_mb = sum(p.stat().st_size for p, _ in csvs) / 1e6
    log.info("Prefiltering %d CSVs (%.0f MB total) across FYs %s",
             len(csvs), total_mb, sorted({fy for _, fy in csvs}))

    frames: list[pd.DataFrame] = []
    for path, fy in csvs:
        log.info("Reading %s [FY%d] (%.0f MB)", path, fy, path.stat().st_size / 1e6)
        frames.append(stream_filter(path, fy))

    df = pd.concat(frames, ignore_index=True)
    log.info("Defense rows total (pre-clean): %d", len(df))

    # Rename federal_action_obligation -> award_amount to match ingest schema.
    if "federal_action_obligation" in df.columns:
        df = df.rename(columns={"federal_action_obligation": "award_amount"})

    # Roll up to sub-agency for meaningful AGENCY nodes.
    if "awarding_sub_agency_name" in df.columns:
        sub = df["awarding_sub_agency_name"].astype("string").str.strip()
        df["awarding_agency_name"] = sub.where(
            sub.notna() & sub.ne(""), df["awarding_agency_name"]
        )

    from pipeline.ingest import clean  # local import to avoid cycles

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
    # Keep fiscal_year through cleaning.
    fy_series = df["fiscal_year"]
    df_for_clean = df[expected].copy()
    cleaned = clean(df_for_clean)
    # Re-attach fiscal_year via the original index since clean drops rows.
    # clean() drops by mask but keeps row order; we re-align by re-running
    # the same row-dropping rules here for safety.
    null_award = (
        cleaned["award_amount"].isna() | (cleaned["award_amount"] <= 0)
    )
    assert not null_award.any(), "ingest.clean did not drop null awards"
    # Easier path: re-do the join by passing fiscal_year through ingest.clean
    # via a side df keyed by an index that survives the clean step. Since
    # clean rebuilds the index, we instead inner-join on the survivable
    # columns. The simplest robust approach: reset original df, run clean,
    # then reattach FY from the surviving row mask.

    # Recompute the mask exactly like ingest.clean does so we can pick the
    # corresponding fiscal_year entries.
    from pipeline.ingest import normalize_vendor, normalize_naics

    orig = df[expected].copy()
    orig_norm = orig.copy()
    orig_norm["recipient_name"] = orig_norm["recipient_name"].map(normalize_vendor)
    orig_norm["naics_code"] = orig_norm["naics_code"].map(normalize_naics)
    orig_norm["awarding_agency_name"] = (
        orig_norm["awarding_agency_name"].astype("string").str.strip().str.upper()
    )
    orig_norm["award_amount"] = pd.to_numeric(orig_norm["award_amount"], errors="coerce")
    drop_mask = (
        orig_norm["award_amount"].isna()
        | (orig_norm["award_amount"] <= 0)
        | orig_norm["recipient_name"].eq("")
        | orig_norm["recipient_name"].isna()
        | orig_norm["naics_code"].eq("")
        | orig_norm["naics_code"].isna()
        | orig_norm["awarding_agency_name"].isna()
        | orig_norm["awarding_agency_name"].eq("")
    )
    surviving_fy = fy_series.loc[~drop_mask].reset_index(drop=True)
    cleaned["fiscal_year"] = surviving_fy.astype("int16").values

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT_PATH, index=False)
    log.info("Wrote %s (%d rows)", OUT_PATH, len(cleaned))
    log.info("FY breakdown:\n%s", cleaned["fiscal_year"].value_counts().sort_index())


if __name__ == "__main__":
    main()
