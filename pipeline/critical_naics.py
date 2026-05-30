"""Defense-critical NAICS reference list.

Aligned with the DoD "Critical Technology Areas" framework (USD(R&E), 2023)
and the DPA Title III investment categories: hard manufacturing, ordnance,
microelectronics, and the aerospace/missile/naval propulsion stack. These are
the categories where a sole-source failure would have an explicit national-
security consequence, as opposed to a generic supply-chain inconvenience.

A vendor's chokepoint risk is much more interesting *within* this set. The
features `critical_naics_count` and `critical_sole_source_count` and the
label `critical_coverage_drop` are computed against this list.

Sources:
  - DoD USD(R&E) Critical Technology Areas (Feb 2022, refreshed 2023)
  - DPA Title III project categories
  - NDAA 2022 Sec. 855 (defense industrial base resilience)
"""
from __future__ import annotations

# NAICS codes are stored as zero-padded 6-digit strings to match how the
# ingest pipeline normalizes them.
CRITICAL_NAICS: dict[str, str] = {
    # Aerospace / aircraft propulsion
    "336411": "Aircraft Manufacturing",
    "336412": "Aircraft Engine and Engine Parts Manufacturing",
    "336413": "Other Aircraft Parts and Auxiliary Equipment Manufacturing",
    # Missiles, space, propulsion
    "336414": "Guided Missile and Space Vehicle Manufacturing",
    "336415": "Guided Missile and Space Vehicle Propulsion Unit Mfg",
    "336419": "Other Guided Missile and Space Vehicle Parts Mfg",
    # Ground combat / armor
    "336992": "Military Armored Vehicle, Tank, and Tank Component Mfg",
    # Naval
    "336611": "Ship Building and Repairing",
    # Power / propulsion / turbines
    "333611": "Turbine and Turbine Generator Set Units Manufacturing",
    # Ordnance & ammunition
    "332992": "Small Arms Ammunition Manufacturing",
    "332993": "Ammunition (except Small Arms) Manufacturing",
    "332994": "Small Arms, Ordnance, and Ordnance Accessories Mfg",
    "332995": "Other Ordnance and Accessories Manufacturing",
    # Microelectronics, guidance, sensors
    "334413": "Semiconductor and Related Device Manufacturing",
    "334418": "Printed Circuit Assembly (Electronic Assembly) Mfg",
    "334511": "Search, Detection, Navigation, Guidance, Aeronautical Systems Mfg",
    "334515": "Instrument Mfg for Measuring and Testing Electricity",
    # Materials / energetics
    "325920": "Explosives Manufacturing",
}

CRITICAL_NAICS_SET: frozenset[str] = frozenset(CRITICAL_NAICS.keys())


def is_critical(naics_code: str) -> bool:
    """Return True if the 6-digit NAICS code is in the critical set."""
    return naics_code in CRITICAL_NAICS_SET


def critical_naics_id(naics_id: str) -> bool:
    """Same check, but for graph node IDs of the form `N::<code>`."""
    if not naics_id.startswith("N::"):
        return False
    return naics_id[3:] in CRITICAL_NAICS_SET
