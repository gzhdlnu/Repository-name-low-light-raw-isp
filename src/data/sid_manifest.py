# =============================================================================
# Project: RAW ISP | Task: SID Manifest Parsing
# Timestamp: 2026-09-20 | Author: Guozh | Version: Phase2-Data-v0.1.0
# =============================================================================
"""Parse the official SID Sony train/val/test list files.

Each official line has the form:
    ./Sony/short/xxx_0.04s.ARW ./Sony/long/xxx_10s.ARW ISO200 F8
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_EXPOSURE_RE = re.compile(r"(?P<seconds>[0-9]+(?:\.[0-9]+)?)s$")
_ISO_RE = re.compile(r"ISO(?P<iso>[0-9]+)$", re.IGNORECASE)
_APERTURE_RE = re.compile(r"F(?P<value>[0-9]+(?:\.[0-9]+)?)$", re.IGNORECASE)


def _parse_exposure(path: str) -> float:
    match = _EXPOSURE_RE.search(Path(path).stem)
    if not match:
        raise ValueError(f"Cannot parse exposure from SID filename: {path}")
    return float(match.group("seconds"))


@dataclass(frozen=True)
class SidPair:
    short_path: Path
    long_path: Path
    short_exposure_s: float
    long_exposure_s: float
    iso: int
    aperture: float

    @property
    def exposure_gain(self) -> float:
        """Linear gain needed to match the long-exposure reference."""
        if self.short_exposure_s <= 0 or self.long_exposure_s <= 0:
            raise ValueError("Exposure must be positive")
        return self.long_exposure_s / self.short_exposure_s


def load_sid_list(list_path: str | Path) -> list[SidPair]:
    """Load an official SID list and resolve paths relative to its directory."""
    list_path = Path(list_path).resolve()
    base = list_path.parent
    pairs: list[SidPair] = []
    for line_number, line in enumerate(list_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{list_path}:{line_number}: expected 4 fields, got {len(fields)}")
        short_rel, long_rel, iso_token, aperture_token = fields
        iso_match = _ISO_RE.fullmatch(iso_token)
        aperture_match = _APERTURE_RE.fullmatch(aperture_token)
        if not iso_match or not aperture_match:
            raise ValueError(f"{list_path}:{line_number}: invalid metadata fields")
        pairs.append(
            SidPair(
                short_path=(base / short_rel).resolve(),
                long_path=(base / long_rel).resolve(),
                short_exposure_s=_parse_exposure(short_rel),
                long_exposure_s=_parse_exposure(long_rel),
                iso=int(iso_match.group("iso")),
                aperture=float(aperture_match.group("value")),
            )
        )
    return pairs
