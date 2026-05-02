"""Canonical physical event identifier for cross-catalogue joins.

The legacy `event_id` is a per-detector-rebuild ordinal and is reused across
detector eras for different physical events (see
`.claude/.../reference_catalogue_event_ids.md`). Joining two catalogues by
`event_id` alone produced silently-wrong data on at least one occasion (the
2026-05-02 Phase 3 sampler bug — bogus refined-pick shifts of months).

`physical_id` is derived deterministically from `(mooring, onset_utc)` and
is **stable within a detector run**. Across eras it may shift by a handful
of milliseconds if STA/LTA parameters changed; for those cases use the
fuzzy-match helper rather than a direct join.

Use:
    >>> from catalogue_id import add_physical_id, safe_join
    >>> phase3 = add_physical_id(pd.read_parquet("phase3_catalogue.parquet"))
    >>> sample = add_physical_id(pd.read_csv("fp_validation_events.csv"))
    >>> joined = safe_join(sample, phase3, on="physical_id")
"""

from __future__ import annotations

import pandas as pd


def physical_id_for(mooring: str, onset_utc) -> str:
    """Return the canonical physical_id for a single event.

    Format: ``"{mooring}_{YYYYmmddTHHMMSS}_{ms:03d}"``
    Example: ``"m3_20190117T144951_744"``
    """
    ts = pd.to_datetime(onset_utc)
    return f"{mooring}_{ts.strftime('%Y%m%dT%H%M%S')}_{ts.microsecond // 1000:03d}"


def add_physical_id(
    df: pd.DataFrame,
    mooring_col: str = "mooring",
    onset_col: str = "onset_utc",
) -> pd.DataFrame:
    """Return a copy of df with a `physical_id` column added.

    Idempotent: if the column is already present and fully populated,
    returns df unchanged.
    """
    if "physical_id" in df.columns and df["physical_id"].notna().all():
        return df
    df = df.copy()
    onset = pd.to_datetime(df[onset_col])
    df["physical_id"] = (
        df[mooring_col].astype(str) + "_"
        + onset.dt.strftime("%Y%m%dT%H%M%S") + "_"
        + (onset.dt.microsecond // 1000).map("{:03d}".format)
    )
    return df


def safe_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "physical_id",
    how: str = "inner",
    **kwargs,
) -> pd.DataFrame:
    """Join two catalogues on physical_id, refusing silent wrong joins.

    Raises ValueError if the join key is missing from either side, with a
    pointer to the migration helper. The default key is `physical_id`; do
    not pass `on='event_id'` for a cross-catalogue merge — `event_id` is
    not stable across detector rebuilds.
    """
    if on != "physical_id":
        raise ValueError(
            f"safe_join refuses to join on {on!r}. event_id is unstable across "
            "detector rebuilds. Use physical_id (add via add_physical_id())."
        )
    for label, df in (("left", left), ("right", right)):
        if on not in df.columns:
            raise ValueError(
                f"{label} dataframe is missing column {on!r}. "
                f"Run add_physical_id() on it before joining. See "
                f"reference_catalogue_event_ids.md for context."
            )
        if df[on].isna().any():
            raise ValueError(
                f"{label} dataframe has NaN in {on!r}. "
                f"Re-run add_physical_id() to populate."
            )
    return left.merge(right, on=on, how=how, **kwargs)
