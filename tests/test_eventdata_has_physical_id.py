"""Smoke test: pack's EventData dataclass exposes physical_id.

Cross-catalogue joins must go through physical_id, not event_id (which is a
per-detector-rebuild ordinal and is reused across eras for different physical
events). This test pins the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACK / "scripts"))

import validation_picker_panel as vpp  # noqa: E402


def test_event_data_has_physical_id():
    fields = list(vpp.EventData.__dataclass_fields__.keys())
    assert "physical_id" in fields, f"physical_id missing from EventData; got {fields}"


def test_catalogue_id_helpers_importable():
    from catalogue_id import physical_id_for, add_physical_id, safe_join  # noqa: F401
    assert physical_id_for("m3", "2019-01-17 14:49:51.744") == "m3_20190117T144951_744"
