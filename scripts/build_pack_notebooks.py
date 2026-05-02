"""build_pack_notebooks.py — regenerate the Maleen-pack notebooks.

Wraps the bravoseis-side notebook generators in `noaa_bravoseis/scripts/` and
substitutes the pack-specific setup cell that auto-detects the repo root via
`_find_repo()` (looks up the tree for pyproject.toml). This way the pack stays
in sync with the canonical event_cells / intro_cells / aggregate_cell logic
without forking it.

Generates:
    notebooks/validation_relabel_50_maleen.ipynb         (Phase A pilot, 50 events)
    notebooks/validation_relabel_50_phase3_maleen.ipynb  (Phase 3, 25 seismic + 25 cryogenic)

Run:
    uv run python scripts/build_pack_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

PACK = Path(__file__).resolve().parent.parent
BRAVOSEIS_REPO = Path("/home/jovyan/repos/specKitScience/noaa_bravoseis")
sys.path.insert(0, str(BRAVOSEIS_REPO / "scripts"))

# Reuse the bravoseis builders' cell helpers so logic stays in one place.
import build_validation_relabel_v2_notebook as v2_builder  # noqa: E402
import build_validation_relabel_phase3_notebook as p3_builder  # noqa: E402

PACK_CSV_DIR = PACK / "outputs" / "figures" / "exploratory" / "validation"
NB_DIR = PACK / "notebooks"


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s)


def pack_setup_cell_phase_a(reviewer: str) -> nbf.NotebookNode:
    return code(
        f"""%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
import pandas as pd
import plotly.io as pio

# Force inline rendering — works on JupyterHub without widget extensions.
pio.renderers.default = "notebook"

# Auto-detect the pack root (the directory containing pyproject.toml).
def _find_repo():
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not find repo root (looking for pyproject.toml)")

REPO = _find_repo()
sys.path.insert(0, str(REPO / "scripts"))
from validation_picker_panel import (
    load_event_data, make_plotly_panel, make_annotated_png,
)

REVIEWER = {reviewer!r}
ANNOTATED_DIR = REPO / "outputs" / "figures" / "exploratory" / "validation" / f"annotated_{{REVIEWER}}"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

events = pd.read_csv(REPO / "outputs/figures/exploratory/validation/fp_validation_events.csv")
print(f"Reviewer: {{REVIEWER}}  |  events: {{len(events)}}  |  output: {{ANNOTATED_DIR}}")
"""
    )


def pack_setup_cell_phase3(reviewer: str) -> nbf.NotebookNode:
    return code(
        f"""%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
import pandas as pd
import plotly.io as pio

pio.renderers.default = "notebook"

def _find_repo():
    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError("Could not find repo root (looking for pyproject.toml)")

REPO = _find_repo()
sys.path.insert(0, str(REPO / "scripts"))
from validation_picker_panel import (
    load_event_data, make_plotly_panel, make_annotated_png,
)

REVIEWER = {reviewer!r}
SAMPLE = "phase3"
SAMPLE_CSV = "fp_validation_events_phase3.csv"

ANNOTATED_DIR = REPO / "outputs" / "figures" / "exploratory" / "validation" / f"annotated_{{SAMPLE}}_{{REVIEWER}}"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

events = pd.read_csv(REPO / "outputs/figures/exploratory/validation" / SAMPLE_CSV)
print(f"Sample: {{SAMPLE}}  |  reviewer: {{REVIEWER}}  |  events: {{len(events)}}  |  output: {{ANNOTATED_DIR}}")
print(f"Class breakdown: {{events['phase3_class'].value_counts().to_dict()}}")
"""
    )


def build_phase_a_maleen() -> Path:
    reviewer = "maleen"
    events = pd.read_csv(PACK_CSV_DIR / "fp_validation_events.csv")

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "bravoseis", "display_name": "BRAVOSEIS", "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = list(v2_builder.intro_cells(reviewer))
    nb.cells.append(pack_setup_cell_phase_a(reviewer))

    for idx in range(1, len(events) + 1):
        ev = events.iloc[idx - 1]
        # Phase A maleen pass is blind — never pass priors.
        nb.cells.extend(v2_builder.event_cells(idx, ev, prior=None))

    nb.cells.extend(v2_builder.aggregate_cell(reviewer))

    out = NB_DIR / f"validation_relabel_50_{reviewer}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


def build_phase3_maleen() -> Path:
    reviewer = "maleen"
    events = pd.read_csv(PACK_CSV_DIR / "fp_validation_events_phase3.csv")

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "bravoseis", "display_name": "BRAVOSEIS", "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.cells = list(p3_builder.intro_cells(reviewer))
    nb.cells.append(pack_setup_cell_phase3(reviewer))

    for idx in range(1, len(events) + 1):
        ev = events.iloc[idx - 1]
        nb.cells.extend(p3_builder.event_cells(idx, ev))

    nb.cells.extend(p3_builder.aggregate_cell())

    out = NB_DIR / f"validation_relabel_50_phase3_{reviewer}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        nbf.write(nb, f)
    return out


def main():
    out_a = build_phase_a_maleen()
    print(f"Wrote {out_a}")
    out_p3 = build_phase3_maleen()
    print(f"Wrote {out_p3}")


if __name__ == "__main__":
    main()
