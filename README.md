# BRAVOSEIS Detection Validation — Independent Reviewer Pack

*AI-generated draft (Claude, Anthropic) — for review. All parameters and figures are derived from version-controlled scripts and data.*

This pack is for the **independent (blind)** inter-rater pass of the BRAVOSEIS detection validation. There are now **two notebooks**, each with 50 events under the same 4-class schema. Your verdicts on each will be compared against a primary reviewer's verdicts via Cohen's κ.

| notebook | what it samples | purpose |
|---|---|---|
| `validation_relabel_50_maleen.ipynb` | Phase A pilot — 50 events stratified by detection band (low / mid / high), drawn 2026-03-03 | Validates the **detection stage** end-to-end (includes the whale-band events that Phase 3 later discards) |
| `validation_relabel_50_phase3_maleen.ipynb` | Phase 3 catalogue sample — 25 seismic + 25 cryogenic, drawn from `phase3_catalogue.parquet` with seed=42 | Validates the **published catalogue** directly (earthquakes / T-phases + icequakes only) |

Run them in either order. They are independent labeling tasks; both produce a labels CSV and a folder of annotated PNGs. You will not see the primary reviewer's labels.

---

## Setup (OOI JupyterHub)

This pack assumes the BRAVOSEIS DAT archive is mounted at `/home/jovyan/my_data/bravoseis/NOAA`. If your environment differs, edit `paths.yaml` — only the `data_root` field needs to change.

```bash
# Clone into your home directory (path can be anywhere; the helper
# auto-detects the repo root from its own file location).
git clone <repo_url> ~/repos/bravoseis-validation-pack
cd ~/repos/bravoseis-validation-pack

# Create a local venv and install deps. This also registers a kernel.
uv sync
uv run python -m ipykernel install --user --name bravoseis-validation \
    --display-name "BRAVOSEIS Validation"
```

Open either `notebooks/validation_relabel_50_maleen.ipynb` (Phase A) or `notebooks/validation_relabel_50_phase3_maleen.ipynb` (Phase 3 catalogue) in JupyterLab and select the `BRAVOSEIS Validation` kernel. The two notebooks share the helper module and write to separate output folders, so you can work on either without disturbing the other.

---

## Picking convention

The visual reference for "true first arrival" is the **first detectable amplitude rise above background noise** in the time-domain (band-filtered or raw) waveform. This follows the PMEL convention (Fox et al. 2001; Dziak et al. 2010) and is the modality the time-domain pickers (STA/LTA, AIC, kurtosis) actually operate in. *Provisional pending B. Dziak's review.*

Spectrogram-domain energy is sometimes visible earlier than the time-domain SNR onset — narrowband components can integrate above the noise floor before the broadband amplitude rise. **Record those in the `notes` field as scientific context, but do not enter them into `visual_onset_s`.** Mixing modalities biases the picker benchmark.

---

## Schema

The Phase A notebook uses a **4-class schema**:

| Label | Meaning |
|---|---|
| **TP** | Real event, AIC pick at the true first arrival (or within ~0.1 s of it) |
| **TP-peak** | Real event, AIC pick lands on peak energy (late) |
| **TP-coda** | Real event, AIC pick lands in the coda or on a multipath arrival (late) |
| **FP** | Spurious trigger, no real signal |

The Phase 3 catalogue notebook adds a **5th class — TP-misclass** — for events where the picker did its job but the catalogue's class label is wrong (e.g., a humpback whale ending up in the cryogenic / icequake catalogue, or fin-whale leakage into the seismic catalogue). Use the `notes` field to record the suspected true class. *Added 2026-05-02 after the first event of the Phase 3 user pass surfaced a humpback in the highband_3 cluster.*

For FP and TP-misclass-without-clear-onset events, set `visual_onset_s = None`.

---

## Per-event workflow

Each event has four cells in the notebook: header markdown, Plotly display, verdict dict, render.

1. **Run the Plotly cell.** Hover over the spectrogram or waveform — the tooltip shows time in seconds within the panel. Drag-rectangle to zoom in for sub-second precision.
2. **Identify the visual true onset** (first time-domain amplitude rise above background).
3. **Edit the verdict cell:**

   ```python
   verdict_N = {
       'event_id': '...',
       'idx': N,
       'label': 'TP',           # TP / TP-peak / TP-coda / FP
       'visual_onset_s': 2.55,  # in seconds within the panel; None for FP
       'notes': 'broadband T-phase onset clearly at 2.55s; AIC pick at 2.62s; '
                'narrowband 8-10 Hz precursor visible from ~2.0s in spectrogram',
   }
   ```

4. **Run the render cell.** It writes:
   - `outputs/figures/exploratory/validation/annotated_maleen/event_NN_annotated.png` — your annotated panel (your magenta onset line + label + notes side panel)
   - `outputs/figures/exploratory/validation/annotated_maleen/event_NN_verdict.json` — your verdict, with `pick_shift_s = visual_onset_s − aic_pick_s` computed automatically

5. Move to the next event. Verdicts are persisted per-event, so you can stop and resume freely.

After all 50: run the **aggregator cell** at the bottom of the notebook. It writes a single CSV at `outputs/figures/exploratory/validation/fp_validation_labels_maleen.csv`.

---

## Adaptive panel windows

Five events (#29, 37, 39, 41, 42) use slightly extended panels (10–14 s) because the AIC pick or the event end falls outside the default 10 s window. The panel time axis still starts at zero and ticks in seconds — just use whatever range you see. The same adaptive logic runs in the primary reviewer's notebook, so your panels and theirs are geometrically identical.

---

## What to send back

When you finish the **Phase A** notebook:

- `outputs/figures/exploratory/validation/fp_validation_labels_maleen.csv`
- `outputs/figures/exploratory/validation/annotated_maleen/`

When you finish the **Phase 3 catalogue** notebook:

- `outputs/figures/exploratory/validation/fp_validation_labels_phase3_maleen.csv`
- `outputs/figures/exploratory/validation/annotated_phase3_maleen/`

Either commit each batch to your branch of this repo and we will pull, or zip the relevant folders plus CSVs and email them back.

---

## Questions

If anything in the schema is ambiguous on a given event, prefer to **label and add a comment in `notes`** rather than skip — disagreements that surface in the κ analysis are exactly the data we need. Reach out via email if you hit a structural problem (cell errors, kernel issues, data-access errors); send a screenshot of the failing cell and the event index.
