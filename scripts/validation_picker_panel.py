"""validation_picker_panel.py

Renderers for the Phase A re-label workflow. Provides:

  - load_event_data(idx)        load DAT segment + spectrogram for event index (1..50)
  - make_plotly_panel(data)     interactive Plotly figure for in-notebook hover/zoom
  - make_annotated_png(...)     matplotlib PNG with STA/LTA + AIC + user-corrected
                                onset lines plus a notes side-panel, for collaborator
                                review

Designed for the validation_relabel_50_v2 notebook. Plotly is used for the input
loop (hover gives sub-second time read-out, no widgets required) and matplotlib
for the shareable artifact.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, spectrogram

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from read_dat import MOORINGS, SAMPLE_RATE, list_mooring_files, read_dat

# Read DATA_ROOT from paths.yaml at the repo root. Falls back to the OOI
# JupyterHub mount point if paths.yaml is missing or doesn't set data_root.
def _load_data_root() -> Path:
    cfg_path = REPO / "paths.yaml"
    default = Path("/home/jovyan/my_data/bravoseis/NOAA")
    if not cfg_path.exists():
        return default
    try:
        import yaml
    except ImportError:
        return default
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    return Path(cfg.get("data_root", default))


DATA_ROOT = _load_data_root()
FIG_DIR = REPO / "outputs" / "figures" / "exploratory" / "validation"
ANNOTATED_DIR = FIG_DIR / "annotated_maleen"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 10
NPERSEG = 256
NOVERLAP = 192
FREQ_MAX = 250

# Okabe-Ito palette
BAND_COLORS = {"low": "#E69F00", "mid": "#56B4E9", "high": "#009E73"}
BAND_FILTERS = {
    "low":  {"btype": "lowpass",  "cutoff": 15},
    "mid":  {"btype": "bandpass", "cutoff": (15, 30)},
    "high": {"btype": "highpass", "cutoff": 30},
}

STA_LTA_COLOR = "#0072B2"   # blue, raw STA/LTA pick
AIC_COLOR     = "#D55E00"   # vermilion, AIC-refined pick (the picker output)
USER_COLOR    = "#CC79A7"   # magenta, user's corrected visual onset

_file_cache: dict = {}
_data_cache: dict = {}


def _band_filter(data, band, fs=SAMPLE_RATE, order=4):
    cfg = BAND_FILTERS[band]
    nyq = fs / 2
    if cfg["btype"] == "lowpass":
        wn = min(cfg["cutoff"] / nyq, 0.999)
    elif cfg["btype"] == "bandpass":
        lo, hi = cfg["cutoff"]
        wn = [max(lo / nyq, 0.001), min(hi / nyq, 0.999)]
    elif cfg["btype"] == "highpass":
        wn = max(cfg["cutoff"] / nyq, 0.001)
    sos = butter(order, wn, btype=cfg["btype"], output="sos")
    return sosfilt(sos, data)


def _get_catalog(mooring_key):
    if mooring_key not in _file_cache:
        info = MOORINGS[mooring_key]
        catalog = list_mooring_files(DATA_ROOT / info["data_dir"], sort_by="timestamp")
        _file_cache[mooring_key] = [(e["timestamp"], e["path"]) for e in catalog]
    return _file_cache[mooring_key]


def _get_data(filepath):
    key = str(filepath)
    if key not in _data_cache:
        if len(_data_cache) >= 3:
            del _data_cache[next(iter(_data_cache))]
        ts, data, _ = read_dat(filepath)
        _data_cache[key] = (ts, data)
    return _data_cache[key]


@dataclass
class EventData:
    idx: int                      # 1-based
    event_id: str
    mooring: str
    band: str
    snr: float
    duration_s: float
    onset_utc: pd.Timestamp       # raw STA/LTA
    onset_utc_refined: pd.Timestamp  # AIC
    peak_freq_hz: float
    onset_grade: str
    # rendering data (panel-relative seconds)
    sta_lta_s: float
    aic_s: float
    end_s: float
    spec_times: np.ndarray
    spec_freqs: np.ndarray
    spec_dB: np.ndarray
    wave_t: np.ndarray
    wave_raw: np.ndarray
    wave_filtered: np.ndarray


def load_event_data(idx: int) -> EventData:
    """Load + compute spectrogram for event idx (1-based, 1..50)."""
    events = pd.read_csv(FIG_DIR / "fp_validation_events.csv")
    events["onset_utc"] = pd.to_datetime(events["onset_utc"])
    events["onset_utc_refined"] = pd.to_datetime(events["onset_utc_refined"])

    if idx < 1 or idx > len(events):
        raise IndexError(f"idx must be 1..{len(events)}, got {idx}")
    ev = events.iloc[idx - 1]

    onset = ev["onset_utc"]
    refined = ev["onset_utc_refined"]
    duration = float(ev["duration_s"])
    band = ev["detection_band"]
    mooring = ev["mooring"]

    # Default: 10s window centered on the STA/LTA event midpoint (preserves the
    # geometry used by every event we've already labeled).
    center = onset + timedelta(seconds=duration / 2)
    t_start = center - timedelta(seconds=WINDOW_SEC / 2)
    t_end = center + timedelta(seconds=WINDOW_SEC / 2)

    # Adaptive override: if the AIC pick would fall too close to (or off) the
    # left edge, or the event end would fall past the right edge, shift and
    # widen the window to put AIC at panel-second PRE_PAD with the full event
    # visible plus POST_PAD seconds of coda context. Only triggers for the
    # subset of events where the default 10s window cannot fit picker + event.
    PRE_PAD = 2.0
    POST_PAD = 2.0
    aic_panel_default = (refined - t_start).total_seconds()
    end_panel_default = (onset - t_start).total_seconds() + duration
    if aic_panel_default < 1.0 or end_panel_default > WINDOW_SEC - 1.0:
        aic_to_end = (onset + timedelta(seconds=duration) - refined).total_seconds()
        window_len = max(WINDOW_SEC, aic_to_end + PRE_PAD + POST_PAD)
        t_start = refined - timedelta(seconds=PRE_PAD)
        t_end = t_start + timedelta(seconds=window_len)

    window_len_s = (t_end - t_start).total_seconds()
    catalog = _get_catalog(mooring)
    segment = None
    for file_ts, filepath in catalog:
        file_end = file_ts + timedelta(seconds=14400)
        if file_ts <= t_start and file_end >= t_end:
            ts, data = _get_data(filepath)
            offset_s = (t_start - ts).total_seconds()
            start_samp = int(offset_s * SAMPLE_RATE)
            end_samp = start_samp + int(window_len_s * SAMPLE_RATE)
            if start_samp < 0 or end_samp > len(data):
                raise RuntimeError(f"Event {ev['event_id']} falls outside file bounds")
            segment = data[start_samp:end_samp]
            break
    if segment is None:
        raise RuntimeError(f"No DAT file covers event {ev['event_id']}")

    sta_lta_s = (onset - t_start).total_seconds()
    aic_s = (refined - t_start).total_seconds()
    end_s = sta_lta_s + duration

    wave_t = np.arange(len(segment)) / SAMPLE_RATE
    filtered = _band_filter(segment.astype(np.float64), band)
    freqs, times, Sxx = spectrogram(segment, fs=SAMPLE_RATE, nperseg=NPERSEG, noverlap=NOVERLAP)
    fmask = freqs <= FREQ_MAX
    freqs = freqs[fmask]
    Sxx_dB = 10 * np.log10(Sxx[fmask, :] + 1e-20)

    return EventData(
        idx=idx,
        event_id=ev["event_id"],
        mooring=mooring,
        band=band,
        snr=float(ev["snr"]),
        duration_s=duration,
        onset_utc=onset,
        onset_utc_refined=refined,
        peak_freq_hz=float(ev["peak_freq_hz"]),
        onset_grade=str(ev.get("onset_grade", "")),
        sta_lta_s=sta_lta_s,
        aic_s=aic_s,
        end_s=end_s,
        spec_times=times,
        spec_freqs=freqs,
        spec_dB=Sxx_dB,
        wave_t=wave_t,
        wave_raw=segment.astype(np.float64),
        wave_filtered=filtered,
    )


def make_plotly_panel(d: EventData):
    """Build the 3-row interactive Plotly panel. Returns plotly.graph_objects.Figure.

    Hover over the spectrogram or waveform to read the time in seconds.
    Zoom in (drag-rectangle) for sub-second precision when picking the visual onset.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    vmin, vmax = np.percentile(d.spec_dB, [5, 95])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04,
        subplot_titles=("Spectrogram (dB)", "Raw waveform", f"Filtered ({d.band})"),
    )

    fig.add_trace(
        go.Heatmap(
            x=d.spec_times, y=d.spec_freqs, z=d.spec_dB,
            zmin=vmin, zmax=vmax, colorscale="Viridis",
            hovertemplate="t = %{x:.3f} s<br>f = %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
            colorbar=dict(title="dB", thickness=10, len=0.5, y=0.78),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=d.wave_t, y=d.wave_raw, mode="lines",
            line=dict(color="#444", width=0.6),
            hovertemplate="t = %{x:.3f} s<br>amp = %{y:.0f}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=d.wave_t, y=d.wave_filtered, mode="lines",
            line=dict(color="#222", width=0.7),
            hovertemplate="t = %{x:.3f} s<br>filt = %{y:.1f}<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )

    # Pick lines on every panel: STA/LTA (dashed blue), AIC (solid vermilion).
    for row in (1, 2, 3):
        fig.add_vline(
            x=d.sta_lta_s, line=dict(color=STA_LTA_COLOR, width=1.5, dash="dash"),
            row=row, col=1,
        )
        fig.add_vline(
            x=d.aic_s, line=dict(color=AIC_COLOR, width=2.2),
            row=row, col=1,
        )
        fig.add_vline(
            x=d.end_s, line=dict(color="gray", width=1, dash="dot"),
            row=row, col=1,
        )

    # Legend proxies
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=f"STA/LTA pick ({d.sta_lta_s:.3f}s)",
        line=dict(color=STA_LTA_COLOR, width=1.5, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=f"AIC pick ({d.aic_s:.3f}s)",
        line=dict(color=AIC_COLOR, width=2.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=f"event end ({d.end_s:.3f}s)",
        line=dict(color="gray", width=1, dash="dot"),
    ), row=1, col=1)

    fig.update_layout(
        title=dict(
            text=(
                f"#{d.idx} — {d.event_id}  |  {d.mooring.upper()} {d.band}  |  "
                f"SNR={d.snr:.1f}  peak={d.peak_freq_hz:.0f} Hz  dur={d.duration_s:.1f}s<br>"
                f"<sub>{d.onset_utc} (UTC)  ·  hover to read time, drag to zoom</sub>"
            ),
            font=dict(size=13),
        ),
        height=720, width=1100,
        margin=dict(l=70, r=20, t=80, b=50),
        legend=dict(orientation="h", x=0, y=-0.08),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time in panel (s)", row=3, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1, range=[0, FREQ_MAX])
    fig.update_yaxes(title_text="Raw amp", row=2, col=1)
    fig.update_yaxes(title_text=f"Filt amp", row=3, col=1)
    return fig


def make_annotated_png(
    d: EventData,
    verdict: dict,
    outpath: Optional[Path] = None,
    prior: Optional[dict] = None,
) -> Path:
    """Render the shareable PNG: spectrogram + waveforms with all three pick lines
    plus a side text panel showing label, pick_shift, notes, and (optional) prior.

    `verdict` keys: label, visual_onset_s, notes
    `prior` keys (optional): label, onset_note, comment
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    label = verdict.get("label") or ""
    v_onset = verdict.get("visual_onset_s")
    notes = verdict.get("notes") or ""
    pick_shift = None if v_onset is None else float(v_onset) - d.aic_s

    if outpath is None:
        outpath = ANNOTATED_DIR / f"event_{d.idx:02d}_annotated.png"
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(
        3, 2, figure=fig, width_ratios=[3.4, 1.0], height_ratios=[3, 2, 2],
        hspace=0.18, wspace=0.04, left=0.06, right=0.98, top=0.92, bottom=0.07,
    )
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_raw = fig.add_subplot(gs[1, 0], sharex=ax_spec)
    ax_filt = fig.add_subplot(gs[2, 0], sharex=ax_spec)
    ax_text = fig.add_subplot(gs[:, 1])
    ax_text.axis("off")

    vmin, vmax = np.percentile(d.spec_dB, [5, 95])
    ax_spec.pcolormesh(
        d.spec_times, d.spec_freqs, d.spec_dB,
        vmin=vmin, vmax=vmax, cmap="viridis", shading="auto", rasterized=True,
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(0, FREQ_MAX)
    ax_spec.set_title(
        f"#{d.idx} — {d.event_id}  |  {d.mooring.upper()} {d.band}  |  "
        f"SNR={d.snr:.1f}  peak={d.peak_freq_hz:.0f} Hz  dur={d.duration_s:.1f}s\n"
        f"{d.onset_utc} (UTC)",
        fontsize=12, fontweight="bold",
    )

    ax_raw.plot(d.wave_t, d.wave_raw, color="0.4", lw=0.4, rasterized=True)
    ax_raw.set_ylabel("Raw amp")

    ax_filt.plot(d.wave_t, d.wave_filtered, color="0.25", lw=0.5, rasterized=True)
    ax_filt.set_ylabel(f"Filt ({d.band})")
    ax_filt.set_xlabel("Time in panel (s)")

    for ax in (ax_spec, ax_raw, ax_filt):
        ax.axvline(d.sta_lta_s, color=STA_LTA_COLOR, lw=1.5, ls="--", alpha=0.95)
        ax.axvline(d.aic_s, color=AIC_COLOR, lw=2.2, alpha=0.95)
        ax.axvline(d.end_s, color="gray", lw=1, ls=":", alpha=0.8)
        if v_onset is not None:
            ax.axvline(float(v_onset), color=USER_COLOR, lw=2.5, alpha=0.95)

    handles = [
        plt.Line2D([0], [0], color=STA_LTA_COLOR, lw=1.5, ls="--",
                   label=f"STA/LTA pick ({d.sta_lta_s:.3f} s)"),
        plt.Line2D([0], [0], color=AIC_COLOR, lw=2.2,
                   label=f"AIC pick ({d.aic_s:.3f} s)"),
        plt.Line2D([0], [0], color="gray", lw=1, ls=":",
                   label=f"event end ({d.end_s:.3f} s)"),
    ]
    if v_onset is not None:
        handles.insert(2, plt.Line2D(
            [0], [0], color=USER_COLOR, lw=2.5,
            label=f"reviewer onset ({float(v_onset):.3f} s)",
        ))
    ax_spec.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    # ----- side text panel -----
    lines = []
    lines.append(("REVIEWER VERDICT", "title"))
    lines.append((f"label: {label or '—'}", "kv"))
    if pick_shift is None:
        lines.append(("pick_shift_s: —", "kv"))
    else:
        sign = "+" if pick_shift >= 0 else ""
        lines.append((f"pick_shift_s: {sign}{pick_shift:.3f}", "kv"))
        lines.append((
            f"  (reviewer onset is {abs(pick_shift):.3f}s "
            f"{'after' if pick_shift > 0 else 'before'} AIC pick)",
            "muted",
        ))
    lines.append(("", ""))
    lines.append(("notes:", "kv"))
    if notes:
        for ln in _wrap(notes, 38):
            lines.append((f"  {ln}", "body"))
    else:
        lines.append(("  —", "muted"))

    if prior is not None:
        lines.append(("", ""))
        lines.append(("PRIOR DETERMINATION", "title"))
        lines.append((f"label: {prior.get('label') or '—'}", "muted"))
        if prior.get("onset_note"):
            lines.append(("onset_note:", "muted"))
            for ln in _wrap(str(prior["onset_note"]), 38):
                lines.append((f"  {ln}", "muted"))
        if prior.get("comment"):
            lines.append(("comment:", "muted"))
            for ln in _wrap(str(prior["comment"]), 38):
                lines.append((f"  {ln}", "muted"))

    y = 0.98
    for text, kind in lines:
        if kind == "title":
            ax_text.text(0.0, y, text, fontsize=11, fontweight="bold",
                         color="#222", transform=ax_text.transAxes, va="top")
            y -= 0.045
        elif kind == "kv":
            ax_text.text(0.0, y, text, fontsize=10, color="#111",
                         transform=ax_text.transAxes, va="top")
            y -= 0.038
        elif kind == "body":
            ax_text.text(0.0, y, text, fontsize=9.5, color="#222",
                         transform=ax_text.transAxes, va="top")
            y -= 0.034
        elif kind == "muted":
            ax_text.text(0.0, y, text, fontsize=9, color="#666",
                         transform=ax_text.transAxes, va="top", style="italic")
            y -= 0.032
        else:
            y -= 0.018

    fig.savefig(outpath, dpi=200, facecolor="white")
    plt.close(fig)
    return outpath


def _wrap(text: str, width: int):
    import textwrap
    return textwrap.wrap(text, width=width) or [""]
