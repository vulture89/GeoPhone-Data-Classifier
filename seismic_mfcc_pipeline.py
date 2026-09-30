#!/usr/bin/env python3
"""
Seismic / Geophone Event Classification Pipeline
=================================================
A production-ready pipeline for detecting, characterising, and clustering
vibration events captured by a geophone sensor, using Audio/Vibration Signal
Processing (MFCCs) rather than 2-D image models.

Modules
-------
1. Data Ingestion & Preprocessing
2. Event Detection  (STA / LTA trigger)
3. Low-Frequency MFCC Feature Extraction
4. Unsupervised Clustering & Labelling Prep
5. Visualisation helpers

Author : Pansilukv
License: MIT
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import signal as sp_signal
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# librosa can be noisy with low-sample-rate data – silence its warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import librosa

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seismic_pipeline")


# ======================================================================
# 0.  CONFIGURATION  (single dataclass – easy to serialise / override)
# ======================================================================
@dataclass
class PipelineConfig:
    """Central configuration for every pipeline stage."""

    # ── data ingestion ────────────────────────────────────────────────
    data_dir: str = "data"                  # folder with .csv chunks
    signal_column: str = "voltage"          # column to analyse
    timestamp_column: str = "timestamp"     # ISO-8601 timestamp column
    expected_fs: float = 188.0              # approximate sampling rate (Hz)
    fs_tolerance: float = 5.0               # Hz tolerance for fs verification
    batch_size: int = 100                   # files per processing batch (memory control)

    # ── preprocessing (bandpass) ──────────────────────────────────────
    highpass_hz: float = 0.5                # remove DC + sub-Hz drift
    lowpass_hz: float = 90.0                # Nyquist-safe ceiling
    filter_order: int = 4                   # Butterworth order

    # ── event detection (STA / LTA) ───────────────────────────────────
    sta_seconds: float = 0.3                # short-term average window
    lta_seconds: float = 5.0                # long-term average window
    trigger_ratio: float = 3.5              # STA/LTA ratio to trigger ON
    detrigger_ratio: float = 1.5            # STA/LTA ratio to trigger OFF
    min_event_duration_s: float = 0.2       # discard micro-glitches
    event_window_s: float = 3.0             # fixed window around peak (seconds)
    event_pad_s: float = 0.5               # extra padding before / after window

    # ── feature extraction (MFCC core) ────────────────────────────────
    n_mfcc: int = 13                        # number of MFCCs to extract
    n_fft: int = 256                        # FFT window length (samples)
    hop_length: int = 64                    # STFT hop (samples)
    fmin: float = 1.0                       # lowest filter-bank edge (Hz)
    fmax: float = 90.0                      # highest filter-bank edge (Hz)
    n_mels: int = 40                        # mel filter-bank channels

    # ── clustering ────────────────────────────────────────────────────
    cluster_method: str = "kmeans"          # "kmeans" or "dbscan"
    n_clusters: int = 5                     # K-Means k  (ignored for DBSCAN)
    dbscan_eps: float = 1.5                # DBSCAN neighbourhood radius
    dbscan_min_samples: int = 3            # DBSCAN min core-point neighbours
    dim_reduction: str = "pca"              # "pca" or "tsne" for visualisation
    pca_components: int = 2                 # components kept for 2-D plot

    # ── output ────────────────────────────────────────────────────────
    output_dir: str = "output"
    output_csv: str = "flagged_events_features.csv"


# ======================================================================
# 1.  DATA INGESTION & PREPROCESSING
# ======================================================================
class DataIngestor:
    """Load CSV chunks, verify sampling rate, apply bandpass filtering.

    Handles production data robustly:
      - Skips empty / corrupt CSV files with warnings
      - Computes fs from a sample of early files (not the full dataset)
      - Loads files in configurable batches to control peak memory
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._sos_filter = None  # cached filter coefficients

    # ── public API ────────────────────────────────────────────────────
    def load_all(self) -> Tuple[np.ndarray, float, pd.DatetimeIndex, List[str]]:
        """
        Returns
        -------
        signal     : 1-D float64 array (filtered voltage)
        fs         : verified sampling frequency (Hz)
        timestamps : DatetimeIndex aligned to *signal*
        filenames  : per-sample source filename list
        """
        data_path = Path(self.cfg.data_dir)
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path.resolve()}")
        log.info("Found %d CSV chunk(s) in %s", len(csv_files), data_path.resolve())

        # ── pass 1: determine fs from first valid files ───────────────
        fs = self._determine_fs(csv_files)
        log.info("Verified sampling rate: %.2f Hz", fs)

        # ── pass 2: batch-load all files ──────────────────────────────
        all_signal: List[np.ndarray] = []
        all_timestamps: List[pd.DatetimeIndex] = []
        all_filenames: List[str] = []
        n_loaded = 0
        n_skipped = 0
        total_samples = 0

        for i, fp in enumerate(csv_files):
            df = self._safe_read_csv(fp)
            if df is None:
                n_skipped += 1
                continue

            n_loaded += 1
            total_samples += len(df)
            all_signal.append(df[self.cfg.signal_column].values.astype(np.float64))
            all_timestamps.append(pd.DatetimeIndex(df[self.cfg.timestamp_column]))
            all_filenames.extend([fp.name] * len(df))

            # Progress logging every 200 files
            if (i + 1) % 200 == 0:
                log.info(
                    "  ... loaded %d / %d files  (%d samples so far)",
                    i + 1, len(csv_files), total_samples,
                )

        if n_skipped:
            log.warning("Skipped %d empty/corrupt file(s)", n_skipped)
        log.info(
            "Loaded %d file(s), %d total samples (%.1f s at %.0f Hz)",
            n_loaded, total_samples, total_samples / fs, fs,
        )

        # ── concatenate & sort by timestamp ───────────────────────────
        signal = np.concatenate(all_signal)
        timestamps = all_timestamps[0].append(all_timestamps[1:])

        # Sort by time (files are name-sorted but let's be safe)
        sort_idx = np.argsort(timestamps)
        signal = signal[sort_idx]
        timestamps = timestamps[sort_idx]
        all_filenames_arr = np.array(all_filenames)[sort_idx]
        filenames_sorted = all_filenames_arr.tolist()

        # ── bandpass filter ───────────────────────────────────────────
        log.info("Applying bandpass filter (%.1f - %.1f Hz)...",
                 self.cfg.highpass_hz, self.cfg.lowpass_hz)
        filtered = self._bandpass(signal, fs)

        return filtered, fs, timestamps, filenames_sorted

    # ── private helpers ───────────────────────────────────────────────
    def _safe_read_csv(self, fp: Path) -> Optional[pd.DataFrame]:
        """
        Read a single CSV file, returning None if it is empty, corrupt,
        or missing required columns.
        """
        try:
            # Quick check: skip truly empty files (0 bytes)
            if fp.stat().st_size == 0:
                log.debug("Skipping empty file: %s", fp.name)
                return None

            df = pd.read_csv(fp, parse_dates=[self.cfg.timestamp_column])

            # Validate required columns exist
            for col in (self.cfg.timestamp_column, self.cfg.signal_column):
                if col not in df.columns:
                    log.warning("Missing column '%s' in %s — skipping", col, fp.name)
                    return None

            if df.empty:
                log.debug("Skipping file with no data rows: %s", fp.name)
                return None

            # Drop rows with NaN in critical columns
            df.dropna(subset=[self.cfg.timestamp_column, self.cfg.signal_column], inplace=True)
            if df.empty:
                return None

            df.sort_values(self.cfg.timestamp_column, inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

        except pd.errors.EmptyDataError:
            log.debug("Skipping empty/headerless file: %s", fp.name)
            return None
        except Exception as exc:
            log.warning("Error reading %s: %s — skipping", fp.name, exc)
            return None

    def _determine_fs(self, csv_files: List[Path]) -> float:
        """Derive fs from the first few valid files (no need to read all)."""
        dt_samples: List[float] = []
        files_checked = 0

        for fp in csv_files:
            if files_checked >= 5:
                break
            df = self._safe_read_csv(fp)
            if df is None or len(df) < 10:
                continue

            dt_s = df[self.cfg.timestamp_column].diff().dt.total_seconds().dropna()
            dt_samples.extend(dt_s.values.tolist())
            files_checked += 1

        if not dt_samples:
            log.warning("Could not compute fs from data, using expected_fs=%.1f", self.cfg.expected_fs)
            return self.cfg.expected_fs

        median_dt = float(np.median(dt_samples))
        fs = round(1.0 / median_dt, 2)

        if abs(fs - self.cfg.expected_fs) > self.cfg.fs_tolerance:
            log.warning(
                "Measured fs=%.2f Hz deviates from expected %.2f Hz by >%.1f Hz",
                fs, self.cfg.expected_fs, self.cfg.fs_tolerance,
            )
        return fs

    def _bandpass(self, sig: np.ndarray, fs: float) -> np.ndarray:
        """
        Zero-phase Butterworth bandpass to remove DC drift and stay
        below the Nyquist limit.
        """
        nyq = fs / 2.0
        lo = self.cfg.highpass_hz / nyq
        hi = min(self.cfg.lowpass_hz, nyq - 1.0) / nyq  # safety margin
        sos = sp_signal.butter(self.cfg.filter_order, [lo, hi], btype="band", output="sos")
        return sp_signal.sosfiltfilt(sos, sig).astype(np.float64)


# ======================================================================
# 2.  EVENT DETECTION  (STA / LTA Trigger)
# ======================================================================
@dataclass
class DetectedEvent:
    """Metadata for a single detected seismic event."""

    index: int                          # ordinal id
    source_file: str                    # originating CSV
    start_sample: int
    end_sample: int
    peak_sample: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_time: pd.Timestamp
    peak_voltage: float
    waveform: np.ndarray = field(repr=False)  # sliced signal window


class EventDetector:
    """
    Classic STA/LTA (Short-Term Average / Long-Term Average) detector
    widely used in seismology. Operates on the *squared* (energy) signal
    to handle both polarities equally.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def detect(
        self,
        signal: np.ndarray,
        fs: float,
        timestamps: pd.DatetimeIndex,
        filenames: List[str],
    ) -> List[DetectedEvent]:
        """Return a list of DetectedEvent objects found in *signal*."""
        sta_len = int(self.cfg.sta_seconds * fs)
        lta_len = int(self.cfg.lta_seconds * fs)
        energy = signal ** 2

        # ── cumulative-sum trick for fast moving averages ─────────────
        cum = np.cumsum(energy)
        cum = np.insert(cum, 0, 0)  # prepend zero for offset indexing

        sta = (cum[sta_len:] - cum[:-sta_len]) / sta_len
        lta = (cum[lta_len:] - cum[:-lta_len]) / lta_len

        # Align: both averages reference the *end* of their window
        offset = lta_len - sta_len
        sta_aligned = sta[offset:]
        lta_aligned = lta[: len(sta_aligned)]

        # Avoid division by zero
        lta_safe = np.where(lta_aligned > 1e-12, lta_aligned, 1e-12)
        ratio = sta_aligned / lta_safe

        # ── trigger logic ─────────────────────────────────────────────
        triggered = False
        on_idx = 0
        raw_events: List[Tuple[int, int]] = []

        sample_offset = lta_len  # first valid ratio sample index in original signal
        for i, r in enumerate(ratio):
            abs_i = i + sample_offset
            if not triggered and r >= self.cfg.trigger_ratio:
                triggered = True
                on_idx = abs_i
            elif triggered and r <= self.cfg.detrigger_ratio:
                triggered = False
                if (abs_i - on_idx) / fs >= self.cfg.min_event_duration_s:
                    raw_events.append((on_idx, abs_i))

        log.info("STA/LTA detected %d raw trigger(s)", len(raw_events))

        # ── slice fixed windows centred on peak ───────────────────────
        half_win = int((self.cfg.event_window_s / 2.0) * fs)
        pad = int(self.cfg.event_pad_s * fs)
        events: List[DetectedEvent] = []

        for idx, (on, off) in enumerate(raw_events):
            peak = on + np.argmax(np.abs(signal[on:off]))
            w_start = max(0, peak - half_win - pad)
            w_end = min(len(signal), peak + half_win + pad)

            events.append(
                DetectedEvent(
                    index=idx,
                    source_file=filenames[peak],
                    start_sample=w_start,
                    end_sample=w_end,
                    peak_sample=peak,
                    start_time=timestamps[w_start],
                    end_time=timestamps[min(w_end, len(timestamps) - 1)],
                    peak_time=timestamps[peak],
                    peak_voltage=float(signal[peak]),
                    waveform=signal[w_start:w_end].copy(),
                )
            )

        log.info("Extracted %d windowed events", len(events))
        return events


# ======================================================================
# 3.  FEATURE EXTRACTION  (MFCC Core + supplementary)
# ======================================================================
class FeatureExtractor:
    """
    Extract a rich 1-D feature vector per event:
      • 13–20 MFCCs  (mean, std, max  → 3×n_mfcc values)
      • Delta-MFCCs  (mean over time   → n_mfcc values)
      • Delta-Delta-MFCCs (mean)       → n_mfcc values)
      • Zero-Crossing Rate  (mean)
      • Spectral Centroid    (mean)
      • Spectral Rolloff     (mean)
      • RMS Energy           (mean)
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> List[str]:
        """Column names for the feature vector (built lazily on first call)."""
        if self._feature_names is None:
            names: List[str] = []
            for stat in ("mean", "std", "max"):
                names += [f"mfcc{i+1}_{stat}" for i in range(self.cfg.n_mfcc)]
            names += [f"delta_mfcc{i+1}_mean" for i in range(self.cfg.n_mfcc)]
            names += [f"delta2_mfcc{i+1}_mean" for i in range(self.cfg.n_mfcc)]
            names += ["zcr_mean", "spectral_centroid_mean",
                       "spectral_rolloff_mean", "rms_mean"]
            self._feature_names = names
        return self._feature_names

    def extract(self, waveform: np.ndarray, fs: float) -> np.ndarray:
        """
        Return a 1-D feature vector for a single event waveform.

        Parameters
        ----------
        waveform : array_like   Filtered signal snippet (float64).
        fs       : float        Sampling rate in Hz.
        """
        y = waveform.astype(np.float32)

        # ── MFCCs ─────────────────────────────────────────────────────
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=fs,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
        )  # shape: (n_mfcc, T)

        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)

        # ── Delta & Delta-Delta MFCCs ─────────────────────────────────
        delta = librosa.feature.delta(mfccs, order=1)
        delta2 = librosa.feature.delta(mfccs, order=2)
        delta_mean = np.mean(delta, axis=1)
        delta2_mean = np.mean(delta2, axis=1)

        # ── Supplementary spectral features ───────────────────────────
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=fs, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=fs, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        rms = librosa.feature.rms(
            y=y, frame_length=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )

        supplementary = np.array([
            zcr.mean(), centroid.mean(), rolloff.mean(), rms.mean()
        ])

        # ── concatenate into single 1-D vector ───────────────────────
        feature_vec = np.concatenate([
            mfcc_mean, mfcc_std, mfcc_max,
            delta_mean, delta2_mean,
            supplementary,
        ])
        return feature_vec

    def extract_batch(
        self, events: List[DetectedEvent], fs: float
    ) -> np.ndarray:
        """Extract features for every event. Returns (N, D) matrix."""
        vectors = []
        for ev in events:
            vec = self.extract(ev.waveform, fs)
            vectors.append(vec)
        mat = np.vstack(vectors)
        log.info(
            "Feature matrix shape: %d events × %d features", mat.shape[0], mat.shape[1]
        )
        return mat


# ======================================================================
# 4.  UNSUPERVISED CLUSTERING & LABELLING PREP
# ======================================================================
class ClusterEngine:
    """
    Standardise → Reduce dimensionality → Cluster → Export summary CSV.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(
        self,
        feature_matrix: np.ndarray,
        events: List[DetectedEvent],
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        Full clustering pipeline.

        Returns
        -------
        summary : DataFrame with event metadata + cluster_id + features.
        """
        # ── standardise ──────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)

        # ── dimensionality reduction (for visualisation) ──────────────
        if self.cfg.dim_reduction == "tsne" and X_scaled.shape[0] > 5:
            perplexity = min(30, X_scaled.shape[0] - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            X_2d = reducer.fit_transform(X_scaled)
            log.info("t-SNE reduction complete")
        else:
            n_comp = min(self.cfg.pca_components, X_scaled.shape[1], X_scaled.shape[0])
            reducer = PCA(n_components=n_comp, random_state=42)
            X_2d = reducer.fit_transform(X_scaled)
            if hasattr(reducer, "explained_variance_ratio_"):
                log.info(
                    "PCA explained variance: %s",
                    np.round(reducer.explained_variance_ratio_, 3),
                )

        # ── clustering ────────────────────────────────────────────────
        if self.cfg.cluster_method == "dbscan":
            clusterer = DBSCAN(
                eps=self.cfg.dbscan_eps,
                min_samples=self.cfg.dbscan_min_samples,
            )
        else:
            k = min(self.cfg.n_clusters, X_scaled.shape[0])
            clusterer = KMeans(n_clusters=k, n_init=10, random_state=42)

        labels = clusterer.fit_predict(X_scaled)
        n_unique = len(set(labels) - {-1})
        log.info(
            "Clustering (%s): %d clusters found  (noise points: %d)",
            self.cfg.cluster_method,
            n_unique,
            int(np.sum(labels == -1)),
        )

        # ── build summary table ───────────────────────────────────────
        records = []
        for i, ev in enumerate(events):
            rec = {
                "event_id": ev.index,
                "filename": ev.source_file,
                "start_time": ev.start_time.isoformat(),
                "end_time": ev.end_time.isoformat(),
                "peak_time": ev.peak_time.isoformat(),
                "peak_voltage": round(ev.peak_voltage, 6),
                "cluster_id": int(labels[i]),
                "dim1": round(float(X_2d[i, 0]), 4),
                "dim2": round(float(X_2d[i, 1]), 4) if X_2d.shape[1] > 1 else 0.0,
            }
            # Append every feature value
            for fname, fval in zip(feature_names, feature_matrix[i]):
                rec[fname] = round(float(fval), 6)
            records.append(rec)

        summary = pd.DataFrame(records)

        # ── persist ───────────────────────────────────────────────────
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.cfg.output_csv
        summary.to_csv(out_path, index=False)
        log.info("Exported annotated summary → %s", out_path.resolve())

        return summary


# ======================================================================
# 5.  VISUALISATION
# ======================================================================
class Visualiser:
    """Plotting helpers for waveforms, MFCC heatmaps, and clusters."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def plot_event(
        self,
        event: DetectedEvent,
        fs: float,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Two-panel figure:
          Top    – raw (filtered) waveform of the sliced event
          Bottom – MFCC spectrogram heatmap
        """
        y = event.waveform.astype(np.float32)
        t = np.arange(len(y)) / fs

        mfccs = librosa.feature.mfcc(
            y=y,
            sr=fs,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            n_mels=self.cfg.n_mels,
        )

        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[1, 1.3], hspace=0.35)

        # ── waveform ─────────────────────────────────────────────────
        ax_wave = fig.add_subplot(gs[0])
        ax_wave.plot(t, y, linewidth=0.5, color="#2b6cb0")
        ax_wave.set_title(
            f"Event #{event.index}  │  {event.source_file}  │  "
            f"Peak {event.peak_voltage:+.4f} V",
            fontsize=10,
            fontweight="bold",
        )
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Voltage (V)")
        ax_wave.grid(True, alpha=0.3)

        # ── MFCC heatmap ─────────────────────────────────────────────
        ax_mfcc = fig.add_subplot(gs[1])
        img = librosa.display.specshow(
            mfccs,
            x_axis="time",
            sr=fs,
            hop_length=self.cfg.hop_length,
            ax=ax_mfcc,
            cmap="magma",
        )
        ax_mfcc.set_title("MFCC Coefficients", fontsize=10)
        ax_mfcc.set_ylabel("MFCC Index")
        fig.colorbar(img, ax=ax_mfcc, format="%+2.0f")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Saved event plot → %s", save_path)
        else:
            plt.show()
        plt.close(fig)

    def plot_clusters(
        self,
        summary: pd.DataFrame,
        method: str = "PCA",
        save_path: Optional[str] = None,
    ) -> None:
        """Scatter plot of events in the 2-D reduced space, coloured by cluster."""
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            summary["dim1"],
            summary["dim2"],
            c=summary["cluster_id"],
            cmap="tab10",
            s=60,
            edgecolors="k",
            linewidths=0.5,
            alpha=0.85,
        )
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(f"Event Clusters ({method.upper()} space)", fontsize=12, fontweight="bold")
        legend = ax.legend(
            *scatter.legend_elements(),
            title="Cluster",
            loc="best",
        )
        ax.add_artist(legend)
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Saved cluster plot → %s", save_path)
        else:
            plt.show()
        plt.close(fig)

    def plot_sta_lta(
        self,
        signal: np.ndarray,
        fs: float,
        timestamps: pd.DatetimeIndex,
        events: List[DetectedEvent],
        save_path: Optional[str] = None,
    ) -> None:
        """Overview plot: full signal with detected event windows highlighted."""
        t = np.arange(len(signal)) / fs
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(t, signal, linewidth=0.3, color="#4a5568", label="Filtered signal")

        for ev in events:
            ax.axvspan(
                ev.start_sample / fs,
                ev.end_sample / fs,
                alpha=0.25,
                color="red",
                label="Event" if ev.index == 0 else None,
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Full Recording — Detected Events", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            log.info("Saved overview plot → %s", save_path)
        else:
            plt.show()
        plt.close(fig)


# ======================================================================
# 6.  MAIN PIPELINE ORCHESTRATION
# ======================================================================
def run_pipeline(cfg: Optional[PipelineConfig] = None) -> pd.DataFrame:
    """
    Execute the full pipeline end-to-end.

    Parameters
    ----------
    cfg : PipelineConfig, optional
        Override the default configuration.

    Returns
    -------
    summary : pd.DataFrame
        The flagged_events_features table.
    """
    if cfg is None:
        cfg = PipelineConfig()

    log.info("=" * 60)
    log.info("  SEISMIC MFCC CLASSIFICATION PIPELINE")
    log.info("=" * 60)

    # ── 1. Ingest & preprocess ────────────────────────────────────────
    log.info("STAGE 1 ▸ Data Ingestion & Preprocessing")
    ingestor = DataIngestor(cfg)
    signal, fs, timestamps, filenames = ingestor.load_all()
    log.info("  Total samples: %d  │  Duration: %.1f s", len(signal), len(signal) / fs)

    # ── 2. Detect events ──────────────────────────────────────────────
    log.info("STAGE 2 ▸ Event Detection (STA / LTA)")
    detector = EventDetector(cfg)
    events = detector.detect(signal, fs, timestamps, filenames)

    if not events:
        log.warning("No events detected — try lowering trigger_ratio or check data.")
        return pd.DataFrame()

    # ── 3. Extract features ───────────────────────────────────────────
    log.info("STAGE 3 ▸ Feature Extraction (MFCC + spectral)")
    extractor = FeatureExtractor(cfg)
    feature_matrix = extractor.extract_batch(events, fs)

    # ── 4. Cluster ────────────────────────────────────────────────────
    log.info("STAGE 4 ▸ Unsupervised Clustering")
    cluster_engine = ClusterEngine(cfg)
    summary = cluster_engine.run(feature_matrix, events, extractor.feature_names)

    # ── 5. Visualise ──────────────────────────────────────────────────
    log.info("STAGE 5 ▸ Visualisation")
    viz = Visualiser(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Cluster scatter
    if len(events) >= 2:
        viz.plot_clusters(summary, method=cfg.dim_reduction, save_path=str(out / "clusters.png"))

    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE  [OK]")
    log.info("  Results → %s", (out / cfg.output_csv).resolve())
    log.info("=" * 60)

    return summary


# ======================================================================
# CLI Entry Point
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Seismic / Geophone MFCC Event Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing CSV chunk files.",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for results and plots.",
    )
    parser.add_argument(
        "--signal-col", default="voltage",
        help="Column name for the signal to analyse.",
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=13,
        help="Number of MFCCs to extract.",
    )
    parser.add_argument(
        "--cluster-method", choices=["kmeans", "dbscan"], default="kmeans",
        help="Clustering algorithm.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of clusters for K-Means.",
    )
    parser.add_argument(
        "--trigger-ratio", type=float, default=3.5,
        help="STA/LTA trigger ratio.",
    )
    parser.add_argument(
        "--event-window", type=float, default=3.0,
        help="Fixed event window duration (seconds).",
    )
    parser.add_argument(
        "--dim-reduction", choices=["pca", "tsne"], default="pca",
        help="Dimensionality reduction method for visualisation.",
    )

    args = parser.parse_args()

    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        signal_column=args.signal_col,
        n_mfcc=args.n_mfcc,
        cluster_method=args.cluster_method,
        n_clusters=args.n_clusters,
        trigger_ratio=args.trigger_ratio,
        event_window_s=args.event_window,
        dim_reduction=args.dim_reduction,
    )

    run_pipeline(config)
