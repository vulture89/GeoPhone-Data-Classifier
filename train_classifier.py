#!/usr/bin/env python3
"""
Seismic Event Classifier Training Script
========================================
Trains a RandomForest model on extracted MFCC features to classify seismic events.
Auto-discovers clusters from the dataset and lets you label them interactively
in the console, or use a pre-defined mapping file.

Author: Pansilukv
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_classifier")

# Metadata columns to strip out before training
METADATA_COLUMNS = [
    "Unnamed: 0",
    "event_id",
    "filename",
    "start_time",
    "end_time",
    "peak_time",
    "peak_voltage",
    "cluster_id",
    "dim1",
    "dim2"
]

# ── Default mapping (used when --no-interactive is set and no mapping file exists)
DEFAULT_CLUSTER_MAPPING = {
    0: "Block Fall / Standard Impact",
    1: "Blast / Major Impact",
    2: "Equipment Operation / Scaling",
    3: "Background Noise",
    4: "Light Impact / Footstep",
}


def _compute_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per cluster to help the user label them."""
    profile_cols = {
        "count": ("event_id", "count"),
        "mean_peak_voltage": ("peak_voltage", lambda x: np.mean(np.abs(x))),
    }
    # Add optional columns if they exist
    optional = {
        "mean_spectral_centroid": ("spectral_centroid_mean", "mean"),
        "mean_rms": ("rms_mean", "mean"),
        "mean_zcr": ("zcr_mean", "mean"),
    }
    for key, (col, agg) in optional.items():
        if col in df.columns:
            profile_cols[key] = (col, agg)

    # Duration if timestamps are available
    if "start_time" in df.columns and "end_time" in df.columns:
        try:
            df["_duration_s"] = (
                pd.to_datetime(df["end_time"]) - pd.to_datetime(df["start_time"])
            ).dt.total_seconds()
            profile_cols["mean_duration_s"] = ("_duration_s", "mean")
        except Exception:
            pass

    profiles = df.groupby("cluster_id").agg(**profile_cols).round(4)
    return profiles


def _interactive_labeling(cluster_ids: list, profiles: pd.DataFrame) -> dict:
    """Prompt the user in the console to label each discovered cluster."""
    print("\n" + "=" * 60)
    print("        INTERACTIVE CLUSTER LABELING")
    print("=" * 60)
    print("\nThe pipeline discovered the following clusters.")
    print("Review the profiles below and assign a label to each.\n")
    print(profiles.to_string())
    print("\n" + "-" * 60)

    mapping = {}
    for cid in sorted(cluster_ids):
        while True:
            try:
                label = input(f"  Label for cluster {cid}: ").strip()
            except EOFError:
                log.error("Non-interactive environment detected. Use --no-interactive or --mapping-file.")
                sys.exit(1)
            if label:
                mapping[cid] = label
                break
            print("    ⚠  Label cannot be empty. Please try again.")

    print("\n" + "-" * 60)
    print("  Your mapping:")
    for cid, label in sorted(mapping.items()):
        print(f"    Cluster {cid} → {label}")
    print("-" * 60 + "\n")

    return mapping


def _load_mapping_file(path: str) -> dict:
    """Load a cluster mapping from a JSON file.  Keys are converted to int."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def train_seismic_classifier(
    input_csv: str,
    output_dir: str,
    n_estimators: int = 100,
    interactive: bool = True,
    mapping_file: str | None = None,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
    test_size: float = 0.20,
):
    """Loads feature data, trains a Random Forest, evaluates it, and exports the model."""

    input_path = Path(input_csv)
    out_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Cannot find input dataset: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading dataset from %s", input_path)

    # 1. Load Data
    df = pd.read_csv(input_path)
    log.info("Loaded %d events with %d columns", df.shape[0], df.shape[1])

    # 2. Discover clusters
    if "cluster_id" not in df.columns:
        raise ValueError("The dataset is missing the 'cluster_id' column needed for labels.")

    cluster_ids = sorted(df["cluster_id"].unique().tolist())
    log.info("Discovered %d clusters: %s", len(cluster_ids), cluster_ids)

    # 3. Determine cluster → label mapping
    if mapping_file:
        log.info("Loading cluster mapping from %s", mapping_file)
        cluster_mapping = _load_mapping_file(mapping_file)
    elif interactive:
        profiles = _compute_cluster_profiles(df)
        cluster_mapping = _interactive_labeling(cluster_ids, profiles)
    else:
        # Fall back to default mapping
        log.info("Using default cluster mapping (non-interactive mode)")
        cluster_mapping = DEFAULT_CLUSTER_MAPPING

    # Save the mapping for future runs
    mapping_out = out_dir / "cluster_mapping.json"
    with open(mapping_out, "w") as f:
        json.dump({str(k): v for k, v in cluster_mapping.items()}, f, indent=2)
    log.info("Saved cluster mapping to %s", mapping_out)

    # 4. Map cluster_id to string labels
    df["label"] = df["cluster_id"].map(cluster_mapping)

    if df["label"].isna().any():
        n_unmapped = df["label"].isna().sum()
        log.warning(
            "Found %d events with cluster IDs not in the mapping. Dropping them.", n_unmapped
        )
        df.dropna(subset=["label"], inplace=True)

    # 5. Prepare Feature Matrix (X) and Target Vector (y)
    cols_to_drop = [col for col in METADATA_COLUMNS if col in df.columns]
    X = df.drop(columns=cols_to_drop + ["label"])
    y = df["label"]

    log.info("Prepared Feature Matrix (X) shape: %s", X.shape)
    log.info("Target Vector (y) distribution:\n%s", y.value_counts().to_string())

    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    log.info("Split data into %d training samples and %d testing samples", len(X_train), len(X_test))

    # 7. Train Random Forest Classifier
    log.info(
        "Training RandomForestClassifier (n_estimators=%d, max_depth=%s, "
        "min_samples_split=%d, min_samples_leaf=%d, max_features=%s)...",
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    log.info("Training complete.")

    # 8. Evaluate the Model
    log.info("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    print("\n" + "=" * 53)
    print("               CLASSIFICATION REPORT")
    print("=" * 53)
    print(report)
    print("=" * 53 + "\n")

    # 9. Generate and Save Confusion Matrix Plot
    cm_path = out_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
    )
    plt.title("Seismic Event Classification - Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info("Saved Confusion Matrix plot to %s", cm_path)

    # 10. Save evaluation report
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("MODE: Unsupervised (cluster labels)\n\n")
        f.write(report)
    log.info("Saved classification report to %s", report_path)

    # 11. Save the Trained Model (dict with metadata for downstream compatibility)
    label_names = {i: name for i, name in enumerate(sorted(y.unique()))}
    model_bundle = {
        "classifier": clf,
        "label_mapping": label_names,
        "feature_names": list(X.columns),
        "training_mode": "unsupervised",
    }
    model_path = out_dir / "seismic_classifier.joblib"
    joblib.dump(model_bundle, model_path)
    log.info("Saved trained model to %s", model_path)
    log.info("Trained in Unsupervised mode using cluster labels.")


# ══════════════════════════════════════════════════════════════════════
#  SUPERVISED MODE
# ══════════════════════════════════════════════════════════════════════

# Required columns (flexible naming)
_LABEL_COL_ALIASES = {"event", "label", "class"}
_SIGNAL_COL_ALIASES = {"voltage", "signal", "raw"}
_TIMESTAMP_ALIASES = {"timestamp", "time", "datetime"}


def _detect_columns(df: pd.DataFrame):
    """Detect signal, timestamp, and optional label columns.

    Returns
    -------
    tuple : (timestamp_col, signal_col, start_col | None, end_col | None, label_col | None)
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Timestamp column (required for both modes)
    ts_col = None
    for alias in _TIMESTAMP_ALIASES:
        if alias in cols_lower:
            ts_col = cols_lower[alias]
            break

    # Signal column
    signal_col = None
    for alias in _SIGNAL_COL_ALIASES:
        if alias in cols_lower:
            signal_col = cols_lower[alias]
            break

    # Label columns (optional – present only in supervised data)
    start_col = cols_lower.get("starttime") or cols_lower.get("start_time")
    end_col = cols_lower.get("endtime") or cols_lower.get("end_time")
    label_col = None
    for alias in _LABEL_COL_ALIASES:
        if alias in cols_lower:
            label_col = cols_lower[alias]
            break

    return ts_col, signal_col, start_col, end_col, label_col


def _has_real_label_values(df: pd.DataFrame, start_col, end_col, label_col) -> bool:
    """Check whether the label columns contain actual (non-empty) values."""
    if not start_col or not end_col or not label_col:
        return False
    # Check that at least some rows have non-null values in all three columns
    mask = df[start_col].notna() & df[end_col].notna() & df[label_col].notna()
    # Also filter out empty strings
    if df[label_col].dtype == object:
        mask = mask & (df[label_col].astype(str).str.strip() != "")
    return mask.sum() >= 1


def is_valid_labeled_file(path: str) -> bool:
    """Quick check: does *path* contain usable startTime + endTime + event
    columns with real values?  If yes → Supervised mode."""
    try:
        # First pass: check column names exist (cheap – only header + few rows)
        header_df = pd.read_csv(path, nrows=0)
        ts_col, signal_col, start_col, end_col, label_col = _detect_columns(header_df)
        if not signal_col or not start_col or not end_col or not label_col:
            return False
        # Second pass: read only the label columns to find real values
        # (labels may be sparse – e.g. only on first row of each event)
        label_df = pd.read_csv(path, usecols=[start_col, end_col, label_col])
        return _has_real_label_values(label_df, start_col, end_col, label_col)
    except Exception:
        return False


def train_supervised(
    labeled_csv: str,
    output_dir: str = "output",
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
    test_size: float = 0.20,
):
    """Train a classifier using a single CSV that contains both the
    continuous signal (timestamp + voltage) and label columns
    (startTime, endTime, event).

    The label columns may be filled on every row of an event or only on
    the first row – both conventions are supported.
    """
    from seismic_mfcc_pipeline import PipelineConfig, FeatureExtractor

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== SUPERVISED MODE ===")
    log.info("Loading data from %s", labeled_csv)

    df = pd.read_csv(labeled_csv)
    ts_col, signal_col, start_col, end_col, label_col = _detect_columns(df)

    if not signal_col:
        raise ValueError("CSV must contain a signal column (voltage / signal / raw).")
    if not ts_col:
        raise ValueError("CSV must contain a timestamp column.")
    if not _has_real_label_values(df, start_col, end_col, label_col):
        raise ValueError("No valid label rows found (startTime + endTime + event must have values).")

    log.info("Detected columns – ts: %s, signal: %s, start: %s, end: %s, label: %s",
             ts_col, signal_col, start_col, end_col, label_col)

    # Parse timestamps
    df[ts_col] = pd.to_datetime(df[ts_col])
    timestamps = pd.DatetimeIndex(df[ts_col])
    full_signal = df[signal_col].values.astype(np.float64)

    # Determine sampling rate from timestamps
    dt = timestamps.to_series().diff().median().total_seconds()
    fs = 1.0 / dt if dt > 0 else 188.0
    log.info("Signal: %d samples at ~%.1f Hz", len(full_signal), fs)

    # ── Extract unique event segments from label columns ──────────────
    label_rows = df.dropna(subset=[start_col, end_col, label_col]).copy()
    if label_rows[label_col].dtype == object:
        label_rows = label_rows[label_rows[label_col].astype(str).str.strip() != ""]
    label_rows[start_col] = pd.to_datetime(label_rows[start_col])
    label_rows[end_col] = pd.to_datetime(label_rows[end_col])

    # Deduplicate: group by (startTime, endTime, event) to handle
    # labels filled on every row of an event
    events_df = label_rows.groupby([start_col, end_col, label_col]).size().reset_index(name="_n")
    events_df = events_df.drop(columns=["_n"])
    log.info("Found %d labeled segments across %d classes",
             len(events_df), events_df[label_col].nunique())

    # ── Extract features per segment ──────────────────────────────────
    cfg = PipelineConfig()
    # Adapt FFT parameters so short segments produce enough STFT frames
    # for librosa delta (needs at least width=9 frames).
    # Minimum samples needed: n_fft + (9-1)*hop_length
    # Use smaller n_fft/hop if the typical segment is short.
    seg_lengths = []
    for _, ev_row in events_df.iterrows():
        mask = (timestamps >= ev_row[start_col]) & (timestamps <= ev_row[end_col])
        seg_lengths.append(int(mask.sum()))
    median_len = int(np.median(seg_lengths)) if seg_lengths else 512
    # Ensure n_fft + 8*hop <= median_len  (so delta width=9 works)
    if median_len < cfg.n_fft + 8 * cfg.hop_length:
        cfg.n_fft = max(32, min(median_len // 4, 256))
        cfg.hop_length = max(8, cfg.n_fft // 4)
        log.info("Adapted FFT params for short segments: n_fft=%d, hop_length=%d",
                 cfg.n_fft, cfg.hop_length)

    extractor = FeatureExtractor(cfg)
    feature_rows = []
    valid_labels = []

    for _, ev_row in events_df.iterrows():
        t_start = ev_row[start_col]
        t_end = ev_row[end_col]
        mask = (timestamps >= t_start) & (timestamps <= t_end)
        indices = np.where(mask)[0]
        if len(indices) < 4:
            log.warning("Segment (%s → %s) has only %d samples – skipping.",
                        t_start, t_end, len(indices))
            continue
        waveform = full_signal[indices[0]:indices[-1] + 1]
        try:
            feat = extractor.extract(waveform, fs)
            feature_rows.append(feat)
            valid_labels.append(ev_row[label_col])
        except Exception as exc:
            log.warning("Feature extraction failed for segment (%s → %s): %s",
                        t_start, t_end, exc)

    if len(feature_rows) < 2:
        raise ValueError("Not enough valid segments to train (need at least 2).")

    X = pd.DataFrame(feature_rows, columns=extractor.feature_names)
    y = pd.Series(valid_labels, name="label")

    log.info("Feature matrix: %d events × %d features", X.shape[0], X.shape[1])
    log.info("Label distribution:\n%s", y.value_counts().to_string())

    # ── Train / test split ────────────────────────────────────────────
    if len(y.unique()) < 2:
        log.warning("Only one class present – training without stratification.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    log.info("Split: %d train / %d test", len(X_train), len(X_test))

    # ── Train Random Forest ───────────────────────────────────────────
    log.info("Training RandomForestClassifier (n_estimators=%d) ...", n_estimators)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    log.info("Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\n" + "=" * 53)
    print("               CLASSIFICATION REPORT (Supervised)")
    print("=" * 53)
    print(report)
    print("=" * 53 + "\n")

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("MODE: Supervised (human labels)\n\n")
        f.write(report)
    log.info("Saved classification report to %s", report_path)

    # ── Confusion matrix ──────────────────────────────────────────────
    cm_path = out_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap="Blues", xticks_rotation=45,
    )
    plt.title("Seismic Event Classification – Confusion Matrix (Supervised)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log.info("Saved confusion matrix to %s", cm_path)

    # ── Save model bundle ─────────────────────────────────────────────
    label_names = {i: name for i, name in enumerate(sorted(y.unique()))}
    model_bundle = {
        "classifier": clf,
        "label_mapping": label_names,
        "feature_names": list(X.columns),
        "training_mode": "supervised",
    }
    model_path = out_dir / "seismic_classifier.joblib"
    joblib.dump(model_bundle, model_path)
    log.info("Saved trained model to %s", model_path)
    log.info("Trained in Supervised mode using labeled data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a supervised classifier on seismic MFCC features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/flagged_events_features.csv",
        help="Path to the extracted features CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save the trained model and plots",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Skip interactive labeling and use default or file-based mapping",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=None,
        help="Path to a JSON file with cluster_id → label mapping (e.g. output/cluster_mapping.json)",
    )

    args = parser.parse_args()

    train_seismic_classifier(
        input_csv=args.input,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        interactive=not args.no_interactive,
        mapping_file=args.mapping_file,
    )
