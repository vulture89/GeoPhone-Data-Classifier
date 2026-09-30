"""
Seismic MFCC Pipeline – Web UI
==============================
Launch:  python web_app.py
"""
from __future__ import annotations
import subprocess, sys

def _ensure_deps():
    _REQUIRED = [
        ("numpy", "numpy>=1.24"), ("pandas", "pandas>=2.0"),
        ("scipy", "scipy>=1.11"), ("librosa", "librosa>=0.10"),
        ("sklearn", "scikit-learn>=1.3"), ("matplotlib", "matplotlib>=3.7"),
        ("joblib", "joblib>=1.3"), ("flask", "flask>=3.0"),
    ]
    missing = []
    for mod, pip_name in _REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[web_app] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

_ensure_deps()

import matplotlib
matplotlib.use("Agg")

import json, logging, os, queue, threading, time
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd

# ── Logging bridge ─────────────────────────────────────────────────────
log_lines: list[str] = []
log_lock = threading.Lock()

class WebLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        with log_lock:
            log_lines.append(msg)

_handler = WebLogHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(message)s", datefmt="%H:%M:%S"))

_running = False

app = Flask(__name__)

# Suppress Flask/Werkzeug GET request logs by default
werkzeug_log = logging.getLogger("werkzeug")
werkzeug_log.setLevel(logging.ERROR)

def _install_log_handler():
    for name in ("seismic_pipeline", "train_classifier"):
        lg = logging.getLogger(name)
        lg.handlers = [h for h in lg.handlers if not isinstance(h, WebLogHandler)]
        lg.addHandler(_handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, WebLogHandler)]
    root.addHandler(_handler)
    root.setLevel(logging.INFO)

# ── API Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/logs")
def api_logs():
    since = int(request.args.get("since", 0))
    with log_lock:
        new = log_lines[since:]
    return jsonify({"lines": new, "total": len(log_lines)})

@app.route("/api/clear_logs", methods=["POST"])
def api_clear_logs():
    with log_lock:
        log_lines.clear()
    return jsonify({"ok": True})

@app.route("/api/status")
def api_status():
    return jsonify({"running": _running})

@app.route("/api/browse")
def api_browse():
    mode = request.args.get("mode", "dir")
    ext = request.args.get("ext", "")
    script = f"""
import tkinter as tk
from tkinter import filedialog
import sys
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
if '{mode}' == 'file':
    filetypes = [('{ext} files', '*{ext}')] if '{ext}' else [('All files', '*.*')]
    res = filedialog.askopenfilename(parent=root, filetypes=filetypes)
else:
    res = filedialog.askdirectory(parent=root)
sys.stdout.write(res)
sys.stdout.flush()
"""
    try:
        import subprocess
        result = subprocess.run(["python", "-c", script], capture_output=True, text=True)
        path = result.stdout.strip()
        if path:
            return jsonify({"path": path})
        return jsonify({"error": "Cancelled"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/run_pipeline", methods=["POST"])
def api_run_pipeline():
    global _running
    if _running:
        return jsonify({"error": "A task is already running."}), 409
    data = request.json
    _running = True
    threading.Thread(target=_pipeline_thread, args=(data,), daemon=True).start()
    return jsonify({"ok": True})

def _pipeline_thread(data):
    global _running
    try:
        _install_log_handler()
        from seismic_mfcc_pipeline import PipelineConfig, run_pipeline
        cfg = PipelineConfig(
            data_dir=data.get("data_dir", "data"),
            output_dir=data.get("output_dir", "output"),
            signal_column=data.get("signal_column", "voltage"),
            timestamp_column=data.get("timestamp_column", "timestamp"),
            expected_fs=float(data.get("expected_fs", 188)),
            highpass_hz=float(data.get("highpass_hz", 0.5)),
            lowpass_hz=float(data.get("lowpass_hz", 90)),
            filter_order=int(data.get("filter_order", 4)),
            sta_seconds=float(data.get("sta_seconds", 0.3)),
            lta_seconds=float(data.get("lta_seconds", 5.0)),
            trigger_ratio=float(data.get("trigger_ratio", 3.5)),
            detrigger_ratio=float(data.get("detrigger_ratio", 1.5)),
            min_event_duration_s=float(data.get("min_event_duration_s", 0.2)),
            event_window_s=float(data.get("event_window_s", 3.0)),
            event_pad_s=float(data.get("event_pad_s", 0.5)),
            n_mfcc=int(data.get("n_mfcc", 13)),
            n_fft=int(data.get("n_fft", 256)),
            hop_length=int(data.get("hop_length", 64)),
            fmin=float(data.get("fmin", 1.0)),
            fmax=float(data.get("fmax", 90.0)),
            n_mels=int(data.get("n_mels", 40)),
            cluster_method=data.get("cluster_method", "kmeans"),
            n_clusters=int(data.get("n_clusters", 5)),
            dbscan_eps=float(data.get("dbscan_eps", 1.5)),
            dbscan_min_samples=int(data.get("dbscan_min_samples", 3)),
            dim_reduction=data.get("dim_reduction", "pca"),
        )
        run_pipeline(cfg)
        with log_lock:
            log_lines.append("✅  Pipeline finished successfully.")
    except Exception as exc:
        with log_lock:
            log_lines.append(f"❌  Pipeline error: {exc}")
    finally:
        _running = False

@app.route("/api/load_profiles", methods=["POST"])
def api_load_profiles():
    csv_path = request.json.get("csv_path", "output/flagged_events_features.csv")
    if not Path(csv_path).exists():
        return jsonify({"error": f"File not found: {csv_path}"}), 404
    try:
        df = pd.read_csv(csv_path)
        if "cluster_id" not in df.columns:
            return jsonify({"error": "CSV has no 'cluster_id' column."}), 400
        for col in ["start_time", "end_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
        if "start_time" in df.columns and "end_time" in df.columns:
            df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
        agg = {"count": ("event_id", "count")}
        if "duration_s" in df.columns:
            agg["mean_duration_s"] = ("duration_s", "mean")
        if "peak_voltage" in df.columns:
            agg["mean_peak_voltage"] = ("peak_voltage", lambda x: np.mean(np.abs(x)))
        for c, n in [("spectral_centroid_mean", "mean_spectral_centroid"),
                     ("rms_mean", "mean_rms"), ("zcr_mean", "mean_zcr")]:
            if c in df.columns:
                agg[n] = (c, "mean")
        profiles = df.groupby("cluster_id").agg(**agg).round(4)
        cluster_ids = sorted(int(x) for x in df["cluster_id"].unique())
        # Build structured data for charts
        chart_data = {}
        for col in profiles.columns:
            chart_data[col] = {str(k): float(v) for k, v in profiles[col].to_dict().items()}
        # Build table data for HTML rendering
        table_rows = []
        for cid in cluster_ids:
            row = {"cluster_id": int(cid)}
            for col in profiles.columns:
                val = profiles.loc[cid, col] if cid in profiles.index else 0
                row[col] = float(val) if hasattr(val, 'item') else val
            table_rows.append(row)
        table_columns = list(profiles.columns)
        return jsonify({"profiles": profiles.to_string(), "cluster_ids": cluster_ids, "chart_data": chart_data, "table_rows": table_rows, "table_columns": table_columns})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route("/api/run_classifier", methods=["POST"])
def api_run_classifier():
    global _running
    if _running:
        return jsonify({"error": "A task is already running."}), 409
    data = request.json
    _running = True
    threading.Thread(target=_classifier_thread, args=(data,), daemon=True).start()
    return jsonify({"ok": True})

def _classifier_thread(data):
    global _running
    try:
        _install_log_handler()
        continuous_csv = data.get("continuous_csv") or None
        output_dir = data.get("output_dir", "output")
        max_feat = data.get("max_features", "sqrt")
        if max_feat == "None":
            max_feat = None
        max_depth_val = int(data.get("max_depth", 0))

        # ── Mode detection: if the CSV has valid label columns → supervised ──
        use_supervised = False
        if continuous_csv:
            from train_classifier import is_valid_labeled_file
            use_supervised = is_valid_labeled_file(continuous_csv)

        if use_supervised:
            from train_classifier import train_supervised
            with log_lock:
                log_lines.append("\U0001f4cb  Training in Supervised mode (using labeled data)")
            train_supervised(
                labeled_csv=continuous_csv,
                output_dir=output_dir,
                n_estimators=int(data.get("n_estimators", 200)),
                max_depth=max_depth_val if max_depth_val > 0 else None,
                min_samples_split=int(data.get("min_samples_split", 2)),
                min_samples_leaf=int(data.get("min_samples_leaf", 1)),
                max_features=max_feat,
                test_size=float(data.get("test_size", 0.20)),
            )
            with log_lock:
                log_lines.append("\u2705  Classifier training finished successfully (Supervised mode using labeled data).")
        else:
            # ── Unsupervised mode (current flow) ──
            from train_classifier import train_seismic_classifier
            with log_lock:
                log_lines.append("\U0001f50d  Training in Unsupervised mode (using cluster labels)")
            mapping = data.get("mapping")
            mapping_file = data.get("mapping_file") or None
            if mapping:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp_map = out_dir / "cluster_mapping.json"
                with open(tmp_map, "w") as f:
                    json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
                mapping_file = str(tmp_map)
            train_seismic_classifier(
                input_csv=data.get("input_csv", "output/flagged_events_features.csv"),
                output_dir=output_dir,
                n_estimators=int(data.get("n_estimators", 200)),
                interactive=False,
                mapping_file=mapping_file,
                max_depth=max_depth_val if max_depth_val > 0 else None,
                min_samples_split=int(data.get("min_samples_split", 2)),
                min_samples_leaf=int(data.get("min_samples_leaf", 1)),
                max_features=max_feat,
                test_size=float(data.get("test_size", 0.20)),
            )
            with log_lock:
                log_lines.append("\u2705  Classifier training finished successfully (Unsupervised mode using cluster labels).")
    except Exception as exc:
        with log_lock:
            log_lines.append(f"\u274c  Classifier error: {exc}")
    finally:
        _running = False

@app.route("/api/save_mapping", methods=["POST"])
def api_save_mapping():
    data = request.json
    mapping = data.get("mapping", {})
    output_dir = data.get("output_dir", "output")
    out_path = Path(output_dir) / "cluster_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    return jsonify({"ok": True, "path": str(out_path)})

@app.route("/api/cluster_image")
def api_cluster_image():
    csv_path = request.args.get("csv_path", "output/flagged_events_features.csv")
    img_dir = str(Path(csv_path).parent)
    img_path = Path(img_dir) / "clusters.png"
    if not img_path.exists():
        return jsonify({"error": "clusters.png not found"}), 404
    from flask import send_file
    return send_file(str(img_path), mimetype="image/png")

@app.route("/api/toggle_werkzeug", methods=["POST"])
def api_toggle_werkzeug():
    wl = logging.getLogger("werkzeug")
    if wl.level == logging.ERROR:
        wl.setLevel(logging.INFO)
        return jsonify({"enabled": True})
    else:
        wl.setLevel(logging.ERROR)
        return jsonify({"enabled": False})

HTML_TEMPLATE = _HTML_TEMPLATE = ""  # populated below

def _build_html():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seismic MFCC Pipeline</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
--bg:#0f111a;--surface:#171a25;--card:#1e2230;--border:#2a2f42;
--accent:#00b4d8;--accent2:#90e0ef;--accent-glow:rgba(0,180,216,.25);
--text:#e2e8f0;--muted:#8b9bb4;--green:#06d6a0;--red:#ef476f;--orange:#ffd166;
--terminal-bg:#090a10;--terminal-text:#60a5fa;--radius:10px;--radius-sm:6px;
--topnav-h:52px}
html,body{height:100%;font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);overflow:hidden}
/* TOP NAV */
.topnav{height:var(--topnav-h);background:var(--surface);border-bottom:1px solid var(--border);
display:flex;align-items:center;padding:0 16px;gap:12px;flex-shrink:0;z-index:100}
.topnav-brand{display:flex;align-items:center;gap:10px;margin-right:16px}
.topnav-brand .brand-icon{font-size:22px;line-height:1}
.topnav-brand .brand-name{font-size:14px;font-weight:700;color:#fff;white-space:nowrap;letter-spacing:.3px}
.topnav-brand .brand-sub{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;white-space:nowrap}
.topnav-divider{width:1px;height:28px;background:var(--border);flex-shrink:0}
.topnav-tabs{display:flex;gap:4px;flex:1;overflow-x:auto}
.topnav-tabs::-webkit-scrollbar{display:none}
.nav-btn{height:36px;padding:0 16px;border:none;border-radius:var(--radius-sm);background:none;
color:var(--muted);cursor:pointer;font-size:13px;font-weight:500;display:flex;align-items:center;
gap:7px;transition:all .2s;white-space:nowrap;flex-shrink:0}
.nav-btn:hover{color:var(--accent2);background:rgba(0,180,216,.1)}
.nav-btn.active{color:#fff;background:var(--accent);box-shadow:0 2px 8px var(--accent-glow)}
.nav-btn .nav-icon{font-size:15px}
/* STATUS */
.status-badge{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;
text-transform:uppercase;letter-spacing:1px;margin-left:auto;flex-shrink:0}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green)}
.status-dot.running{background:var(--orange);box-shadow:0 0 8px var(--orange);animation:pulse 1.2s infinite}
.status-label{font-size:11px;color:var(--muted)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
/* APP LAYOUT */
.app{display:flex;flex-direction:column;height:100vh;width:100vw}
.app-body{display:flex;flex:1;overflow:hidden;min-height:0}
.main-panel{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.terminal-panel{width:420px;min-width:200px;background:var(--terminal-bg);
border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.resize-handle{width:5px;cursor:col-resize;background:var(--border);flex-shrink:0;transition:background .2s}
.resize-handle:hover,.resize-handle.active{background:var(--accent)}
/* LOADING BAR */
.loading-bar{position:fixed;bottom:0;left:0;width:100%;height:3px;background:var(--border);z-index:2000;display:none}
.loading-bar.active{display:block}
.loading-bar .bar-fill{height:100%;width:30%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:0 2px 2px 0;animation:loadSlide 1.5s ease-in-out infinite}
@keyframes loadSlide{0%{width:10%;margin-left:0}50%{width:40%;margin-left:30%}100%{width:10%;margin-left:90%}}
/* SUCCESS TOAST */
.toast{position:fixed;top:calc(var(--topnav-h) + 12px);left:50%;transform:translateX(-50%) translateY(-20px);
background:var(--card);border:1px solid var(--green);border-radius:var(--radius);
padding:14px 24px;display:flex;align-items:center;gap:10px;
box-shadow:0 8px 32px rgba(6,214,160,.2);z-index:3000;
opacity:0;transition:opacity .35s,transform .35s;pointer-events:none;min-width:280px}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0);pointer-events:auto}
.toast.error-toast{border-color:var(--red);box-shadow:0 8px 32px rgba(239,71,111,.2)}
.toast-icon{font-size:20px;flex-shrink:0}
.toast-content{flex:1}
.toast-title{font-size:13px;font-weight:700;color:#fff;margin-bottom:2px}
.toast-msg{font-size:12px;color:var(--muted)}
.toast-close{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:0 4px;flex-shrink:0}
.toast-close:hover{color:var(--text)}
/* TAB CONTENT */
.tab-content{display:none;flex:1;overflow-y:auto;padding:20px 24px;flex-direction:column;gap:16px}
.tab-content.active{display:flex}
.tab-content::-webkit-scrollbar{width:6px}
.tab-content::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
/* CARDS */
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:18px;margin-bottom:0}
.card h3{font-size:12px;font-weight:600;color:var(--accent2);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:14px;
display:flex;align-items:center;gap:8px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px 16px}
/* FIELDS */
.field{display:flex;flex-direction:column;gap:3px;min-width:0}
.field label{font-size:11px;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.5px}
.field input,.field select{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);
padding:7px 10px;color:var(--text);font-size:13px;outline:none;transition:border-color .2s;width:100%}
.field input:focus,.field select:focus{border-color:var(--accent)}
.field select{cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b9bb4' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
background-repeat:no-repeat;background-position:right 8px center;padding-right:28px}
.field-browse{grid-column:span 2}
.field-row{display:flex;align-items:stretch}
.field-row input{flex:1;min-width:0;border-radius:var(--radius-sm) 0 0 var(--radius-sm)}
.field-row button{flex-shrink:0;padding:0 12px;background:var(--surface);border:1px solid var(--border);
border-left:none;border-radius:0 var(--radius-sm) var(--radius-sm) 0;color:var(--muted);cursor:pointer;
font-size:12px;white-space:nowrap;transition:all .2s;display:flex;align-items:center}
.field-row button:hover{border-color:var(--accent);color:var(--accent2);background:var(--card)}
/* BUTTONS */
.btn{padding:10px 28px;border:none;border-radius:var(--radius-sm);font-size:13px;font-weight:600;
cursor:pointer;transition:all .2s;text-transform:uppercase;letter-spacing:.8px}
.btn-primary{background:var(--accent);color:#fff;box-shadow:0 4px 15px var(--accent-glow)}
.btn-primary:hover{background:#0096b7;transform:translateY(-1px);box-shadow:0 6px 20px var(--accent-glow)}
.btn-primary:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-secondary{background:var(--surface);color:var(--accent2);border:1px solid var(--border)}
.btn-secondary:hover{border-color:var(--accent);background:var(--card)}
.btn-group{display:flex;gap:10px;justify-content:center;padding:12px 0}
/* PROFILES */
.profiles-output{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);
padding:12px;font-family:'Cascadia Code','Fira Code',monospace;font-size:12px;color:var(--terminal-text);
white-space:pre;overflow-x:auto;min-height:80px;max-height:400px;overflow-y:auto}
/* CLUSTER LABELS */
.cluster-labels{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.cluster-label-item{display:flex;align-items:center;gap:8px;background:var(--bg);border:1px solid var(--border);
border-radius:var(--radius-sm);padding:6px 10px}
.cluster-label-item span{font-size:12px;color:var(--accent2);font-weight:600;white-space:nowrap}
.cluster-label-item input{flex:1;background:transparent;border:none;color:var(--text);font-size:13px;outline:none}
/* TERMINAL */
.terminal-header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;
border-bottom:1px solid var(--border)}
.terminal-header h3{font-size:13px;font-weight:600;color:var(--accent2);letter-spacing:.5px}
.terminal-actions{display:flex;gap:8px;align-items:center}
.terminal-actions button{background:none;border:1px solid var(--border);border-radius:var(--radius-sm);
color:var(--muted);padding:4px 10px;font-size:11px;cursor:pointer;transition:all .2s}
.terminal-actions button:hover{border-color:var(--accent);color:var(--accent2)}
.terminal-body{flex:1;overflow-y:auto;padding:12px 16px;font-family:'Cascadia Code','Fira Code',monospace;
font-size:12px;line-height:1.6;color:var(--terminal-text)}
.terminal-body::-webkit-scrollbar{width:5px}
.terminal-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.log-line{padding:1px 0;word-break:break-all}
.log-line.success{color:var(--green)}
.log-line.error{color:var(--red)}
.toggle{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted);cursor:pointer}
.toggle-track{width:32px;height:16px;background:var(--border);border-radius:8px;position:relative;transition:background .2s}
.toggle-track.on{background:var(--accent)}
.toggle-knob{width:12px;height:12px;background:#fff;border-radius:50%;position:absolute;top:2px;left:2px;transition:left .2s}
.toggle-track.on .toggle-knob{left:18px}
/* RESPONSIVE */
@media(max-width:900px){
  .field-browse{grid-column:span 2}
}
@media(max-width:768px){
  .topnav{padding:0 10px;gap:8px}
  .brand-sub{display:none}
  .nav-btn{padding:0 10px;font-size:12px}
  .nav-btn .nav-label{display:none}
  .terminal-panel{width:100%!important;border-left:none;border-top:1px solid var(--border);max-height:38vh}
  .app-body{flex-direction:column}
  .resize-handle{height:5px;width:100%;cursor:row-resize}
  .tab-content{padding:14px 12px}
  .card{padding:14px}
  .card-grid{grid-template-columns:repeat(auto-fill,minmax(140px,1fr))}
  .field-browse{grid-column:span 2}
}
@media(max-width:480px){
  .nav-btn .nav-icon{font-size:13px}
  .topnav-brand .brand-name{font-size:12px}
  .card-grid{grid-template-columns:1fr}
  .field-browse{grid-column:span 1}
  .btn{padding:9px 18px;font-size:12px}
}
</style>
</head>
<body>
<!-- TOAST NOTIFICATION -->
<div class="toast" id="toastEl">
  <div class="toast-icon" id="toastIcon">&#x2705;</div>
  <div class="toast-content"><div class="toast-title" id="toastTitle">Done!</div><div class="toast-msg" id="toastMsg"></div></div>
  <button class="toast-close" onclick="dismissToast()">&#x00D7;</button>
</div>

<div class="app">
<!-- TOP NAV -->
<nav class="topnav">
  <div class="topnav-brand">
    <span class="brand-icon">&#x1F30D;</span>
    <div><div class="brand-name">Geophone</div><div class="brand-sub">Seismic MFCC Pipeline</div></div>
  </div>
  <div class="topnav-divider"></div>
  <div class="topnav-tabs">
    <button class="nav-btn active" id="navbtn-pipeline" onclick="switchTab('pipeline',this)">
      <span class="nav-icon">&#x2699;&#xFE0F;</span><span class="nav-label">Pipeline</span>
    </button>
    <button class="nav-btn" id="navbtn-profiles" onclick="switchTab('profiles',this)">
      <span class="nav-icon">&#x1F4CB;</span><span class="nav-label">Profiles</span>
    </button>
    <button class="nav-btn" id="navbtn-classifier" onclick="switchTab('classifier',this)">
      <span class="nav-icon">&#x1F916;</span><span class="nav-label">Classifier</span>
    </button>
  </div>
  <div class="status-badge">
    <div class="status-dot" id="statusDot"></div>
    <span class="status-label" id="statusLabel">Idle</span>
  </div>
</nav>
<div class="app-body">
<div class="main-panel">

  <!-- ═══ TAB 1: PIPELINE ═══ -->
  <div class="tab-content active" id="tab-pipeline">
    <div class="card"><h3>&#x1F4C2; Input / Output</h3>
      <div class="card-grid">
        <div class="field field-browse"><label>Data Folder</label><div class="field-row"><input id="data_dir" value="data"><button class="browse-btn" onclick="browseHint(this, 'dir')">Browse</button></div></div>
        <div class="field field-browse"><label>Output Folder</label><div class="field-row"><input id="output_dir" value="output"><button class="browse-btn" onclick="browseHint(this, 'dir')">Browse</button></div></div>
        <div class="field"><label>Signal Column</label><input id="signal_column" value="voltage"></div>
        <div class="field"><label>Timestamp Column</label><input id="timestamp_column" value="timestamp"></div>
      </div>
    </div>
    <div class="card"><h3>&#x1F50A; Signal &amp; Bandpass Filter</h3>
      <div class="card-grid">
        <div class="field"><label>Expected Fs (Hz)</label><input id="expected_fs" type="number" value="188" step="1"></div>
        <div class="field"><label>Highpass (Hz)</label><input id="highpass_hz" type="number" value="0.5" step="0.1"></div>
        <div class="field"><label>Lowpass (Hz)</label><input id="lowpass_hz" type="number" value="90" step="1"></div>
        <div class="field"><label>Filter Order</label><input id="filter_order" type="number" value="4" min="1" max="10"></div>
      </div>
    </div>
    <div class="card"><h3>&#x26A1; Event Detection (STA/LTA)</h3>
      <div class="card-grid">
        <div class="field"><label>STA (s)</label><input id="sta_seconds" type="number" value="0.3" step="0.05"></div>
        <div class="field"><label>LTA (s)</label><input id="lta_seconds" type="number" value="5.0" step="0.5"></div>
        <div class="field"><label>Trigger Ratio</label><input id="trigger_ratio" type="number" value="3.5" step="0.1"></div>
        <div class="field"><label>De-trigger</label><input id="detrigger_ratio" type="number" value="1.5" step="0.1"></div>
        <div class="field"><label>Min Duration (s)</label><input id="min_event_duration_s" type="number" value="0.2" step="0.05"></div>
        <div class="field"><label>Window (s)</label><input id="event_window_s" type="number" value="3.0" step="0.5"></div>
        <div class="field"><label>Pad (s)</label><input id="event_pad_s" type="number" value="0.5" step="0.1"></div>
      </div>
    </div>
    <div class="card"><h3>&#x1F3B5; MFCC Features</h3>
      <div class="card-grid">
        <div class="field"><label>Num MFCCs</label><input id="n_mfcc" type="number" value="13" min="1" max="40"></div>
        <div class="field"><label>FFT Size</label><input id="n_fft" type="number" value="256" step="64"></div>
        <div class="field"><label>Hop Length</label><input id="hop_length" type="number" value="64" step="16"></div>
        <div class="field"><label>F-min (Hz)</label><input id="fmin" type="number" value="1.0" step="0.5"></div>
        <div class="field"><label>F-max (Hz)</label><input id="fmax" type="number" value="90" step="1"></div>
        <div class="field"><label>Mel Bands</label><input id="n_mels" type="number" value="40" min="10" max="128"></div>
      </div>
    </div>
    <div class="card"><h3>&#x1F4CA; Clustering</h3>
      <div class="card-grid">
        <div class="field"><label>Method</label><select id="cluster_method"><option value="kmeans">K-Means</option><option value="dbscan">DBSCAN</option></select></div>
        <div class="field"><label>K (clusters)</label><input id="n_clusters" type="number" value="5" min="2" max="50"></div>
        <div class="field"><label>DBSCAN eps</label><input id="dbscan_eps" type="number" value="1.5" step="0.1"></div>
        <div class="field"><label>DBSCAN min samples</label><input id="dbscan_min_samples" type="number" value="3" min="1"></div>
        <div class="field"><label>Dim Reduction</label><select id="dim_reduction"><option value="pca">PCA</option><option value="tsne">t-SNE</option></select></div>
      </div>
    </div>
    <div class="btn-group"><button class="btn btn-primary" id="btnRunPipeline" onclick="runPipeline()">&#x25B6; Run Pipeline</button></div>
  </div>

  <!-- ═══ TAB 2: PROFILES ═══ -->
  <div class="tab-content" id="tab-profiles">
    <div class="card"><h3>&#x1F4CB; Cluster Profiles</h3>
      <div class="field" style="margin-bottom:12px"><label>Features CSV</label>
        <div class="field-row"><input id="prof_csv" value="output/flagged_events_features.csv"><button class="browse-btn" onclick="loadProfiles()">Load Profiles</button></div>
      </div>
      <div id="profilesTableContainer" style="overflow-x:auto;max-height:300px;overflow-y:auto"><p style="color:var(--muted);font-size:13px">Click "Load Profiles" to view cluster summaries.</p></div>
    </div>
    <div class="card" id="clusterNamingCard" style="display:none"><h3>&#x1F3F7; Cluster Names</h3>
      <p style="font-size:12px;color:var(--muted);margin-bottom:10px">Click a chip or a bar in the charts below to name each cluster. Duplicate names are not allowed.</p>
      <div id="clusterNamingContainer" style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px"></div>
      <div id="namingError" style="color:var(--red);font-size:12px;margin-bottom:8px;display:none"></div>
      <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <button class="btn btn-primary" onclick="saveClusterMapping()">&#x1F4BE; Save Mapping</button>
        <button class="btn btn-secondary" id="btnShowScatter" style="display:none" onclick="openScatterPopup()">&#x1F5BC; View Scatter Plot</button>
      </div>
    </div>
    <div class="card" id="chartsCard" style="display:none"><h3>&#x1F4CA; Cluster Data Visualization</h3>
      <p style="font-size:11px;color:var(--muted);margin-bottom:8px">Click a bar to name that cluster.</p>
      <div id="chartsContainer" style="display:flex;flex-wrap:wrap;gap:16px"></div>
    </div>
  </div>

  <!-- Scatter Plot Popup -->
  <div id="scatterPopup" onclick="if(event.target===this)closeScatterPopup()" style="display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.7);z-index:1000;align-items:center;justify-content:center">
    <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;max-width:90vw;max-height:90vh;display:flex;flex-direction:column;position:relative">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-shrink:0">
        <h3 style="font-size:14px;color:var(--accent2);text-transform:uppercase;letter-spacing:1px">&#x1F5BC; Cluster Scatter Plot</h3>
        <button onclick="closeScatterPopup()" style="background:none;border:1px solid var(--border);border-radius:var(--radius-sm);color:var(--muted);padding:4px 12px;cursor:pointer;font-size:14px">&times;</button>
      </div>
      <div style="text-align:center;overflow:auto;flex:1;min-height:0">
        <img id="clusterImage" src="" alt="Cluster visualization" style="max-width:100%;max-height:calc(90vh - 80px);object-fit:contain;border-radius:var(--radius-sm);border:1px solid var(--border);display:none">
        <div id="clusterImagePlaceholder" style="background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:40px 12px;color:var(--muted);font-size:13px">Cluster plot will appear here after loading profiles.</div>
      </div>
    </div>
  </div>

  <!-- Cluster Naming Popup -->
  <div id="namingPopup" onclick="if(event.target===this)closeNamingPopup()" style="display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.7);z-index:1001;align-items:center;justify-content:center">
    <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:24px;min-width:340px;max-width:450px;position:relative">
      <h3 id="namingPopupTitle" style="font-size:14px;color:var(--accent2);margin-bottom:16px">Name Cluster</h3>
      <div class="field" style="margin-bottom:12px"><label>Cluster Name</label>
        <input id="namingPopupInput" style="width:100%;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius-sm);padding:8px 12px;color:var(--text);font-size:14px;outline:none" placeholder="Enter cluster name...">
      </div>
      <div style="margin-bottom:12px"><label style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px">Quick Presets</label>
        <div id="namingPresets" style="display:flex;flex-wrap:wrap;gap:6px;margin-top:6px"></div>
      </div>
      <div id="namingPopupError" style="color:var(--red);font-size:12px;margin-bottom:10px;display:none"></div>
      <div style="display:flex;gap:10px;justify-content:flex-end">
        <button class="btn btn-secondary" onclick="closeNamingPopup()">Cancel</button>
        <button class="btn btn-primary" onclick="confirmNamingPopup()">Apply</button>
      </div>
    </div>
  </div>

  <!-- ═══ TAB 3: CLASSIFIER ═══ -->
  <div class="tab-content" id="tab-classifier">
    <div class="card"><h3>&#x1F916; Classifier Settings</h3>
      <div class="card-grid">
        <div class="field field-browse"><label>Features CSV</label><div class="field-row"><input id="cls_input" value="output/flagged_events_features.csv"><button class="browse-btn" onclick="browseHint(this, 'file', '.csv')">Browse</button></div></div>
        <div class="field field-browse"><label>Output Folder</label><div class="field-row"><input id="cls_output" value="output"><button class="browse-btn" onclick="browseHint(this, 'dir')">Browse</button></div></div>
        <div class="field field-browse"><label>Labeled Data (optional&nbsp;–&nbsp;enables Supervised mode)</label><div class="field-row"><input id="cls_continuous" placeholder="CSV with timestamp, voltage, startTime, endTime, event columns" value=""><button class="browse-btn" onclick="browseHint(this, 'file', '.csv')">Browse</button></div></div>
        <div class="field field-browse"><label>Mapping JSON</label><div class="field-row"><input id="cls_mapping" value=""><button class="browse-btn" onclick="browseHint(this, 'file', '.json')">Browse</button></div></div>
        <div class="field"><label>Trees (n_estimators)</label><input id="n_estimators" type="number" value="200" min="10" step="10"></div>
        <div class="field"><label>Max Depth (0=None)</label><input id="max_depth" type="number" value="0" min="0"></div>
        <div class="field"><label>Min Split</label><input id="min_samples_split" type="number" value="2" min="2"></div>
        <div class="field"><label>Min Leaf</label><input id="min_samples_leaf" type="number" value="1" min="1"></div>
        <div class="field"><label>Max Features</label><select id="max_features"><option value="sqrt">sqrt</option><option value="log2">log2</option><option value="None">None</option></select></div>
        <div class="field"><label>Test Size</label><input id="test_size" type="number" value="0.20" step="0.05" min="0.05" max="0.50"></div>
      </div>
    </div>
    <div class="btn-group"><button class="btn btn-primary" id="btnRunClassifier" onclick="runClassifier()">&#x25B6; Train Classifier</button></div>
  </div>
</div><!-- /main-panel -->


<!-- RESIZE HANDLE + TERMINAL -->
<div class="resize-handle" id="resizeHandle"></div>
<div class="terminal-panel" id="terminalPanel">
  <div class="terminal-header">
    <h3>&#x1F4DF; Terminal</h3>
    <div class="terminal-actions">
      <div class="toggle" onclick="toggleWerkzeug()"><span>HTTP logs</span>
        <div class="toggle-track" id="werkzeugToggle"><div class="toggle-knob"></div></div></div>
      <button onclick="clearLogs()">Clear</button>
      <button id="btnHideConsole" onclick="toggleConsole()">Hide</button>
    </div>
  </div>
  <div class="terminal-body" id="terminalBody"></div>
</div>
<button id="btnShowConsole" onclick="toggleConsole()" style="display:none;position:fixed;right:12px;bottom:12px;z-index:100;padding:8px 16px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius-sm);font-size:12px;font-weight:600;cursor:pointer;box-shadow:0 4px 15px var(--accent-glow)">&#x1F4DF; Show Console</button>
</div><!-- /app-body -->
</div><!-- /app -->
<div class="loading-bar" id="loadingBar"><div class="bar-fill"></div></div>

<script>
function switchTab(name,btn){
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}
function showLoading(){document.getElementById('loadingBar').classList.add('active');}
function hideLoading(){document.getElementById('loadingBar').classList.remove('active');}
async function browseHint(btn, type, ext){
  const inp=btn.parentElement.querySelector('input');
  try {
    const res = await fetch(`/api/browse?mode=${type}&ext=${ext || ''}`);
    const data = await res.json();
    if(data.path) { inp.value = data.path; }
  } catch(e) {
    const v = prompt('Enter ' + (type === 'file' ? 'file' : 'folder') + ' path:', inp.value);
    if(v !== null) inp.value = v;
  }
}
function getVal(id){return document.getElementById(id).value}
function setStatus(running){
  const dot=document.getElementById('statusDot');
  const lbl=document.getElementById('statusLabel');
  if(running){dot.classList.add('running');if(lbl)lbl.textContent='Running…';showLoading();}
  else{dot.classList.remove('running');if(lbl)lbl.textContent='Idle';hideLoading();}
}
let _toastTimer=null;
function showToast(title,msg,isError){
  var el=document.getElementById('toastEl');
  var ic=document.getElementById('toastIcon');
  var ti=document.getElementById('toastTitle');
  var tm=document.getElementById('toastMsg');
  ic.textContent=isError?'❌':'✅';
  ti.textContent=title;
  tm.textContent=msg||'';
  el.classList.toggle('error-toast',!!isError);
  el.classList.add('show');
  if(_toastTimer)clearTimeout(_toastTimer);
  _toastTimer=setTimeout(dismissToast,isError?8000:5000);
}
function dismissToast(){
  document.getElementById('toastEl').classList.remove('show');
  if(_toastTimer){clearTimeout(_toastTimer);_toastTimer=null;}
}
let logOffset=0;
function pollLogs(){
  fetch('/api/logs?since='+logOffset).then(r=>r.json()).then(d=>{
    const tb=document.getElementById('terminalBody');
    d.lines.forEach(l=>{const div=document.createElement('div');div.className='log-line';
      if(l.includes('\u2705')){div.classList.add('success');
        // Show success toast
        var taskName=l.includes('Classifier')?'Classifier Training':'Pipeline';
        showToast(taskName+' Complete!',l.replace(/^.*\u2705\s*/,''),false);}
      else if(l.includes('\u274c')||l.includes('ERROR')){div.classList.add('error');
        // Show error toast
        showToast('Error',l.replace(/^.*\u274c\s*/,''),true);}
      div.textContent=l;tb.appendChild(div);});
    if(d.lines.length>0)tb.scrollTop=tb.scrollHeight;logOffset=d.total;
  }).catch(()=>{});setTimeout(pollLogs,500);
}
pollLogs();
function checkStatus(){
  fetch('/api/status').then(r=>r.json()).then(d=>{setStatus(d.running);
    if(!d.running){document.getElementById('btnRunPipeline').disabled=false;
      document.getElementById('btnRunClassifier').disabled=false;}}).catch(()=>{});
}
setInterval(checkStatus,1500);
function runPipeline(){
  const btn=document.getElementById('btnRunPipeline');btn.disabled=true;setStatus(true);
  const fields=['data_dir','output_dir','signal_column','timestamp_column','expected_fs',
    'highpass_hz','lowpass_hz','filter_order','sta_seconds','lta_seconds','trigger_ratio',
    'detrigger_ratio','min_event_duration_s','event_window_s','event_pad_s',
    'n_mfcc','n_fft','hop_length','fmin','fmax','n_mels',
    'cluster_method','n_clusters','dbscan_eps','dbscan_min_samples','dim_reduction'];
  const body={};fields.forEach(f=>{body[f]=getVal(f)});
  fetch('/api/run_pipeline',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(d=>{if(d.error)alert(d.error)}).catch(e=>alert(e));
}
const CHART_COLORS=['#6c5ce7','#00cec9','#fdcb6e','#ff6b6b','#a29bfe','#55efc4','#fab1a0','#74b9ff','#fd79a8','#636e72'];
const LABEL_PRESETS=['Footstep','Vehicle','Blast','Drill','Natural','Machinery','Unknown','Animal','Wind','Rain'];
let _chartBarRects=[];  // [{canvasId,clusterIdx,clusterIdVal,x,y,w,h}]
function fmtVal(v){if(Math.abs(v)>=1000)return v.toFixed(0);if(Math.abs(v)>=10)return v.toFixed(1);if(Math.abs(v)>=1)return v.toFixed(2);return v.toPrecision(3);}
function drawBarChart(canvasId,labels,values,title,cids){
  var c=document.getElementById(canvasId);if(!c)return;var ctx=c.getContext('2d');
  var W=c.width,H=c.height,pad=55,bpad=50,top=30;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#111528';ctx.fillRect(0,0,W,H);
  ctx.fillStyle='#a29bfe';ctx.font='bold 12px sans-serif';ctx.textAlign='center';ctx.fillText(title,W/2,18);
  if(values.length===0)return;
  var maxV=Math.max(...values)*1.15||1;
  var barW=Math.max(18,Math.min(40,Math.floor((W-pad*2)/(values.length*1.5))));
  var gap=Math.floor(barW*0.5);
  var totalW=values.length*barW+(values.length-1)*gap;
  var startX=(W-totalW)/2;
  ctx.strokeStyle='#1e2545';ctx.lineWidth=1;
  for(var i=0;i<5;i++){var y=top+(H-top-bpad)*(i/4);ctx.beginPath();ctx.moveTo(pad-5,y);ctx.lineTo(W-10,y);ctx.stroke();
    ctx.fillStyle='#6b7394';ctx.font='9px sans-serif';ctx.textAlign='right';ctx.fillText(fmtVal(maxV*(1-i/4)),pad-8,y+3);}
  for(var i=0;i<values.length;i++){
    var bh=(values[i]/maxV)*(H-top-bpad);var x=startX+i*(barW+gap);var y=H-bpad-bh;
    ctx.fillStyle=CHART_COLORS[i%CHART_COLORS.length];ctx.beginPath();ctx.roundRect(x,y,barW,bh,3);ctx.fill();
    ctx.fillStyle='#cdd1e0';ctx.font='9px sans-serif';ctx.textAlign='center';ctx.fillText(fmtVal(values[i]),x+barW/2,y-4);
    ctx.fillStyle='#8892b0';ctx.fillText(labels[i],x+barW/2,H-bpad+14);
    _chartBarRects.push({canvasId:canvasId,clusterIdx:i,clusterIdVal:cids[i],x:x,y:y,w:barW,h:bh});
  }
}
function renderCharts(chartData,cids){
  _chartBarRects=[];
  var container=document.getElementById('chartsContainer');container.innerHTML='';
  var card=document.getElementById('chartsCard');card.style.display='block';
  var metrics=Object.keys(chartData);
  var labels=cids.map(function(c){return 'C'+c;});
  metrics.forEach(function(metric,idx){
    var vals=cids.map(function(c){return chartData[metric][String(c)]||0;});
    var wrap=document.createElement('div');wrap.style.cssText='background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px';
    var cvs=document.createElement('canvas');cvs.id='chart_'+idx;cvs.width=340;cvs.height=220;
    cvs.style.cursor='pointer';
    cvs.addEventListener('click',function(e){var rect=cvs.getBoundingClientRect();var mx=e.clientX-rect.left;var my=e.clientY-rect.top;
      var sx=cvs.width/rect.width,sy=cvs.height/rect.height;mx*=sx;my*=sy;
      for(var b=0;b<_chartBarRects.length;b++){var br=_chartBarRects[b];if(br.canvasId!==cvs.id)continue;
        if(mx>=br.x&&mx<=br.x+br.w&&my>=br.y&&my<=br.y+br.h){promptClusterName(br.clusterIdVal);return;}}
    });
    wrap.appendChild(cvs);container.appendChild(wrap);
    setTimeout(function(){drawBarChart(cvs.id,labels,vals,metric.replace(/_/g,' '),cids);},0);
  });
}
function renderProfilesTable(tableRows,tableColumns){
  var container=document.getElementById('profilesTableContainer');
  var html='<table style="width:100%;border-collapse:collapse;font-size:12px">';
  html+='<thead><tr style="border-bottom:2px solid var(--border)">';
  html+='<th style="padding:6px 10px;text-align:left;color:var(--accent2);font-size:11px;text-transform:uppercase">Cluster</th>';
  tableColumns.forEach(function(col){html+='<th style="padding:6px 10px;text-align:right;color:var(--accent2);font-size:11px;text-transform:uppercase">'+col.replace(/_/g,' ')+'</th>';});
  html+='</tr></thead><tbody>';
  tableRows.forEach(function(row,i){var bg=i%2===0?'transparent':'rgba(255,255,255,0.02)';
    html+='<tr style="border-bottom:1px solid var(--border);background:'+bg+'">';
    html+='<td style="padding:6px 10px;color:var(--accent);font-weight:600">Cluster '+row.cluster_id+'</td>';
    tableColumns.forEach(function(col){var v=row[col];html+='<td style="padding:6px 10px;text-align:right;color:var(--text)">'+(typeof v==='number'?fmtVal(v):v)+'</td>';});
    html+='</tr>';});
  html+='</tbody></table>';container.innerHTML=html;
}
function loadProfiles(){
  var csvPath=getVal('prof_csv');showLoading();
  fetch('/api/load_profiles',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({csv_path:csvPath})})
    .then(r=>r.json()).then(d=>{hideLoading();if(d.error){alert(d.error);return}
      // Render table
      if(d.table_rows&&d.table_columns)renderProfilesTable(d.table_rows,d.table_columns);
      else document.getElementById('profilesTableContainer').innerHTML='<pre style="color:var(--terminal-text)">'+d.profiles+'</pre>';
      // Render charts
      if(d.chart_data&&d.cluster_ids)renderCharts(d.chart_data,d.cluster_ids);
      // Load cluster image for popup
      var img=document.getElementById('clusterImage');
      var ph=document.getElementById('clusterImagePlaceholder');
      img.src='/api/cluster_image?csv_path='+encodeURIComponent(csvPath)+'&t='+Date.now();
      img.onload=function(){img.style.display='block';ph.style.display='none';document.getElementById('btnShowScatter').style.display='inline-block';};
      img.onerror=function(){img.style.display='none';ph.style.display='block';ph.textContent='clusters.png not found in output folder.';};
      // Auto-load cluster naming
      if(d.cluster_ids)buildClusterNaming(d.cluster_ids);
    }).catch(e=>{hideLoading();alert(e);});
}
let clusterIds=[];
let clusterNames={};
function buildClusterNaming(ids){
  clusterIds=ids;clusterNames={};
  var card=document.getElementById('clusterNamingCard');card.style.display='block';
  var c=document.getElementById('clusterNamingContainer');c.innerHTML='';
  clusterIds.forEach(cid=>{
    clusterNames[cid]='';
    var chip=document.createElement('div');chip.id='cname_chip_'+cid;
    chip.style.cssText='display:inline-flex;align-items:center;gap:6px;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:6px 12px;cursor:pointer;transition:border-color .2s';
    chip.onmouseenter=function(){if(!clusterNames[cid])chip.style.borderColor='var(--accent)'};
    chip.onmouseleave=function(){if(!clusterNames[cid])chip.style.borderColor='var(--border)'};
    chip.onclick=function(){promptClusterName(cid);};
    var dot=document.createElement('span');dot.style.cssText='width:10px;height:10px;border-radius:50%;background:'+CHART_COLORS[clusterIds.indexOf(cid)%CHART_COLORS.length];
    var lbl=document.createElement('span');lbl.id='cname_lbl_'+cid;lbl.style.cssText='font-size:12px;color:var(--text)';
    lbl.textContent='Cluster '+cid+': (click to name)';
    chip.appendChild(dot);chip.appendChild(lbl);c.appendChild(chip);
  });
}
let _namingCid=null;
function promptClusterName(cid){
  _namingCid=cid;
  var popup=document.getElementById('namingPopup');popup.style.display='flex';
  document.getElementById('namingPopupTitle').textContent='Name Cluster '+cid;
  document.getElementById('namingPopupInput').value=clusterNames[cid]||'';
  document.getElementById('namingPopupError').style.display='none';
  // Build presets
  var pc=document.getElementById('namingPresets');pc.innerHTML='';
  LABEL_PRESETS.forEach(function(p){
    var b=document.createElement('button');b.textContent=p;
    b.style.cssText='padding:4px 10px;background:var(--surface);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:12px;cursor:pointer;transition:all .2s';
    b.onmouseenter=function(){b.style.borderColor='var(--accent)';b.style.color='var(--accent2)'};
    b.onmouseleave=function(){b.style.borderColor='var(--border)';b.style.color='var(--text)'};
    b.onclick=function(){document.getElementById('namingPopupInput').value=p;};
    pc.appendChild(b);
  });
  setTimeout(function(){document.getElementById('namingPopupInput').focus();},50);
}
function closeNamingPopup(){document.getElementById('namingPopup').style.display='none';_namingCid=null;}
function confirmNamingPopup(){
  var name=document.getElementById('namingPopupInput').value.trim();
  var errEl=document.getElementById('namingPopupError');
  if(!name){errEl.textContent='Name cannot be empty.';errEl.style.display='block';return;}
  for(var k in clusterNames){if(String(k)!==String(_namingCid)&&clusterNames[k]===name){errEl.textContent='Duplicate: "'+name+'" is already used for Cluster '+k+'.';errEl.style.display='block';return;}}
  clusterNames[_namingCid]=name;
  var lbl=document.getElementById('cname_lbl_'+_namingCid);if(lbl)lbl.textContent='Cluster '+_namingCid+': '+name;
  var chip=document.getElementById('cname_chip_'+_namingCid);if(chip){chip.style.borderColor='var(--green)';chip.style.boxShadow='0 0 6px rgba(0,206,201,.3)';}
  document.getElementById('namingError').style.display='none';
  closeNamingPopup();
}
function openScatterPopup(){document.getElementById('scatterPopup').style.display='flex';}
function closeScatterPopup(){document.getElementById('scatterPopup').style.display='none';}
function saveClusterMapping(){
  var errEl=document.getElementById('namingError');
  // Validate all named
  var unnamed=[];
  clusterIds.forEach(function(cid){if(!clusterNames[cid])unnamed.push(cid);});
  if(unnamed.length>0){errEl.textContent='Please name all clusters. Missing: '+unnamed.map(function(c){return 'Cluster '+c;}).join(', ');errEl.style.display='block';return;}
  // Check duplicates again
  var seen={};for(var k in clusterNames){var v=clusterNames[k];if(seen[v]){errEl.textContent='Duplicate name "'+v+'" used for multiple clusters.';errEl.style.display='block';return;}seen[v]=true;}
  errEl.style.display='none';
  var mapping={};clusterIds.forEach(function(cid){mapping[String(cid)]=clusterNames[cid];});
  fetch('/api/save_mapping',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mapping:mapping,output_dir:getVal('prof_csv').replace(/\/[^\/]*$/,'')})}).then(r=>r.json()).then(d=>{
    if(d.error){errEl.textContent=d.error;errEl.style.display='block';return;}
    errEl.style.display='none';
    var msg=document.getElementById('namingError');msg.style.display='block';msg.style.color='var(--green)';msg.textContent='\u2705 Mapping saved to '+d.path;
  }).catch(e=>{errEl.textContent=String(e);errEl.style.display='block';});
}
function toggleConsole(){
  var tp=document.getElementById('terminalPanel'),rh=document.getElementById('resizeHandle'),sb=document.getElementById('btnShowConsole');
  if(tp.style.display==='none'){tp.style.display='';rh.style.display='';sb.style.display='none';}
  else{tp.style.display='none';rh.style.display='none';sb.style.display='block';}
}
function runClassifier(){
  const btn=document.getElementById('btnRunClassifier');btn.disabled=true;setStatus(true);
  const body={input_csv:getVal('cls_input'),output_dir:getVal('cls_output'),mapping_file:getVal('cls_mapping'),
    continuous_csv:getVal('cls_continuous'),
    n_estimators:getVal('n_estimators'),max_depth:getVal('max_depth'),
    min_samples_split:getVal('min_samples_split'),min_samples_leaf:getVal('min_samples_leaf'),
    max_features:getVal('max_features'),test_size:getVal('test_size')};
  if(clusterIds.length>0&&Object.keys(clusterNames).length>0){
    var allNamed=true;clusterIds.forEach(function(cid){if(!clusterNames[cid])allNamed=false;});
    if(allNamed){body.mapping={};clusterIds.forEach(function(cid){body.mapping[String(cid)]=clusterNames[cid];});}
  }
  fetch('/api/run_classifier',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(r=>r.json()).then(d=>{if(d.error)alert(d.error)}).catch(e=>alert(e));
}
function clearLogs(){fetch('/api/clear_logs',{method:'POST'}).then(()=>{
  document.getElementById('terminalBody').innerHTML='';logOffset=0;});}
function toggleWerkzeug(){fetch('/api/toggle_werkzeug',{method:'POST'}).then(r=>r.json()).then(d=>{
  document.getElementById('werkzeugToggle').classList.toggle('on',d.enabled);});}
(function(){var h=document.getElementById('resizeHandle'),p=document.getElementById('terminalPanel'),d=false,sx,sw;
h.addEventListener('mousedown',function(e){d=true;sx=e.clientX;sw=p.offsetWidth;h.classList.add('active');document.body.style.cursor='col-resize';document.body.style.userSelect='none';e.preventDefault();});
document.addEventListener('mousemove',function(e){if(!d)return;var nw=sw-(e.clientX-sx);p.style.width=Math.max(200,Math.min(window.innerWidth*0.7,nw))+'px';});
document.addEventListener('mouseup',function(){if(d){d=false;h.classList.remove('active');document.body.style.cursor='';document.body.style.userSelect='';}});})();
</script>
</body>
</html>"""

HTML_TEMPLATE = _build_html()

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("\n  Geophone - Seismic MFCC Pipeline Web UI")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=False, port=5000)
