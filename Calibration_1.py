#!/usr/bin/env python3
"""Step 1 for NorSand calibration: user-data review and provisional CSL fitting.

This script is intentionally neutral:
- it reads user-supplied triaxial data,
- performs only basic data checks,
- uses the final recorded point of each test as the candidate endpoint,
- fits CSL models automatically,
- reports the fitted parameters,
- and produces clean figures.

It does NOT decide whether critical state was truly reached.
That responsibility stays with the user.

Required columns (case insensitive, common aliases accepted)
-------------------------------------------------------------
tid, e1, p, q, e

Recommended columns
-------------------
ev for drained tests, u for undrained tests

Conventions
-----------
- axial and volumetric strain in %
- stresses in kPa
- p is effective mean stress p'
- final recorded point of each test is used as the candidate endpoint
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

# --------------------------------------------------------------------------------------
# Style and configuration
# --------------------------------------------------------------------------------------
P_REF_STRAIGHT = 500.0  # e_cs = Gamma500 - lambda10 * log10(p'/500)
P_REF_CURVED = 101.3    # e_cs = Ca - Cb * (p'/101.3)^Cc

COLUMN_ALIASES = {
    "tid": ["tid", "testid", "test_id", "test", "testname", "specimen", "sample"],
    "e1": ["e1", "eps1", "axialstrain", "axial_strain", "axial_strain_pct", "axialstrainpct", "strain"],
    "ev": ["ev", "epsv", "volumetricstrain", "volumetric_strain", "volumetric_strain_pct", "volumetricstrainpct", "ev_pct"],
    "p":  ["p", "pprime", "p_kpa", "mean_effective_stress", "meaneffectivestress", "meanstress"],
    "q":  ["q", "q_kpa", "deviatorstress", "deviatoricstress"],
    "e":  ["e", "voidratio", "void_ratio"],
    "u":  ["u", "du", "delta_u", "porepressure", "excessporepressure", "pwp"],
}

PLOT_COLORS = {"CID": "0.10", "CIU": "0.48", "UNKNOWN": "0.70"}
PLOT_LINESTYLES = {"CID": "-", "CIU": "--", "UNKNOWN": ":"}
PLOT_MARKERS = {"CID": "o", "CIU": "s", "UNKNOWN": "^"}

FS, AXIS_FS, TICK_FS, LEGEND_FS = 19, 22, 18, 17
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": FS,
    "axes.labelsize": AXIS_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
    "legend.title_fontsize": LEGEND_FS,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "axes.grid": True,
    "grid.color": "0.90",
    "grid.linewidth": 0.6,
    "legend.edgecolor": "0.5",
    "legend.fancybox": False,
    "lines.linewidth": 1.8,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
    "pdf.fonttype": 42,
})


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def panel(ax, s: str) -> None:
    ax.text(0.0, 1.015, s, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS)


def save(fig, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=0.20)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def logx(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())


def infer_test_type(name: str) -> str:
    m = re.search(r"(^|[^A-Za-z])(CID|CIU)(?=[^A-Za-z]|$)", str(name), re.I)
    return m.group(2).upper() if m else "UNKNOWN"


# --------------------------------------------------------------------------------------
# Input and checking
# --------------------------------------------------------------------------------------
def load_data(filename: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(filename, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError("Uploaded file contains no data.")

    original = {normalize(c): c for c in raw.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = normalize(alias)
            if key in original:
                rename[original[key]] = canonical
                break

    df = raw.rename(columns=rename).copy()
    required = ["tid", "e1", "p", "q", "e"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Required columns are tid, e1, p, q, e."
        )

    for col in ["e1", "ev", "p", "q", "e", "u"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tid"] = df["tid"].astype("string").str.strip()
    df["source_row"] = np.arange(2, len(df) + 2)

    invalid = df["tid"].isna() | df["tid"].eq("")
    invalid |= ~np.isfinite(df[["e1", "p", "q", "e"]]).all(axis=1)
    bad_rows = df.loc[invalid, ["source_row", "tid"]].copy()

    clean = df.loc[~invalid].copy()
    if clean.empty:
        raise ValueError("No usable rows found after basic input checking.")

    return clean, bad_rows, rename


def summarize_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, g in df.groupby("tid", sort=False):
        g = g.copy()
        typ = infer_test_type(tid)
        de1 = np.diff(g["e1"].to_numpy(float))
        warnings = []

        if (g["p"] <= 0).any():
            warnings.append(f"{int((g['p'] <= 0).sum())} nonpositive p' row(s)")
        if (de1 < -1e-9).any():
            warnings.append(f"{int((de1 < -1e-9).sum())} reversed strain increment(s)")
        if (np.abs(de1) <= 1e-9).any():
            warnings.append(f"{int((np.abs(de1) <= 1e-9).sum())} duplicate strain increment(s)")
        if typ == "CID" and g["ev"].isna().all():
            warnings.append("CID volumetric strain missing")
        if typ == "CIU" and g["u"].isna().all():
            warnings.append("CIU pore pressure missing")
        if typ == "UNKNOWN":
            warnings.append("test type not identified from test name")
        if abs(float(g["q"].iloc[0])) > max(2.0, 0.01 * abs(float(g["q"].max()))):
            warnings.append("nonzero initial q: verify offset/anisotropy")

        rows.append({
            "test_id": str(tid),
            "type": typ,
            "n_rows": int(len(g)),
            "e1_start_pct": float(g["e1"].iloc[0]),
            "e1_end_pct": float(g["e1"].iloc[-1]),
            "p0_kPa": float(g["p"].iloc[0]),
            "q0_kPa": float(g["q"].iloc[0]),
            "e0": float(g["e"].iloc[0]),
            "p_end_kPa": float(g["p"].iloc[-1]),
            "q_end_kPa": float(g["q"].iloc[-1]),
            "e_end": float(g["e"].iloc[-1]),
            "eta_end": float(g["q"].iloc[-1] / g["p"].iloc[-1]) if float(g["p"].iloc[-1]) > 0 else np.nan,
            "basic_data_warning": "; ".join(warnings) if warnings else "No basic data flags",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------
def straight_csl(p, gamma500, lambda10):
    p = np.asarray(p, float)
    return gamma500 - lambda10 * np.log10(p / P_REF_STRAIGHT)


def curved_csl(p, ca, cb, cc):
    p = np.asarray(p, float)
    return ca - cb * (p / P_REF_CURVED) ** cc


def endpoint_M(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    mask = np.isfinite(p) & np.isfinite(q) & (p > 0)
    p, q = p[mask], q[mask]
    if len(p) < 2:
        return np.nan, np.nan
    # Through-origin fit q = M p.
    M = float(np.dot(p, q) / np.dot(p, p))
    rmse = float(np.sqrt(np.mean((q - M * p) ** 2)))
    return M, rmse


def fit_models(endpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    v = endpoints[np.isfinite(endpoints["p_end_kPa"]) & np.isfinite(endpoints["e_end"]) & (endpoints["p_end_kPa"] > 0)].copy()
    if len(v) < 4:
        raise ValueError("At least four candidate endpoints with positive p' are required.")

    p = v["p_end_kPa"].to_numpy(float)
    e = v["e_end"].to_numpy(float)
    q = v["q_end_kPa"].to_numpy(float)

    # straight CSL
    x = np.log10(p / P_REF_STRAIGHT)
    X = np.c_[np.ones(len(p)), -x]
    theta, *_ = np.linalg.lstsq(X, e, rcond=None)
    gamma500, lambda10 = map(float, theta)
    pred_straight = straight_csl(p, gamma500, lambda10)

    # curved CSL with multi-start least squares
    candidates = []
    for ca0 in [max(e) + 0.03, max(e) + 0.08, 1.0]:
        for cb0 in [0.03, 0.08, 0.15]:
            for cc0 in [0.20, 0.35, 0.50, 0.80]:
                try:
                    fit = least_squares(
                        lambda a: curved_csl(p, *a) - e,
                        x0=[ca0, cb0, cc0],
                        bounds=([0.0, 1e-8, 0.01], [4.0, 4.0, 2.0]),
                        max_nfev=20000,
                        xtol=1e-12,
                        ftol=1e-12,
                        gtol=1e-12,
                    )
                    if fit.success:
                        candidates.append(fit)
                except ValueError:
                    pass
    curved_ok = len(candidates) > 0
    if curved_ok:
        best = min(candidates, key=lambda r: np.sum(r.fun ** 2))
        ca, cb, cc = map(float, best.x)
        pred_curved = curved_csl(p, ca, cb, cc)
    else:
        ca = cb = cc = np.nan
        pred_curved = np.full_like(e, np.nan)

    total_ss = float(np.sum((e - e.mean()) ** 2))
    rows = []
    model_params = {
        "semi_log": {"Gamma500": gamma500, "lambda10": lambda10, "lambda_ln": lambda10 / np.log(10), "Gamma100": float(straight_csl(100.0, gamma500, lambda10))}
    }

    for name, pred, k in [("semi_log", pred_straight, 2), ("curved", pred_curved, 3)]:
        if name == "curved" and not curved_ok:
            continue
        rss = float(np.sum((e - pred) ** 2))
        rows.append({
            "model": name,
            "n": len(e),
            "number_parameters": k,
            "RMSE": float(np.sqrt(rss / len(e))),
            "residual_sd": float(np.sqrt(rss / max(len(e) - k, 1))),
            "R2": float(1 - rss / total_ss) if total_ss > 0 else np.nan,
            "AIC": float(len(e) * np.log(max(rss / len(e), 1e-30)) + 2 * k),
            "BIC": float(len(e) * np.log(max(rss / len(e), 1e-30)) + k * np.log(len(e))),
            "p_min_kPa": float(p.min()),
            "p_max_kPa": float(p.max()),
        })
    if curved_ok:
        model_params["curved"] = {"Ca": ca, "Cb": cb, "Cc": cc}

    stats = pd.DataFrame(rows)
    stats = stats.merge(pd.DataFrame(model_params).T.reset_index().rename(columns={"index": "model"}), on="model", how="left")

    v["e_pred_semi_log"] = pred_straight
    if curved_ok:
        v["e_pred_curved"] = pred_curved

    M_end, M_rmse = endpoint_M(p, q)
    stress_space = {"M_endpoint": M_end, "RMSE_q": M_rmse}
    return stats, v, model_params, stress_space


def selected_csl_values(p, model_params: dict, selected_model: str):
    if selected_model == "semi_log":
        par = model_params["semi_log"]
        return straight_csl(p, par["Gamma500"], par["lambda10"])
    par = model_params["curved"]
    return curved_csl(p, par["Ca"], par["Cb"], par["Cc"])


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_fig1_endpoints_and_fits(endpoints: pd.DataFrame, model_params: dict, stress_space: dict, outdir: Path):
    # Compare both candidate CSL forms against the user-supplied final points.
    fig, ax = plt.subplots(figsize=(12, 9))
    lo = float(endpoints["p_end_kPa"].min())
    hi = float(endpoints["p_end_kPa"].max())
    pp = np.geomspace(lo, hi, 250)

    for typ, g in endpoints.groupby("type", sort=False):
        ax.scatter(g["p_end_kPa"], g["e_end"], marker=PLOT_MARKERS[typ], s=125,
                   fc="white" if typ == "CID" else "0.55", ec="k", lw=1.2, zorder=3)

    s = model_params["semi_log"]
    ax.plot(pp, straight_csl(pp, s["Gamma500"], s["lambda10"]), "k-", lw=2.3)
    if "curved" in model_params:
        c = model_params["curved"]
        ax.plot(pp, curved_csl(pp, c["Ca"], c["Cb"], c["Cc"]),
                color="0.35", ls="--", lw=2.3)

    logx(ax)
    ax.set_xlim(lo * 0.85, hi * 1.15)
    ax.set_xlabel(r"Mean effective stress, $p^{\prime}$ (kPa)")
    ax.set_ylabel("Candidate endpoint void ratio, $e$")

    handles = [Line2D([], [], color="k", lw=2.3, label="Semi-log CSL")]
    if "curved" in model_params:
        handles.append(Line2D([], [], color="0.35", ls="--", lw=2.3, label="Curved CSL"))
    handles.extend([
        Line2D([], [], marker="o", mfc="white", mec="k", ms=11, ls="", label="CID endpoints"),
        Line2D([], [], marker="s", mfc="0.55", mec="k", ms=11, ls="", label="CIU endpoints"),
    ])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005),
               ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, outdir, "Fig1_CSL_endpoint_fits")


def plot_fig2_paths(df: pd.DataFrame, model_params: dict, stress_space: dict, selected_model: str, outdir: Path):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(23, 9))

    positive_p = df.loc[df["p"] > 0, "p"]
    pmin = float(max(1.0, positive_p.min()))
    pmax = float(positive_p.max())
    pp = np.geomspace(pmin, pmax, 250)
    efit = selected_csl_values(pp, model_params, selected_model)

    for tid, g in df.groupby("tid", sort=False):
        typ = infer_test_type(tid)
        col = PLOT_COLORS[typ]
        ls = PLOT_LINESTYLES[typ]

        gp = g[g["p"] > 0]
        if len(gp) >= 2:
            ax.plot(gp["p"], gp["e"], color=col, ls=ls, lw=1.4)
            ax.plot(gp["p"].iloc[0], gp["e"].iloc[0], "o", mfc="white", mec="k", ms=6)
            ax.plot(gp["p"].iloc[-1], gp["e"].iloc[-1], "s", mfc=col, mec="k", ms=7)

        bx.plot(g["p"], g["q"], color=col, ls=ls, lw=1.4)
        bx.plot(g["p"].iloc[0], g["q"].iloc[0], "o", mfc="white", mec="k", ms=6)
        bx.plot(g["p"].iloc[-1], g["q"].iloc[-1], "s", mfc=col, mec="k", ms=7)

    ax.plot(pp, efit, "k-", lw=2.4)
    M = stress_space["M_endpoint"]
    bx.plot([0, pmax * 1.05], [0, M * pmax * 1.05], "k-", lw=2.0)

    logx(ax)
    ax.set_xlim(pmin * 0.85, pmax * 1.15)
    ax.set_xlabel(r"Mean effective stress, $p^{\prime}$ (kPa)")
    ax.set_ylabel("Void ratio, $e$")
    bx.set_xlim(0, None)
    bx.set_ylim(0, None)
    bx.set_xlabel(r"Mean effective stress, $p^{\prime}$ (kPa)")
    bx.set_ylabel("Deviatoric stress, $q$ (kPa)")
    panel(ax, "(a)")
    panel(bx, "(b)")

    csl_label = "Selected CSL: semi-log" if selected_model == "semi_log" else "Selected CSL: curved"
    handles = [
        Line2D([], [], color="0.10", lw=1.8, ls="-", label="CID"),
        Line2D([], [], color="0.48", lw=1.8, ls="--", label="CIU"),
        Line2D([], [], marker="o", mfc="white", mec="k", ls="", ms=9, label="First recorded point"),
        Line2D([], [], marker="s", mfc="0.4", mec="k", ls="", ms=9, label="Final recorded point"),
        Line2D([], [], color="k", lw=2.4, label=csl_label),
        Line2D([], [], color="k", lw=2.0, label=rf"$q = M_{{end}} p^\prime$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, outdir, "Fig2_State_and_stress_paths")


def plot_individual_test(g: pd.DataFrame, model_params: dict, stress_space: dict, selected_model: str, outdir: Path):
    tid = str(g["tid"].iloc[0])
    typ = infer_test_type(tid)
    fig, axs = plt.subplots(2, 2, figsize=(21, 15))
    fig.suptitle(f"{tid} ({typ})", fontsize=FS, y=1.01)

    # common references for this test
    gp = g[g["p"] > 0]
    pmin = float(max(1.0, gp["p"].min())) if len(gp) else 1.0
    pmax = float(gp["p"].max()) if len(gp) else 10.0
    M = stress_space["M_endpoint"]

    # (a) q - e1
    ax = axs[0, 0]
    ax.plot(g["e1"], g["q"], "k-", lw=2.0)
    ax.plot(g["e1"].iloc[-1], g["q"].iloc[-1], "s", mfc="k", mec="k", ms=7)
    ax.set_xlabel(r"Axial strain, $\varepsilon_1$ (%)")
    ax.set_ylabel("Deviatoric stress, $q$ (kPa)")
    panel(ax, "(a)")

    # (b) q - p' with q = M p'
    ax = axs[0, 1]
    ax.plot(g["p"], g["q"], "k-", lw=2.0)
    ax.plot(g["p"].iloc[-1], g["q"].iloc[-1], "s", mfc="k", mec="k", ms=7)
    pspan = np.array([0.0, max(float(g["p"].max()) * 1.05, 1.0)])
    ax.plot(pspan, M * pspan, color="0.35", ls="--", lw=1.8)
    ax.set_xlabel(r"Mean effective stress, $p^{\prime}$ (kPa)")
    ax.set_ylabel("Deviatoric stress, $q$ (kPa)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    panel(ax, "(b)")

    # (c) ev or du - e1
    ax = axs[1, 0]
    if typ == "CID" and g["ev"].notna().any():
        ax.plot(g["e1"], g["ev"], "k-", lw=2.0)
        ax.set_ylabel(r"Volumetric strain, $\varepsilon_v$ (%)")
        ax.axhline(0, color="0.5", ls=":", lw=1.0)
    elif typ == "CIU" and g["u"].notna().any():
        ax.plot(g["e1"], g["u"], "k-", lw=2.0)
        ax.set_ylabel(r"Excess pore pressure, $\Delta u$ (kPa)")
    else:
        ax.text(0.5, 0.5, "No auxiliary response available", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Auxiliary response")
    ax.set_xlabel(r"Axial strain, $\varepsilon_1$ (%)")
    panel(ax, "(c)")

    # (d) e - log p' laboratory path only
    ax = axs[1, 1]
    if len(gp):
        ax.plot(gp["p"], gp["e"], "k-", lw=2.0)
        ax.plot(gp["p"].iloc[-1], gp["e"].iloc[-1], "s", mfc="k", mec="k", ms=7)
        logx(ax)
        ax.set_xlim(pmin * 0.85, pmax * 1.15)
    ax.set_xlabel(r"Mean effective stress, $p^{\prime}$ (kPa)")
    ax.set_ylabel("Void ratio, $e$")
    panel(ax, "(d)")

    handles = [
        Line2D([], [], color="k", lw=2.0, label="Laboratory record"),
        Line2D([], [], marker="s", mfc="k", mec="k", ls="", ms=9, label="Final recorded point"),
        Line2D([], [], color="0.35", ls="--", lw=1.8, label=("Selected CSL" if selected_model in ["semi_log", "curved"] else "Reference line")),
        Line2D([], [], color="0.35", ls="--", lw=1.8, label=rf"$q = M_{{end}} p^\prime$"),
    ]
    # Deduplicate same style entry names by rebuilding explicit handles below
    handles = [
        Line2D([], [], color="k", lw=2.0, label="Laboratory record"),
        Line2D([], [], marker="s", mfc="k", mec="k", ls="", ms=9, label="Final recorded point"),
        Line2D([], [], color="0.35", ls="--", lw=1.8, label=rf"$q = M_{{end}} p^\prime$ (panel b)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    safe_tid = re.sub(r"[^A-Za-z0-9_-]", "_", tid)
    save(fig, outdir, f"Test_{safe_tid}")


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def write_summary(stats: pd.DataFrame, quality: pd.DataFrame, invalid: pd.DataFrame,
                  renamed: dict, model_params: dict, stress_space: dict,
                  selected_model: str, outdir: Path) -> None:
    sf = stats.set_index("model")
    lines = []
    lines.append("NORSAND STEP 1: USER DATA REVIEW AND PROVISIONAL CSL FIT")
    lines.append("")
    lines.append(f"Number of tests: {len(quality)}")
    lines.append(f"Number of valid data rows: {int(quality['n_rows'].sum())}")
    lines.append(f"Invalid / incomplete rows omitted: {len(invalid)}")
    lines.append("Column mapping: " + ", ".join(f"{src} -> {dst}" for src, dst in renamed.items()))
    lines.append("")
    lines.append("ASSUMPTION USED HERE")
    lines.append("The final recorded point of each uploaded test is treated as the candidate endpoint.")
    lines.append("This script does not decide whether critical state was truly reached.")
    lines.append("")
    lines.append("FITTED CSL PARAMETERS")
    s = sf.loc["semi_log"]
    lines.append("Semi-log CSL: e_cs = Gamma500 - lambda10 log10(p'/500)")
    lines.append(f"  Gamma500 = {s['Gamma500']:.6f}")
    lines.append(f"  lambda10 = {s['lambda10']:.6f}")
    lines.append(f"  lambda_ln = {s['lambda_ln']:.6f}")
    lines.append(f"  Gamma100 = {s['Gamma100']:.6f}")
    lines.append(f"  RMSE = {s['RMSE']:.6f}; R2 = {s['R2']:.6f}")
    if "curved" in sf.index:
        c = sf.loc["curved"]
        lines.append("")
        lines.append("Curved CSL: e_cs = Ca - Cb (p'/101.3)^Cc")
        lines.append(f"  Ca = {c['Ca']:.6f}")
        lines.append(f"  Cb = {c['Cb']:.6f}")
        lines.append(f"  Cc = {c['Cc']:.6f}")
        lines.append(f"  RMSE = {c['RMSE']:.6f}; R2 = {c['R2']:.6f}")
    lines.append("")
    lines.append(f"Stress-space endpoint line: q = M_end p' with M_end = {stress_space['M_endpoint']:.6f}")
    lines.append(f"Selected model used in full-path figures: {selected_model}")
    lines.append("")
    lines.append("BASIC TEST WARNINGS")
    for _, r in quality.iterrows():
        lines.append(f"{r['test_id']}: {r['basic_data_warning']}")
    lines.append("")
    lines.append("NOTES")
    lines.append("- This script completes the first step: import, basic checking, endpoint extraction, and provisional CSL fitting.")
    lines.append("- It does not calibrate M_tc, N, chi_tc, stiffness, or hardening parameters.")
    lines.append("- The quality and physical validity of the fitted CSL depend on the uploaded laboratory data.")
    (outdir / "Step1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="NorSand Step 1: review user triaxial data and fit provisional CSLs.")
    parser.add_argument("--input", default="tx_data_FRS_12tests.csv", help="User-supplied CSV file")
    parser.add_argument("--output", default="NorSand_Step1_output_revised", help="Results directory")
    parser.add_argument("--selected-model", default="curved", choices=["semi_log", "curved"], help="Model to show on full-path and individual overlay figures")
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    data, invalid, renamed = load_data(args.input)
    quality = summarize_tests(data)

    if (quality["p_end_kPa"] <= 0).any():
        bad = quality.loc[quality["p_end_kPa"] <= 0, "test_id"].tolist()
        raise ValueError("Tests with nonpositive final p' cannot be used in log-pressure CSL fitting: " + ", ".join(bad))

    stats, endpoints, model_params, stress_space = fit_models(quality)

    # outputs
    invalid.to_csv(outdir / "Table0_invalid_input_rows.csv", index=False)
    quality.to_csv(outdir / "Table1_test_quality_summary.csv", index=False)
    endpoints.to_csv(outdir / "Table2_candidate_endpoints.csv", index=False)
    stats.to_csv(outdir / "Table3_CSL_parameters_and_fit_statistics.csv", index=False)

    plot_fig1_endpoints_and_fits(endpoints, model_params, stress_space, outdir)
    plot_fig2_paths(data, model_params, stress_space, args.selected_model, outdir)
    for _, g in data.groupby("tid", sort=False):
        plot_individual_test(g, model_params, stress_space, args.selected_model, outdir)
    write_summary(stats, quality, invalid, renamed, model_params, stress_space, args.selected_model, outdir)

    print(f"Saved outputs to: {outdir.resolve()}")
    print("\nCSL parameters and fit statistics:\n")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
