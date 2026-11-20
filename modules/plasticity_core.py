"""Calibration step 4: plasticity (Jefferies & Been 2016, sections 9.3.4 to 9.3.5).

Browser version of PartII_Plasticity.py, without matplotlib:
- measured dilatancy D by Eq. 9.2, the secant from row n-2 to n+2 in recorded order,
  using eps_q = e1 - ev/3 (no smoothing, undefined at the first and last two rows),
- Dmin, eta_max and psi at Dmin taken independently from each dense CID test,
- eta_max = Mtc - (1 - N) Dmin (Eq. 9.5) and Dmin = chi_tc psi through the origin (Eq. 9.6),
- late q/p' and D of the other tests as an independent check on Mtc.

psi uses the CSL accepted in step 3. Least squares is a starting line, not a
substitute for engineering review.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd


def process_test(g: pd.DataFrame, csl_fn) -> pd.DataFrame:
    """Add eps_q, eta, psi and D to one test, keeping the recorded row order."""
    r = g.reset_index(drop=True).copy()
    r["eps_q"] = r["e1"] - r["ev"] / 3.0
    r["eta"] = r["q"] / r["p"]
    r["psi"] = r["e"] - csl_fn(r["p"].to_numpy(float))
    d = np.full(len(r), np.nan)
    if len(r) >= 6 and r["ev"].notna().any():
        ev = r["ev"].to_numpy(float)
        eq = r["eps_q"].to_numpy(float)
        num = ev[4:] - ev[:-4]
        den = eq[4:] - eq[:-4]
        mid = np.full(len(den), np.nan)
        ok = np.abs(den) > 1e-12
        mid[ok] = num[ok] / den[ok]
        d[2:-2] = mid
    r["D"] = d
    return r


def peak_point(r: pd.DataFrame) -> dict | None:
    """Dmin and eta_max, each from its own row (they need not coincide)."""
    eta = r["eta"].to_numpy(float)
    D = r["D"].to_numpy(float)
    if not np.isfinite(D).any() or not np.isfinite(eta).any():
        return None
    i_d = int(np.nanargmin(D))
    i_eta = int(np.nanargmax(eta))
    return {
        "Dmin": float(D[i_d]), "psi_at_Dmin": float(r["psi"].iloc[i_d]),
        "eta_at_Dmin": float(eta[i_d]), "eta_max": float(eta[i_eta]),
        "D_at_eta_max": float(D[i_eta]) if math.isfinite(D[i_eta]) else float("nan"),
        "strain_at_Dmin": float(r["e1"].iloc[i_d]), "strain_at_eta_max": float(r["e1"].iloc[i_eta]),
        "gap": float(abs(r["e1"].iloc[i_d] - r["e1"].iloc[i_eta])),
        "source_row_Dmin": int(r["source_row"].iloc[i_d]), "source_row_eta_max": int(r["source_row"].iloc[i_eta]),
        "psi0": float(r["psi"].iloc[0]),
    }


def propose_dense(peak: dict | None) -> bool:
    """Starting suggestion only: dense of critical (psi0 < 0) and dilating (Dmin < 0).

    PartII_Plasticity.py also required '-D' in the test name, which only suits the
    Fraser River sand file. The user confirms every test on the page.
    """
    return bool(peak and math.isfinite(peak["psi0"]) and peak["psi0"] < 0 and peak["Dmin"] < 0)


def rmse(y, pred) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((y - pred) ** 2))) if len(y) else float("nan")


def fit(points: list[dict]) -> dict:
    """Least-squares Mtc, N (Eq. 9.5) and chi_tc (Eq. 9.6, through the origin)."""
    out = {"n": len(points), "Mtc": float("nan"), "N": float("nan"), "chi_tc": float("nan"), "messages": []}
    d = np.array([z["Dmin"] for z in points], float)
    eta = np.array([z["eta_max"] for z in points], float)
    psi = np.array([z["psi_at_Dmin"] for z in points], float)
    if len(d) < 3 or np.ptp(d) < 1e-8:
        out["messages"].append("Select at least three dense CID tests with different Dmin to fit Mtc and N.")
    else:
        (m0, slope), *_ = np.linalg.lstsq(np.c_[np.ones_like(d), d], eta, rcond=None)
        out["Mtc"], out["N"] = float(m0), float(1.0 + slope)
    if len(psi) and float(psi @ psi) > 1e-12:
        out["chi_tc"] = float(psi @ d / (psi @ psi))
    elif len(points):
        out["messages"].append("chi_tc needs at least one point with psi not equal to zero.")
    return out


def refit_n(points: list[dict], mtc: float) -> float:
    """Best N for a fixed Mtc (what the Review.html Mtc slider does)."""
    d = np.array([z["Dmin"] for z in points], float)
    eta = np.array([z["eta_max"] for z in points], float)
    if not len(d) or float(d @ d) < 1e-12:
        return float("nan")
    return float(1.0 + d @ (eta - mtc) / (d @ d))


def errors(points: list[dict], mtc: float, n: float, chi: float) -> dict:
    d = np.array([z["Dmin"] for z in points], float)
    eta = np.array([z["eta_max"] for z in points], float)
    psi = np.array([z["psi_at_Dmin"] for z in points], float)
    return {"RMSE_eta": rmse(eta, mtc - (1 - n) * d), "RMSE_D": rmse(d, chi * psi)}


def late_check(r: pd.DataFrame, typ: str) -> dict:
    """Late q/p' and D, as candidate critical stress ratios (not verified critical state)."""
    j = max(5, int(np.ceil(0.10 * len(r))))
    tail = r.iloc[-j:]
    eta_tail = tail["eta"].to_numpy(float)
    d_tail = tail["D"].dropna().to_numpy(float) if typ == "CID" else np.array([])
    d_all = r["D"].dropna()
    return {
        "eta_last": float(r["eta"].iloc[-1]),
        "eta_late_median": float(np.median(eta_tail)),
        "eta_late_range": float(np.ptp(eta_tail)),
        "D_last": float(d_all.iloc[-1]) if typ == "CID" and len(d_all) else float("nan"),
        "D_late_median": float(np.median(d_tail)) if len(d_tail) else float("nan"),
    }


def residuals_csv(points: list[dict], mtc: float, n: float, chi: float) -> str:
    rows = []
    for z in points:
        eta_pred = mtc - (1 - n) * z["Dmin"]
        d_pred = chi * z["psi_at_Dmin"]
        rows.append({"test_id": z["tid"], "Dmin": z["Dmin"], "eta_max": z["eta_max"],
                     "psi_at_Dmin": z["psi_at_Dmin"], "eta_pred": eta_pred, "eta_residual": z["eta_max"] - eta_pred,
                     "D_pred": d_pred, "D_residual": z["Dmin"] - d_pred,
                     "strain_at_Dmin_pct": z["strain_at_Dmin"], "strain_at_eta_max_pct": z["strain_at_eta_max"],
                     "source_row_Dmin": z["source_row_Dmin"], "source_row_eta_max": z["source_row_eta_max"]})
    return pd.DataFrame(rows).to_csv(index=False)


def adopted_json(ls: dict, adopted: dict, err: dict, reason: str, csl: dict) -> str:
    return json.dumps({
        "Mtc": adopted["m"], "N": adopted["n"], "chi_tc": adopted["chi"], "reason": reason,
        "RMSE_eta": err["RMSE_eta"], "RMSE_D": err["RMSE_D"],
        "least_squares": {"Mtc": ls["Mtc"], "N": ls["N"], "chi_tc": ls["chi_tc"]},
        "csl": csl, "source": "Jefferies & Been 2016, sections 9.3.4 to 9.3.5",
    }, indent=2)


def report_text(points, ls, ls_err, adopted, err, reason, checks, warnings, csl_note) -> str:
    def f(v):
        return f"{v:.6f}" if v is not None and math.isfinite(v) else "n/a"
    lines = [
        "NorSand calibration, step 4: plasticity (Jefferies & Been 2016, sections 9.3.4 to 9.3.5)",
        "Measured dilatancy: Eq. 9.2, rows n-2 to n+2, recorded order. No smoothing or interpolation.",
        "Mtc and N: Eq. 9.5. chi_tc: Eq. 9.6 through the origin. Dmin and eta_max taken independently.",
        "Only selected dense CID tests are used in the regressions. Other tests are an independent check only.",
        "Least squares is a starting line. Manual adoption requires evidence.",
        "",
        f"CSL used for psi: {csl_note}",
        f"Dense tests: {', '.join(z['tid'] for z in points) or 'none'}",
        "",
        f"Least squares: Mtc={f(ls['Mtc'])}, N={f(ls['N'])}, chi={f(ls['chi_tc'])}, "
        f"RMSE_eta={f(ls_err['RMSE_eta'])}, RMSE_D={f(ls_err['RMSE_D'])}",
        f"Adopted:       Mtc={f(adopted['m'])}, N={f(adopted['n'])}, chi={f(adopted['chi'])}, "
        f"RMSE_eta={f(err['RMSE_eta'])}, RMSE_D={f(err['RMSE_D'])}",
        f"Adoption reason: {reason or 'Not supplied'}",
        "",
        "Dense-test peak points:",
    ]
    for z in points:
        lines.append(f"  {z['tid']}: Dmin={z['Dmin']:.4f} at e1={z['strain_at_Dmin']:.2f}%, "
                     f"eta_max={z['eta_max']:.4f} at e1={z['strain_at_eta_max']:.2f}%, psi at Dmin={z['psi_at_Dmin']:.4f}")
    lines += ["", "Independent Mtc check (candidate q/p' only, unless the endpoint was judged critical state):"]
    for c in checks:
        lines.append(f"  {c['tid']} ({c['type']}, step 3: {c['decision']}): last q/p'={c['eta_last']:.4f}, "
                     f"late median={c['eta_late_median']:.4f}, late D median={f(c['D_late_median'])}")
    acc = [c["eta_last"] for c in checks if c["decision"] == "accepted"]
    if acc:
        lines.append(f"  Accepted step 3 endpoints (n={len(acc)}): median q/p'={float(np.median(acc)):.4f}")
    lines += ["", "Data warnings:"] + (["- " + w for w in warnings] or ["None"])
    return "\n".join(lines) + "\n"


def safe_name(s) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(s))
