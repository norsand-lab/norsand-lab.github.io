"""Calibration step 6: common hardening and stiffness across all tests.

Browser version of PartIII_Vincent_Common_Relationships.py. The individually fitted
values of each test (from the step 5 trials, or typed in) are regressed against the
initial state parameter:

    H_i    = H0 - Hpsi * psi0_i                  (common hardening, OLS)
    Gref_i = a + b * psi0_i                      (state-dependent stiffness, OLS)
    Gref   = geometric mean of Gref_i            (one stiffness, G = Gref (p'/pref)^m)

with in-sample RMSE and R2 and leave-one-out errors. Every test can then be re-run
with the common H0 and Hpsi and either stiffness law. No fitting to the laboratory
curves happens here, and the results are candidates, not adopted values.

The individual Gref values are manual fits at a common reference pressure, not
stiffness measurements, so m cannot be identified from them. m stays fixed.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

# Vincent's manual best fits for the 12 Fraser River sand tests (Seequent 2021 NorSand
# webinar), from Vincent_12_individual_best_fits.csv. A worked example, not defaults.
VINCENT_FRS = pd.read_csv(io.StringIO("""test_id,psi0_Vincent,p0_Vincent_kPa,Gref_bestfit_kPa,H_bestfit,OCR,S
CID-D050,-0.15,50,23500,100,1.1,0
CID-D113,-0.181,114,24000,110,1.2,0
CID-D400,-0.158,410,21000,100,1.1,0
CID-D515,-0.109,515,21000,80,1.1,0
CID-L100,0.025,102,15000,60,1.1,0
CID-L300,0.127,303,10000,40,1.2,0
CID-L600,0.019,603,14000,45,1.2,0
CIU-L200,0.076,202,10100,35,1.1,1
CIU-M200,0.0,200,14000,70,1.2,0
CIU-L300,0.088,301,9000,40,1.1,1
CIU-L390,0.041,388,15000,40,1.05,1
CIU-M400,-0.032,393,18000,70,1.15,0
"""))


def r2_and_rmse(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[good], pred[good]
    if len(obs) < 2:
        return float("nan"), float("nan")
    err = obs - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    den = float(np.sum((obs - obs.mean()) ** 2))
    return rmse, (float(1 - np.sum(err ** 2) / den) if den > 0 else float("nan"))


def loocv_linear(x, y):
    """Leave-one-out predictions of a straight-line fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or np.ptp(x) < 1e-9:
        return np.full(len(x), np.nan)
    out = []
    for i in range(len(x)):
        ok = np.arange(len(x)) != i
        co = np.polyfit(x[ok], y[ok], 1)
        out.append(float(np.polyval(co, x[i])))
    return np.array(out)


def fit_relations(rows: list[dict], m: float, pref: float) -> dict:
    """rows: dicts with tid, psi0, p0, gref, h. Needs three or more tests with different psi0."""
    psi = np.array([r["psi0"] for r in rows], float)
    H = np.array([r["h"] for r in rows], float)
    g = np.array([r["gref"] for r in rows], float)
    p0 = np.array([r["p0"] for r in rows], float)
    if len(rows) < 3 or np.ptp(psi) < 1e-9:
        raise ValueError("Use at least three tests with different ψ₀.")
    if (g <= 0).any():
        raise ValueError("Every Gref must be above zero.")
    hslope, hinter = np.polyfit(psi, H, 1)
    gslope, ginter = np.polyfit(psi, g, 1)
    gconst = float(np.exp(np.mean(np.log(g))))  # equal relative errors: fit of log(Gref)
    h_fit = hinter + hslope * psi
    g_state = ginter + gslope * psi
    hs, gs, gc = r2_and_rmse(H, h_fit), r2_and_rmse(g, g_state), r2_and_rmse(g, np.full(len(g), gconst))
    return {
        "n": len(rows), "m": m, "pref": pref,
        "H0": float(hinter), "Hpsi": float(-hslope),
        "H_rmse": hs[0], "H_r2": hs[1], "H_loocv_rmse": r2_and_rmse(H, loocv_linear(psi, H))[0],
        "Gref_const": gconst, "Gref_const_rmse": gc[0], "Gref_const_r2": gc[1],
        "Gref_a": float(ginter), "Gref_b": float(gslope),
        "Gref_state_rmse": gs[0], "Gref_state_r2": gs[1],
        "Gref_state_loocv_rmse": r2_and_rmse(g, loocv_linear(psi, g))[0],
        "points": [{**r, "H_fit": float(hf), "Gref_state_fit": float(gsf), "Gref_const_fit": gconst,
                    "G_p0_input": float(r["gref"] * (r["p0"] / pref) ** m),
                    "G_p0_state": float(gsf * (r["p0"] / pref) ** m),
                    "G_p0_const": float(gconst * (r["p0"] / pref) ** m)}
                   for r, hf, gsf in zip(rows, h_fit, g_state)],
    }


def gref_for(fit: dict, law: str, psi0: float) -> float:
    if law == "state":
        return fit["Gref_a"] + fit["Gref_b"] * psi0
    return fit["Gref_const"]


def points_csv(fit: dict) -> str:
    return pd.DataFrame(fit["points"]).to_csv(index=False)


def summary_lines(fit: dict) -> list[str]:
    f = fit
    return [
        f"Common hardening: H = {f['H0']:.3f} - ({f['Hpsi']:.3f}) psi   (R2 {f['H_r2']:.3f}, RMSE {f['H_rmse']:.3f}, "
        f"leave-one-out RMSE {f['H_loocv_rmse']:.3f})",
        "  In the solver H uses the current psi, not psi0: H = max(H0 - Hy psi, 0.5/lambda_local).",
        f"One stiffness: Gref = {f['Gref_const']:.1f} kPa (geometric mean), G = Gref (p'/{f['pref']:g})^{f['m']:g}   "
        f"(R2 {f['Gref_const_r2']:.3f}, RMSE {f['Gref_const_rmse']:.1f} kPa)",
        f"State-dependent stiffness: Gref = {f['Gref_a']:.1f} + ({f['Gref_b']:.1f}) psi0 kPa   "
        f"(R2 {f['Gref_state_r2']:.3f}, RMSE {f['Gref_state_rmse']:.1f} kPa, leave-one-out RMSE {f['Gref_state_loocv_rmse']:.1f} kPa)",
        f"Based on {f['n']} tests, all in-sample. m is fixed, not fitted. Candidate values, not independent validation.",
    ]
