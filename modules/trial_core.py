"""Calibration step 5: manual NorSand trials (browser version of PartIII_Manual_Trial.py).

For one test at a time the user chooses trial values of Gref, H0, Hpsi, OCR, nu, m and S.
NorSand is run to the last recorded axial strain and compared with the laboratory curves
at the recorded strains. No optimisation runs and nothing is adopted automatically.

The CSL comes from step 3 and Mtc, N and chi_tc from step 4. Reference: Jefferies & Been
(2016), section 9.3.6, iterative forward modelling.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

from modules.norsand_v3 import NorSandParams, simulate

DEFAULT_TRIAL = {"gref": 20000.0, "h0": 60.0, "hy": 0.0, "ocr": 1.10, "nu": 0.20, "m": 0.47, "s": 0}

LOG_FIELDS = ["run", "test_id", "type", "status", "gref", "h0", "hy", "ocr", "nu", "m", "s",
              "psi0", "p0_kPa", "strain_end_pct", "fit_to_pct", "q_RMSE_kPa", "q_R2",
              "volume_RMSE_pct", "u_RMSE_kPa", "psi_RMSE", "q_peak_lab_kPa", "q_peak_sim_kPa", "message"]


def validate_trial(t: dict, typ: str) -> None:
    for key in ("gref", "h0", "hy", "ocr", "nu", "m"):
        if not math.isfinite(float(t[key])):
            raise ValueError(f"{key} must be a number")
    if not (t["gref"] > 0 and t["h0"] >= 0 and t["ocr"] >= 1):
        raise ValueError("Gref must be above 0, H0 at least 0 and OCR at least 1")
    if not (-1 < t["nu"] < 0.5 and 0 <= t["m"] <= 1):
        raise ValueError("Poisson's ratio must be between -1 and 0.5, and m between 0 and 1")
    if t["s"] not in (0, 1):
        raise ValueError("S must be 0 or 1")
    if typ == "CID" and t["s"] == 1:
        raise ValueError("S = 1 only applies to undrained tests. Set S = 0 for a drained test.")


def score_at_lab_points(model_x, model_y, lab_x, lab_y, cutoff=None) -> dict:
    """RMSE and R2 of the model, interpolated at the recorded laboratory strains."""
    x, y = np.asarray(lab_x, float), np.asarray(lab_y, float)
    mx, my = np.asarray(model_x, float), np.asarray(model_y, float)
    good = np.isfinite(mx) & np.isfinite(my)
    mx, my = mx[good], my[good]
    if len(mx) < 2:
        return {"RMSE": None, "R2": None, "n_points": 0}
    ix = np.argsort(mx, kind="stable")
    mx, my = mx[ix], my[ix]
    mx, unique_ix = np.unique(mx, return_index=True)
    my = my[unique_ix]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= mx[0] - 1e-7) & (x <= mx[-1] + 1e-7)
    if cutoff is not None:
        ok &= x <= cutoff
    if ok.sum() < 3:
        return {"RMSE": None, "R2": None, "n_points": int(ok.sum())}
    obs = y[ok]
    pred = np.interp(np.clip(x[ok], mx[0], mx[-1]), mx, my)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    den = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1 - float(np.sum((pred - obs) ** 2)) / den if den > 1e-12 else None
    return {"RMSE": rmse, "R2": r2, "n_points": int(ok.sum())}


def run_trial(g: pd.DataFrame, typ: str, trial: dict, csl: dict, plast: dict, csl_fn,
              cutoff=None, rtol=0.001):
    """Run one manual trial. Returns (record, model, lab) where lab has psi and the
    volume or pore-pressure series used for comparison. model is None if the run failed."""
    g = g.sort_values("e1", kind="stable")
    p0, e0, q0 = float(g["p"].iloc[0]), float(g["e"].iloc[0]), float(g["q"].iloc[0])
    end = float(g["e1"].max())
    lab = pd.DataFrame({"e1": g["e1"].to_numpy(float), "q": g["q"].to_numpy(float), "p": g["p"].to_numpy(float),
                        "psi": g["e"].to_numpy(float) - csl_fn(g["p"].to_numpy(float)),
                        "other": (g["ev"] if typ == "CID" else g["du"]).to_numpy(float)})
    psi0 = e0 - float(csl_fn(np.array([p0]))[0])
    record = {k: None for k in LOG_FIELDS}
    record.update(status="FAILED", type=typ, psi0=psi0, p0_kPa=p0, strain_end_pct=end,
                  fit_to_pct=cutoff if cutoff is not None else end, message="Trial values, not adopted",
                  **{k: trial[k] for k in DEFAULT_TRIAL})
    try:
        validate_trial(trial, typ)
        if p0 <= 0 or end <= 0:
            raise ValueError("The first p′ and the last axial strain must both be positive")
        params = dict(name=str(g["tid"].iloc[0]), drainage="drained" if typ == "CID" else "undrained",
                      p0=p0, psi0=psi0, eta0=q0 / p0, ocr=trial["ocr"], h0=trial["h0"], hy=trial["hy"],
                      s=int(trial["s"]), gref=trial["gref"], nu=trial["nu"], m=trial["m"],
                      mtc=plast["mtc"], n=plast["n"], chi=plast["chi"], **csl)
        model = simulate(NorSandParams(**params), axial_target=end / 100.0, rtol=rtol)
    except Exception as exc:  # noqa: BLE001 - reported on the page and in the log
        record["message"] = f"Simulation failed: {type(exc).__name__}: {exc}"
        return record, None, lab
    other = model["epsv_pct"] if typ == "CID" else model["delta_u_kpa"]
    qm = score_at_lab_points(model["eps1_pct"], model["q_kpa"], lab["e1"], lab["q"], cutoff)
    vm = score_at_lab_points(model["eps1_pct"], other, lab["e1"], lab["other"], cutoff)
    pm = score_at_lab_points(model["eps1_pct"], model["psi"], lab["e1"], lab["psi"], cutoff)
    record.update(status="OK", q_RMSE_kPa=qm["RMSE"], q_R2=qm["R2"],
                  volume_RMSE_pct=vm["RMSE"] if typ == "CID" else None,
                  u_RMSE_kPa=vm["RMSE"] if typ == "CIU" else None,
                  psi_RMSE=pm["RMSE"], q_peak_lab_kPa=float(np.nanmax(lab["q"])),
                  q_peak_sim_kPa=float(model["q_kpa"].max()))
    return record, model, lab


def log_csv(records: list[dict]) -> str:
    return pd.DataFrame(records, columns=LOG_FIELDS).to_csv(index=False)


def model_csv(model: pd.DataFrame) -> str:
    buf = io.StringIO()
    model.to_csv(buf, index=False)
    return buf.getvalue()
