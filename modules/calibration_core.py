"""Calibration steps 1 to 3: import, review and provisional CSL fitting.

This is the browser version of Calibration_1.py. The logic is the same:
- read user-supplied triaxial data and map its columns,
- run basic data checks,
- take the final recorded point of each test as the candidate critical-state point,
- fit a semi-log and a curved CSL through the accepted points and report fit statistics.

It still does NOT decide whether critical state was reached. The user accepts
or rejects each candidate point on the page.

Differences from Calibration_1.py
- No matplotlib or scipy, so it runs in PyScript with numpy and pandas only.
  The curved CSL is fitted by solving Ca and Cb exactly for each Cc and searching
  over Cc, which reaches the same least-squares optimum as the multi-start fit.
- One reference pressure (pref, default 100 kPa) is used for both CSL forms,
  to match the NorSand solver. The semi-log Gamma is also reported at 1 kPa,
  which is the value modules/geomechanics.py expects.
- Void ratio may be derived from e0 and volumetric strain instead of being required.
- Stresses can be derived from principal stresses, and units and signs converted.
- Part of a test can be excluded, and the endpoint is the last point that is kept.

Conventions after conversion: strains in %, compression positive, stresses in kPa,
p is the mean effective stress p'.
"""
from __future__ import annotations

import io
import math
import re

import numpy as np
import pandas as pd

PREF_DEFAULT = 100.0

# Order matters: a column is given to the first quantity that claims it.
QUANTITY_ORDER = [
    "tid", "type", "e0", "p0", "ocr",
    "e1", "ev", "q", "p", "du",
    "s1e", "s3e", "s1", "s3", "u", "e",
]

COLUMN_ALIASES = {
    "tid": ["tid", "testid", "test", "testname", "testno", "specimen", "sample", "sampleid", "id"],
    "type": ["testtype", "type", "drainage", "drainagetype", "condition"],
    "e0": ["e0", "initialvoidratio", "einitial", "einit"],
    "p0": ["p0", "initialp", "pinitial", "p0prime", "initialmeaneffectivestress"],
    "ocr": ["ocr", "overconsolidationratio"],
    "e1": ["e1", "eps1", "epsa", "ea", "epsilon1", "epsilona", "axialstrain", "axstrain", "strain"],
    "ev": ["ev", "epsv", "epsilonv", "volumetricstrain", "volstrain", "volumestrain"],
    "q": ["q", "deviatorstress", "deviatoricstress", "devstress", "deviator"],
    "p": ["p", "pprime", "peff", "meaneffectivestress", "meaneffstress", "effectivemeanstress", "meanstress"],
    "du": ["du", "deltau", "excessporepressure", "excessporewaterpressure", "excesspwp", "u"],
    "s1e": ["sigma1eff", "sigma1prime", "sigmaprime1", "s1eff", "s1prime", "sig1eff", "sig1prime", "effectivesigma1", "sigma1effective"],
    "s3e": ["sigma3eff", "sigma3prime", "sigmaprime3", "s3eff", "s3prime", "sig3eff", "sig3prime", "effectivesigma3", "sigma3effective"],
    "s1": ["sigma1", "s1", "sig1", "totalsigma1", "sigma1total"],
    "s3": ["sigma3", "s3", "sig3", "totalsigma3", "sigma3total"],
    "u": ["porepressure", "porewaterpressure", "pwp", "uabs", "totalporepressure"],
    "e": ["e", "voidratio"],
}

DEFAULT_SETTINGS = {
    "stress_unit": "kpa",       # kpa | mpa
    "strain_unit": "pct",       # pct | decimal
    "sign": "compression",      # compression | tension (axial strain and stresses)
    "epsv_sign": "contraction", # contraction | dilation
}

INTERNAL_COLUMNS = ("__source", "__row")


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def normalize(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


_UNIT_SUFFIX = re.compile(r"(kpa|mpa|pct|percent)$")


def _names(column) -> set:
    n = normalize(column)
    return {n, _UNIT_SUFFIX.sub("", n)}


def finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def infer_test_type(name) -> str | None:
    m = re.search(r"(^|[^A-Za-z])(CID|CIU)(?=[^A-Za-z]|$)", str(name), re.I)
    return m.group(2).upper() if m else None


def parse_type(value) -> str | None:
    s = normalize(value)
    if not s or s == "nan":
        return None
    if "ciu" in s or "undrained" in s or s in ("u", "cu"):
        return "CIU"
    if "cid" in s or "drained" in s or s in ("d", "cd"):
        return "CID"
    return None


def _first_finite(values):
    for v in values:
        if finite(v):
            return float(v)
    return float("nan")


# --------------------------------------------------------------------------------------
# Reading and column detection
# --------------------------------------------------------------------------------------
def read_table(text: str, name: str = "data") -> pd.DataFrame:
    """Read CSV, tab-separated (pasted from Excel) or semicolon-separated text."""
    text = text.lstrip("﻿")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{name}: needs a header row and at least one data row.")
    head = lines[0]
    if "\t" in head:
        sep = "\t"
    elif head.count(";") > head.count(","):
        sep = ";"
    else:
        sep = ","
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, dtype=str, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError(f"{name}: contains no data.")
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", str(name))
    df["__source"] = stem
    df["__row"] = np.arange(2, len(df) + 2)
    return df


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True, sort=False)


def data_columns(raw: pd.DataFrame) -> list[str]:
    return [c for c in raw.columns if c not in INTERNAL_COLUMNS]


def detect_mapping(columns: list[str]) -> dict:
    """Match each quantity to a column by name. The first alias match wins."""
    names = {c: _names(c) for c in columns}
    used = set()
    mapping = {}
    for key in QUANTITY_ORDER:
        mapping[key] = None
        for alias in COLUMN_ALIASES[key]:
            hit = next((c for c in columns if c not in used and alias in names[c]), None)
            if hit is not None:
                mapping[key] = hit
                used.add(hit)
                break
    return mapping


# --------------------------------------------------------------------------------------
# Building the standard data set
# --------------------------------------------------------------------------------------
def build(raw: pd.DataFrame, mapping: dict, settings: dict | None = None,
          overrides: dict | None = None) -> dict:
    """Convert the raw table into tid, e1, q, p, ev, du, e with per-test metadata.

    Returns a dict with data (clean rows), bad_rows, tests (one dict per test),
    status (per quantity: found, derived, file, missing, needed or unused) and errors.
    """
    st = {**DEFAULT_SETTINGS, **(settings or {})}
    overrides = overrides or {}
    stress_f = 1000.0 if st["stress_unit"] == "mpa" else 1.0
    strain_f = 100.0 if st["strain_unit"] == "decimal" else 1.0
    sign = -1.0 if st["sign"] == "tension" else 1.0
    ev_sign = -1.0 if st["epsv_sign"] == "dilation" else 1.0

    n = len(raw)
    status = {k: "unused" for k in QUANTITY_ORDER}
    errors = []

    def num(key, factor=1.0):
        col = mapping.get(key)
        if not col or col not in raw.columns:
            return None
        status[key] = "found"
        return pd.to_numeric(raw[col], errors="coerce").to_numpy(float) * factor

    # Test ID: a column, otherwise the file name.
    if mapping.get("tid") and mapping["tid"] in raw.columns:
        tid = raw[mapping["tid"]].astype("string").str.strip().fillna("")
        status["tid"] = "found"
    else:
        tid = raw["__source"].astype("string").fillna("")
        status["tid"] = "file"
    tid = tid.to_numpy(object)

    e1 = num("e1", strain_f * sign)
    ev = num("ev", strain_f * ev_sign)
    q = num("q", stress_f * sign)
    p = num("p", stress_f * sign)
    du = num("du", stress_f)
    s1e = num("s1e", stress_f * sign)
    s3e = num("s3e", stress_f * sign)
    s1 = num("s1", stress_f * sign)
    s3 = num("s3", stress_f * sign)
    u = num("u", stress_f)
    e_given = num("e")
    e0_col = num("e0")
    p0_col = num("p0", stress_f)
    ocr_col = num("ocr")
    type_col = raw[mapping["type"]].to_numpy(object) if mapping.get("type") in raw.columns else None
    if type_col is not None:
        status["type"] = "found"

    # Effective principal stresses from total stresses and pore pressure.
    if (s1e is None or s3e is None) and s1 is not None and s3 is not None and u is not None:
        s1e, s3e = s1 - u, s3 - u
        status["s1e"] = status["s3e"] = "derived"

    if q is None:
        if s1e is not None and s3e is not None:
            q = s1e - s3e
            status["q"] = "derived"
        elif s1 is not None and s3 is not None:
            q = s1 - s3
            status["q"] = "derived"
    if p is None and s1e is not None and s3e is not None:
        p = (s1e + 2.0 * s3e) / 3.0
        status["p"] = "derived"

    if e1 is None:
        status["e1"] = "missing"
        errors.append("Axial strain was not found. Choose its column in the mapping table.")
    if q is None:
        status["q"] = "missing"
        errors.append("Deviator stress q was not found and cannot be derived. Map q, or both principal stresses.")
    if p is None:
        status["p"] = "missing"
        errors.append("Mean effective stress p′ was not found and cannot be derived. Map p′, or both effective principal stresses.")

    nan = np.full(n, np.nan)
    ev = ev if ev is not None else nan.copy()
    df = pd.DataFrame({
        "tid": tid,
        "e1": e1 if e1 is not None else nan.copy(),
        "q": q if q is not None else nan.copy(),
        "p": p if p is not None else nan.copy(),
        "ev": ev,
        "du": du if du is not None else nan.copy(),
        "e": e_given if e_given is not None else nan.copy(),
        "source": raw["__source"].to_numpy(object),
        "source_row": raw["__row"].to_numpy(int),
    })

    # Excess pore pressure from pore pressure, relative to the start of each test.
    if mapping.get("du") is None and u is not None:
        df["du"] = u
        df["du"] = df["du"] - df.groupby("tid", sort=False)["du"].transform(lambda s: _first_finite(s))
        status["du"] = "derived"

    # Per-test metadata. Values typed on the page override the file.
    tests = []
    e_derived_any = False
    named = df["tid"] != ""
    for t, idx in df[named].groupby("tid", sort=False).groups.items():
        idx = np.asarray(idx)
        ov = overrides.get(t, {})
        g_ev = df.loc[idx, "ev"].to_numpy(float)
        g_du = df.loc[idx, "du"].to_numpy(float)

        typ, typ_src = None, "page"
        if ov.get("type") in ("CID", "CIU"):
            typ = ov["type"]
        if typ is None and type_col is not None:
            typ = next((parse_type(v) for v in type_col[idx] if parse_type(v)), None)
            typ_src = "file"
        if typ is None:
            typ = infer_test_type(t)
            typ_src = "name"
        if typ is None:
            has_ev, has_du = np.isfinite(g_ev).any(), np.isfinite(g_du).any()
            if has_ev and not has_du:
                typ, typ_src = "CID", "columns"
            elif has_du and not has_ev:
                typ, typ_src = "CIU", "columns"
        if typ is None:
            typ, typ_src = "UNKNOWN", "none"

        e0 = ov.get("e0") if finite(ov.get("e0")) else float("nan")
        if not finite(e0) and e0_col is not None:
            e0 = _first_finite(e0_col[idx])
        if not finite(e0):
            e0 = _first_finite(df.loc[idx, "e"])

        p0 = ov.get("p0") if finite(ov.get("p0")) else float("nan")
        if not finite(p0) and p0_col is not None:
            p0 = abs(_first_finite(p0_col[idx]))
        if not finite(p0):
            p0 = _first_finite(df.loc[idx, "p"])

        ocr = ov.get("ocr") if finite(ov.get("ocr")) else float("nan")
        if not finite(ocr) and ocr_col is not None:
            ocr = _first_finite(ocr_col[idx])
        if not finite(ocr):
            ocr = 1.0

        # Void ratio: given values, else e = e0 - (1 + e0) ev / 100.
        # An undrained test has no volume change, so a missing ev means e stays at e0.
        e_now = df.loc[idx, "e"].to_numpy(float)
        if finite(e0) and not np.isfinite(e_now).all():
            ev_use = np.where(np.isfinite(g_ev), g_ev, 0.0 if typ == "CIU" else np.nan)
            derived = float(e0) - (1.0 + float(e0)) * ev_use / 100.0
            fill = ~np.isfinite(e_now) & np.isfinite(derived)
            if fill.any():
                e_now = np.where(fill, derived, e_now)
                df.loc[idx, "e"] = e_now
                e_derived_any = True

        tests.append({"tid": t, "type": typ, "type_source": typ_src,
                      "p0": float(p0), "e0": float(e0), "ocr": float(ocr)})

    if status["e"] == "found" and e_derived_any:
        status["e"] = "derived"
    elif status["e"] != "found":
        status["e"] = "derived" if e_derived_any else "missing"

    # Rows that cannot be used (the void ratio is allowed to be missing here:
    # those tests can still be reviewed but cannot give a CSL point).
    if errors:
        invalid = np.ones(n, bool)
    else:
        invalid = (df["tid"] == "").to_numpy() | ~np.isfinite(df[["e1", "p", "q"]].to_numpy(float)).all(axis=1)
    bad = df.loc[invalid, ["source", "source_row", "tid"]].copy()
    bad["reason"] = np.where(bad["tid"] == "", "missing test ID", "non-numeric strain or stress")
    clean = df.loc[~invalid].reset_index(drop=True)

    # Basic checks, as in Calibration_1.summarize_tests.
    types = {t["tid"]: t["type"] for t in tests}
    for t in tests:
        g = clean[clean["tid"] == t["tid"]]
        t["n_rows"] = int(len(g))
        t["warnings"] = test_warnings(g, t["type"])
        t["has_e"] = bool(len(g) and np.isfinite(g["e"]).any())

    status_needed(status, types)
    return {"data": clean, "bad_rows": bad, "tests": tests, "status": status, "errors": errors}


def status_needed(status: dict, types: dict) -> None:
    """Mark ev and du as needed when tests of that type exist but the quantity is not available."""
    kinds = set(types.values())
    if status["ev"] == "unused" and "CID" in kinds:
        status["ev"] = "needed"
    if status["du"] == "unused" and "CIU" in kinds:
        status["du"] = "needed"


def test_warnings(g: pd.DataFrame, typ: str) -> list[str]:
    if g.empty:
        return ["no usable rows"]
    warnings = []
    de1 = np.diff(g["e1"].to_numpy(float))
    if (g["p"] <= 0).any():
        warnings.append(f"{int((g['p'] <= 0).sum())} non-positive p′ row(s)")
    if (de1 < -1e-9).any():
        warnings.append(f"{int((de1 < -1e-9).sum())} reversed strain increment(s)")
    if (np.abs(de1) <= 1e-9).any():
        warnings.append(f"{int((np.abs(de1) <= 1e-9).sum())} repeated strain value(s)")
    if typ == "CID" and g["ev"].isna().all():
        warnings.append("volumetric strain missing for a drained test")
    if typ == "CIU" and g["du"].isna().all():
        warnings.append("pore pressure missing for an undrained test")
    if typ == "UNKNOWN":
        warnings.append("test type not identified: choose CID or CIU")
    if not np.isfinite(g["e"]).any():
        warnings.append("void ratio missing: enter e₀")
    if abs(float(g["q"].iloc[0])) > max(2.0, 0.01 * abs(float(g["q"].max()))):
        warnings.append("non-zero initial q: check offset or anisotropic consolidation")
    return warnings


# --------------------------------------------------------------------------------------
# Exclusions, candidate points and dilatancy
# --------------------------------------------------------------------------------------
def kept_mask(g: pd.DataFrame, ranges) -> np.ndarray:
    mask = np.ones(len(g), bool)
    e1 = g["e1"].to_numpy(float)
    for a, b in ranges or []:
        lo, hi = min(a, b), max(a, b)
        mask &= ~((e1 >= lo) & (e1 <= hi))
    return mask


def _change_pct(e1, values, window):
    """Change over the last `window` % of axial strain, as a percentage of the end value."""
    if e1[-1] - np.nanmin(e1) < window:
        return float("nan")
    idx = np.where(e1 <= e1[-1] - window)[0]
    if not len(idx):
        return float("nan")
    ref, end = values[idx[-1]], values[-1]
    if not (finite(ref) and finite(end)) or abs(end) < 1e-9:
        return float("nan")
    return float((end - ref) / abs(end) * 100.0)


def candidate_point(g: pd.DataFrame, typ: str, window: float = 2.0, d_window: float = 1.0) -> dict | None:
    """The final kept point of a test, with simple checks on whether it has levelled off."""
    g = g[g["p"] > 0]
    if g.empty:
        return None
    e1 = g["e1"].to_numpy(float)
    last = g.iloc[-1]

    d_end = float("nan")
    if typ != "CIU":
        gw = g[(g["e1"] >= e1[-1] - d_window) & g["ev"].notna()]
        if len(gw) >= 2:
            eq = gw["e1"] - gw["ev"] / 3.0
            deq = float(eq.iloc[-1] - eq.iloc[0])
            if abs(deq) > 1e-9:
                d_end = float(gw["ev"].iloc[-1] - gw["ev"].iloc[0]) / deq

    dq = _change_pct(e1, g["q"].to_numpy(float), window)
    dp = _change_pct(e1, g["p"].to_numpy(float), window)
    if not finite(dq):
        check = "short"
    else:
        steady = abs(dq) <= 5.0
        if finite(d_end):
            steady &= abs(d_end) <= 0.1
        if typ == "CIU" and finite(dp):
            steady &= abs(dp) <= 5.0
        check = "steady" if steady else "changing"

    return {
        "tid": str(last["tid"]), "type": typ,
        "e1": float(last["e1"]), "p": float(last["p"]), "q": float(last["q"]), "e": float(last["e"]),
        "eta": float(last["q"] / last["p"]),
        "D_end": d_end, "dq_pct": dq, "dp_pct": dp, "check": check,
    }


def dilatancy_series(g: pd.DataFrame):
    """D = d(ev)/d(eq) with eq = e1 - ev/3, by central differences over a few points."""
    g = g[g["ev"].notna()]
    if len(g) < 3:
        return None, None
    e1 = g["e1"].to_numpy(float)
    ev = g["ev"].to_numpy(float)
    eq = e1 - ev / 3.0
    k = max(1, len(g) // 40)
    d = np.full(len(g), np.nan)
    for i in range(k, len(g) - k):
        deq = eq[i + k] - eq[i - k]
        if abs(deq) > 1e-9:
            d[i] = (ev[i + k] - ev[i - k]) / deq
    return e1, d


# --------------------------------------------------------------------------------------
# CSL fitting
# --------------------------------------------------------------------------------------
def semi_log_e(p, gamma_ref, lambda10, pref=PREF_DEFAULT):
    return gamma_ref - lambda10 * np.log10(np.asarray(p, float) / pref)


def curved_e(p, ca, cb, cc, pref=PREF_DEFAULT):
    return ca - cb * (np.asarray(p, float) / pref) ** cc


CA_BOUNDS, CB_BOUNDS, CC_BOUNDS = (0.0, 4.0), (1e-8, 4.0), (0.01, 2.0)


def _curved_given_cc(x, e):
    """Best Ca and Cb for a fixed Cc (the model is linear in them), within the bounds."""
    A = np.c_[np.ones_like(x), -x]
    (ca, cb), *_ = np.linalg.lstsq(A, e, rcond=None)
    cb = min(max(cb, CB_BOUNDS[0]), CB_BOUNDS[1])
    ca = float(np.mean(e + cb * x))
    if not CA_BOUNDS[0] <= ca <= CA_BOUNDS[1]:
        ca = min(max(ca, CA_BOUNDS[0]), CA_BOUNDS[1])
        cb = float(np.dot(ca - e, x) / np.dot(x, x))
        cb = min(max(cb, CB_BOUNDS[0]), CB_BOUNDS[1])
    r = ca - cb * x - e
    return float(ca), float(cb), float(r @ r)


def fit_curved(p, e, pref=PREF_DEFAULT):
    ratio = np.asarray(p, float) / pref
    e = np.asarray(e, float)

    def sse(cc):
        return _curved_given_cc(ratio ** cc, e)[2]

    grid = np.linspace(CC_BOUNDS[0], CC_BOUNDS[1], 200)
    values = [sse(c) for c in grid]
    i = int(np.argmin(values))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    # Golden-section refinement around the best grid value.
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = sse(d)
    cc = (a + b) / 2.0
    if values[i] < sse(cc):
        cc = float(grid[i])
    ca, cb, _ = _curved_given_cc(ratio ** cc, e)
    return ca, cb, float(cc)


def _stats(e, pred, k):
    n = len(e)
    rss = float(np.sum((e - pred) ** 2))
    tss = float(np.sum((e - e.mean()) ** 2))
    return {
        "n": n, "k": k,
        "RMSE": math.sqrt(rss / n),
        "residual_sd": math.sqrt(rss / max(n - k, 1)),
        "R2": 1.0 - rss / tss if tss > 0 else float("nan"),
        "AIC": n * math.log(max(rss / n, 1e-30)) + 2 * k,
        "BIC": n * math.log(max(rss / n, 1e-30)) + k * math.log(n),
    }


def endpoint_M(p, q):
    """Through-origin fit q = M p' to the candidate points (a rough check only)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    mask = np.isfinite(p) & np.isfinite(q) & (p > 0)
    p, q = p[mask], q[mask]
    if len(p) < 2:
        return float("nan"), float("nan")
    M = float(np.dot(p, q) / np.dot(p, p))
    return M, float(np.sqrt(np.mean((q - M * p) ** 2)))


def fit_models(points: list[dict], pref: float = PREF_DEFAULT) -> dict:
    """Fit both CSL forms to the accepted candidate points.

    The semi-log form needs two points at different p'. The curved form needs four.
    """
    pts = [c for c in points if finite(c["p"]) and finite(c["e"]) and c["p"] > 0]
    out = {"n": len(pts), "pref": pref, "semi_log": None, "curved": None,
           "M_end": float("nan"), "M_rmse": float("nan"), "messages": []}
    if not pref or pref <= 0:
        out["messages"].append("The reference pressure must be positive.")
        return out
    p = np.array([c["p"] for c in pts])
    e = np.array([c["e"] for c in pts])
    q = np.array([c["q"] for c in pts])

    if len(pts) < 2 or np.ptp(np.log10(p)) < 1e-6:
        out["messages"].append("Accept at least two points at different p′ to fit the CSL.")
        return out

    x = np.log10(p / pref)
    (gamma_ref, lambda10), *_ = np.linalg.lstsq(np.c_[np.ones(len(p)), -x], e, rcond=None)
    gamma_ref, lambda10 = float(gamma_ref), float(lambda10)
    out["semi_log"] = {
        "Gamma_ref": gamma_ref,
        "lambda10": lambda10,
        "lambda_ln": lambda10 / math.log(10.0),
        # NorSand's semi-log CSL is e_c = Gamma - lambda10 log10(p'), so Gamma is e_c at 1 kPa.
        "Gamma_1kPa": gamma_ref + lambda10 * math.log10(pref),
        **_stats(e, semi_log_e(p, gamma_ref, lambda10, pref), 2),
    }

    if len(pts) >= 4:
        ca, cb, cc = fit_curved(p, e, pref)
        out["curved"] = {"Ca": ca, "Cb": cb, "Cc": cc, **_stats(e, curved_e(p, ca, cb, cc, pref), 3)}
    else:
        out["messages"].append("The curved CSL needs at least four accepted points.")

    out["M_end"], out["M_rmse"] = endpoint_M(p, q)
    return out


def csl_values(p, fit: dict, model: str):
    pref = fit["pref"]
    if model == "curved" and fit.get("curved"):
        c = fit["curved"]
        return curved_e(p, c["Ca"], c["Cb"], c["Cc"], pref)
    s = fit["semi_log"]
    return semi_log_e(p, s["Gamma_ref"], s["lambda10"], pref)


def norsand_csl(fit: dict, model: str) -> dict:
    """CSL parameters in the form NorSandParams expects."""
    if model == "curved" and fit.get("curved"):
        c = fit["curved"]
        return {"csl_type": "curved", "ca": c["Ca"], "cb": c["Cb"], "cc": c["Cc"], "pref": fit["pref"]}
    s = fit["semi_log"]
    return {"csl_type": "semi_log", "gamma": s["Gamma_1kPa"], "lambda10": s["lambda10"], "pref": fit["pref"]}


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------
def candidates_csv(points: list[dict], accepted: dict, fit: dict | None) -> str:
    rows = []
    for c in points:
        row = {
            "test_id": c["tid"], "type": c["type"], "accepted": bool(accepted.get(c["tid"])),
            "e1_end_pct": c["e1"], "p_end_kPa": c["p"], "q_end_kPa": c["q"], "e_end": c["e"],
            "eta_end": c["eta"], "D_end": c["D_end"], "q_change_last2pct": c["dq_pct"],
            "p_change_last2pct": c["dp_pct"], "check": c["check"],
        }
        if fit and fit.get("semi_log"):
            row["e_pred_semi_log"] = float(csl_values(c["p"], fit, "semi_log"))
        if fit and fit.get("curved"):
            row["e_pred_curved"] = float(csl_values(c["p"], fit, "curved"))
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


def summary_text(built: dict, mapping: dict, points: list[dict], accepted: dict,
                 excluded: dict, fit: dict | None, model: str) -> str:
    tests = built["tests"]
    lines = [
        "NORSAND CALIBRATION, STEPS 1 TO 3: DATA REVIEW AND PROVISIONAL CSL FIT", "",
        f"Number of tests: {len(tests)}",
        f"Number of valid data rows: {len(built['data'])}",
        f"Invalid or incomplete rows omitted: {len(built['bad_rows'])}",
        "Column mapping: " + ", ".join(f"{col} -> {key}" for key, col in mapping.items() if col),
        "",
        "ASSUMPTION USED HERE",
        "The final kept point of each test is the candidate critical-state point.",
        "Whether critical state was reached was decided by the user, point by point.",
        "",
    ]
    lines.append("CANDIDATE POINTS")
    for c in points:
        flag = "accepted" if accepted.get(c["tid"]) else "rejected"
        lines.append(f"  {c['tid']} ({c['type']}): e1 = {c['e1']:.3f} %, p' = {c['p']:.2f} kPa, "
                     f"q = {c['q']:.2f} kPa, e = {c['e']:.4f}  [{flag}]")
    ex = {t: r for t, r in excluded.items() if r}
    if ex:
        lines.append("")
        lines.append("EXCLUDED STRAIN RANGES")
        for t, r in ex.items():
            lines.append(f"  {t}: " + ", ".join(f"{min(a, b):g} to {max(a, b):g} %" for a, b in r))
    lines.append("")
    lines.append("FITTED CSL PARAMETERS")
    if not fit or not fit.get("semi_log"):
        lines.append("  No fit: " + " ".join(fit["messages"] if fit else ["no accepted points"]))
    else:
        pref = fit["pref"]
        s = fit["semi_log"]
        lines += [
            f"Semi-log CSL: e_c = Gamma - lambda10 log10(p'/{pref:g})",
            f"  Gamma (at {pref:g} kPa) = {s['Gamma_ref']:.6f}",
            f"  Gamma (at 1 kPa, NorSand input) = {s['Gamma_1kPa']:.6f}",
            f"  lambda10 = {s['lambda10']:.6f}",
            f"  lambda_ln = {s['lambda_ln']:.6f}",
            f"  RMSE = {s['RMSE']:.6f}; R2 = {s['R2']:.6f}; AIC = {s['AIC']:.3f}; BIC = {s['BIC']:.3f}",
        ]
        if fit.get("curved"):
            c = fit["curved"]
            lines += [
                "",
                f"Curved CSL: e_c = Ca - Cb (p'/{pref:g})^Cc",
                f"  Ca = {c['Ca']:.6f}",
                f"  Cb = {c['Cb']:.6f}",
                f"  Cc = {c['Cc']:.6f}",
                f"  RMSE = {c['RMSE']:.6f}; R2 = {c['R2']:.6f}; AIC = {c['AIC']:.3f}; BIC = {c['BIC']:.3f}",
            ]
        lines += [
            "",
            f"Stress ratio through the candidate points: q = M_end p' with M_end = {fit['M_end']:.6f}",
            f"CSL selected for calibration: {model}",
        ]
    lines.append("")
    lines.append("BASIC TEST WARNINGS")
    for t in tests:
        lines.append(f"  {t['tid']} ({t['type']}): " + ("; ".join(t["warnings"]) or "no basic data flags"))
    lines += [
        "",
        "NOTES",
        "- M_end is a rough check. M_tc is calibrated from the dilatancy data in step 4.",
        "- The quality of the fitted CSL depends on the uploaded laboratory data.",
    ]
    return "\n".join(lines) + "\n"
