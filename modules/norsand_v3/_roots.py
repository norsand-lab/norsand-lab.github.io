"""Brent's root finder, a drop-in replacement for scipy.optimize.brentq.

The Part III solver only needs brentq from scipy. Loading scipy in the browser is a
large download, so this pure-Python version (same algorithm as scipy's C brentq)
is used instead.
"""
from __future__ import annotations

import math


def brentq(f, a, b, xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if fa == 0:
        return a
    if fb == 0:
        return b
    xpre, xcur, fpre, fcur = a, b, fa, fb
    xblk = fblk = spre = scur = 0.0
    for _ in range(maxiter):
        if fpre * fcur < 0:
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)          # secant
            else:
                dpre = (fpre - fcur) / (xpre - xcur)                  # inverse quadratic
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta):
                spre, scur = scur, stry
            else:
                spre, scur = sbis, sbis
        else:
            spre, scur = sbis, sbis
        xpre, fpre = xcur, fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta
        fcur = f(xcur)
        if math.isnan(fcur):
            raise ValueError("function returned NaN")
    raise RuntimeError("brentq did not converge")
