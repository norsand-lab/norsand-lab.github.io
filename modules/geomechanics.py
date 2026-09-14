"""Core NorSand helper equations for conventional triaxial validation.

Scope of this starter package
-----------------------------
- Conventional axisymmetric triaxial compression only.
- Built first for GeoStudio / Excel-VBA benchmark reproduction.
- Stress units: kPa.
- Strains inside the calculation: decimal.
- Output strains: percent.

The implementation is intentionally small and readable so it can be debugged
against the supplied MATLAB code and the GeoStudio example before any website
or app integration.
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

CSLType = Literal["curved", "semi_log"]

PREF_DEFAULT = 100.0          # kPa
LODE_ANGLE_TC = math.pi / 6.0 # +30 degrees, triaxial compression
SPACING_RATIO = math.e
BULK_WATER = 2.0e6            # kPa


def e_crit(
    p_mean: float,
    csl_type: CSLType,
    *,
    gamma: float = 0.0,
    lambda_nat: float | None = None,
    lambda10: float | None = None,
    ca: float = 0.0,
    cb: float = 0.0,
    cc: float = 0.0,
    pref: float = PREF_DEFAULT,
) -> Tuple[float, float]:
    """Return critical void ratio and local positive CSL slope.

    Curved CSL used in the GeoStudio example:
        e_c = C_a - C_b (p'/p_ref)^C_c
        lambda_local = - d e_c / d ln(p') = C_b C_c (p'/p_ref)^C_c

    Semi-log CSL option:
        e_c = Gamma - lambda ln(p')
    If lambda10 is supplied, lambda10 is converted by
        lambda = lambda10 / ln(10)
    so that Gamma - lambda ln(p') is equivalent to
        Gamma - lambda10 log10(p').
    """
    if p_mean <= 0:
        raise ValueError("Mean effective stress p' must be positive.")
    if pref <= 0:
        raise ValueError("Reference pressure must be positive.")

    if csl_type == "curved":
        ratio = p_mean / pref
        ec = ca - cb * ratio**cc
        lambda_local = cb * cc * ratio**cc
    elif csl_type == "semi_log":
        if lambda_nat is None:
            if lambda10 is None:
                raise ValueError("Semi-log CSL requires lambda_nat or lambda10.")
            lambda_nat = lambda10 / math.log(10.0)
        if lambda_nat <= 0:
            raise ValueError("Semi-log CSL slope must be positive.")
        ec = gamma - lambda_nat * math.log(p_mean)
        lambda_local = lambda_nat
    else:
        raise ValueError("csl_type must be 'curved' or 'semi_log'.")

    return ec, lambda_local


def elastic_shear_modulus(
    p_mean: float,
    *,
    gref: float,
    m: float,
    pref: float = PREF_DEFAULT,
) -> float:
    """Return pressure-dependent elastic shear modulus, kPa.

    For GeoStudio benchmark inputs, Gref is defined at p_ref = 100 kPa:
        G = G_ref (p'/p_ref)^m

    This is the convention used here because the GeoStudio example table labels
    the input as Gref [kPa] and states that the reference stress is p_ref=100 kPa.
    """
    if p_mean <= 0 or gref <= 0 or pref <= 0:
        raise ValueError("p_mean, gref, and pref must be positive.")
    if not (0.0 <= m <= 1.0):
        raise ValueError("Elastic exponent m must satisfy 0 <= m <= 1.")
    return gref * (p_mean / pref) ** m


def m_psi_v3(
    mtc: float,
    n: float,
    dmin: float,
    lode_angle: float = LODE_ANGLE_TC,
) -> float:
    """Return state-dependent image stress ratio Mi.

    This mirrors the supplied MATLAB M_psi_v3.m. For triaxial compression the
    Lode angle is +30 degrees, so the base critical friction ratio is Mtc.
    """
    if mtc <= 0:
        raise ValueError("Mtc must be positive.")
    if n < 0:
        raise ValueError("N must be non-negative.")

    t_tc = math.pi / 6.0
    mte = mtc / (1.0 + mtc / 3.0)

    if lode_angle >= t_tc - 1e-8:
        m_base = mtc
    elif lode_angle <= -t_tc + 1e-8:
        m_base = mte
    else:
        m_base = mtc - (mtc - mte) * math.cos(1.5 * (lode_angle + t_tc))

    # Original v3 correction. For triaxial compression this is equivalent to
    # Mi = Mtc - N*abs(Dmin).
    mi = m_base * (1.0 - n * abs(dmin) / mtc)
    if mi <= 0:
        raise ValueError("Calculated image stress ratio Mi is non-positive.")
    return mi


def hardening_modulus(psi: float, lambda_local: float, h0: float, hy: float) -> float:
    """Return state-dependent hardening modulus.

    Mirrors the supplied MATLAB H_psi.m:
        H = max(H0 - Hy*psi, 0.5/lambda_local)
    """
    if lambda_local <= 0:
        raise ValueError("lambda_local must be positive.")
    return max(h0 - hy * psi, 0.5 / lambda_local)


def book_softening_term(
    *,
    ik: float,
    lambda_local: float,
    chi: float,
    mi_tc: float,
    dp: float,
    eta: float,
    eta_l: float,
) -> float:
    """Return book softening contribution used for undrained loose case.

    It is inactive unless the response is contractive, i.e., Dp > 0.
    """
    if dp <= 0:
        return 0.0
    if abs(eta_l) < 1e-12:
        raise ValueError("eta_l is too close to zero in book_softening_term.")
    lambda_factor = 1.0 - chi * lambda_local / mi_tc
    return -lambda_factor * dp * eta * ik / eta_l
