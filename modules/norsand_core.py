"""Clean NorSand triaxial material-point simulator.

This file is the one to debug first. It contains no Dash code, no website code,
no calibration code, and no DSS/shear-test code. It is a direct conventional
triaxial runner intended for GeoStudio/Excel-VBA benchmark validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import numpy as np
import pandas as pd

from .geomechanics import (
    BULK_WATER,
    LODE_ANGLE_TC,
    PREF_DEFAULT,
    SPACING_RATIO,
    book_softening_term,
    e_crit,
    elastic_shear_modulus,
    hardening_modulus,
    m_psi_v3,
)

Drainage = Literal["drained", "undrained"]
CSLType = Literal["curved", "semi_log"]
StrainMode = Literal["small", "large"]


@dataclass
class NorSandParams:
    """Input parameters for one conventional triaxial compression simulation."""

    name: str
    drainage: Drainage 

    # Initial conditions
    p0: float                # initial mean effective stress, kPa
    psi0: float              # initial state parameter
    ocr: float = 1.0         # initial overconsolidation ratio for yield-surface size
    eta0: float = 0.0        # initial stress ratio q/p'

    # Critical-state line
    csl_type: CSLType = "curved" 
    pref: float = PREF_DEFAULT 
    gamma: float = 0.0       # semi-log CSL intercept, if used
    lambda10: float | None = None
    ca: float = 0.0           # curved CSL parameter C_a
    cb: float = 0.0          # curved CSL parameter C_b
    cc: float = 0.0          # curved CSL parameter C_c

    # Strength and plasticity
    mtc: float = 1.45 
    n: float = 0.45 
    chi: float = 5.3 
    h0: float = 61.0 
    hy: float = 247.0 
    s: int = 0               # GeoStudio extra softening switch; only used for undrained when s=1

    # Elasticity
    gref: float = 22900.0    # kPa at pref, matching GeoStudio table convention
    nu: float = 0.2 
    m: float = 0.47 

    # Numerical controls
    max_strain: float = 0.20 # target plastic deviatoric strain used by the original triaxial driver
    n_steps: int = 501       # GeoStudio shear stage uses 500 load steps; 501 rows include initial row
    delta_qp: float = 3.0    # drained triaxial compression at constant sigma3': dq/dp'=3
    strain_mode: StrainMode = "small"


def _validate_params(params: NorSandParams) -> None:
    if params.p0 <= 0:
        raise ValueError("p0 must be positive.")
    if params.gref <= 0:
        raise ValueError("gref must be positive.")
    if params.mtc <= 0:
        raise ValueError("mtc must be positive.")
    if params.ocr < 1:
        raise ValueError("OCR must be >= 1.")
    if not (0 <= params.m <= 1):
        raise ValueError("m must satisfy 0 <= m <= 1.")
    if not (-1 < params.nu < 0.5):
        raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5.")
    if params.n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if params.max_strain <= 0:
        raise ValueError("max_strain must be positive.")
    if params.drainage not in ("drained", "undrained"):
        raise ValueError("drainage must be 'drained' or 'undrained'.")


def run_triaxial(params: NorSandParams) -> pd.DataFrame:
    """Run one conventional axisymmetric triaxial compression simulation.

    Returns a pandas DataFrame containing the model history. The output is
    intentionally verbose because the website will eventually need more than
    just q-eps and p-q curves.
    """
    _validate_params(params)

    # ------------------------------------------------------------------
    # Initial stresses from p0 and eta0.
    # For the GeoStudio examples eta0=0, so sigma1'=sigma3'=p0.
    # ------------------------------------------------------------------
    sig_m0 = params.p0
    sig_m = sig_m0
    sig1 = sig_m0 * (1.0 + 2.0 * params.eta0 / 3.0)
    sig3 = sig_m0 * (1.0 - params.eta0 / 3.0)
    k0 = sig3 / sig1
    sig_q = sig1 - sig3
    eta = sig_q / sig_m

    # ------------------------------------------------------------------
    # Initial void ratio from current CSL and psi0.
    # ------------------------------------------------------------------
    ec0, lambda_local = e_crit(
        sig_m,
        params.csl_type,
        gamma=params.gamma,
        lambda10=params.lambda10,
        ca=params.ca,
        cb=params.cb,
        cc=params.cc,
        pref=params.pref,
    )
    e = ec0 + params.psi0
    e0 = e
    if e <= 0:
        raise ValueError("Initial void ratio is non-positive.")

    ep_g = 0.0
    ep_v = 0.0
    ep1 = 0.0
    sample_ht = 1.0
    one_over_v0 = 1.0 / (1.0 + e0)

    # ------------------------------------------------------------------
    # Initial image/yield quantities.
    # The initial estimate follows the supplied MATLAB code:
    # Dmin first from current state, Mi from M_psi_v3, then p_i/p'.
    # ------------------------------------------------------------------
    psi = e - ec0
    dmin = params.chi * psi
    mi = m_psi_v3(params.mtc, params.n, dmin, LODE_ANGLE_TC)
    mi_tc = mi

    pi_over_p_yield = math.exp(eta / mi - 1.0)
    pimax_over_p_check = math.exp(-dmin / mi)
    ocr_max = pimax_over_p_check / pi_over_p_yield
    actual_ocr = min(params.ocr, ocr_max)

    pimg_over_p = actual_ocr * math.exp(eta / mi - 1.0)
    pimx_over_p = math.exp(-dmin / mi_tc)

    k_over_g = 2.0 * (1.0 + params.nu) / (3.0 * (1.0 - 2.0 * params.nu))
    bulk_pore_fluid = 0.0 if params.drainage == "drained" else BULK_WATER
    use_book_softening = params.s == 1 and params.drainage == "undrained"

    rows: list[dict] = []

    def save_row(step: int, dp_current: float | None, drainage_now: Drainage) -> None:
        ec_now, lambda_now = e_crit(
            sig_m,
            params.csl_type,
            gamma=params.gamma,
            lambda10=params.lambda10,
            ca=params.ca,
            cb=params.cb,
            cc=params.cc,
            pref=params.pref,
        )
        psi_now = e - ec_now
        eta_now = sig_q / sig_m
        pimg = sig_m * pimg_over_p
        pi_max = sig_m * pimx_over_p

        # Display-only diagnostics recalculated at the saved row.
        # These do not affect the stress-strain integration.
        ec_i_now, _ = e_crit(
            pimg,
            params.csl_type,
            gamma=params.gamma,
            lambda10=params.lambda10,
            ca=params.ca,
            cb=params.cb,
            cc=params.cc,
            pref=params.pref,
        )
        psi_i_now = e - ec_i_now
        den_chi_now = 1.0 - lambda_now * params.chi / mi
        chi_i_now = params.chi / den_chi_now
        dmin_current_approx = chi_i_now * psi_i_now
        pimax_over_p_current_approx = math.exp(-dmin_current_approx / mi)
        dp_consistent = mi - eta_now
        g_now = elastic_shear_modulus(sig_m, gref=params.gref, m=params.m, pref=params.pref)
        k_now = g_now * k_over_g
        delta_u = sig_m0 + sig_q / 3.0 - sig_m  # valid for initial q0=0
        rows.append(
            {
                "step": step,
                "eps1_pct": 100.0 * ep1 if params.strain_mode == "large" else 100.0 * (1.0 - sample_ht),
                "epsv_pct": 100.0 * ep_v if params.strain_mode == "large" else 100.0 * (-(e - e0) * one_over_v0),
                "p_kpa": sig_m,
                "q_kpa": sig_q,
                "e": e,
                "ec": ec_now,
                "psi": psi_now,
                "eta": eta_now,
                "Mi": mi,
                "Dp": dp_consistent,
                "Dmin": dmin_current_approx,
                "psi_i": psi_i_now,
                "chi_i": chi_i_now,
                "pi_over_p": pimg_over_p,
                "p_i_kpa": pimg,
                "pimax_over_p": pimax_over_p_current_approx,
                "p_i_max_kpa": sig_m * pimax_over_p_current_approx,
                "lambda_local": lambda_now,
                "G_kpa": g_now,
                "K_kpa": k_now,
                "delta_u_kpa": 0.0 if drainage_now == "drained" else delta_u,
                "drainage": drainage_now,
                "OCR_used": actual_ocr,
                "k0": k0,
                "Dp_used_for_increment": np.nan if dp_current is None else dp_current,
                "Dmin_used_for_increment": dmin,
                "pimax_over_p_used_for_increment": pimx_over_p,
            }
        )

    save_row(step=0, dp_current=None, drainage_now=params.drainage)

    # ------------------------------------------------------------------
    # Elastic loading to yield surface when OCR > 1.
    # ------------------------------------------------------------------
    j_plastic = 1
    if actual_ocr > 1.001:
        if params.drainage == "drained":
            if params.delta_qp > 30000:
                dsig_m = 0.0
                eta = mi * (1.0 + math.log(pimg_over_p))
            else:
                sig_m_top = sig_m * pimg_over_p * SPACING_RATIO
                sig_m_btm = 0.0
                pimg_const = pimg_over_p * sig_m
                p_ys = sig_m
                eta_ys = eta
                for _ in range(30):
                    p_ys = 0.5 * (sig_m_top + sig_m_btm)
                    dsig_m_trial = p_ys - sig_m
                    q_ys = sig_q + dsig_m_trial * params.delta_qp
                    eta_ys = mi * (1.0 + math.log(pimg_const / p_ys))
                    if q_ys > p_ys * eta_ys:
                        sig_m_top = p_ys
                    else:
                        sig_m_btm = p_ys

                dsig_m = p_ys - sig_m0
                sig_m = p_ys
                eta = eta_ys

            g = elastic_shear_modulus(0.5 * (sig_m0 + sig_m), gref=params.gref, m=params.m, pref=params.pref)
            k = g * k_over_g
            dsig_q = eta * sig_m - sig_q
            sig_q += dsig_q
        else:
            dsig_m = 0.0
            g = elastic_shear_modulus(sig_m0, gref=params.gref, m=params.m, pref=params.pref)
            k = g * k_over_g
            eta_ys = mi * (1.0 + math.log(pimg_over_p))
            dsig_q = (eta_ys - eta) * sig_m0
            sig_q += dsig_q
            eta = eta_ys

        dep_vp = 0.0
        dep_gp = 0.0
        dep_ve = dsig_m / k
        dep_ge = dsig_q / (3.0 * g)
        dep_v = dep_vp + dep_ve
        dep_g = dep_gp + dep_ge
        dep1 = dep_g + dep_v / 3.0

        ep_v += dep_v
        ep_g += dep_g
        ep1 += dep1
        e = e - (1.0 + e) * dep_v
        sample_ht *= 1.0 - dep1
        sig_q = sig_m * eta

        ec_now, _ = e_crit(sig_m, params.csl_type, gamma=params.gamma, lambda10=params.lambda10, ca=params.ca, cb=params.cb, cc=params.cc, pref=params.pref)
        psi = e - ec_now
        save_row(step=1, dp_current=None, drainage_now=params.drainage)
        j_plastic = 2

    # ------------------------------------------------------------------
    # Plastic strain-controlled integration.
    # ------------------------------------------------------------------
    dep_gp = params.max_strain / (params.n_steps - 1)

    for step in range(j_plastic, params.n_steps):
        sig_q_old = sig_q
        sig_m_old = sig_m

        # 1. Current state.
        ec, lambda_local = e_crit(
            sig_m,
            params.csl_type,
            gamma=params.gamma,
            lambda10=params.lambda10,
            ca=params.ca,
            cb=params.cb,
            cc=params.cc,
            pref=params.pref,
        )
        psi = e - ec

        # 2. Image state and limiting dilatancy.
        pimg = sig_m * pimg_over_p
        if pimg <= 0:
            raise ValueError(f"Image pressure became non-positive at step {step}.")
        ec_i, _ = e_crit(
            pimg,
            params.csl_type,
            gamma=params.gamma,
            lambda10=params.lambda10,
            ca=params.ca,
            cb=params.cb,
            cc=params.cc,
            pref=params.pref,
        )
        psi_i = e - ec_i

        den_chi = 1.0 - lambda_local * params.chi / mi_tc
        if abs(den_chi) < 1e-12:
            raise ValueError(f"chi_i denominator too close to zero at step {step}.")
        chi_i = params.chi / den_chi
        dmin = chi_i * psi_i

        # 3. Elastic stiffness.
        g = elastic_shear_modulus(sig_m, gref=params.gref, m=params.m, pref=params.pref)
        k = g * k_over_g
        ir = g / sig_m
        ik = ir * k_over_g
        k_tot = bulk_pore_fluid * k / (bulk_pore_fluid + k) if bulk_pore_fluid > 0 else 0.0

        # 4. Flow rule / stress-dilatancy.
        mi_old = mi
        mi_tc = m_psi_v3(params.mtc, params.n, dmin, LODE_ANGLE_TC)
        mi = mi_tc
        dmi_over_mi = 1.0 - mi_old / mi
        dp = mi - eta
        dep_vp = dp * dep_gp

        # 5. Image-pressure hardening/softening.
        if use_book_softening:
            eta_l = mi_tc - dmin
            dpmx_dep_gp = book_softening_term(
                ik=k_tot / sig_m,
                lambda_local=lambda_local,
                chi=params.chi,
                mi_tc=mi_tc,
                dp=dp,
                eta=eta,
                eta_l=eta_l,
            )
        else:
            dpmx_dep_gp = 0.0

        pimx_over_p = math.exp(-dmin / mi_tc)
        dpimg_over_pimg = (
            hardening_modulus(psi, lambda_local, params.h0, params.hy)
            * pimg_over_p ** (-2.0)
            * (pimx_over_p - pimg_over_p)
            + dpmx_dep_gp
        ) * dep_gp

        # 6. Consistency and stress update.
        if params.drainage == "drained":
            denom_path = params.delta_qp - eta
            if abs(denom_path) < 1e-12:
                raise ValueError(f"delta_qp - eta too close to zero at step {step}.")
            eta_ratio = 1.0 + mi / denom_path
            deta = (eta * dmi_over_mi + mi * dpimg_over_pimg) / eta_ratio
            eta = eta + deta

            denom_path_new = params.delta_qp - eta
            if abs(denom_path_new) < 1e-12:
                raise ValueError(f"Updated delta_qp - eta too close to zero at step {step}.")
            dsig_m = sig_m * deta / denom_path_new
            dsig_q = sig_m * deta + eta * dsig_m
            sig_m = sig_m + dsig_m
            if sig_m <= 0:
                raise ValueError(f"Mean effective stress became non-positive at step {step}.")
            sig_q = sig_q + dsig_q
            eta = sig_q / sig_m
            pimg_over_p = math.exp(eta / mi - 1.0)
        else:
            dsig_m = -dep_vp * k_tot
            sig_m = sig_m + dsig_m
            if sig_m < 0.1:
                sig_m = 0.1
                dsig_m = sig_m - sig_m_old

            pimg_over_p = pimg_over_p * (1.0 + dpimg_over_pimg)
            if pimg_over_p <= 0:
                raise ValueError(f"p_i/p' became non-positive at step {step}.")
            pimg_over_p = pimg_over_p * sig_m_old / sig_m
            eta = mi * (1.0 + math.log(pimg_over_p))
            sig_q = sig_m * eta
            dsig_q = sig_q - sig_q_old

        # 7. Elastic and total strain update.
        dep_ge = dsig_q / (3.0 * g)
        dep_ve = dsig_m / k
        dep_v = dep_vp + dep_ve
        dep_g = dep_gp + dep_ge
        dep1 = dep_g + dep_v / 3.0

        ep_v += dep_v
        ep_g += dep_g
        ep1 += dep1
        e = e - (1.0 + e) * dep_v
        if e <= 0:
            raise ValueError(f"Void ratio became non-positive at step {step}.")
        sample_ht *= 1.0 - dep1

        save_row(step=step, dp_current=dp, drainage_now=params.drainage)

    return pd.DataFrame(rows)
