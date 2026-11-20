# Copied from 3_NorSand_PartIII_Manual_Trial_VSCode/.../NorSand_Python/src/norsand_core.py.
# Only change: brentq comes from ._roots instead of scipy, so it runs in the browser.
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
    ca: float = 0.0          # curved CSL parameter C_a
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

    # Versioned research alternatives; default preserves original MATLAB behavior.
    chi_conversion: Literal["legacy_mi", "book_c3b_mtc"] = "book_c3b_mtc"
    initialization: Literal["legacy", "image_consistent_isotropic", "coupled_yield"] = "coupled_yield"
    image_ratio_rule: Literal["symmetric_c13", "geostudio_loose_m"] = "geostudio_loose_m"


def _validate_params(params: NorSandParams) -> None:
    # Reject NaN/Inf before inequality tests: NaN silently passes x <= 0 checks.
    finite_fields = ("p0", "psi0", "ocr", "eta0", "pref", "ca", "cb", "cc",
                     "mtc", "n", "chi", "h0", "hy", "gref", "nu", "m",
                     "max_strain", "delta_qp")
    for field in finite_fields:
        value = getattr(params, field)
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
    if params.pref <= 0:
        raise ValueError("Reference mean stress must be positive")
    if params.csl_type == "curved" and (params.cb <= 0 or params.cc <= 0):
        raise ValueError("Curved CSL requires cb>0 and cc>0")
    if params.csl_type == "semi_log" and (params.lambda10 is None or not math.isfinite(params.lambda10) or params.lambda10 <= 0):
        raise ValueError("Semi-log CSL requires a finite positive lambda10")
    if params.chi < 0 or not (0 <= params.n <= 1):
        raise ValueError("Expect chi>=0 and 0<=N<=1 for this supported formulation")
    if params.h0 < 0 or params.hy < 0:
        raise ValueError("H0 and Hy must be non-negative")
    if params.s not in (0, 1):
        raise ValueError("S must be 0 or 1")
    if params.strain_mode != "small":
        raise ValueError("This verified driver supports engineering/small strain mode only")
    if params.image_ratio_rule not in ("symmetric_c13", "geostudio_loose_m"):
        raise ValueError("Unknown image-stress-ratio rule")
    if params.chi_conversion not in ("legacy_mi", "book_c3b_mtc"):
        raise ValueError("Unknown chi conversion")
    if params.initialization not in ("legacy", "image_consistent_isotropic", "coupled_yield"):
        raise ValueError("Unknown initialization")
    if params.initialization == "image_consistent_isotropic" and (abs(params.eta0)>1e-10 or abs(params.ocr-1)>1e-10):
        raise ValueError("Experimental image-consistent initialization is restricted to eta0=0, OCR=1")
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


def run_triaxial(params: NorSandParams, *, adaptive: bool = False, rtol: float = 0.003, atol: float = 0.00001, control: Literal["plastic_deviatoric", "total_axial"] = "plastic_deviatoric", total_axial_target: float | None = None) -> pd.DataFrame:
    """Run one conventional axisymmetric triaxial compression simulation.

    Returns a pandas DataFrame containing the model history. The output is
    intentionally verbose because the website will eventually need more than
    just q-eps and p-q curves.
    """
    _validate_params(params)

    def image_ratio(dmin_value: float) -> float:
        """Jefferies Appendix C: symmetric C.13 or loose M(θ) alternative.

        geostudio_loose_m is a *benchmark-compatible model variant*, not
        a universal replacement for the symmetric C.13 formulation.
        """
        if params.image_ratio_rule == "geostudio_loose_m" and dmin_value > 0.0:
            return params.mtc  # axisymmetric triaxial compression only
        return m_psi_v3(params.mtc, params.n, dmin_value, LODE_ANGLE_TC)

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
    plastic_progress = 0.0
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
    mi = image_ratio(dmin)
    mi_tc = mi

    if params.initialization == "image_consistent_isotropic":
        # EXPERIMENTAL: isotope OCR=1 branch only, not a general yield initialization.
        # Iteratively enforce Mi = M(chi_i*(e - ec(p/e))) at initial image pressure.
        pimg_guess = sig_m * math.exp(-1.0)
        for _ in range(60):
            ec_image,_ = e_crit(pimg_guess,params.csl_type,gamma=params.gamma,
                  lambda10=params.lambda10,ca=params.ca,cb=params.cb,cc=params.cc,pref=params.pref)
            slope_for_chi = lambda_local
            chi_denom = 1.0 - slope_for_chi*params.chi/params.mtc
            if chi_denom <= 0: raise ValueError("Invalid image-state chi conversion denominator during init")
            di = params.chi/chi_denom * (e-ec_image)
            mi_new = image_ratio(di)
            next_pi=sig_m*math.exp(eta/mi_new-1)
            if abs(next_pi-pimg_guess)/sig_m < 1e-11:
                break
            pimg_guess = 0.5*pimg_guess+0.5*next_pi
        mi=mi_new
        mi_tc=mi_new
        dmin=di

    # Coupled OCR initialization (experimental development branch).
    # For OCR = p_i0 / p_iyield, yield p_iyield = p0 exp(eta0/Mi - 1).
    # Mi must be evaluated from the image state at that same p_i0.
    # For eta0=0 this simplifies to p_i0 = OCR*p0/e and is explicit.
    if params.initialization == "coupled_yield":
        if params.chi_conversion != "book_c3b_mtc":
            raise ValueError("coupled_yield currently requires the explicitly approximated book_c3b_mtc chi conversion")
        from ._roots import brentq  # was scipy.optimize.brentq
        den0=1.0-lambda_local*params.chi/params.mtc
        if den0<=0:
            raise ValueError("Invalid C.3b denominator in initial state")
        chi_i0=params.chi/den0
        def eval_initial(mi_candidate):
            pi0=params.ocr*sig_m*math.exp(eta/mi_candidate-1.0)
            ei,_=e_crit(pi0,params.csl_type,gamma=params.gamma,lambda10=params.lambda10,
                          ca=params.ca,cb=params.cb,cc=params.cc,pref=params.pref)
            d0=chi_i0*(e-ei)
            mi_new=image_ratio(d0)
            return mi_candidate-mi_new,pi0,d0
        grid=np.geomspace(max(params.mtc*0.005,1e-5),params.mtc,300)
        a=b=None
        last=None
        for x in grid:
            try: fx,_,_=eval_initial(float(x))
            except (ValueError,OverflowError): continue
            if last is not None and fx*last[1]<0:
                a,b=last[0],float(x);break
            last=(float(x),fx)
        if a is None:
            if abs(eval_initial(params.mtc)[0]) < 1e-10:
                mi = params.mtc  # exact fixed-M root of loose-state alternative
            else:
                raise ValueError('Coupled image-state initialization has no admissible bracketed root for Mi')
        else:
            mi=brentq(lambda x: eval_initial(x)[0],a,b,xtol=1e-12)
        _,pimg_initial,dmin=eval_initial(mi)
        mi_tc=mi

    pi_over_p_yield = math.exp(eta / mi - 1.0)
    pimax_over_p_check = math.exp(-dmin / mi)
    ocr_max = pimax_over_p_check / pi_over_p_yield
    actual_ocr = min(params.ocr, ocr_max)

    pimg_over_p = actual_ocr * math.exp(eta / mi - 1.0)
    if params.initialization == "coupled_yield":
        actual_ocr=params.ocr  # Do not silently clamp OCR; require explicit admissibility.
        pimg_over_p=pimg_initial/sig_m
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
        den_chi_now = 1.0 - lambda_now * params.chi / (params.mtc if params.chi_conversion == "book_c3b_mtc" else mi)
        chi_i_now = params.chi / den_chi_now
        dmin_current_approx = chi_i_now * psi_i_now
        pimax_over_p_current_approx = math.exp(-dmin_current_approx / mi)
        dp_consistent = mi - eta_now
        g_now = elastic_shear_modulus(sig_m, gref=params.gref, m=params.m, pref=params.pref)
        k_now = g_now * k_over_g
        delta_u = sig_m0 * (1.0 - params.eta0 / 3.0) - (sig_m - sig_q / 3.0)  # constant total sigma3; supports nonzero initial eta0
        rows.append(
            {
                "step": step,
                "eps_qp": plastic_progress,
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
    if params.initialization == "coupled_yield" and actual_ocr > 1.0+1.e-10:
        # Elastic-to-yield intersection with the IMAGE-state dependent Mi.
        # Keep p_i constant during elastic loading (no plastic hardening).
        from ._roots import brentq  # was scipy.optimize.brentq
        pimg_fixed=sig_m*pimg_over_p
        pstart=sig_m
        qstart=sig_q
        e_start=e
        gstart=elastic_shear_modulus(pstart,gref=params.gref,m=params.m,pref=params.pref)
        kstart=gstart*k_over_g
        def stage_at(t):
            if params.drainage=='drained':
                p_new=pstart+t
                q_new=qstart+params.delta_qp*t
                if p_new<=0: raise ValueError('Elastic path reached nonpositive pressure')
                if params.m==1:
                    vol=(pstart/kstart)*math.log(p_new/pstart)
                else:
                    vol=(pstart/kstart)*((p_new/pstart)**(1-params.m)-1)/(1-params.m)
                en=e_start-(1+e_start)*vol
            else:
                p_new=pstart
                q_new=qstart+t
                vol=0.0
                en=e_start
            eci,_=e_crit(pimg_fixed,params.csl_type,gamma=params.gamma,lambda10=params.lambda10,
                         ca=params.ca,cb=params.cb,cc=params.cc,pref=params.pref)
            den=1.0-lambda_local*params.chi/params.mtc
            dil=(params.chi/den)*(en-eci)
            M=image_ratio(dil)
            yield_gap=q_new/p_new-M*(1+math.log(pimg_fixed/p_new))
            return yield_gap,p_new,q_new,en,vol,M,dil
        startgap=stage_at(0)[0]
        if startgap>1.e-8:raise ValueError('Initial stress state lies outside the NorSand yield surface')
        bound=max(10*pstart,1000.)
        bracket=1e-6
        while bracket<bound and stage_at(bracket)[0]<0: bracket*=2
        if bracket>=bound and stage_at(bracket)[0]<0:
            raise ValueError('Unable to locate an elastic yield intersection')
        t_yield=brentq(lambda t:stage_at(t)[0],0,bracket,xtol=1.e-11)
        # The elastic-to-yield segment is a REAL resolved path, not an instantaneous
        # jump. Integrate engineering elastic strains analytically for the specified
        # stress-dependent G(p) and K(p), and save intermediate points. The
        # constitutive plastic update begins only after the yield intersection.
        # The number of output samples does not control constitutive accuracy.
        _,p_y,q_y,e_y,vol_y,mi_y,d_y=stage_at(t_yield)
        g0=elastic_shear_modulus(pstart,gref=params.gref,m=params.m,pref=params.pref)
        def elastic_eps_q(t):
            if params.drainage == 'undrained':
                return t/(3.0*g0)
            p=pstart+t
            if abs(1.0-params.m)<1e-12:
                return (params.delta_qp*pstart/(3.0*g0))*math.log(p/pstart)
            return (params.delta_qp*pstart/(3.0*g0))*(((p/pstart)**(1.0-params.m)-1.0)/(1.0-params.m))
        eps_y=elastic_eps_q(t_yield)+vol_y/3.0
        # 24 outputs minimum, about one point per 0.05% axial strain otherwise.
        nelastic=min(1000,max(24,int(math.ceil(eps_y/0.0005))))
        for ie in range(1,nelastic+1):
            ti=t_yield*ie/nelastic
            _,pnew,qnew,enew,vol,mi_new,dnew=stage_at(ti)
            ge=elastic_eps_q(ti)
            sig_m=pnew;sig_q=qnew;eta=qnew/pnew;e=enew;mi=mi_new;mi_tc=mi_new;dmin=dnew
            ep_v=vol;ep_g=ge;ep1=ge+vol/3.0;sample_ht=1.0-ep1
            pimg_over_p=pimg_fixed/sig_m
            save_row(step=ie,dp_current=None,drainage_now=params.drainage)
        j_plastic=nelastic+1
    elif actual_ocr > 1.001:
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
        plastic_progress += dep_gp
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

    def advance(step: int, dep_gp: float, record: bool = True) -> float:
        nonlocal sig_q, sig_m, eta, e, pimg_over_p, pimx_over_p, mi, mi_tc, dmin, ep_v, ep_g, ep1, sample_ht, plastic_progress
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

        den_chi = 1.0 - lambda_local * params.chi / (params.mtc if params.chi_conversion == "book_c3b_mtc" else mi_tc)
        if den_chi <= 1e-12:
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
        mi_tc = image_ratio(dmin)
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
            if sig_m <= 0.1:
                raise ValueError(f"Undrained effective stress dropped to inadmissible p={sig_m:.6g} kPa at step {step}; rejecting increment")

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
        plastic_progress += dep_gp
        ep1 += dep1
        e = e - (1.0 + e) * dep_v
        if e <= 0:
            raise ValueError(f"Void ratio became non-positive at step {step}.")
        sample_ht *= 1.0 - dep1

        if record:
            save_row(step=step, dp_current=dp, drainage_now=params.drainage)
        return dp

    def state_vector() -> tuple:
        return (sig_q, sig_m, eta, e, pimg_over_p, pimx_over_p, mi, mi_tc, dmin, ep_v, ep_g, ep1, sample_ht, plastic_progress)

    def put_state(v: tuple) -> None:
        nonlocal sig_q, sig_m, eta, e, pimg_over_p, pimx_over_p, mi, mi_tc, dmin, ep_v, ep_g, ep1, sample_ht, plastic_progress
        (sig_q, sig_m, eta, e, pimg_over_p, pimx_over_p, mi, mi_tc, dmin, ep_v, ep_g, ep1, sample_ht, plastic_progress) = v

    if control not in ('plastic_deviatoric','total_axial'):
        raise ValueError('control must be plastic_deviatoric or total_axial')
    if control == 'total_axial':
        # Axial-strain-controlled trial driver: the incremental imposed ENGINEERING
        # axial strain is inverted for plastic deviatoric strain, at fixed
        # drainage boundary conditions.  We solve eps1(h)=target by scalar root,
        # then estimate numerical error using one full target axial step versus
        # two half target axial steps, with full state rollback.
        # NOTE: the constitutive update is still the archived explicit algorithm;
        # this is a new boundary-control wrapper, not a new implicit constitutive law.
        from ._roots import brentq  # was scipy.optimize.brentq
        target=params.max_strain if total_axial_target is None else total_axial_target
        if not (0 < target < 0.7):
            raise ValueError('Engineering axial target must be between 0 and 0.7')
        start_axial=1.0-sample_ht
        if target <= start_axial+1.e-12:
            raise ValueError(f'Target axial strain {target:.7g} is at/below elastic-yield strain {start_axial:.7g}; elastic-only loading not implemented in this research branch')
        def axial_read():
            return 1.0-sample_ht
        def invert_axial(axial_step:float,step_index:int):
            init=state_vector()
            initial_ax=axial_read()
            if axial_step<=0:raise ValueError('Axial increment must be positive')
            # Continuity check at zero plastic increment. If the constitutive
            # method changes stress at h->0, no amount of time stepping can fix it.
            def residual(h):
                put_state(init)
                try:
                    advance(step_index,h,record=False)
                    val=axial_read()-initial_ax-axial_step
                    if not math.isfinite(val): raise ValueError('nonfinite axial strain')
                    return val
                except Exception:
                    put_state(init)
                    raise
            r0=residual(0.0)
            put_state(init)
            if r0>min(1.e-10,0.01*axial_step):
                raise ValueError(f'Nonzero-strain update discontinuity in constitutive law: residual at zero plastic increment={r0:.4g}')
            # Find bracket; if a trial is inadmissible, reduce the external
            # axial increment (handled in the outer rejection loop).
            upper=min(max(axial_step*1.2,1.e-9),0.05)
            fup=None
            for _ in range(36):
                try: fup=residual(upper)
                except (ValueError,OverflowError,ZeroDivisionError):
                    put_state(init)
                    raise
                put_state(init)
                if fup>=0:break
                upper*=2
                if upper>0.2:raise ValueError('No admissible positive plastic increment produces the target axial strain')
            else:raise ValueError('Could not bracket plastic increment for axial control')
            root=brentq(residual,0.,upper,xtol=1.e-13,rtol=1.e-12)
            put_state(init)
            last_dp=advance(step_index,root,record=False)
            achieved=axial_read()-initial_ax
            if abs(achieved-axial_step)>max(1.e-10,1.e-7*axial_step):
                raise ValueError(f'Failed prescribed axial-strain constraint: error={achieved-axial_step}')
            return root,last_dp

        achieved_axial=start_axial
        hax=min(0.002,target-start_axial)
        accepted=0;rejected=0;errors=[];max_attempts=500000;max_accepted=250000
        hmin=max(1.e-10,1.e-9*(target-start_axial))
        while achieved_axial<target-1.e-12:
            if accepted+rejected>max_attempts or accepted>max_accepted:
                raise ValueError('Axial-controlled integration exceeded maximum attempts')
            h=min(hax,target-achieved_axial)
            old=state_vector()
            full=None; failure=None
            try:
                invert_axial(h,accepted+j_plastic)
                full=state_vector()
            except (ValueError,OverflowError,ZeroDivisionError,FloatingPointError) as err:
                failure=str(err)
            put_state(old)
            if failure is None:
                try:
                    _,lastdp=invert_axial(h/2,accepted+j_plastic)
                    _,lastdp=invert_axial(h/2,accepted+j_plastic)
                    half=state_vector()
                except (ValueError,OverflowError,ZeroDivisionError,FloatingPointError) as err:
                    failure=str(err)
            if failure:
                put_state(old);rejected+=1;hax=0.5*h
                if hax<hmin:
                    raise ValueError(f'Axial-controlled minimum increment exhausted at eps1={achieved_axial:.8g}; {failure}')
                continue
            idx=[0,1,3,4,6,10,11]
            floors=[1.,1.,0.02,0.02,0.02,0.001,0.001]
            scaled=[]
            for ix,floor in zip(idx,floors):
                a=float(full[ix]);b=float(half[ix]);c=float(old[ix])
                scale=atol*floor+rtol*max(abs(b),abs(c),floor)
                scaled.append(abs(a-b)/scale)
            err=max(scaled)
            if not math.isfinite(err) or err>1:
                put_state(old);rejected+=1
                hax=h*max(0.2,min(0.8,0.8/math.sqrt(err) if math.isfinite(err) else 0.2))
                if hax<hmin:
                    raise ValueError(f'Axial local error stalled at eps1={achieved_axial:.8g}; scaled_error={err:g}')
                continue
            accepted+=1
            achieved_axial=axial_read()
            errors.append(err)
            save_row(step=accepted+j_plastic-1,dp_current=lastdp,drainage_now=params.drainage)
            growth=2. if err<0.1 else min(2.,max(0.8,0.9/math.sqrt(max(err,1.e-12))))
            hax=h*growth
        df=pd.DataFrame(rows)
        df.attrs['adaptive_diagnostics']={
            'accepted_steps':accepted,'rejected_steps':rejected,
            'achieved_engineering_axial_strain':achieved_axial,
            'requested_engineering_axial_strain':target,
            'max_scaled_local_error':max(errors) if errors else None,
            'local_rtol':rtol,'local_atol':atol,
            'control':'prescribed total engineering axial strain (scalar inversion of plastic increment)',
            'method':'explicit constitutive step, scalar axial-control root, local step doubling',
            'initial_elastic_stage':'resolved elastic strain/stress history to yield using stress-dependent elastic moduli'}
        return df

    if not adaptive:
        for step in range(j_plastic, params.n_steps):
            advance(step, dep_gp)
        return pd.DataFrame(rows)

    # Adaptive step doubling integrates the ORIGINAL plastic-deviatoric-strain
    # driver with local rejection and rollback. It does not impose total axial strain.
    # One full Euler increment versus two half increments provides a local error estimate.
    total_plastic = (params.n_steps - j_plastic) * dep_gp
    progressed = 0.0
    trial_h = min(dep_gp, total_plastic)
    accepted = 0
    rejected = 0
    errors = []
    minimum_h = max(1e-13, total_plastic * 1e-11)
    max_accepted = 250000
    max_attempts = 1000000
    attempt = 0
    while progressed < total_plastic - 1e-14 * max(1.0,total_plastic):
        attempt += 1
        if attempt > max_attempts or accepted > max_accepted:
            raise ValueError('Adaptive integration exceeded maximum steps; no convergence')
        h = min(trial_h, total_plastic-progressed)
        old = state_vector()
        failure = None
        full = None
        try:
            advance(accepted + j_plastic, h, record=False)
            full = state_vector()
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as err:
            failure = str(err)
        put_state(old)
        if failure is None:
            try:
                advance(accepted + j_plastic, 0.5*h, record=False)
                last_dp = advance(accepted + j_plastic, 0.5*h, record=False)
                half = state_vector()
            except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as err:
                failure = str(err)
        if failure is not None:
            put_state(old)
            rejected += 1
            trial_h = 0.5*h
            if trial_h < minimum_h:
                raise ValueError(f'Adaptive minimum step exhausted at plastic strain {progressed:.8g}; last error: {failure}')
            continue
        # Scaled difference in physical observables, not just p/q.
        # chi conversion must be well-conditioned in its own right.
        idx=[0,1,3,4,6,10,11] # q,p,e,p_i/p,Mi,eps_g,eps1
        floors=[1.,1.,0.02,0.02,0.02,0.001,0.001]
        per=[]
        for ix, floor in zip(idx,floors):
            v1=float(full[ix]); v2=float(half[ix]); v0=float(old[ix])
            scale=atol*floor + rtol*max(abs(v2),abs(v0),floor)
            per.append(abs(v1-v2)/scale)
        err=max(per)
        if not math.isfinite(err):
            err=math.inf
        if err > 1.0:
            put_state(old)
            rejected += 1
            trial_h = max(0.2, min(0.8,0.8/(math.sqrt(err) if math.isfinite(err) else 5.0)))*h
            if trial_h < minimum_h:
                raise ValueError(f'Adaptive local error failed at plastic strain {progressed:.8g}; normalized error {err:g}')
            continue
        accepted += 1
        progressed += h
        # use accepted two-half-steps state, which is already in scope
        errors.append(err)
        save_row(step=accepted+j_plastic-1,dp_current=last_dp,drainage_now=params.drainage)
        # First-order Euler: difference of full/two-half scales as O(h^2) locally.
        growth=2.0 if err < 0.1 else min(2.,max(0.8,0.9/math.sqrt(max(err,1.e-12))))
        trial_h=h*growth
    result=pd.DataFrame(rows)
    result.attrs['adaptive_diagnostics']={
      'accepted_steps':accepted,'rejected_steps':rejected,
      'max_scaled_local_error':float(max(errors)) if errors else None,
      'plastic_strain_integrated':progressed,
      'local_rtol':rtol,'local_atol':atol,
      'method':'explicit Euler step doubling, accepted solution from two half steps',
      'control':'plastic deviatoric strain; NOT imposed total axial strain'}
    return result
