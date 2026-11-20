"""NorSand solver used by calibration step 5 (Part III manual trials).

norsand_core.py and geomechanics.py are copies of the Part III solver. simulate() is
the same as Part III's simulation_api.simulate: monotonic CID/CIU, state-based image
rule, C.3b chi conversion, coupled yield initialization, strain-controlled to a total
axial strain target. No fitting.
"""
from dataclasses import replace

from .norsand_core import NorSandParams, run_triaxial


def simulate(params: NorSandParams, *, axial_target: float = 0.18, rtol: float = 0.0003):
    if not 1e-6 <= rtol <= 0.01:
        raise ValueError("rtol must lie between 1e-6 and 0.01")
    standard = replace(params, chi_conversion="book_c3b_mtc",
                       initialization="coupled_yield", image_ratio_rule="geostudio_loose_m")
    result = run_triaxial(standard, control="total_axial", total_axial_target=axial_target, rtol=rtol)
    result.attrs["model_definition"] = {
        "image_rule": "Mi = Mtc for positive image limiting dilatancy; symmetric dense branch otherwise",
        "reference": "Jefferies and Been (2016), Appendix C: GeoStudio-compatible loose-M variant",
        "chi_conversion": "Approximate C.3b with local curved-CSL slope",
        "S_softening": standard.s,
        "hardening_H": "max(H0-Hy*psi, 0.5/lambda_local)",
        "calibration_performed": False,
    }
    return result


__all__ = ["NorSandParams", "run_triaxial", "simulate"]
