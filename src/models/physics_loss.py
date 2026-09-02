"""
Physics-consistency loss  (CLAUDE.md §5.3).

The training loss needs gradients w.r.t. predicted T and S, so it uses the
**EOS-80 (UNESCO 1983) equation of state implemented in torch** — this is a
standard published polynomial (Fofonoff & Millard 1983), not a hand-rolled
approximation; it is expressed in torch only so autograd can flow. `gsw`
(full TEOS-10) is used for the non-differentiable physics *diagnostics* in
src/evaluate.py, so the honest final numbers still come from TEOS-10.

Penalties:
  1. density inversions  — rho must be non-decreasing with depth in a stable
     column; any negative d(rho)/d(depth) between valid adjacent levels is
     penalised (this is the §5.3 term).
  2. (optional) stability roughness — discourages physically implausible
     spikes in the buoyancy-frequency profile; weak, off by default.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# EOS-80 (UNESCO 1983) in torch  — density of seawater, kg/m^3
# SP: practical salinity, t: in-situ temperature (deg C), p: sea pressure (dbar)
# --------------------------------------------------------------------------- #

def density_eos80(SP: torch.Tensor, t: torch.Tensor, p_dbar: torch.Tensor) -> torch.Tensor:
    T, S = t, SP.clamp_min(0.0)
    P = p_dbar / 10.0                                   # dbar -> bar

    rho_w = (999.842594 + 6.793952e-2 * T - 9.095290e-3 * T**2
             + 1.001685e-4 * T**3 - 1.120083e-6 * T**4 + 6.536332e-9 * T**5)

    A = (8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T**2
         - 8.2467e-7 * T**3 + 5.3875e-9 * T**4)
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2
    C = 4.8314e-4
    rho_0 = rho_w + A * S + B * S**1.5 + C * S**2       # rho(S,T,p=0)

    Kw = (19652.21 + 148.4206 * T - 2.327105 * T**2
          + 1.360477e-2 * T**3 - 5.155288e-5 * T**4)
    K0 = (Kw + (54.6746 - 0.603459 * T + 1.09987e-2 * T**2 - 6.1670e-5 * T**3) * S
          + (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T**2) * S**1.5)
    Aw = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * T**2 - 5.77905e-7 * T**3
    A_ = (Aw + (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * T**2) * S
          + 1.91075e-4 * S**1.5)
    Bw = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * T**2
    B_ = Bw + (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * T**2) * S
    K = K0 + A_ * P + B_ * P**2

    return rho_0 / (1.0 - P / K)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #

def physics_consistency_loss(pred_T: torch.Tensor, pred_S: torch.Tensor,
                             pressure_dbar: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             stability_weight: float = 0.0) -> torch.Tensor:
    """
    pred_T, pred_S : [B, L]  predicted profiles in physical units (deg C, PSU)
    pressure_dbar  : [B, L] or [L]  sea pressure of each level
    mask           : [B, L]  1 where the level is valid (both adjacent levels
                     must be valid for a pair to count)
    """
    if pressure_dbar.dim() == 1:
        pressure_dbar = pressure_dbar.unsqueeze(0).expand_as(pred_T)

    rho = density_eos80(pred_S, pred_T, pressure_dbar)          # [B, L]
    drho = rho[:, 1:] - rho[:, :-1]                             # want >= 0

    if mask is not None:
        pair = (mask[:, 1:] * mask[:, :-1])
    else:
        pair = torch.ones_like(drho)

    denom = pair.sum().clamp_min(1.0)
    inversion_penalty = (torch.relu(-drho) * pair).sum() / denom

    loss = inversion_penalty
    if stability_weight > 0:
        # second difference of rho — discourage jagged N^2
        d2 = drho[:, 1:] - drho[:, :-1]
        pair2 = pair[:, 1:] * pair[:, :-1]
        loss = loss + stability_weight * (d2.pow(2) * pair2).sum() / pair2.sum().clamp_min(1.0)
    return loss


# --------------------------------------------------------------------------- #
# TEOS-10 diagnostics (numpy / gsw) — evaluation only, no gradients
# --------------------------------------------------------------------------- #

def physics_diagnostics_gsw(T, S, pressure_dbar, lon, lat, mask=None) -> dict:
    """Fraction of adjacent-level pairs that are density-inverted, per TEOS-10."""
    import gsw

    T = np.asarray(T, "float64"); S = np.asarray(S, "float64")
    p = np.broadcast_to(np.asarray(pressure_dbar, "float64"), T.shape)
    lon = np.broadcast_to(np.asarray(lon, "float64")[:, None], T.shape)
    lat = np.broadcast_to(np.asarray(lat, "float64")[:, None], T.shape)
    if mask is None:
        mask = np.ones_like(T)

    SA = gsw.SA_from_SP(S, p, lon, lat)
    CT = gsw.CT_from_t(SA, T, p)
    sigma = gsw.sigma0(SA, CT)
    dsig = np.diff(sigma, axis=1)
    pair = mask[:, 1:] * mask[:, :-1]
    inverted = (dsig < -1e-4) & (pair > 0)
    return {
        "inversion_fraction": float(inverted.sum() / max(pair.sum(), 1)),
        "mean_negative_dsigma": float(-dsig[(dsig < 0) & (pair > 0)].mean())
        if ((dsig < 0) & (pair > 0)).any() else 0.0,
    }
