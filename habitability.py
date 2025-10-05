# habitability.py
import math
import matplotlib.pyplot as plt
from typing import Optional, Tuple

T_SUN_K = 5772.0
RSUN_AU = 0.00465047  # 1 R_sun in AU

# Habitable-zone bounds in stellar flux S (relative to Earth)
HZ_CONSERVATIVE = (0.36, 1.11)
HZ_OPTIMISTIC   = (0.32, 1.77)

def luminosity_rel(radius_rsun: float, teff_k: float) -> float:
    """L*/Lsun = (R/Rsun)^2 * (Teff/5772 K)^4"""
    return (radius_rsun**2) * ((teff_k / T_SUN_K)**4)

def a_from_a_over_rstar(a_over_rstar: float, rstar_rsun: float) -> float:
    """Semi-major axis in AU from (a/R*) and R* (Rsun)."""
    return a_over_rstar * rstar_rsun * RSUN_AU

def a_from_period_mass(period_days: float, mstar_msun: float) -> float:
    """Kepler's 3rd law: a(AU) = (P_year^2 * M_*)^(1/3)"""
    P_year = period_days / 365.25
    return ((P_year**2) * mstar_msun) ** (1/3)

def insolation_rel(L_rel: float, a_au: float) -> float:
    """S (relative to Earth) = (L*/Lsun) / a^2"""
    return L_rel / (a_au**2)

def equilibrium_temp_k(S: float, albedo: float = 0.30) -> float:
    """Airless blackbody equilibrium temperature."""
    return 278.5 * ((1 - albedo)**0.25) * (S**0.25)

EARTH_T_EQ = equilibrium_temp_k(1.0, 0.30)

def _choose_a_au(
    a_au: Optional[float],
    a_over_rstar: Optional[float],
    rstar_rsun: Optional[float],
    period_days: Optional[float],
    mstar_msun: Optional[float],
) -> Optional[float]:
    # Prefer explicit a(AU); else a/R* + R*; else Kepler (P + M*)
    if a_au is not None and a_au > 0:
        return float(a_au)
    if a_over_rstar is not None and rstar_rsun is not None:
        return a_from_a_over_rstar(float(a_over_rstar), float(rstar_rsun))
    if period_days is not None and mstar_msun is not None:
        return a_from_period_mass(float(period_days), float(mstar_msun))
    return None

def _nice_bounds_log(xmin_guess: float, xmax_guess: float) -> tuple[float, float]:
    """Round to clean decades for log scaling."""
    xmin = max(0.01, xmin_guess)
    xmax = max(xmin * 1.2, xmax_guess)
    lo = 10 ** math.floor(math.log10(xmin))
    hi = 10 ** math.ceil(math.log10(xmax))
    return lo, hi

def make_plot(
    name: str = "Exoplanet",
    # Direct flux (optional)
    S: Optional[float] = None,
    # Star + orbit (preferred if sufficient info is present)
    star_teff_k: Optional[float] = None,
    star_radius_rsun: Optional[float] = None,
    a_au: Optional[float] = None,
    a_over_rstar: Optional[float] = None,
    period_days: Optional[float] = None,
    mstar_msun: Optional[float] = None,
    # Visual/physics
    albedo: float = 0.30,
    # Axis control: if None, auto-enable log when dynamic range is large
    log_x: Optional[bool] = None,
) -> Tuple[plt.Figure, dict]:
    """
    Plot planet vs Earth on (S, T_eq). If star+orbit information is sufficient,
    it is used to compute S; otherwise falls back to direct S.
    """
    # 1) Try to compute S from star + orbit
    S_used = None
    if star_teff_k and star_radius_rsun:
        a_try = _choose_a_au(a_au, a_over_rstar, star_radius_rsun, period_days, mstar_msun)
        if a_try and a_try > 0:
            Lrel = luminosity_rel(float(star_radius_rsun), float(star_teff_k))
            S_used = insolation_rel(Lrel, float(a_try))

    # 2) Fallback to direct S if needed
    if S_used is None and S is not None and S > 0:
        S_used = float(S)

    if S_used is None or S_used <= 0:
        raise ValueError("Insufficient info: provide (Teff, R*, and a) or (a/R*+R*) or (P+M*) OR direct S>0.")

    Teq = equilibrium_temp_k(S_used, albedo)

    # --- Axis bounds ---
    lin_xmin = min(HZ_OPTIMISTIC[0] * 0.8, S_used * 0.6)
    lin_xmax = max(HZ_OPTIMISTIC[1] * 1.5, S_used * 1.6)
    lin_xmin = max(0.01, lin_xmin)

    # Choose scaling
    use_log = log_x if log_x is not None else ((lin_xmax / lin_xmin) > 30)
    if use_log:
        x_min, x_max = _nice_bounds_log(lin_xmin, lin_xmax)
    else:
        x_min, x_max = lin_xmin, lin_xmax

    y_min = 150
    y_max = max(400, Teq * 1.25, EARTH_T_EQ * 1.25)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    if use_log:
        ax.set_xscale("log")

    # Background regions
    # Too cold (left of optimistic HZ)
    if x_min < HZ_OPTIMISTIC[0]:
        ax.axvspan(x_min, HZ_OPTIMISTIC[0], facecolor="#93c5fd", alpha=0.25,
                   hatch="///", edgecolor="#93c5fd", label="Too cold")
    # Too hot (right of optimistic HZ)
    if x_max > HZ_OPTIMISTIC[1]:
        ax.axvspan(HZ_OPTIMISTIC[1], x_max, facecolor="#fca5a5", alpha=0.20,
                   hatch="\\\\\\\\", edgecolor="#fca5a5", label="Too hot")

    # Optimistic & Conservative HZ bands
    ax.axvspan(HZ_OPTIMISTIC[0], HZ_OPTIMISTIC[1], color="#3b82f6", alpha=0.20, label="Optimistic HZ")
    ax.axvspan(HZ_CONSERVATIVE[0], HZ_CONSERVATIVE[1], color="#10b981", alpha=0.30, label="Conservative HZ")

    # Boundary lines + labels for optimistic HZ
    ax.axvline(HZ_OPTIMISTIC[0], color="#2563eb", linestyle="--", lw=1)
    ax.axvline(HZ_OPTIMISTIC[1], color="#2563eb", linestyle="--", lw=1)
    ax.text(HZ_OPTIMISTIC[0], y_max*0.97, "HZ outer (cold edge)", rotation=90, va="top", ha="right",
            color="#2563eb", fontsize=8)
    ax.text(HZ_OPTIMISTIC[1], y_max*0.97, "HZ inner (hot edge)", rotation=90, va="top", ha="left",
            color="#2563eb", fontsize=8)

    # Reference Earth & candidate
    ax.axvline(1.0, linestyle=":", linewidth=1, color="gray")
    ax.scatter([1.0], [EARTH_T_EQ], s=40, label="Earth", zorder=3, color="#1f2937")
    ax.scatter([S_used], [Teq], s=60, label=name, zorder=4, color="#f59e0b")

    # Labels and limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Starlight S (relative to Earth)" + ("  [log scale]" if use_log else ""))
    ax.set_ylabel("Equilibrium temperature (K) — airless blackbody")
    ax.grid(True, linestyle=":", linewidth=0.7, which="both")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title(f"Habitability snapshot: {name}  (A={albedo:.2f})", fontsize=10)

    # Region captions (readable even when zones are wide)
    # NEW: captions in axes coordinates (never overlap with HZ labels)
    ax.text(0.03, 0.92, "Too cold", transform=ax.transAxes,
            color="#1f2937", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    ax.text(0.97, 0.92, "Too hot", transform=ax.transAxes, ha="right",
            color="#1f2937", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))


    return fig, {"S": S_used, "Teq": Teq}
