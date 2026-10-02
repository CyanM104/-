import numpy as np
import astropy.units as u
import astropy.constants as cst
from astropy.modeling.physical_models import BlackBody

from src.density import MultiBrokenPowerLawProfile
from src.solver import solve_nlte_profile
from src.transport_jax import Photosphere, LineTransition

def synthesize_kilonova_spectrum(wav_grid_AA, epoch_days, T_phot, v_phot, v_max, X_Sr, alpha2=5.0):
    """
    Unified entry point executing full microphysics NLTE followed by 3D ray-tracing.
    """
    t_sec = epoch_days * 86400.0
    density_profile = MultiBrokenPowerLawProfile(exponents=[2.0, alpha2, 15.0], v_breaks=[0.25, 0.38])

    # 1. Compute Level Populations and Sobolev depths along radial velocity shells
    v_shells = np.linspace(v_phot, v_max, 20)
    transitions, shell_data = solve_nlte_profile(
        epoch_days=epoch_days,
        T_phot=T_phot,
        v_phot=v_phot,
        v_shells=v_shells,
        mass_fraction=X_Sr,
        density_profile=density_profile
    )

    # 2. Build Photosphere and integrate rays
    continuum = BlackBody(temperature=T_phot * u.K, scale=1.0 * u.Unit("erg/(s cm2 Hz sr)"))
    photosphere = Photosphere(
        v_phot=v_phot * cst.c.cgs.value,
        v_max=v_max * cst.c.cgs.value,
        t_d=t_sec,
        continuum=continuum,
        line_list=transitions,
        T_phot_comoving=T_phot
    )

    calc_wav, calc_flux = photosphere.calc_spectrum(
        start_wav=np.min(wav_grid_AA) * u.AA,
        end_wav=np.max(wav_grid_AA) * u.AA,
        n_points=len(wav_grid_AA)
    )

    return np.interp(wav_grid_AA, calc_wav, calc_flux)
