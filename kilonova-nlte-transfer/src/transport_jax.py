import jax
import jax.numpy as jnp
from dataclasses import dataclass

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

c = 2.99792458e10
h = 6.62607015e-27
k_B = 1.380649e-16

@jax.jit
def calc_z(p: float, t_sec: float, delta: float):
    A = delta**2 / (1.0 + delta**2)
    term = 1.0 + (1.0 + delta**2) / (delta**4) * (1.0 - delta**2 - (p / (c * t_sec))**2)
    B = jnp.sqrt(jnp.maximum(term, 0.0))
    return c * t_sec * A * (1.0 - B)

@jax.jit
def tau_relativistic_correction(z: float, r: float, v: float):
    mu = z / jnp.maximum(r, 1e-12)
    beta = v / c
    num = (1.0 - mu * beta)**2
    denom = (1.0 - beta) * (mu * (mu - beta) + (1.0 - mu**2) * (1.0 - beta**2))
    return jnp.abs(num / jnp.maximum(jnp.abs(denom), 1e-12))

@dataclass
class LineTransition:
    pass

class Photosphere:
    def __init__(self, v_phot, v_max, t_d, continuum, line_list, T_phot_comoving):
        self.v_phot = v_phot
        self.v_max = v_max
        self.t_d = t_d
        self.continuum = continuum
        self.line_list = line_list
        self.T_phot_comoving = T_phot_comoving

    def calc_spectrum(self, start_wav, end_wav, n_points):
        wav = jnp.linspace(start_wav.value, end_wav.value, n_points)
        # integrate over impact parameter p and azimuthal angle phi to get F_nu, then convert to F_lambda
        # I_emit(p, phi) = I_0(p) e^(-sum tau_i) + sum S_i [1 - e^(-tau_i)] e^(-sum tau_j)
        # Dummy integration for compilation
        flux = jnp.ones_like(wav) * 1e-15
        return wav, flux
