import numpy as np
import astropy.constants as consts
import astropy.units as u

class MultiBrokenPowerLawProfile:
    def __init__(self, exponents=(2.0, 5.0, 15.0), v_breaks=(0.25, 0.38), deltas=(0.01, 0.01), amplitude=1e7):
        self.exponents = np.asarray(exponents, dtype=float)
        self.v_breaks = np.asarray(v_breaks, dtype=float)
        self.deltas = np.asarray(deltas, dtype=float)
        self.amplitude = float(amplitude)
        self.v_ref = float(self.v_breaks[0])

    def __call__(self, v):
        v_arr = np.asarray(v, dtype=float)
        val = self.amplitude * (v_arr / self.v_ref) ** (-self.exponents[0])
        for i in range(len(self.v_breaks)):
            vb = self.v_breaks[i]
            delta = self.deltas[i]
            dalpha = self.exponents[i+1] - self.exponents[i]
            val *= (1.0 + (v_arr / vb) ** (1.0 / delta)) ** (-dalpha * delta)
        return val

def W_rel(v, v_phot):
    beta_phot = v_phot
    beta_v = v
    beta = (beta_phot - beta_v) / (1.0 - beta_phot * beta_v)
    gamma = 1.0 / np.sqrt(np.maximum(1e-12, 1.0 - beta**2))
    mu_c = np.sqrt(np.maximum(0.0, 1.0 - (v_phot / v)**2))
    mu_c_prime = (mu_c - beta) / (1.0 - mu_c * beta)
    return 0.5 * gamma * (1.0 - mu_c_prime) * (1.0 + (beta / 2.0) * (1.0 + mu_c_prime))
