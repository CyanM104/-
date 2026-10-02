import numpy as np

def load_collision_strengths(filename):
    # Load temperature-dependent effective collision strengths Omega_ij
    return {}

def calculate_photoionization_rate(sigma, J_nu, nu):
    # integrate level-resolved cross sections sigma(v) against diluted photospheric field J_nu
    h = 6.62607015e-27
    return np.trapz(4 * np.pi * sigma / (h * nu) * J_nu, nu)

def calculate_non_thermal_ionization(t_d, v_max, M_ej=0.04, eta=1.0):
    # Model radioactive beta-decay heating
    t_e = 15.0 * (eta * M_ej / 0.01)**(2.0/3.0) * (v_max / 0.2)**(-2)
    f_beta = 0.2 * (1.0 + t_d / t_e)**(-1.5)
    q_dep = 1e10 * t_d**(-1.3) * f_beta
    # Work functions
    w_SrI = 124 # eV
    w_SrII = 272 # eV
    w_SrIII = 444 # eV
    w_SrIV = 608 # eV
    return q_dep

def calculate_recombination_rate(T):
    # Temperature-dependent rates for Sr II -> I, Sr III -> II, etc
    return 1e-13 * (T / 1e4)**(-0.5)
