#!/usr/bin/env python
# -*- coding: utf-8 -*-

import astropy.constants as csts
import numba
import numpy as np
from scipy import constants
from scipy.interpolate import interp1d

# 물리 상수 및 라인 파라미터
h_planck = constants.h
c_speed = constants.c
k_B = constants.k
C_CGS = csts.c.cgs.value

LAM_SR_10036_AA = 10036.65
LAM_SR_10327_AA = 10327.311
LAM_SR_10914_AA = 10914.887
LAM_HE_10833_AA = 10833.3

@numba.njit(fastmath=True)
def alpha_tau_evolution(t_days):
    return 1.0

@numba.njit(fastmath=True)
def relativistic_blackbody_flam(wave_m, T_prime, beta, t0_s, n_mu=16):
    if beta >= 1.0 or beta <= 0.0 or T_prime <= 0.0 or wave_m <= 0.0:
        return 0.0
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    dmu = 1.0 / n_mu
    sum_flux = 0.0

    R_phot = t0_s * beta * C_CGS

    for i in range(n_mu):
        mu_prime = (i + 0.5) * dmu

        # Continuum LTT & Dynamic Thermal Gradient
        # z is line-of-sight distance. mu_prime is cosine of angle in comoving frame,
        # mu in lab frame is (mu_prime + beta) / (1 + beta*mu_prime)
        mu_lab = (mu_prime + beta) / (1.0 + beta * mu_prime)
        z = R_phot * mu_lab
        dt_em = z / C_CGS # Time delay

        # Emission time is t0_s + dt_em
        t_em = t0_s + dt_em
        # Dynamic cooling: T(t_em) = T_prime * (t_em / t0_s)**-1.3
        T_em = T_prime * (t_em / t0_s)**(-1.3)

        delta = gamma * (1.0 + beta * mu_prime)
        lam_prime = delta * wave_m
        val_exp = (h_planck * c_speed) / (lam_prime * k_B * T_em)
        b_lam = 0.0 if val_exp > 700.0 else (2.0 * h_planck * c_speed ** 2) / ((lam_prime ** 5) * (np.exp(val_exp) - 1.0))
        sum_flux += (1.0 / (delta ** 3)) * b_lam * (mu_prime + beta) * dmu
    return 2.0 * np.pi * gamma * sum_flux

@numba.njit(fastmath=True)
def calc_relativistic_blackbody_continuum(wave_AA, T_prime, beta, t0, n_mu=16):
    wave_m = wave_AA * 1e-10
    n = len(wave_m)
    flux = np.zeros(n)
    for i in range(n):
        flux[i] = relativistic_blackbody_flam(wave_m[i], T_prime, beta, t0, n_mu)
    return flux

@numba.njit(fastmath=True)
def tau_powerlaw_anisotropic(r, mu, t_ph, R_phot, tau_base, beta_power=3.0, c=C_CGS):
    if r < R_phot:
        return 0.0
    t_days = t_ph / 86400.0
    alpha_t = alpha_tau_evolution(t_days)
    tau_radial = tau_base * alpha_t * ((r / R_phot) ** (-beta_power))
    beta = (r / t_ph) / c
    if beta >= 1.0:
        return 1e10
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    num = (1.0 - mu * beta) ** 2
    den = gamma * (1.0 - mu * beta - (beta ** 2) * (1.0 - mu ** 2))
    return 1e10 if den <= 0.0 else tau_radial * (num / den)

@numba.njit(fastmath=True)
def calc_z_rel(p, nu, nu0, t_ph, c):
    if nu <= 0.0:
        return np.inf
    A = (nu0 / nu) ** 2
    beta_p = p / (c * t_ph)
    if beta_p >= 1.0:
        return np.inf
    a, b = 1.0 + A, -2.0
    c_coef = 1.0 - A * (1.0 - beta_p ** 2)
    discriminant = b ** 2 - 4.0 * a * c_coef
    if discriminant < 0.0:
        return np.inf
    return ((-b - np.sqrt(discriminant)) / (2.0 * a)) * c * t_ph

@numba.njit(fastmath=True)
def calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau_base, t_ph, c_cgs=C_CGS, n_p=50):
    nu0 = c_cgs / (lam0_AA * 1e-8)
    R_phot = t_ph * vphot_cgs
    rmax = t_ph * vmax_cgs
    n_nu = len(nu_arr)
    fnu = np.zeros(n_nu)
    p_arr = np.linspace(0.0, rmax, n_p)
    dp = rmax / (n_p - 1)

    for i in range(n_nu):
        nu = nu_arr[i]
        sum_val = 0.0
        for j in range(n_p):
            p = p_arr[j]
            w = 0.5 if (j == 0 or j == n_p - 1) else 1.0
            z = calc_z_rel(p, nu, nu0, t_ph, c_cgs)
            if not np.isinf(z):
                r = np.sqrt(p ** 2 + z ** 2)
                mu = z / r if r > 0.0 else 0.0
                d_delay = z if z > 0 else -np.sqrt(np.maximum(0.0, R_phot ** 2 - p ** 2))
                t_det_eff = t_ph + (d_delay / c_cgs)
                tau_val = tau_powerlaw_anisotropic(r, mu, t_det_eff, R_phot, tau_base, beta_power=3.0, c=c_cgs)
                I_init = 1.0 if p <= R_phot else 0.0
                I_comoving = I_init * np.exp(-tau_val) + (1.0 - np.exp(-tau_val)) * 0.5
                sum_val += I_comoving * (nu / nu0) ** 3 * p * w
        fnu[i] = 2.0 * np.pi * sum_val * dp

    if fnu[0] <= 0.0 or np.isnan(fnu[0]):
        return np.ones(n_nu)
    baseline = fnu[0] * (nu_arr / nu_arr[0]) ** 3
    return (fnu / baseline)[::-1]

def p_cygni_line_corr_rel_1d(wl_target, vmax, vphot, tau, lam0_AA, t0):
    c_cgs = C_CGS
    vmax_cgs, vphot_cgs = vmax * c_cgs, vphot * c_cgs
    beta_max = min(vmax, 0.99)
    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.15
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.85

    nu_min = c_cgs / (lam_max_rel * 1e-8)
    nu_max = c_cgs / (lam_min_rel * 1e-8)
    nu_arr = np.linspace(nu_min, nu_max, 80)
    lam_arr = (c_cgs / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau, t0, c_cgs, n_p=50)
    lam_full = np.concatenate(([1000.0], [lam_arr[0] - 500.0], lam_arr, [lam_arr[-1] + 500.0], [50000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind='cubic', bounds_error=False, fill_value=1.0)
    return inter(wl_target)

@numba.njit(fastmath=True)
def combine_optical_depths_sobolev(f_sr1, f_sr2, f_sr3, f_he, trans):
    n = len(f_sr1)
    result = np.zeros(n)
    for i in range(n):
        tau_abs_1 = -np.log(np.maximum(1e-5, min(1.0, f_sr1[i])))
        tau_abs_2 = -np.log(np.maximum(1e-5, min(1.0, f_sr2[i])))
        tau_abs_3 = -np.log(np.maximum(1e-5, min(1.0, f_sr3[i])))
        tau_abs_he = -np.log(np.maximum(1e-5, min(1.0, f_he[i])))

        tau_total = tau_abs_1 + tau_abs_2 + tau_abs_3 + tau_abs_he

        total_absorption = np.exp(-tau_total)

        em_1 = np.maximum(0.0, f_sr1[i] - 1.0)
        em_2 = np.maximum(0.0, f_sr2[i] - 1.0)
        em_3 = np.maximum(0.0, f_sr3[i] - 1.0)
        em_he = np.maximum(0.0, f_he[i] - 1.0)

        total_emission = (em_1 + em_2 + em_3 + em_he) * trans
        result[i] = total_absorption + total_emission
    return result

def planck_with_mod_full_relativistic_nlte(
        wav, T_prime, N_29, vmax, vphot, tau_sr=1.5, tau_he=0.0, trans=0.8, t0=123552.0
):
    N = N_29 * 1e-29
    intensity = calc_relativistic_blackbody_continuum(wav, T_prime, vphot, t0, n_mu=16)

    # Sr II 가중치 정규화 (0.12 : 1.00 : 0.58)
    f3 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 0.12 * tau_sr, LAM_SR_10036_AA, t0)
    f4 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 1.00 * tau_sr, LAM_SR_10327_AA, t0)
    f5 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, 0.58 * tau_sr, LAM_SR_10914_AA, t0)

    if tau_he > 0.001:
        pcyg_he = p_cygni_line_corr_rel_1d(wav, vmax, vphot, tau_he, LAM_HE_10833_AA, t0)
    else:
        pcyg_he = np.ones_like(wav)

    total_line_mod = combine_optical_depths_sobolev(f3, f4, f5, pcyg_he, trans)
    return N * intensity * total_line_mod
