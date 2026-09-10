#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import multiprocessing as mp
import os
import re
import urllib.request
import warnings

import astropy.constants as csts
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy import constants
from scipy.interpolate import interp1d
from scipy.optimize import minimize

try:
    import corner
except ImportError:
    corner = None

try:
    import emcee
except ImportError:
    raise ImportError("MCMC 실행을 위해 'emcee' 라이브러리가 필요합니다.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# [물리 상수 및 라인 정적 파라미터]
# ---------------------------------------------------------------------
h_planck = constants.h
c_speed = constants.c
k_B = constants.k
C_CGS = csts.c.cgs.value

# Sneppen et al. (2023) 근적외선 가우시안 보정 성분
CEN1_AA, SIG1_AA = 15500.0, 580.0
CEN2_AA, SIG2_AA = 20200.0, 800.0

# Chiba et al. (2026) / Arya et al. (2026) 원소별 중심 파장 (Rest Wavelength)
LAM_SR_10036_AA = 10036.65
LAM_SR_10327_AA = 10327.311
LAM_SR_10914_AA = 10914.887
LAM_HE_10833_AA = 10833.3  # He I 2^3S - 2^3P


# ---------------------------------------------------------------------
# [1] 복합 Sobolev 복사전달 엔진 (NLTE 및 He I 스위치 지원)
# ---------------------------------------------------------------------

@numba.njit(fastmath=True)
def alpha_tau_evolution(t_days, use_nlte=True):
    if not use_nlte:
        return 1.0
    if 1.5 < t_days <= 3.0:
        return t_days / 1.5
    elif 3.0 < t_days <= 5.0:
        return 2.0 - 0.5 * (t_days - 3.0)
    else:
        return 1.0


@numba.njit(fastmath=True)
def relativistic_blackbody_flam(wave_m, T_prime, beta, n_mu=16):
    if beta >= 1.0 or beta <= 0.0 or T_prime <= 0.0 or wave_m <= 0.0:
        return 0.0
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    dmu = 1.0 / n_mu
    sum_flux = 0.0
    for i in range(n_mu):
        mu_prime = (i + 0.5) * dmu
        delta = gamma * (1.0 + beta * mu_prime)
        lam_prime = delta * wave_m
        val_exp = (h_planck * c_speed) / (lam_prime * k_B * T_prime)
        b_lam = 0.0 if val_exp > 700.0 else (2.0 * h_planck * c_speed ** 2) / (
                    (lam_prime ** 5) * (np.exp(val_exp) - 1.0))
        sum_flux += (1.0 / (delta ** 3)) * b_lam * (mu_prime + beta) * dmu
    return 2.0 * np.pi * gamma * sum_flux


@numba.njit(fastmath=True)
def calc_relativistic_blackbody_continuum(wave_AA, T_prime, beta, n_mu=16):
    wave_m = wave_AA * 1e-10
    n = len(wave_m)
    flux = np.zeros(n)
    for i in range(n):
        flux[i] = relativistic_blackbody_flam(wave_m[i], T_prime, beta, n_mu)
    return flux


@numba.njit(fastmath=True)
def tau_powerlaw_anisotropic(r, mu, t_ph, R_phot, tau_base, beta_power=3.0, c=C_CGS, use_nlte=True):
    if r < R_phot:
        return 0.0

    t_days = t_ph / 86400.0
    alpha_t = alpha_tau_evolution(t_days, use_nlte=use_nlte)
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
def calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau_base, t_ph, c_cgs=C_CGS, n_p=40, use_nlte=True):
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

                tau_val = tau_powerlaw_anisotropic(r, mu, t_det_eff, R_phot, tau_base, beta_power=3.0, c=c_cgs, use_nlte=use_nlte)
                I_init = 1.0 if p <= R_phot else 0.0
                I_comoving = I_init * np.exp(-tau_val) + (1.0 - np.exp(-tau_val)) * 0.5
                sum_val += I_comoving * (nu / nu0) ** 3 * p * w

        fnu[i] = 2.0 * np.pi * sum_val * dp

    if fnu[0] <= 0.0 or np.isnan(fnu[0]):
        return np.ones(n_nu)
    baseline = fnu[0] * (nu_arr / nu_arr[0]) ** 3
    return (fnu / baseline)[::-1]


def p_cygni_line_corr_rel_1d(wl_target, vmax, vphot, tau, lam0_AA, t0, use_nlte=True):
    c_cgs = C_CGS
    vmax_cgs, vphot_cgs = vmax * c_cgs, vphot * c_cgs
    beta_max = min(vmax, 0.99)
    lam_max_rel = lam0_AA * np.sqrt((1.0 + beta_max) / (1.0 - beta_max)) * 1.08
    lam_min_rel = lam0_AA * np.sqrt((1.0 - beta_max) / (1.0 + beta_max)) * 0.92

    nu_min = c_cgs / (lam_max_rel * 1e-8)
    nu_max = c_cgs / (lam_min_rel * 1e-8)
    nu_arr = np.linspace(nu_min, nu_max, 40)
    lam_arr = (c_cgs / nu_arr[::-1]) * 1e8

    f_normed = calc_rel_line_profile_with_ltt(nu_arr, lam0_AA, vmax_cgs, vphot_cgs, tau, t0, c_cgs, n_p=40, use_nlte=use_nlte)
    lam_full = np.concatenate(([1000.0], [lam_arr[0] - 200.0], lam_arr, [lam_arr[-1] + 200.0], [50000.0]))
    f_full = np.concatenate(([1.0], [1.0], f_normed, [1.0], [1.0]))

    inter = interp1d(lam_full, f_full, kind='linear', bounds_error=False, fill_value=1.0)
    return inter(wl_target)


def planck_with_mod_full_relativistic(
        wav, T_prime, N_29, vmax, vphot, tau_sr=3.80, tau_he=0.0, trans=1.0, amp1=0.31, amp2=0.44, t0=123552.0,
        use_nlte=True, use_he=True
):
    N = N_29 * 1e-29
    intensity = calc_relativistic_blackbody_continuum(wav, T_prime, vphot, n_mu=16)

    pcyg_sr3 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (1.0 / 13.8) * tau_sr, LAM_SR_10036_AA, t0, use_nlte=use_nlte)
    pcyg_sr4 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (8.1 / 13.8) * tau_sr, LAM_SR_10327_AA, t0, use_nlte=use_nlte)
    pcyg_sr5 = p_cygni_line_corr_rel_1d(wav, vmax, vphot, (4.7 / 13.8) * tau_sr, LAM_SR_10914_AA, t0, use_nlte=use_nlte)
    corr_sr = pcyg_sr3 * pcyg_sr4 * pcyg_sr5

    if use_he and tau_he > 0.001:
        pcyg_he = p_cygni_line_corr_rel_1d(wav, vmax, vphot, tau_he, LAM_HE_10833_AA, t0, use_nlte=use_nlte)
    else:
        pcyg_he = np.ones_like(wav)

    total_line_corr = corr_sr * pcyg_he
    mask_emission = total_line_corr > 1.0
    total_line_corr[mask_emission] = (total_line_corr[mask_emission] - 1.0) * trans + 1.0

    gauss1 = amp1 * np.exp(-0.5 * ((wav - CEN1_AA) / SIG1_AA) ** 2)
    gauss2 = amp2 * np.exp(-0.5 * ((wav - CEN2_AA) / SIG2_AA) ** 2)

    total_mod = total_line_corr + gauss1 + gauss2
    return N * intensity * total_mod


def lum_dist_arr(N_29_array, vphot_array, trans_array=1.0, n_days=1.427, dt=0.0):
    c_m = constants.c
    N = np.maximum(N_29_array * 1e-29, 1e-35)
    theta = 2.0 * np.sqrt(N * 5.48e6)
    v = vphot_array * c_m
    t = (n_days - dt) * (3600.0 * 24.0)
    r = v * t

    D = (r / theta) * 2.0
    D_mpc = D * (3.2408e-23)
    return D_mpc


# ---------------------------------------------------------------------
# [2] MCMC 확률 클래스 및 데이터 로더
# ---------------------------------------------------------------------

class MCMCProbabilityWrapper(object):
    def __init__(self, x_fit, y_fit, err_fit, time_s, bounds, use_nlte=True, use_he=True):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.err_fit = err_fit
        self.time_s = time_s
        self.bounds = bounds
        self.use_nlte = use_nlte
        self.use_he = use_he

    def log_prior(self, theta):
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return -np.inf

        if self.use_he:
            T_prime, N_29, vmax, vphot, tau_sr, tau_he, trans, amp1, amp2 = theta
        else:
            T_prime, N_29, vmax, vphot, tau_sr, trans, amp1, amp2 = theta
            tau_he = 0.0

        for val, (low, high) in zip(theta, self.bounds):
            if not (low <= val <= high):
                return -np.inf
        if vphot >= vmax - 0.005 or tau_sr <= 0.001 or N_29 <= 0.0:
            return -np.inf
        return 0.0

    def log_likelihood(self, theta):
        if self.use_he:
            T_prime, N_29, vmax, vphot, tau_sr, tau_he, trans, amp1, amp2 = theta
        else:
            T_prime, N_29, vmax, vphot, tau_sr, trans, amp1, amp2 = theta
            tau_he = 0.0

        try:
            model = planck_with_mod_full_relativistic(
                wav=self.x_fit, T_prime=T_prime, N_29=N_29, vmax=vmax, vphot=vphot,
                tau_sr=tau_sr, tau_he=tau_he, trans=trans, amp1=amp1, amp2=amp2,
                t0=self.time_s, use_nlte=self.use_nlte, use_he=self.use_he
            )
            if np.any(np.isnan(model)) or np.any(np.isinf(model)):
                return -np.inf
            total_chi2 = np.sum(((self.y_fit - model) / self.err_fit) ** 2)
            return -0.5 * total_chi2
        except Exception:
            return -np.inf

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll if np.isfinite(ll) else -np.inf

    def chi2_for_minimizer(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return 1e12
        ll = self.log_likelihood(theta)
        return -2.0 * ll if np.isfinite(ll) else 1e12


def load_data(url, local_filename="temp_spectrum.dat"):
    if os.path.exists(local_filename):
        try:
            df = pd.read_csv(local_filename, sep=r'\s+', comment='#', header=None, on_bad_lines='skip')
            wave_raw = pd.to_numeric(df[0], errors='coerce').values
            flux_raw = pd.to_numeric(df[1], errors='coerce').values
            err_raw = pd.to_numeric(df[3], errors='coerce').values
            valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
            wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
            if wave.max() < 3000:
                wave = wave * 10.0
            exc_reg = (~((wave > 13100) & (wave < 14400))) & (~((wave > 17550) & (wave < 19200))) & (
                ~((wave > 5330) & (wave < 5740))) & (~((wave > 9840) & (wave < 10300))) & (wave >= 3800) & (wave <= 21500)
            return wave[exc_reg], flux[exc_reg], err[exc_reg]
        except Exception:
            pass

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        raw_data = response.read().decode('utf-8')

    try:
        with open(local_filename, 'w', encoding='utf-8') as f:
            f.write(raw_data)
    except Exception:
        pass

    raw_data = re.sub(r'(?<=\d)[Dd](?=[+-]?\d)', 'E', raw_data)
    df = pd.read_csv(io.StringIO(raw_data), sep=r'\s+', comment='#', header=None, on_bad_lines='skip')
    wave_raw, flux_raw, err_raw = pd.to_numeric(df[0], errors='coerce').values, pd.to_numeric(df[1], errors='coerce').values, pd.to_numeric(df[3], errors='coerce').values
    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0
    exc_reg = (~((wave > 13100) & (wave < 14400))) & (~((wave > 17550) & (wave < 19200))) & (
        ~((wave > 5330) & (wave < 5740))) & (~((wave > 9840) & (wave < 10300))) & (wave >= 3800) & (wave <= 21500)
    return wave[exc_reg], flux[exc_reg], err[exc_reg]


# ---------------------------------------------------------------------
# [3] 메인 실행 함수 (4개 케이스 순회 연산 및 3종 플롯 완전 저장)
# ---------------------------------------------------------------------

def main():
    target_base_dir = r"C:\Users\juneh\PyCharmMiscProject\20260909"
    if target_base_dir.startswith("C:") and os.path.exists("/mnt/c"):
        target_base_dir = target_base_dir.replace("\\", "/").replace("C:", "/mnt/c")
    target_base_dir = os.path.normpath(target_base_dir)

    fit_cases = [
        {"case_id": "Case1_LTE_noHe", "use_nlte": False, "use_he": False},
        {"case_id": "Case2_LTE_withHe", "use_nlte": False, "use_he": True},
        {"case_id": "Case3_NLTE_noHe", "use_nlte": True, "use_he": False},
        {"case_id": "Case4_NLTE_withHe", "use_nlte": True, "use_he": True},
    ]

    phases_template = [
        {
            "label": "Phase +1.43d (OB1)", "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "bounds_withHe": [(4200.0, 5800.0), (0.50, 1.80), (0.28, 0.38), (0.220, 0.270), (1.0, 12.0), (0.0, 0.300), (0.0, 1.5), (0.0, 0.60), (0.20, 0.70)],
            "bounds_noHe": [(4200.0, 5800.0), (0.50, 1.80), (0.28, 0.38), (0.220, 0.270), (1.0, 12.0), (0.0, 1.5), (0.0, 0.60), (0.20, 0.70)]
        },
        {
            "label": "Phase +2.42d (OB2)", "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "bounds_withHe": [(3000.0, 3600.0), (1.50, 3.20), (0.25, 0.35), (0.180, 0.250), (1.0, 15.0), (0.0, 2.00), (0.0, 2.0), (0.0, 0.80), (0.0, 0.80)],
            "bounds_noHe": [(3000.0, 3600.0), (1.50, 3.20), (0.25, 0.35), (0.180, 0.250), (1.0, 15.0), (0.0, 2.0), (0.0, 0.80), (0.0, 0.80)]
        },
        {
            "label": "Phase +3.41d (OB3)", "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "bounds_withHe": [(2629.0, 3029.0), (2.10, 3.10), (0.24, 0.32), (0.180, 0.220), (1.0, 15.0), (0.0, 3.00), (0.0, 2.5), (0.0, 1.00), (0.0, 1.00)],
            "bounds_noHe": [(2629.0, 3029.0), (2.10, 3.10), (0.24, 0.32), (0.180, 0.220), (1.0, 15.0), (0.0, 2.5), (0.0, 1.00), (0.0, 1.00)]
        },
        {
            "label": "Phase +4.40d (OB4)", "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "bounds_withHe": [(2407.0, 2807.0), (2.50, 4.00), (0.20, 0.28), (0.100, 0.185), (1.0, 20.0), (0.0, 4.00), (0.0, 3.0), (0.0, 1.20), (0.0, 1.20)],
            "bounds_noHe": [(2407.0, 2807.0), (2.50, 4.00), (0.20, 0.28), (0.100, 0.185), (1.0, 20.0), (0.0, 3.0), (0.0, 1.20), (0.0, 1.20)]
        }
    ]

    ncpu = max(1, mp.cpu_count() - 2)

    for case_cfg in fit_cases:
        case_id = case_cfg["case_id"]
        use_nlte = case_cfg["use_nlte"]
        use_he = case_cfg["use_he"]

        target_save_dir = os.path.join(target_base_dir, case_id)
        os.makedirs(target_save_dir, exist_ok=True)

        if use_he:
            labels = ["T_prime", "N_29", "vmax", "vphot", "tau_sr", "tau_he", "trans", "amp1", "amp2"]
            corner_labels = [r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$", r"$\tau_{\text{Sr II}}$",
                             r"$\tau_{\text{He I}}$", r"$\text{trans}$", r"$\text{amp}_1$", r"$\text{amp}_2$"]
        else:
            labels = ["T_prime", "N_29", "vmax", "vphot", "tau_sr", "trans", "amp1", "amp2"]
            corner_labels = [r"$T^\prime$", r"$N_{29}$", r"$v_{\max}$", r"$v_{\text{phot}}$", r"$\tau_{\text{Sr II}}$",
                             r"$\text{trans}$", r"$\text{amp}_1$", r"$\text{amp}_2$"]

        results_summary = []
        spectra_data = []

        print(f"\n========================================================")
        print(f" [피팅 진행 케이스: {case_id} (NLTE={use_nlte}, He={use_he})]")
        print(f"========================================================\n")

        for p_info in phases_template:
            label, days, url = p_info["label"], p_info["days"], p_info["url"]
            local_file = os.path.join(target_save_dir, f"OB_{days}d.dat")
            bounds = p_info["bounds_withHe"] if use_he else p_info["bounds_noHe"]
            time_s = days * 24.0 * 3600.0

            bounds_arr = np.array(bounds)
            low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]
            midpoint_guess = 0.5 * (low_b + high_b)

            print(f"--> [{label}] 데이터 로드 및 초기 피팅 시작...")
            wave, flux, err = load_data(url, local_file)
            eff_err = np.maximum(err, 0.05 * np.abs(flux))
            x_fit, y_fit, err_fit = wave[::6], flux[::6], eff_err[::6]

            prob_wrapper = MCMCProbabilityWrapper(x_fit, y_fit, err_fit, time_s, bounds, use_nlte=use_nlte, use_he=use_he)

            opt_res = minimize(prob_wrapper.chi2_for_minimizer, midpoint_guess, method='Nelder-Mead',
                               options={'maxiter': 2500, 'xatol': 1e-4, 'fatol': 1e-2})
            center_point = opt_res.x if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10) else midpoint_guess

            ndim, nwalkers = len(bounds), 44
            spans = high_b - low_b
            pos = []
            for _ in range(nwalkers):
                while True:
                    cand = center_point + spans * 0.005 * np.random.randn(ndim)
                    cand = np.clip(cand, low_b + 0.001 * spans, high_b - 0.001 * spans)
                    if prob_wrapper.log_prior(cand) > -1e10:
                        pos.append(cand)
                        break
            pos = np.array(pos)

            print(f"    MCMC 샘플링 진행 중 ({nwalkers} Walkers x 10,000 Steps)...")
            with mp.Pool(processes=ncpu) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_wrapper, pool=pool)
                sampler.run_mcmc(pos, 10000, progress=False)

            flat_samples = sampler.get_chain(discard=1500, thin=2, flat=True)

            popt = {}
            for i in range(ndim):
                mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                popt[labels[i]] = mcmc[1]
            if not use_he:
                popt["tau_he"] = 0.0

            if corner is not None:
                safe_ranges = []
                for col_idx in range(flat_samples.shape[1]):
                    col_data = flat_samples[:, col_idx]
                    ptp_val = np.ptp(col_data)
                    med_val = np.median(col_data)
                    if ptp_val < 1e-5:
                        span_pad = max(abs(med_val) * 0.05, 0.01)
                        safe_ranges.append((med_val - span_pad, med_val + span_pad))
                    else:
                        safe_ranges.append(0.999)

                fig_corner = corner.corner(
                    flat_samples,
                    labels=corner_labels,
                    range=safe_ranges,
                    quantiles=[0.16, 0.50, 0.84],
                    show_titles=True,
                    title_fmt='.3f',
                    smooth=1.0,
                    levels=(0.68, 0.95),
                    fill_contours=True,
                    plot_datapoints=True
                )
                label_fn = label.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "")
                fig_corner.savefig(os.path.join(target_save_dir, f"{label_fn}_corner.png"), dpi=200)
                plt.close(fig_corner)

            model_fit = planck_with_mod_full_relativistic(
                x_fit, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
                amp1=popt["amp1"], amp2=popt["amp2"], t0=time_s, use_nlte=use_nlte, use_he=use_he
            )
            chi2_fit = np.sum(((y_fit - model_fit) / err_fit) ** 2)
            red_chi2_fit = chi2_fit / (len(x_fit) - ndim)

            dl_samples = lum_dist_arr(flat_samples[:, 1], flat_samples[:, 3], flat_samples[:, 5 if not use_he else 6], n_days=days)
            dl_med = np.median(dl_samples)

            print(f"    결과 요약 [{label}]: Red.Chi2={red_chi2_fit:.2f} | D_L={dl_med:.2f} Mpc | T'={popt['T_prime']:.0f}K\n")

            results_summary.append({
                "days": days, "label": label, "popt": popt, "red_chi2": red_chi2_fit, "dl_med": dl_med
            })
            spectra_data.append({
                "days": days, "label": label, "wave": wave, "flux": flux, "x_fit": x_fit, "model_fit": model_fit
            })

        print(f"--> [{case_id}] 시각화 플롯 3종 생성 및 저장 시작...")

        # =====================================================================
        # PLOT ①: 전체 시계열 스펙트럼 피팅 플롯
        # =====================================================================
        fig1, ax1 = plt.subplots(figsize=(10, 11))
        offset_step = 4.0e-16
        telluric_bands = [(5330, 5740), (9800, 10250), (13100, 14400), (17550, 19200)]
        for b_low, b_high in telluric_bands:
            ax1.axvspan(b_low, b_high, color='gray', alpha=0.15, zorder=1)

        ax1.text(13750, 1.50e-15, 'Telluric Band', rotation=90, color='gray', fontsize=8.5, ha='center', va='top')
        ax1.text(18375, 1.50e-15, 'Telluric Band', rotation=90, color='gray', fontsize=8.5, ha='center', va='top')

        for idx, sdata in enumerate(spectra_data):
            offset = (len(spectra_data) - 1 - idx) * offset_step
            res = results_summary[idx]
            popt = res["popt"]

            wave_grid = np.linspace(3800, 21500, 1000)
            model_grid = planck_with_mod_full_relativistic(
                wave_grid, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
                amp1=popt["amp1"], amp2=popt["amp2"], t0=sdata["days"] * 86400.0,
                use_nlte=use_nlte, use_he=use_he
            )

            ax1.plot(sdata["wave"], sdata["flux"] + offset, color='#cccccc', alpha=0.7, lw=0.8, zorder=2)
            ax1.plot(wave_grid, model_grid + offset, color='#c2185b', lw=2.0, zorder=3,
                     label=rf"{sdata['label']}: $D_L={res['dl_med']:.1f}\mathrm{{Mpc}}$, $\tau_{{\mathrm{{Sr}}}}={popt['tau_sr']:.2f}, \tau_{{\mathrm{{He}}}}={popt['tau_he']:.2f}$")
            ax1.text(4000, offset + 0.35e-16, f"+{sdata['days']:.3f}d", fontsize=11, fontweight='bold', color='black')

        ax1.set_xlim(3500, 22000)
        ax1.set_ylim(-0.02e-15, 1.68e-15)
        ax1.set_xlabel('Rest Wavelength [Å]', fontsize=12)
        ax1.set_ylabel(r'Flux [$erg / s / cm^2 / \AA$] + Offset', fontsize=12)
        ax1.set_title(f'AT2017gfo Spectrum Fit ({case_id})', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8.2, framealpha=0.9, facecolor='white')
        ax1.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(target_save_dir, "Plot1_Stacked_Spectra_Fit.png"), dpi=250)
        plt.close(fig1)

        # =====================================================================
        # PLOT ②: 라인 프로파일 진화 및 구성 요소 분해 플롯
        # =====================================================================
        fig2, ax2 = plt.subplots(figsize=(11, 7.8))
        wave_zoom = np.linspace(7000, 12500, 500)
        ax2.axvspan(9650, 10300, color='gray', alpha=0.18, zorder=1)

        ax2.axhline(1.0, color='black', ls=':', lw=1.1, zorder=2, label='Normalized Continuum (1.0)')
        ax2.axvline(10327.311, color='slateblue', ls='-.', lw=1.2, zorder=2, label=r'Rest $\mathrm{Sr\ II}\ (1.0327\mu\mathrm{m})$')
        ax2.axvline(10833.3, color='forestgreen', ls='-.', lw=1.2, zorder=2, label=r'Rest $\mathrm{He\ I}\ (1.0833\mu\mathrm{m})$')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        global_max_flux = 1.0
        global_min_flux = 1.0

        for idx, sdata in enumerate(spectra_data):
            res = results_summary[idx]
            popt = res["popt"]
            t_ph = sdata["days"] * 86400.0
            c = colors[idx % len(colors)]

            model_full = planck_with_mod_full_relativistic(
                wave_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau_sr=popt["tau_sr"], tau_he=popt["tau_he"], trans=popt["trans"],
                amp1=popt["amp1"], amp2=popt["amp2"], t0=t_ph, use_nlte=use_nlte, use_he=use_he
            )
            model_sr = planck_with_mod_full_relativistic(
                wave_zoom, popt["T_prime"], popt["N_29"], popt["vmax"], popt["vphot"],
                tau_sr=popt["tau_sr"], tau_he=0.0, trans=popt["trans"],
                amp1=popt["amp1"], amp2=popt["amp2"], t0=t_ph, use_nlte=use_nlte, use_he=False
            )

            cont_zoom = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(wave_zoom, popt["T_prime"], popt["vphot"])
            prof_full = model_full / cont_zoom
            prof_sr = model_sr / cont_zoom

            global_max_flux = max(global_max_flux, np.max(prof_full))
            global_min_flux = min(global_min_flux, np.min(prof_full))

            ax2.plot(wave_zoom, prof_full, color=c, ls='-', lw=2.2, zorder=4,
                     label=rf"+{sdata['days']:.2f}d (Full Fit: $\tau_{{\mathrm{{Sr}}}}={popt['tau_sr']:.2f}, \tau_{{\mathrm{{He}}}}={popt['tau_he']:.2f}$)")
            ax2.plot(wave_zoom, prof_sr, color=c, ls='--', lw=1.3, alpha=0.75, zorder=3,
                     label=rf"+{sdata['days']:.2f}d (Pure Sr II)")

        y_upper_limit = max(2.10, global_max_flux * 1.10)
        y_lower_limit = max(0.40, global_min_flux - 0.10)
        ax2.set_xlim(7000, 12500)
        ax2.set_ylim(y_lower_limit, y_upper_limit)
        ax2.text(9975, y_upper_limit - 0.08, 'Masked', ha='center', va='center', color='gray', fontsize=10, fontweight='bold')

        ax2.set_xlabel('Rest Wavelength [Å]', fontsize=12)
        ax2.set_ylabel(r'Normalized Flux ($F_\lambda / F_{\mathrm{cont}}$)', fontsize=12)
        ax2.set_title(f'AT2017gfo Normalized Line Profile Evolution ({case_id})', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=7.2, ncol=2, frameon=True, facecolor='white', framealpha=0.95, borderpad=0.3, labelspacing=0.25, handlelength=1.5)
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(target_save_dir, "Plot2_Sr_He_Feature_Decomposition.png"), dpi=250)
        plt.close(fig2)

        # =====================================================================
        # PLOT ③: 광학적 두께(tau) 시계열 진화 추이 플롯
        # =====================================================================
        days_arr = [r["days"] for r in results_summary]
        tau_sr_arr = [r["popt"]["tau_sr"] for r in results_summary]
        tau_he_arr = [r["popt"]["tau_he"] for r in results_summary]

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(days_arr, tau_sr_arr, 'o-', color='royalblue', ms=8, lw=2.2, label=r'$\mathrm{Sr\ II}$ Optical Depth ($\tau_{\mathrm{Sr}}$)')
        if use_he:
            ax3.plot(days_arr, tau_he_arr, 's--', color='darkgreen', ms=8, lw=2.2, label=r'$\mathrm{He\ I}$ Optical Depth ($\tau_{\mathrm{He}}$)')

        for d, ts, th in zip(days_arr, tau_sr_arr, tau_he_arr):
            ax3.annotate(f"{ts:.2f}", (d, ts), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', color='royalblue')
            if use_he:
                ax3.annotate(f"{th:.2f}", (d, th), textcoords="offset points", xytext=(0, -15), ha='center', fontweight='bold', color='darkgreen')

        ax3.set_xlabel('Phase [Days post-merger]', fontsize=12)
        ax3.set_ylabel(r'Optical Depth ($\tau$)', fontsize=12)
        ax3.set_title(f'Temporal Evolution of Optical Depth ({case_id})', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(target_save_dir, "Plot3_Optical_Depth_Evolution.png"), dpi=250)
        plt.close(fig3)

    print("\n========================================================")
    print(" [총 4개 피팅 케이스 연산 및 각 케이스별 3종 플롯 완전 저장 완료]")
    print("========================================================\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()