#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
import emcee
import warnings

from config.settings import fit_cases, phases_template
from src.data_loader import load_data
from src.mcmc_wrapper import MCMCProbabilityWrapper
from src.models import planck_with_mod_full_relativistic, lum_dist_arr
from utils.plotting import generate_all_plots

warnings.filterwarnings("ignore")

def main():
    target_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_results")
    os.makedirs(target_base_dir, exist_ok=True)

    ncpu = max(1, mp.cpu_count() - 2)
    mcmc_steps = 10000

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
        flat_samples_dict = {}
        labels_dict = {"labels": labels, "corner_labels": corner_labels}

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

            prob_wrapper = MCMCProbabilityWrapper(x_fit, y_fit, err_fit, time_s, bounds, use_nlte=use_nlte, use_he=use_he, days=days)

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

            print(f"    MCMC 샘플링 진행 중 ({nwalkers} Walkers x {mcmc_steps} Steps)...")
            with mp.Pool(processes=ncpu) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, prob_wrapper, pool=pool)
                sampler.run_mcmc(pos, mcmc_steps, progress=False)

            flat_samples = sampler.get_chain(discard=min(1500, int(mcmc_steps*0.15)), thin=max(1, int(mcmc_steps*0.0002)), flat=True)
            flat_samples_dict[days] = flat_samples

            popt = {}
            for i in range(ndim):
                mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
                popt[labels[i]] = mcmc[1]
            if not use_he:
                popt["tau_he"] = 0.0

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

        print(f"--> [{case_id}] 시각화 플롯 15종 생성 및 저장 시작...")
        generate_all_plots(spectra_data, results_summary, case_id, use_nlte, use_he, flat_samples_dict, labels_dict, target_save_dir)

    print("\n========================================================")
    print(" [총 4개 피팅 케이스 연산 및 각 케이스별 15종 플롯 완전 저장 완료]")
    print("========================================================\n")

if __name__ == "__main__":
    mp.freeze_support()
    main()
