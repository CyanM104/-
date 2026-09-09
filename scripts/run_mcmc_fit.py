#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import multiprocessing as mp
import os
import re
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import minimize

from src.probability import MCMCProbabilityWrapper
from src.radiation_engine import (
    calc_relativistic_blackbody_continuum,
    planck_with_mod_full_relativistic_nlte,
)

try:
    import corner
except ImportError:
    corner = None

try:
    import emcee
except ImportError:
    raise ImportError("MCMC 실행을 위해 'emcee' 라이브러리가 필요합니다.")

warnings.filterwarnings("ignore")


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


def load_data(url, local_filename="temp_spectrum.dat"):
    if os.path.exists(local_filename):
        try:
            df = pd.read_csv(
                local_filename,
                sep=r"\s+",
                comment="#",
                header=None,
                on_bad_lines="skip",
            )
            wave_raw = pd.to_numeric(df[0], errors="coerce").values
            flux_raw = pd.to_numeric(df[1], errors="coerce").values
            err_raw = pd.to_numeric(df[3], errors="coerce").values
            valid = (
                ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
            )
            wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
            if wave.max() < 3000:
                wave = wave * 10.0
            exc_reg = (
                (~((wave > 13100) & (wave < 14400)))
                & (~((wave > 17550) & (wave < 19200)))
                & (~((wave > 5330) & (wave < 5740)))
                & (~((wave > 9800) & (wave < 10400)))
                & (wave >= 3800)
                & (wave <= 21500)
            )
            return wave[exc_reg], flux[exc_reg], err[exc_reg]
        except Exception:
            pass

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        raw_data = response.read().decode("utf-8")

    try:
        with open(local_filename, "w", encoding="utf-8") as f:
            f.write(raw_data)
    except Exception:
        pass

    raw_data = re.sub(r"(?<=\d)[Dd](?=[+-]?\d)", "E", raw_data)
    df = pd.read_csv(
        io.StringIO(raw_data),
        sep=r"\s+",
        comment="#",
        header=None,
        on_bad_lines="skip",
    )
    wave_raw, flux_raw, err_raw = (
        pd.to_numeric(df[0], errors="coerce").values,
        pd.to_numeric(df[1], errors="coerce").values,
        pd.to_numeric(df[3], errors="coerce").values,
    )
    valid = ~np.isnan(wave_raw) & ~np.isnan(flux_raw) & ~np.isnan(err_raw)
    wave, flux, err = wave_raw[valid], flux_raw[valid], err_raw[valid]
    if wave.max() < 3000:
        wave = wave * 10.0
    exc_reg = (
        (~((wave > 13100) & (wave < 14400)))
        & (~((wave > 17550) & (wave < 19200)))
        & (~((wave > 5330) & (wave < 5740)))
        & (~((wave > 9800) & (wave < 10400)))
        & (wave >= 3800)
        & (wave <= 21500)
    )
    return wave[exc_reg], flux[exc_reg], err[exc_reg]


def main():
    target_save_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(target_save_dir, exist_ok=True)

    phases = [
        {
            "label": "Phase +1.43d (OB1)",
            "days": 1.427,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57983.969_Phase%2B1.43d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB1_1.43d.dat"),
            "bounds": [
                (4200.0, 5200.0),
                (0.80, 1.80),
                (0.38, 0.50),
                (0.27, 0.35),
                (0.8, 2.5),
                (0.0, 0.05),
                (0.15, 1.2),
            ],
            "init_guess": [4800.0, 1.20, 0.42, 0.29, 1.80, 0.05, 0.35],
        },
        {
            "label": "Phase +2.42d (OB2)",
            "days": 2.417,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57984.969_Phase%2B2.42d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB2_2.42d.dat"),
            "bounds": [
                (2900.0, 3600.0),
                (1.50, 3.20),
                (0.32, 0.42),
                (0.21, 0.29),
                (0.5, 2.5),
                (0.0, 1.20),
                (0.15, 1.2),
            ],
            "init_guess": [3200.0, 2.10, 0.36, 0.24, 1.50, 0.30, 0.45],
        },
        {
            "label": "Phase +3.41d (OB3)",
            "days": 3.413,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57985.974_Phase%2B3.41d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB3_3.41d.dat"),
            "bounds": [
                (2500.0, 3200.0),
                (1.80, 3.50),
                (0.27, 0.37),
                (0.17, 0.25),
                (0.2, 2.5),
                (0.2, 2.20),
                (0.15, 1.2),
            ],
            "init_guess": [2800.0, 2.60, 0.31, 0.20, 1.20, 0.80, 0.55],
        },
        {
            "label": "Phase +4.40d (OB4)",
            "days": 4.403,
            "url": "https://sid.erda.dk/share_redirect/df1fMhon6Z/dereddened%2Bderedshifted_spectra/AT2017gfo_ENGRAVE_v1.0_XSHOOTER_MJD-57986.974_Phase%2B4.40d_deredz.dat",
            "local_file": os.path.join(target_save_dir, "OB4_4.40d.dat"),
            "bounds": [
                (2200.0, 2900.0),
                (2.00, 4.00),
                (0.23, 0.33),
                (0.14, 0.21),
                (0.1, 2.5),
                (0.3, 2.50),
                (0.15, 1.2),
            ],
            "init_guess": [2500.0, 3.10, 0.27, 0.16, 1.00, 1.20, 0.60],
        },
    ]

    labels = ["T_prime", "N_29", "vmax", "vphot", "tau_sr", "tau_he", "trans"]
    corner_labels = [
        r"$T^\prime$",
        r"$N_{29}$",
        r"$v_{\max}$",
        r"$v_{\text{phot}}$",
        r"$\tau_{\text{Sr II}}$",
        r"$\tau_{\text{He I}}$",
        r"$\text{trans}$",
    ]
    ncpu = max(1, mp.cpu_count() - 2)

    results_summary = []
    spectra_data = []

    print("\n========================================================")
    print(" [Arya+2026 7D Pure NLTE 복합 MCMC 피팅 시작]")
    print("========================================================\n")

    for p_info in phases:
        label, days, url, local_file, bounds = (
            p_info["label"],
            p_info["days"],
            p_info["url"],
            p_info["local_file"],
            p_info["bounds"],
        )
        init_guess = p_info["init_guess"]
        time_s = days * 24.0 * 3600.0

        bounds_arr = np.array(bounds)
        low_b, high_b = bounds_arr[:, 0], bounds_arr[:, 1]

        print(f"--> [{label}] 데이터 로드 및 초기 피팅 시작...")
        wave, flux, err = load_data(url, local_file)
        eff_err = np.maximum(err, 0.05 * np.abs(flux))
        x_fit, y_fit, err_fit = wave[::5], flux[::5], eff_err[::5]

        prob_wrapper = MCMCProbabilityWrapper(
            x_fit, y_fit, err_fit, time_s, bounds, days
        )

        opt_res = minimize(
            prob_wrapper.chi2_for_minimizer,
            init_guess,
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-4, "fatol": 1e-2},
        )
        center_point = (
            opt_res.x
            if (opt_res.success and prob_wrapper.log_prior(opt_res.x) > -1e10)
            else init_guess
        )

        ndim, nwalkers = len(bounds), 32
        nsteps = 9000
        spans = high_b - low_b
        pos = []
        for _ in range(nwalkers):
            while True:
                cand = center_point + spans * 0.01 * np.random.randn(ndim)
                cand = np.clip(
                    cand, low_b + 0.01 * spans, high_b - 0.01 * spans
                )
                if prob_wrapper.log_prior(cand) > -1e10:
                    pos.append(cand)
                    break
        pos = np.array(pos)

        print(
            f"    MCMC 샘플링 진행 중 ({nwalkers} Walkers x {nsteps} Steps)..."
        )
        with mp.Pool(processes=ncpu) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, prob_wrapper, pool=pool
            )
            sampler.run_mcmc(pos, nsteps, progress=True)

        flat_samples = sampler.get_chain(discard=2000, thin=3, flat=True)

        popt = {}
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            popt[labels[i]] = mcmc[1]

        if corner is not None:
            fig_corner = corner.corner(
                flat_samples,
                labels=corner_labels,
                quantiles=[0.16, 0.50, 0.84],
                show_titles=True,
                title_fmt=".3f",
                smooth=1.0,
                levels=(0.68, 0.95),
                fill_contours=True,
                plot_datapoints=False,
            )
            safe_label = re.sub(r"[^a-zA-Z0-9_.]", "_", label)
            safe_label = re.sub(r"_+", "_", safe_label).strip("_")
            fig_corner.savefig(
                os.path.join(target_save_dir, f"{safe_label}_7D_NLTE_corner.png"),
                dpi=200,
            )
            plt.close(fig_corner)

        dl_samples = lum_dist_arr(
            flat_samples[:, 1],
            flat_samples[:, 3],
            flat_samples[:, 6],
            n_days=days,
        )
        dl_med = np.median(dl_samples)

        results_summary.append(
            {"days": days, "label": label, "popt": popt, "dl_med": dl_med}
        )
        spectra_data.append(
            {"days": days, "label": label, "wave": wave, "flux": flux}
        )

    # 시각화 플롯 렌더링
    print("--> 시각화 플롯 생성 및 저장 중...")

    masked_regions = [
        (5330, 5740, "Telluric/Noise"),
        (9800, 10400, "Telluric/Noise"),
        (13100, 14400, "Telluric Band"),
        (17550, 19200, "Telluric Band"),
    ]

    fig1, ax1 = plt.subplots(figsize=(11, 12))
    offset_step = 4.0e-16
    max_flux_val = 0.0

    for idx, (m_start, m_end, m_label) in enumerate(masked_regions):
        ax1.axvspan(m_start, m_end, color="gray", alpha=0.18, zorder=1)
        if idx == 2 or idx == 3:
            ax1.text(
                (m_start + m_end) / 2.0,
                1.55e-15,
                m_label,
                rotation=90,
                ha="center",
                va="top",
                fontsize=8.5,
                color="dimgray",
                alpha=0.8,
            )

    for idx, sdata in enumerate(spectra_data):
        res = results_summary[idx]
        popt = res["popt"]
        offset = (len(spectra_data) - 1 - idx) * offset_step

        wave_grid = np.linspace(3800, 21500, 1200)
        model_grid = planck_with_mod_full_relativistic_nlte(
            wave_grid,
            popt["T_prime"],
            popt["N_29"],
            popt["vmax"],
            popt["vphot"],
            tau_sr=popt["tau_sr"],
            tau_he=popt["tau_he"],
            trans=popt["trans"],
            t0=sdata["days"] * 86400.0,
        )

        current_max = np.max(model_grid + offset)
        if current_max > max_flux_val:
            max_flux_val = current_max

        ax1.plot(
            sdata["wave"],
            sdata["flux"] + offset,
            color="gray",
            alpha=0.35,
            lw=0.8,
            zorder=2,
        )
        ax1.plot(
            wave_grid,
            model_grid + offset,
            color="crimson",
            lw=2.0,
            zorder=3,
            label=rf"{sdata['label']}: $D_L={res['dl_med']:.1f}\mathrm{{Mpc}}$, $\tau_{{\mathrm{{Sr}}}}={popt['tau_sr']:.2f}, \tau_{{\mathrm{{He}}}}={popt['tau_he']:.2f}$",
        )

        ax1.text(
            4000,
            offset + 0.4e-16,
            f"+{sdata['days']}d",
            fontsize=13,
            fontweight="bold",
            color="black",
        )

    ax1.set_xlim(3500, 22000)
    ax1.set_ylim(-0.5e-16, max_flux_val * 1.15)
    ax1.set_xlabel("Rest Wavelength [Å]", fontsize=13)
    ax1.set_ylabel(r"Flux [$erg / s / cm^2 / \AA$] + Offset", fontsize=13)
    ax1.set_title(
        "AT2017gfo Pure NLTE Fit (Arya+2026 Compliant with Masked Regions)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9.5)
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plot1_path = os.path.join(
        target_save_dir, "Plot1_Stacked_NLTE_Spectra_Fit_Masked.png"
    )
    plt.savefig(plot1_path, dpi=250)
    plt.close(fig1)

    # PLOT 2
    fig2, ax2 = plt.subplots(figsize=(11, 7))
    wave_zoom = np.linspace(7000, 12500, 800)
    color_map = {
        1.427: "#1f77b4",
        2.417: "#ff7f0e",
        3.413: "#2ca02c",
        4.403: "#d62728",
    }

    for m_start, m_end, m_label in masked_regions:
        if m_end >= 7000 and m_start <= 12500:
            ax2.axvspan(m_start, m_end, color="gray", alpha=0.2, zorder=1)
            ax2.text(
                (m_start + m_end) / 2.0,
                1.48,
                "Masked",
                rotation=0,
                ha="center",
                va="top",
                fontsize=9,
                color="dimgray",
                fontweight="bold",
                alpha=0.7,
            )

    for idx, sdata in enumerate(spectra_data):
        res = results_summary[idx]
        popt = res["popt"]
        t_ph = sdata["days"] * 86400.0
        c_color = color_map.get(sdata["days"], "black")

        model_full = planck_with_mod_full_relativistic_nlte(
            wave_zoom,
            popt["T_prime"],
            popt["N_29"],
            popt["vmax"],
            popt["vphot"],
            tau_sr=popt["tau_sr"],
            tau_he=popt["tau_he"],
            trans=popt["trans"],
            t0=t_ph,
        )

        model_sr = planck_with_mod_full_relativistic_nlte(
            wave_zoom,
            popt["T_prime"],
            popt["N_29"],
            popt["vmax"],
            popt["vphot"],
            tau_sr=popt["tau_sr"],
            tau_he=0.0,
            trans=popt["trans"],
            t0=t_ph,
        )

        cont_zoom = (popt["N_29"] * 1e-29) * calc_relativistic_blackbody_continuum(
            wave_zoom, popt["T_prime"], popt["vphot"]
        )

        norm_full = model_full / cont_zoom
        norm_sr = model_sr / cont_zoom

        ax2.plot(
            wave_zoom,
            norm_full,
            color=c_color,
            ls="-",
            lw=2.2,
            zorder=3,
            label=rf"+{sdata['days']:.2f}d (Full Fit: $\tau_{{\mathrm{{Sr}}}}={popt['tau_sr']:.2f}, \tau_{{\mathrm{{He}}}}={popt['tau_he']:.2f}$)",
        )
        ax2.plot(
            wave_zoom,
            norm_sr,
            color=c_color,
            ls="--",
            lw=1.2,
            alpha=0.65,
            zorder=2,
            label=rf"+{sdata['days']:.2f}d (Pure $\mathrm{{Sr\ II}}$)",
        )

    ax2.axhline(
        1.0,
        color="black",
        ls=":",
        lw=1.5,
        alpha=0.7,
        label="Normalized Continuum (1.0)",
    )
    ax2.axvline(
        10327.311,
        color="navy",
        ls="-.",
        alpha=0.5,
        label=r"Rest $\mathrm{Sr\ II}\ (1.0327\mu\mathrm{m})$",
    )
    ax2.axvline(
        10833.3,
        color="darkgreen",
        ls="-.",
        alpha=0.5,
        label=r"Rest $\mathrm{He\ I}\ (1.0833\mu\mathrm{m})$",
    )

    ax2.set_xlim(7000, 12500)
    ax2.set_ylim(0.30, 1.55)
    ax2.set_xlabel(r"Rest Wavelength [$\AA$]", fontsize=13)
    ax2.set_ylabel(r"Normalized Flux ($F_{\lambda} / F_{\text{cont}}$)", fontsize=13)
    ax2.set_title(
        r"AT2017gfo Normalized Line Profile Evolution ($7000\AA - 12500\AA$ Overlaid)",
        fontsize=14,
        fontweight="bold",
    )

    ax2.legend(
        loc="lower left",
        fontsize=8.0,
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
    )
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()

    plot2_path = os.path.join(
        target_save_dir, "Plot2_Normalized_Line_Profile_7000AA_Masked.png"
    )
    plt.savefig(plot2_path, dpi=250)
    plt.close(fig2)

    # PLOT 3
    days_arr = [r["days"] for r in results_summary]
    tau_sr_arr = [r["popt"]["tau_sr"] for r in results_summary]
    tau_he_arr = [r["popt"]["tau_he"] for r in results_summary]

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(
        days_arr,
        tau_sr_arr,
        "o-",
        color="royalblue",
        ms=8,
        lw=2.2,
        label=r"$\mathrm{Sr\ II}$ Optical Depth ($\tau_{\mathrm{Sr}}$)",
    )
    ax3.plot(
        days_arr,
        tau_he_arr,
        "s--",
        color="darkgreen",
        ms=8,
        lw=2.2,
        label=r"$\mathrm{He\ I}$ Optical Depth ($\tau_{\mathrm{He}}$)",
    )

    for d, ts, th in zip(days_arr, tau_sr_arr, tau_he_arr):
        ax3.annotate(
            f"{ts:.2f}",
            (d, ts),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontweight="bold",
            color="royalblue",
        )
        ax3.annotate(
            f"{th:.2f}",
            (d, th),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontweight="bold",
            color="darkgreen",
        )

    ax3.set_xlabel("Phase [Days post-merger]", fontsize=13)
    ax3.set_ylabel(r"Optical Depth ($\tau$)", fontsize=13)
    ax3.set_title(
        r"Temporal Evolution of $\tau_{\mathrm{Sr\ II}}$ and $\tau_{\mathrm{He\ I}}$ (Pure NLTE)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.legend(loc="upper left", fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plot3_path = os.path.join(
        target_save_dir, "Plot3_Optical_Depth_Evolution_NLTE.png"
    )
    plt.savefig(plot3_path, dpi=250)
    plt.close(fig3)

    print(f"\n========================================================")
    print(f" [피팅 및 시각화 저장 완료]")
    print(f" 저장 경로: {target_save_dir}")
    print(f"========================================================\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
