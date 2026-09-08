#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from src.radiation_engine import planck_with_mod_full_relativistic_nlte


class MCMCProbabilityWrapper(object):
    def __init__(self, x_fit, y_fit, err_fit, time_s, bounds, days):
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.err_fit = err_fit
        self.time_s = time_s
        self.bounds = bounds
        self.days = days

    def log_prior(self, theta):
        T_prime, N_29, vmax, vphot, tau_sr, tau_he, trans = theta
        for val, (low, high) in zip(theta, self.bounds):
            if not (low <= val <= high):
                return -np.inf

        # 1. 유체역학적 제약: vmax - vphot >= 0.08c
        if (vmax - vphot) < 0.08 or N_29 <= 0.0:
            return -np.inf

        # 2. Phase별 물리적 광학 두께 조건 (Arya+2026 / Chiba+2026)
        if self.days < 2.0:
            # Phase +1.43d: Sr II 지배, He I 과도 피팅 차단
            if tau_sr < 0.8 or tau_he > 0.35:
                return -np.inf
        elif self.days < 3.0:
            # Phase +2.42d: Sr II 지배 지속
            if tau_sr < tau_he:
                return -np.inf

        # 3. 총 광학 두께 상한 제약
        if (tau_sr + tau_he) > 3.5:
            return -np.inf

        return 0.0

    def log_likelihood(self, theta):
        T_prime, N_29, vmax, vphot, tau_sr, tau_he, trans = theta
        try:
            model = planck_with_mod_full_relativistic_nlte(
                wav=self.x_fit,
                T_prime=T_prime,
                N_29=N_29,
                vmax=vmax,
                vphot=vphot,
                tau_sr=tau_sr,
                tau_he=tau_he,
                trans=trans,
                t0=self.time_s,
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
