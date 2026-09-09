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

        # 1. 유체역학적 제약: 0.03 <= vmax - vphot <= 0.25c, and N_29 > 0.0
        v_diff = vmax - vphot
        if not (0.03 <= v_diff <= 0.25) or N_29 <= 0.0:
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
