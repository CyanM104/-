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

        # 3. [핵심] 시계열 물리적 제약 (Sr II / He I 축퇴 및 지그재그 방지)
        if self.days < 2.0:
            # Phase +1.43d: Sr II 지배적, He I 활성화 억제
            if tau_sr < 1.0 or tau_he > 0.25:
                return -np.inf
        elif self.days < 3.0:
            # Phase +2.42d: Sr II가 He I보다 여전히 우세해야 함
            if tau_sr < tau_he:
                return -np.inf
        elif self.days < 4.0:
            # Phase +3.41d: Day 2보다 tau_sr이 폭증(1.68)하는 역주행 차단
            if tau_sr > 1.2:
                return -np.inf
        else:
            # Phase +4.40d: 광구 감속 조건 강제 (Day 3보다 빨라지는 비물리적 해 배제)
            if vphot > 0.14:
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
