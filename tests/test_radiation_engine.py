import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.radiation_engine import (
    planck_with_mod_full_relativistic_nlte,
    LAM_SR_10036_AA,
    LAM_SR_10327_AA,
    LAM_SR_10914_AA,
    LAM_HE_10833_AA
)
from src.probability import MCMCProbabilityWrapper

def test_sr_ii_line_triplet_weights():
    wav = np.linspace(5000, 15000, 100)
    T_prime = 5000.0
    N_29 = 1.0
    vmax = 0.3
    vphot = 0.2
    tau_sr = 2.0
    tau_he = 0.5
    trans = 0.8
    t0 = 123552.0

    with patch('src.radiation_engine.p_cygni_line_corr_rel_1d') as mock_p_cygni:
        # Mock the return value to be a valid array so that combine_optical_depths_beer_lambert doesn't fail
        mock_p_cygni.return_value = np.ones_like(wav)

        planck_with_mod_full_relativistic_nlte(
            wav=wav,
            T_prime=T_prime,
            N_29=N_29,
            vmax=vmax,
            vphot=vphot,
            tau_sr=tau_sr,
            tau_he=tau_he,
            trans=trans,
            t0=t0
        )

        # Extract the arguments with which p_cygni_line_corr_rel_1d was called
        # There should be 4 calls: 3 for Sr, 1 for He
        assert mock_p_cygni.call_count == 4

        calls = mock_p_cygni.call_args_list

        # Call 1: 10036 AA
        args1, _ = calls[0]
        assert np.isclose(args1[3], 0.12 * tau_sr)
        assert np.isclose(args1[4], LAM_SR_10036_AA)

        # Call 2: 10327 AA
        args2, _ = calls[1]
        assert np.isclose(args2[3], 1.00 * tau_sr)
        assert np.isclose(args2[4], LAM_SR_10327_AA)

        # Call 3: 10914 AA
        args3, _ = calls[2]
        assert np.isclose(args3[3], 0.58 * tau_sr)
        assert np.isclose(args3[4], LAM_SR_10914_AA)

        # Call 4: He
        args4, _ = calls[3]
        assert np.isclose(args4[3], tau_he)
        assert np.isclose(args4[4], LAM_HE_10833_AA)

def test_mcmc_probability_wrapper_priors():
    x_fit = np.array([10000.0])
    y_fit = np.array([1.0])
    err_fit = np.array([0.1])
    time_s = 123552.0

    bounds = [
        (3000, 8000),    # T_prime
        (0.1, 10.0),     # N_29
        (0.1, 0.5),      # vmax
        (0.05, 0.4),     # vphot
        (0.0, 5.0),      # tau_sr
        (0.0, 5.0),      # tau_he
        (0.0, 1.0)       # trans
    ]

    # hydrodynamic constraint vmax - vphot >= 0.05
    wrapper = MCMCProbabilityWrapper(x_fit, y_fit, err_fit, time_s, bounds, days=1.5)

    # Valid theta
    valid_theta = [5000.0, 1.0, 0.3, 0.2, 1.0, 0.2, 0.8]  # vmax-vphot = 0.1 (>= 0.05)
    assert wrapper.log_prior(valid_theta) == 0.0

    # Invalid theta: hydrodynamic constraint vmax - vphot < 0.05
    invalid_theta_hydro = [5000.0, 1.0, 0.24, 0.2, 1.0, 0.2, 0.8] # vmax-vphot = 0.04
    assert wrapper.log_prior(invalid_theta_hydro) == -np.inf

    # Check phase independence of tau constraints (i.e. they shouldn't exist anymore)
    valid_theta_high_tau = [5000.0, 1.0, 0.3, 0.2, 4.0, 3.0, 0.8]
    assert wrapper.log_prior(valid_theta_high_tau) == 0.0
