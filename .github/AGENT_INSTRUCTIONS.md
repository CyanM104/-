# Physics & Coding Rules for AI Agent (Jules)

1. **Sr II Triplet Weighting**:
   - Always normalize Sr II optical depths relative to $10327.31\,\AA$ ($0.12 : 1.00 : 0.58$).
   - Never divide by raw sum (e.g., $13.8$) directly in MCMC parameters.

2. **Phase-Dependent Priors**:
   - Phase < 2.0d: Force $\tau_{\mathrm{Sr}} \ge 0.8$ and $\tau_{\mathrm{He}} \le 0.35$.
   - Maintain hydrodynamic constraint: $v_{\max} - v_{\text{phot}} \ge 0.08c$.

3. **Radiative Transfer Coupling**:
   - Use Beer-Lambert exponential combination (`combine_optical_depths_beer_lambert`).
   - Do NOT use simple multiplicative profiles (`corr_sr * pcyg_he`).
