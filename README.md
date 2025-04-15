# 🪐 Estimating Cosmological Parameters using Type Ia Supernovae and MHMCMC

This project estimates two key cosmological parameters — the matter density parameter \( \Omega_m \) and the dimensionless Hubble constant \( h \) — by fitting observational data from Type Ia supernovae to a flat ΛCDM model using the **Metropolis-Hastings Markov Chain Monte Carlo (MHMCMC)** algorithm.

This work is based on an assignment that explores how supernovae, as standard candles, can be used to measure the expansion of the Universe.

---

## 📚 Theoretical Background

In a **flat ΛCDM cosmology**, the luminosity distance \( D_L(z) \) is related to redshift by:

\[
D_L(z) = \frac{c(1+z)}{H_0} \left[ \eta(1, \Omega_m) - \eta\left(\frac{1}{1+z}, \Omega_m \right) \right]
\]

where \( \eta(a, \Omega_m) \) is a fitting function from Pen (1999), accurate to better than 0.4% for \( 0.2 \leq \Omega_m \leq 1 \).

The distance modulus is:

\[
\mu = 25 - 5 \log_{10} h + 5 \log_{10} D_L^*(z)
\]

where \( D_L^* \) is the luminosity distance assuming \( h = 1 \).  

The MCMC samples from the posterior distribution of parameters by evaluating the likelihood:

\[
\mathcal{L} \propto \exp\left[ -\frac{1}{2} (\mu - \mu_{\text{th}})^T C^{-1} (\mu - \mu_{\text{th}}) \right]
\]

where:
- \( \mu \): Observed distance modulus
- \( \mu_{\text{th}} \): Theoretical prediction for given parameters
- \( C \): Covariance matrix from data

---

## 🧰 Project Structure

project_root/ ├── src/ │ └── MHMCMC.py # Main implementation of Metropolis-Hastings MCMC │ ├── data/ │ ├── jla_mub_0.txt # Observed distance moduli (μ) and redshift values │ └── jla_mub_covmatrix.txt # 31x31 covariance matrix │ ├── Jupyter_notebook/ │ └── MCMC_plots.ipynb # Visualizations of MCMC chains, histograms, contour plots │ ├── requirements.txt # List of Python dependencies └── README.md # You're reading it