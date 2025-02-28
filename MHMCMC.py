import numpy as np
import pandas as pd

print("MHCMCMC run begins")


# Defining Functions

# Eta function
def eta(a, omega_m):
    s = np.cbrt((1 - omega_m) / omega_m)
    eta_calculated_fit = 2 * (np.sqrt(s**3 + 1)) * ((1 / a**4) - ((0.1540 * s) / a**3) + ((0.4304 * s**2) / a**2) + ((0.19097 * s**3) / a) + (0.066941 * s**4)) ** (-1 / 8)
    return eta_calculated_fit

# Luminosity Distance
def Luminosity_distance(z, omega_m_lum):
    D_L = 3000 * (1 + z) * (eta(1, omega_m_lum) - eta(1 / (1 + z), omega_m_lum))
    return D_L

# Distance Modulus
def mu(z, omega_m_mu, h):
    D = Luminosity_distance(z, omega_m_mu)
    Distance_modulus = 25 - (5 * np.log10(h)) + (5 * np.log10(D))
    return Distance_modulus

# Likelihood Function
def likelihood(Omga, h, redshift, Distance, InvCovariance_Matrix):
    if not (0.0 <= Omga <= 1 and 0.0 <= h <= 1):
        return -np.inf
    
    DistanceDifference = Distance - mu(redshift, Omga, h)
    return -0.5 * np.dot(DistanceDifference, np.dot(InvCovariance_Matrix, DistanceDifference))

# MH-MCMC Step Function
def Metro_Monte(Omega_MC_previous, h_MC_previous, Var_Om_proposal, Var_h_proposal, CovarianceMatrix, redshift, Distance):
    Omega_next = Omega_MC_previous + Var_Om_proposal * np.random.randn()
    h_next = h_MC_previous + Var_h_proposal * np.random.randn()

    log_posterior_old = likelihood(Omega_MC_previous, h_MC_previous, redshift, Distance, CovarianceMatrix)
    log_posterior_new = likelihood(Omega_next, h_next, redshift, Distance, CovarianceMatrix)

    # Taking Ratio to get The alpha ratio.
    Acceptance_ratio = log_posterior_new - log_posterior_old
    if np.log(np.random.uniform(0, 1)) < Acceptance_ratio:  # accepting values even if it's below 1.
        return Omega_next, h_next, 'Yes'
    else:
        return Omega_MC_previous, h_MC_previous, 'No'

# MCMC Run Function
def MHMC_run(Variance_Om, Variance_h, Guess_Om, Guess_h, CovarianceMatrix, Iteration, redshift, Distance):
    Initial_Omega_guess = [Guess_Om]
    Initial_h_guess = [Guess_h]
    likelihood_values = []
    
    for _ in range(Iteration):  # Iterations begin here
        omega_p = Initial_Omega_guess[-1]
        h_p = Initial_h_guess[-1]

        omega_next, h_next, condition = Metro_Monte(omega_p, h_p, Variance_Om, Variance_h, CovarianceMatrix, redshift, Distance)
        if condition == 'Yes':
            Initial_Omega_guess.append(omega_next)  # Appending values that are accepted along with its likelihood
            Initial_h_guess.append(h_next)
            likelihood_values.append(likelihood(omega_next, h_next, redshift, Distance, CovarianceMatrix))
        
    acceptance_percent = (len(likelihood_values) / Iteration) * 100
    
    
    return Initial_h_guess, Initial_Omega_guess, acceptance_percent

# Data Loading
Data_z_and_u = np.genfromtxt('jla_mub_0.txt')
Covariance = np.genfromtxt('jla_mub_covmatrix.txt')

# Extract redshift and Distance Modulus
redshift = Data_z_and_u[:, 0]
Distance = Data_z_and_u[:, 1]

# Reshape Covariance Matrix and Compute Inverse
Reshaped_covariance_matrix = np.reshape(Covariance, (31, 31))
Inverse_Covariance_matrix = np.linalg.inv(Reshaped_covariance_matrix)

# Run MCMC
h_chain, Omega_m_chain, acceptance_rate = MHMC_run(0.01, 0.01, 0.1, 0.1, Inverse_Covariance_matrix, 5000, redshift, Distance)

print(
    f"""Chain finished
accetance rate = {acceptance_rate}
    """)

# Save MCMC results to CSV
df = pd.DataFrame({'h': h_chain, 'Omega_m': Omega_m_chain})
df.to_csv('MCMC_results.csv', index=False)
