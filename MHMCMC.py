import numpy as np
import pandas as pd

# Defining  Functions

# Eta function
def eta(a,omega_m):
    s = np.cbrt((1 - omega_m)/omega_m)
    eta_calculated_fit = 2 *(np.sqrt(s**3 + 1) ) * ( (1/a**4) - ( (0.1540*s)/a**3) + ((0.4304*s**2)/a**2) + ((0.19097*s**3)/a) + (0.066941*s**4)  )**(-1/8)
    return eta_calculated_fit

#Luminosity Distance
def Luminosity_distance(z,omega_m_lum):
    D_L = 3000 * (1 + z) * (eta(1,omega_m_lum) - eta(1 / (1 + z),omega_m_lum) )
    return D_L 

# Distance Modulus 
def mu(z,omega_m_mu,h):
    
    D = Luminosity_distance(z,omega_m_mu)
    Distance_modulus = 25 - (5 * np.log10(h)) + (5 * np.log10(D))
    return Distance_modulus


def likelihood(Omga, h, redshift, Distance, InvCovarinace_Matrix):
    
    if not ( 0.0 <= Omga <= 1 and 0.0 <= h <= 1):
        return -np.inf
    
    else:
        DistanceDifference = Distance - mu(redshift, Omga, h)
        likelihood = -0.5 * np.dot(DistanceDifference, np.dot(InvCovarinace_Matrix, DistanceDifference))
        return likelihood


#MH-MCMC 
def Metro_Monte(Omega_MC_previous,h_MC_previous,Var_Om_proposal,Var_h_proposal, CovarianceMatrix): 
    
    Omega_next = np.abs(Omega_MC_previous + Var_Om_proposal * np.random.randn() )
    h_next = np.abs(h_MC_previous + Var_h_proposal * np.random.randn() )

    log_posterior_old = likelihood(Omega_MC_previous, h_MC_previous, redshift, Distance, CovarianceMatrix) # Calculating likelihood of the point.
    log_posterior_new = likelihood(Omega_next, h_next, redshift, Distance, CovarianceMatrix)

    #Taking Ratio to get The alpha ratio.
    Acceptance_ratio = log_posterior_new - log_posterior_old
    if np.log(np.random.normal(0,1)) < Acceptance_ratio: # accepting values even if it's below 1.
        return Omega_next,h_next,'Yes'
    else:
        return Omega_MC_previous,h_MC_previous,'No'


def MHMC_run(Variance_Om, Variance_h, Guess_Om, Guess_h, CovarianceMatrix, Iteration):
    Initial_Omega_guess = [Guess_Om]
    Initial_h_guess = [Guess_h]
    likelihood_values = []
    N = Iteration   # Number of iterations
    
    for i in range(1, N):                       # Iterations begin here
        omega_p = Initial_Omega_guess[-1]
        h_p = Initial_h_guess[-1]

        omega_next, h_next, condition = Metro_Monte(omega_p, h_p, Variance_Om, Variance_h, CovarianceMatrix)
        if condition == 'Yes':
            Initial_Omega_guess.append(omega_next)  # Appending values that are accepted along with it's likelihood
            Initial_h_guess.append(h_next)
            likelihood_values.append(likelihood(omega_next, h_next, redshift, Distance))
        else:
            continue 
        
    acceptance_percent = (len(likelihood_values) / N) * 100

    return Initial_h_guess, Initial_Omega_guess, acceptance_percent

# Data
Data_z_and_u = np.genfromtxt('jla_mub_0.txt')
Covariance = np.genfromtxt('jla_mub_covmatrix.txt')


# Reshaped the Covariance Matrix and took inverse of it as it is needed in Likelihood
Reshaped_covariance_matrix = np.reshape(Covariance,(31,31))
Inverse_Covariance_matrix = np.linalg.inv(Reshaped_covariance_matrix)

# Run MHMCMC
h_chain, Omega_m_chain, acceptance_rate = MHMC_run(0.01,0.01,0.1,0.1,Inverse_Covariance_matrix,1000) 

# Save MCMC results to CSV
df = pd.DataFrame({'h': h_chain, 'Omega_m': Omega_m_chain})
df.to_csv('MCMC_results.csv', index=False)
