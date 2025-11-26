# Name - Ishan Jha
# Roll - IMT2022562
# Date - 25/11/2025
# Question - 2


import numpy as np
import random
import matplotlib.pyplot as plt

    
sample_count = 100      # No of measurements
param_count = 2         # No of unknown parameters
trial_count = 20000     # No of iterations

def graphical_plots(estimates, thetas, crlb_dig):
    plt.figure(figsize=(10, 6))

    # Plot histogram
    _, bins, _ = plt.hist(estimates[:, 0],
                          bins=50, density=True,
                          alpha=0.6, color='green',
                          label='Simulated Estimatations')

    # Overlay theoretical Gaussian curve predicted by CRLB
    mean_val = thetas[0]
    std_dev = np.sqrt(crlb_dig[0])

    pdf_vals = (1 / (std_dev * np.sqrt(2 * np.pi))) * \
               np.exp(- (bins - mean_val)**2 / (2 * std_dev**2))

    plt.plot(bins, pdf_vals, linewidth=3,
             color='red', label='CRLB Predicted PDF')

    plt.axvline(mean_val, color='black',
                linestyle='--', label='True θ₀')

    plt.title(f'Demonstration of Estimator Efficiency for θ₀\n'
              f'(No of Samples={sample_count} & No of Trials={trial_count})')
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

def statistics(estimations, crlb_dig):
    cov = np.cov(estimations, rowvar=False)
    var = np.diag(cov)

    print ("Estimated Variance : ", var)
    print("=========================================================================")
    print("Variance Ratio (Empirical/CRLB) : ", (var/crlb_dig))
    print("=========================================================================")

def crlb_calculations(H, noise):
    noise_cov = noise @ noise.T
    noise_cov_inv = np.linalg.inv(noise_cov)
    crlb = np.linalg.inv(H.T @ noise_cov_inv @ H)                   # Theoretical CRLB = (Hᵀ C⁻¹ H)⁻¹
    crlb_diagonal = np.diag(crlb)

    print("CRLB (variance limits for the parameters) : ", crlb_diagonal)
    print("=========================================================================")
    return crlb, crlb_diagonal, noise_cov, noise_cov_inv

def estimations_func(H, signal, thetas, noise_cov, noise_cov_inv, crlb):
    estimations = []
    estimation_matrix = crlb @ H.T @ noise_cov_inv

    for i in range(trial_count):
        noise = np.random.multivariate_normal(np.zeros(sample_count), noise_cov)
        observation = H @ thetas + signal + noise
        estimations.append(estimation_matrix @ (observation - signal))
    
    return np.array(estimations)

def q2_simulation():

    reals = []
    for i in range(param_count):
        reals.append(random.uniform(-5, 5))

    theta_real = np.array(reals)                         # Ground Truth for 3 parameters

    np.random.seed(42)
    design_matrix = np.random.randn(sample_count, param_count)      # Observation Matrix - H
    signal_vec = np.random.randn(sample_count)                      # Deterministic signal component
    random_block = np.random.randn(sample_count, sample_count)      # Generation of noise covariance matrix

    crlb, crlb_diagonal, noise_cov, noise_cov_inv = crlb_calculations(design_matrix, random_block)
    parameter_estimates = estimations_func(design_matrix, signal_vec, theta_real, noise_cov, noise_cov_inv, crlb)
    statistics(parameter_estimates, crlb_diagonal)
    graphical_plots(parameter_estimates, theta_real, crlb_diagonal)

q2_simulation()
