# Name - Ishan Jha
# Roll - IMT2022562
# Date - 25/11/2025
# Question - 1(b)

import numpy as np
import random
import matplotlib.pyplot as plt

def display(mu, variance, rho, samples, trials):
    print("Following are the quantities for the current simulation : ")
    print("1) True Value                : ", mu)
    print("2) Variance                  : ", variance)
    print("3) Rho                       : ", rho)
    print("4) No of samples             : ", samples)
    print("5) No of trials/iterations   : ", trials)
    print("==============================================================")

def generate_toeplitz_cov(size, rho, variance):
    idx = np.arange(size)
    return variance * (rho ** np.abs(idx[:, None] - idx[None, :]))

def gls_estimator(y_vec, cov_inv):
    ones = np.ones(len(y_vec))
    denominator = ones @ cov_inv @ ones
    return (ones @ cov_inv @ y_vec) / denominator

def run_correlated_experiment(true_mu, variance, rho, N, trials):
    Sigma = generate_toeplitz_cov(N, rho, variance)
    Sigma_inv = np.linalg.inv(Sigma)

    ones = np.ones(N)
    denom = ones @ Sigma_inv @ ones
    crlb_val = 1.0 / denom

    sample_mean_vals = []
    gls_vals = []

    L = np.linalg.cholesky(Sigma)

    for _ in range(trials):
        z = np.random.randn(N)
        noise = L @ z
        obs = true_mu + noise

        sample_mean_vals.append(np.mean(obs))
        gls_vals.append(gls_estimator(obs, Sigma_inv))

    sm = np.array(sample_mean_vals)
    gls = np.array(gls_vals)

    return sm, gls, crlb_val

def compare_histograms(sm_arr, gls_arr, true_mu):
    plt.figure()
    plt.hist(sm_arr, bins=50, alpha=0.5, density=True, label="Sample Mean", color="brown")
    plt.hist(gls_arr, bins=50, alpha=0.5, density=True, label="GLS", color="skyblue")
    plt.axvline(true_mu, color='k', linestyle='--', linewidth=2, label="True Mean")
    plt.title("Estimators for Correlated Noise")
    plt.xlabel("Estimated Value")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def correlated_noise_simulation():

    true_mu = random.uniform(1, 5)
    variance = random.uniform(0.5, 5)
    rho = random.uniform(0.1, 1)
    N = 100
    trials = 10000

    display(true_mu, variance, rho, N, trials)

    sm_arr, gls_arr, crlb_val = run_correlated_experiment(true_mu, variance, rho, N, trials)

    mse_sm = np.mean((sm_arr - true_mu) ** 2)
    mse_gls = np.mean((gls_arr - true_mu) ** 2)

    print(f"MSE (Sample Mean) : {mse_sm:.6f}")
    print(f"MSE (GLS)         : {mse_gls:.6f}")
    print(f"CRLB              : {crlb_val:.6f}")

    compare_histograms(sm_arr, gls_arr, true_mu)


correlated_noise_simulation()
