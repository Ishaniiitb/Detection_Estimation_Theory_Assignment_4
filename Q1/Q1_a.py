# Name - Ishan Jha
# Roll - IMT2022562
# Date - 25/11/2025
# Question - 1(a)

import matplotlib.pyplot as plt
import numpy as np
import random

def display(true, sigma, trials):
    print("Following are the quantities for the current simulation : ")
    print("1) True Value        : ", true)
    print("2) Variance of Noise : ", sigma)
    print("3) Number of Trials  : ", trials)
    print("==============================================================")

def compute_crlb(noise_power, sample_count):
    return noise_power / sample_count                               # Theoretical CRLB = σ²/N.

def generate_dc_measurements(true_bias, noise_dev, total_samples):
    noise_vec = np.random.normal(0, noise_dev, total_samples)
    return true_bias + noise_vec                                    # Generate x[n] = A + w[n]

def estimate_dc_level(data_vector):
    """Use the sample average as estimator."""
    return np.mean(data_vector)

def run_dc_trials(actual_offset, noise_dev, sample_count, trial_count):

    estimates = []                                                  # Run multiple iterations for DC estimation.

    for _ in range(trial_count):
        samples = generate_dc_measurements(actual_offset, noise_dev, sample_count)
        est = estimate_dc_level(samples)
        estimates.append(est)

    estimates = np.array(estimates)
    mse = np.mean((estimates - actual_offset) ** 2)
    crlb_val = compute_crlb(noise_dev**2, sample_count)

    return mse, crlb_val, estimates

def plot_mse_vs_crlb(sample_sizes, mse_vals, crlb_vals):
    plt.figure(figsize=(10, 6))
    plt.loglog(sample_sizes, crlb_vals, 'r-', linewidth=2, label="CRLB (1/N)")
    plt.loglog(sample_sizes, mse_vals, 'bo--', markersize=7, label="Empirical MSE")
    plt.xlabel("Number of Observations (N)")
    plt.ylabel("Mean Squared Error")
    plt.title("DC Estimator Performance: MSE vs CRLB")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

def execute_dc_demo():

    true_value = random.uniform(2.5, 7.5)
    noise_sigma = random.uniform(0.5, 5)
    sample_grid = [10, 50, 100, 500, 1000, 5000]
    trials = 1000

    display(true_value, noise_sigma, trials)

    mse_results = []
    crlb_results = []

    print(f"{'N':<10} | {'MSE Estimate':<20} | {'CRLB':<20}")
    print("==============================================================")

    for N in sample_grid:
        mse, crlb, _ = run_dc_trials(true_value, noise_sigma, N, trials)
        mse_results.append(mse)
        crlb_results.append(crlb)

        print(f"{N:<10} | {mse:<20.6f} | {crlb:<20.6f}")
    
    print("==============================================================")

    plot_mse_vs_crlb(sample_grid, mse_results, crlb_results)


execute_dc_demo()
