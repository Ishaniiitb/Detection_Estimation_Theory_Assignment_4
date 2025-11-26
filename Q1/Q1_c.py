# Name - Ishan Jha
# Roll - IMT2022562
# Date - 25/11/2025
# Question - 1(c)

import matplotlib.pyplot as plt
import numpy as np
import random

def display(true, sigma, samples, trials):
    print("Following are the quantities for the current simulation : ")
    print("1) True Value                : ", true)
    print("2) Variance                  : ", sigma**2)
    print("3) No of samples             : ", samples)
    print("4) No of trials/iterations   : ", trials)
    print("==============================================================")

def generate_noisy_dc(A, noise_std, N):
    noise_vec = np.random.normal(0, noise_std, N)
    return A + noise_vec

def estimator_mean(data):
    return np.mean(data)

def estimator_single(data):
    return data[0]

def run_suboptimal_trials(true_val, sigma, N, runs):

    mean_list = []
    single_list = []

    for _ in range(runs):
        samples = generate_noisy_dc(true_val, sigma, N)
        mean_list.append(estimator_mean(samples))
        single_list.append(estimator_single(samples))

    return np.array(mean_list), np.array(single_list)

def plot_distribution_comparison(mean_arr, single_arr, true_val):
    plt.figure(figsize=(10, 6))
    plt.hist(single_arr, bins=50, alpha=0.5, label="Single Sample")
    plt.hist(mean_arr, bins=50, alpha=0.5, label="Sample Mean")
    plt.axvline(true_val, color='k', linestyle='--', linewidth=2, label="True DC")
    plt.title("Non-MVUE Estimator Comparison: Single Measurement vs Mean")
    plt.legend()
    plt.show()

def nonMVUE_simulation():

    true_dc = random.uniform(2.5, 7.5)
    sigma = random.uniform(0.5, 2.25)
    N = 100
    runs = 20000

    display(true_dc, sigma, N, runs)

    mean_arr, single_arr = run_suboptimal_trials(true_dc, sigma, N, runs)

    mse_mean = np.var(mean_arr)
    mse_single = np.var(single_arr)

    print("Non-MVUE Estimator Simulation : ")
    print(f"MSE (Single Sample) : {mse_single:.4f}")
    print(f"MSE (Sample Mean)   : {mse_mean:.4f}")
    print(f"CRLB (Single)       : {sigma**2:.4f}")
    print(f"CRLB (Mean)         : {sigma**2 / N:.4f}")

    plot_distribution_comparison(mean_arr, single_arr, true_dc)


nonMVUE_simulation()
