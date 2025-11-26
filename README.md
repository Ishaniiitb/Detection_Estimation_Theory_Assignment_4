# Detection & Estimation Theory — Assignment 4  
**Author:** Ishan Jha  
**Roll Number:** IMT2022562  
**Institution:** IIIT Bangalore  
**Date:** 25 November 2025  

This repository contains the complete solution set for **Assignment 4** of *Detection and Estimation Theory*.  
All problems involve a combination of:

- Mathematical derivations  
- Simulation-driven verification  
- Performance analysis of estimators  
- Use of the Cramér–Rao Lower Bound (CRLB)  

The full written report is available as:

📄 **Report.pdf**

---

# 📁 Repository Structure

Detection_Estimation_Theory_Assignment_4/
│
├── Q1a.py # Problem 1(a) – DC-level estimation with i.i.d. Gaussian noise
├── Q1b.py # Problem 1(b) – GLS estimation under correlated noise
├── Q1c.py # Problem 1(c) – Sample mean vs. single-sample estimator
├── Q2.py # Problem 2 – MLE & CRLB verification for multi-parameter case
│
├── Report.pdf # Detailed theoretical and simulation write-up
└── README.md

# 📘 Problem 1: Estimation of a DC Level

## 🔹 Problem 1(a): DC Estimation with i.i.d. Gaussian Noise

### **Model**
$$
y[n] = A + w[n], \qquad w[n] \sim \mathcal{N}(0,\sigma^2)
$$

Estimator:
\[
\hat{A} = \frac{1}{N}\sum_{n=0}^{N-1} y[n]
\]

CRLB:
\[
\mathrm{Var}(\hat{A}) \ge \frac{\sigma^2}{N}
\]

### **Functionality of Q1a.py**
- Generates noisy measurements of a constant \(A\)  
- Computes the empirical MSE of \(\hat{A}\)  
- Computes the theoretical CRLB (\(\sigma^2 / N\))  
- Evaluates estimator performance over multiple sample sizes  
- Produces a log–log plot comparing MSE and CRLB  

### **Result**
The sample mean reaches the CRLB, confirming it is:
- **Efficient**
- **MVUE**
- The statistically optimal estimator for this i.i.d. scenario.

---

## 🔹 Problem 1(b): Estimation with Correlated Gaussian Noise

### **Model**
\[
y[n] = \mu + w[n], \qquad 
\mathbf{w} \sim \mathcal{N}(0,\Sigma)
\]
\[
\Sigma[n,m] = \sigma^2 \rho^{|n-m|}
\]

### **Estimators Compared**
1. **Sample Mean**
   \[
   \hat{\mu}_{SM} = \frac{1}{N}\sum y[n]
   \]

2. **GLS (Generalized Least Squares) Estimator**
   \[
   \hat{\mu}_{GLS}
    = \frac{\mathbf{1}^T \Sigma^{-1} \mathbf{y}}
           {\mathbf{1}^T \Sigma^{-1} \mathbf{1}}
   \]

### **CRLB**
\[
\mathrm{Var}(\hat{\mu}) \ge 
\left( \mathbf{1}^T \Sigma^{-1} \mathbf{1} \right)^{-1}
\]

### **Functionality of Q1b.py**
- Builds a Toeplitz covariance structure  
- Generates correlated Gaussian noise  
- Computes both \(\hat{\mu}_{SM}\) and \(\hat{\mu}_{GLS}\)  
- Computes empirical variances from Monte-Carlo trials  
- Plots histograms of their distributions  
- Compares both MSEs to the CRLB  

### **Result**
- Sample Mean → **Not efficient**, does **not** achieve CRLB  
- GLS Estimator → **Efficient**, **achieves CRLB**  

The GLS estimator is therefore the MVUE under correlated noise.

---

## 🔹 Problem 1(c): Comparison Between Two Unbiased Estimators

This highlights that **“unbiased” does not imply “optimal.”**

### **Two Estimators**
1. **Sample Mean**
   \[
   \hat{A}_{mean} = \frac{1}{N}\sum y[n]
   \]

2. **First Sample**
   \[
   \hat{A}_{single} = y[1]
   \]

### **Variances**
\[
\mathrm{Var}(\hat{A}_{mean}) = \frac{\sigma^2}{N}
\]
\[
\mathrm{Var}(\hat{A}_{single}) = \sigma^2
\]

### **Functionality of Q1c.py**
- Runs many Monte-Carlo trials  
- Computes the empirical variances of both estimators  
- Plots distribution histograms  

### **Conclusion**
- Both estimators are unbiased  
- The single-sample estimator has **much higher variance**  
- Sample mean is **significantly better** and **MVUE** in the i.i.d. case  

# 📘 Problem 2: Multi-Parameter Estimation Using the MLE

## 🔹 Model Description

In this problem, the observation model is a **linear Gaussian system**:

\[
\mathbf{y} = H\boldsymbol{\theta} + \mathbf{s} + \mathbf{w},
\]

where:

- \( H \) is a known \( N \times p \) observation matrix  
- \( \boldsymbol{\theta} \) is an unknown parameter vector of size \( p \times 1 \)  
- \( \mathbf{s} \) is a known deterministic signal  
- \( \mathbf{w} \sim \mathcal{N}(0, C) \) is Gaussian noise with known covariance \( C \)

This represents a **multiple-parameter estimation** problem in the presence of correlated noise.

---

## 🔹 Maximum Likelihood Estimator (MLE)

For the above linear Gaussian model, the log-likelihood is maximized by:

\[
\hat{\boldsymbol{\theta}}_{\mathrm{MLE}}
    = (H^T C^{-1} H)^{-1}
      H^T C^{-1} (\mathbf{y} - \mathbf{s}).
\]

This is also the **Generalized Least Squares (GLS)** estimator and is unbiased.

---

## 🔹 Cramér–Rao Lower Bound (CRLB)

For any unbiased estimator of \( \boldsymbol{\theta} \), the covariance must satisfy:

\[
\mathrm{Cov}(\hat{\boldsymbol{\theta}}) 
    \ge (H^T C^{-1} H)^{-1}.
\]

The matrix on the right is the **CRLB matrix**, which sets the minimum achievable variance for each component of the vector parameter.

---

## 🔹 What Q2.py Performs

The script carries out a detailed simulation to verify the CRLB for a vector parameter:

### **1. Synthetic Data Generation**
- Randomly generates:
  - True unknown parameter vector \( \boldsymbol{\theta}_{\mathrm{true}} \)
  - Observation matrix \( H \)
  - Covariance matrix \( C \) with controllable correlation structure  

### **2. Computes Theoretical Values**
- Computes the theoretical CRLB matrix  
- Computes the MLE \( \hat{\boldsymbol{\theta}}_{\mathrm{MLE}} \)

### **3. Monte-Carlo Simulation (≈ 20,000 trials)**
For each trial:
- Generate noise via Cholesky decomposition of \( C \)  
- Create a noisy measurement vector \( \mathbf{y} \)  
- Estimate \( \boldsymbol{\theta} \) using the MLE formula  
- Store the estimate  

### **4. Empirical Performance Analysis**
- Estimate empirical covariance across all trials  
- Compare each variance term to the CRLB diagonal  
- Plot histogram of estimator distribution vs. Gaussian predicted by CRLB  
- Confirm efficiency visually and numerically

---

## 🔹 Key Observations and Results

- The empirical variances of the parameter estimates **align extremely well** with the CRLB predictions.  
- This verifies that the **MLE/GLS estimator is efficient** for the Gaussian linear model.  
- In multi-parameter estimation, each component:
  - is unbiased  
  - achieves its individual CRLB value  
- Thus, the estimator is **MVUE** for vector-valued parameters as well.

---

## 🔹 Final Conclusion for Problem 2

This section demonstrates that:

- For Gaussian models, the **MLE is optimal** (minimum variance)  
- The theoretical CRLB matrix is **achievable**  
- Monte-Carlo simulations fully confirm the theoretical derivations  
- The GLS framework naturally generalizes scalar estimation to vector estimation  

Problem 2 therefore extends the ideas of unbiasedness and efficiency to **multi-dimensional parameter spaces**, showing that the same CRLB principles apply.

# 📊 Summary Table

| Problem | Estimator | Efficient? | CRLB Achieved? | Notes |
|--------|-----------|------------|----------------|-------|
| 1(a) | Sample Mean | ✔ Yes | ✔ Yes | Optimal for i.i.d. Gaussian noise |
| 1(b) | Sample Mean | ✘ No | ✘ No | Not MVUE under correlated noise |
| 1(b) | GLS Estimator | ✔ Yes | ✔ Yes | Achieves CRLB with correlated covariance |
| 1(c) | Single-Sample Estimator | ✘ No | ✘ No | Unbiased but very high variance |
| 1(c) | Sample Mean | ✔ Yes | ✔ Yes | Best unbiased estimator in i.i.d. case |
| 2 | MLE / GLS (Vector Case) | ✔ Yes | ✔ Yes | Efficient multi-parameter estimator |

---

# Summary Notes

- ✔ **Efficient** = estimator attains CRLB  
- ✘ **Not efficient** = estimator is unbiased but has variance above CRLB  
- **GLS** dominates in the presence of correlated noise  
- **Sample Mean** is optimal only for i.i.d. Gaussian noise  
- **MLE/GLS** in Problem 2 achieves CRLB for each component of the parameter vector

# 📌 Academic Purpose

This assignment highlights several core ideas in statistical estimation theory:

- Application of classical estimation principles  
- Derivation and validation of unbiased estimators  
- Practical computation and interpretation of the Cramér–Rao Lower Bound (CRLB)  
- Understanding how noise covariance affects estimator performance  
- Efficiency analysis using large-scale Monte-Carlo simulations  
- Estimation of both scalar and multi-dimensional parameter vectors  

These elements collectively demonstrate how theoretical bounds and practical estimators interact in real detection and estimation problems.

---

# 📬 Contact

**Author:** Ishan Jha  
**Roll Number:** IMT2022562  
**Institution:** IIIT Bangalore  
