## Table of Contents
- [1. Today's Topic](#1-todays-topic)
  - [2. Getting Started](#2-getting-started)
    - [2.1 Prerequisites](#21-prerequisites)
  - [3. Conclusion](#3-conclusion)
  - [4. Final](#4-final)
  - [📊 Dataset](#-dataset)
  - [🟦 Step 1: Histogram Estimation](#-step-1-histogram-estimation)
  - [🟩 Step 2: KDE Formula](#-step-2-kde-formula)
  - [🟩 Step 3: Calculation at (x = 60)](#-step-3-calculation-at-x--60)
  - [🧩 Comparison](#-comparison)
  - [🔑 Benefits of KDE](#-benefits-of-kde)

# 1. Today's Topic
1) Sampling
2) Estimation
    - Parametric Estimation
    - Non Parametric Estimation
        - 1. Empirical Distribution Function (EDF)
        - 2. Histogram Estimator
        - 3. Kernel Density Estimation (KDE)
3) Maximum Likelihood Estimation
4) Entropy
      - Cross-Entroy Loss
      - Loss Mimimaztion
5) Markov Model
   - First Order Markov Model
   - Second Markov Model
6) Probabilty Distribution
   - Discrete Probability Distribution
     - probability mass function (PMF).
     - Bernoulli Distribution → One trial, success/failure.
     - Binomial Distribution → Number of successes in 𝑛 independent trials.
     - Poisson Distribution (count data and rate Parameter ) → Number of events in a fixed interval at constant rate.
     - Geometric Distribution → Number of trials until first success.
     - Hypergeometric Distribution → Successes in draws without replacement.
     - Discrete Uniform Distribution → All outcomes equally likely (fair die).
   - Continuous Probability Distribution
     - Probability density function (PDF).
     - Normal (Gaussian) Distribution (1D and MultiVariant)→ Bell curve, symmetric around mean.
     - Exponential Distribution → Time until next event (memoryless property).
     - Uniform Distribution → Equal chance for all values in an interval.
     - t-Distribution → Like normal but heavier tails, used for small samples.
     - Chi-Square Distribution → Sum of squared normal variables, used in tests of independence
     - F-Distribution → Ratio of variances, used in ANOVA.
     - Gamma/Beta Distributions → Flexible families used in Bayesian statistics.
7) Fundamentals of Probability & Random Variables

   * **Probability Distributions & Density Functions (PDF / PMF):**
   * **Continuous Distributions:** Gaussian (Normal) Distribution (1D and Multivariate). Understand parameters like mean ($\mu$) and variance ($\sigma^2$), standard deviation, and density curves.


   * **Discrete Distributions:** Bernoulli Distribution (success/failure trials) and Poisson Distribution (count data, rate parameter $\lambda$).




   * **Independence & Data Modeling:**
   * **i.i.d. Assumption:** What *Independent and Identically Distributed* variables mean mathematically, why it allows multiplying individual probabilities into a joint likelihood function, and concept of **Distribution Shift** (Train vs. Test data).




   * **Joint, Marginal, and Conditional Distributions:**
   * How to compute marginal distributions $P(X)$ and conditional distributions $P(Y \mid X)$ from a joint distribution $P(X, Y)$.


   * Understanding how conditioning reduces variance.
8) Parameter Estimation Techniques

   * **Maximum Likelihood Estimation (MLE):**
     * Formulating the Likelihood function $L(\theta) = f(x_1, \dots, x_n \mid \theta)$.


     * Converting likelihood to Log-Likelihood to simplify derivation.
     * Analytical MLE derivations for Bernoulli ($p=\frac{\sum x_i}{n}$), Poisson ($\lambda = \bar{x}$), and Gaussian ($\mu = \bar{x}$).




   * **Maximum A-Posteriori (MAP) Estimation:**
     * Formulating the MAP objective: $\arg\max_\theta P(\theta \mid X)$.


     * Analytical derivation of MAP for Gaussian mean with a Gaussian prior (Precision-weighted average).




   * **Loss Minimization & Regularization Connection:**
   * **Negative Log-Likelihood (NLL):** Showing how maximizing MLE equals minimizing loss (e.g., Cross-Entropy Loss for Bernoulli, Mean Squared Error for Gaussian).


   * **Regularization:** Showing how MAP estimation with a Gaussian prior on weights leads directly to $L_2$ Regularization / Ridge Regression ($\lambda = \frac{\sigma^2}{\tau^2}$).

9) Non-Parametric Density Estimation

   * **Parametric vs. Non-Parametric:** Why parametric models can fail if the family assumption is wrong, and how non-parametric parameters grow with sample size $n$.


   * **Histograms:** Bin width $h$, origin dependency, discontinuity, and the curse of dimensionality.


   * **Kernel Density Estimation (KDE / Parzen Windows):**
   * Kernel functions $K(u)$ (e.g., Gaussian Kernel) and bandwidth selection ($h$).

10. Multivariable Linear Algebra & Calculus

    * **Vector Calculus & Optimization:**
      * Gradient operations, setting derivatives/partial derivatives to zero ($\frac{dJ}{d\theta} = 0$) to solve stationary points.



    * **Multivariate Normal Distribution:**
      * Mean vectors $\boldsymbol{\mu}$ and Covariance matrices $\boldsymbol{\Sigma}$.


      * Matrix Inversion, Transpose operations, and Determinants ($\vert{}\boldsymbol{\Sigma}\vert{}$).


      * Covariance vs. Correlation ($\rho$), Variance-Covariance diagonal vs. off-diagonal entries.


    * **Mahalanobis Distance** and $\chi^2$ (Chi-squared) distribution for anomaly detection.

11. Discriminative vs. Generative Modeling (ML Applications)

    * **Generative/Joint Modeling $P(X, Y)$:** Modeling features and targets jointly, conditioning to obtain linear regression, and reasons why it breaks down (one-hot vectors, singular matrices, non-Gaussian features).


    * **Discriminative/Conditional Modeling $P(Y \mid X)$:**
      * Linear Regression formulation: $y = \mathbf{w}^T \mathbf{x} + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.


      * Ordinary Least Squares (OLS) closed-form solution: $\hat{\mathbf{w}}_{\text{ML}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.


      * Ridge Regression closed-form solution: $\hat{\mathbf{w}}_{\text{MAP}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$.


    * **Information Theory Basics:** Kullback-Leibler (KL) Divergence and its equivalence to MLE/Log-Likelihood maximization.
  
## 2. Getting Started
Content goes here...

### 2.1 Prerequisites
Content goes here...

## 3. Conclusion
Content goes here...

## 4. Final
Content

Got it, Ravi — let’s rewrite the **full KDE worked example** with proper LaTeX block formatting (`$$ ... $$`) so it looks clean in your Markdown notes.

---

## 📊 Dataset
Scores of 10 students:  
\(\{50, 52, 53, 55, 60, 61, 62, 65, 70, 72\}\)

---

## 🟦 Step 1: Histogram Estimation
With bin width \(h = 5\):  

$$
p(x) = \frac{K_j}{n h}
$$

- [50–55): \(K_1 = 4 \Rightarrow p(x) = \frac{4}{10 \cdot 5} = 0.08\)  
- [60–65): \(K_2 = 4 \Rightarrow p(x) = 0.08\)  
- [70–75): \(K_3 = 2 \Rightarrow p(x) = 0.04\)  

👉 This gives a **blocky estimate**: flat bars showing density in each bin.

---

## 🟩 Step 2: KDE Formula
$$
\hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h}\right)
$$

- \(n = 10\)  
- Bandwidth \(h = 2\)  
- Kernel (Gaussian):  

$$
K(u) = \frac{1}{\sqrt{2\pi}} e^{-\frac{u^2}{2}}
$$

---

## 🟩 Step 3: Calculation at \(x = 60\)
$$
\hat{f}(60) = \frac{1}{10 \cdot 2} \sum_{i=1}^{10} K\left(\frac{60 - X_i}{2}\right)
$$

- For \(X_i = 60\): \(K(0) = 0.3989\)  
- For \(X_i = 61\): \(K(-0.5) \approx 0.3521\)  
- For \(X_i = 62\): \(K(-1) \approx 0.2420\)  
- For \(X_i = 65\): \(K(-2.5) \approx 0.0175\)  
- Far points (like 50, 72) ≈ 0  

Sum ≈ 1.01  

$$
\hat{f}(60) \approx \frac{1.01}{20} = 0.0505
$$

---

## 🧩 Comparison
- **Histogram at 60:** density = 0.08 (flat bar).  
- **KDE at 60:** density ≈ 0.05 (smooth curve, influenced by nearby points).  

👉 KDE avoids abrupt jumps at bin edges — it smoothly transitions.

---

## 🔑 Benefits of KDE
1. **Smooth curve** → avoids blocky jumps of histogram.  
2. **Local influence** → nearby points contribute more, far points less.  
3. **Flexible** → adapts to data without assuming Normal distribution.  
4. **Main use:** estimating the **probability density function (PDF)** directly from data in a continuous way.  

---

✅ **Takeaway:**  
- Histogram = piecewise constant approximation.  
- KDE = smooth, continuous approximation.  
- KDE is widely used in **machine learning, statistics, and data visualization** to understand the underlying distribution of data.

---

Would you like me to also give you a **Python code snippet (with `seaborn` or `matplotlib`)** that plots both histogram and KDE for this dataset? That way you’ll have a ready‑to‑run example for your notes.