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