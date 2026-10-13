# 📘 Curse of Dimensionality

## 🗂️ Table of Contents
- [📘 Curse of Dimensionality](#-curse-of-dimensionality)
  - [🗂️ Table of Contents](#️-table-of-contents)
  - [ 🔮 The Curse of Dimensionality](#--the-curse-of-dimensionality)
  - [ 🌌 Data Sparsity Problem](#--data-sparsity-problem)
  - [ ⚠️ Consequences](#-️-consequences)
  - [ 🛠️ Mitigation Strategies](#-️-mitigation-strategies)
    - [ ✂️ Feature Selection](#-️-feature-selection)
    - [ 🔄 Feature Extraction](#--feature-extraction)
- [🎯 Principal Component Analysis (PCA)](#-principal-component-analysis-pca)
  - [ 📘 Introduction](#--introduction)
  - [ 🎯 Goal of PCA](#--goal-of-pca)
  - [ ⚡ Core Benefits](#--core-benefits)
  - [ 🔍 Feature Selection vs. Feature Extraction](#--feature-selection-vs-feature-extraction)
  - [ 📊 Role of Variance](#--role-of-variance)
  - [ 🧮 Mathematical Formula](#--mathematical-formula)

---

## <a id="curse"></a> 🔮 The Curse of Dimensionality
- Occurs when a model is fed too many features (dimensions) beyond an optimal threshold.  
- Adding unnecessary features does **not** improve accuracy.  
- Leads to increased model complexity and decreased performance.  

---

## <a id="sparsity"></a> 🌌 Data Sparsity Problem
- As dimensions increase, data points become more spread out in feature space.  
- Makes it difficult for models to calculate meaningful distances between points.  
- Distance-based algorithms (e.g., k-NN, clustering) suffer heavily.  

---

## <a id="consequences"></a> ⚠️ Consequences
- Poor model performance.  
- High computational costs.  
- Training becomes slow and inefficient.  

---

## <a id="mitigation"></a> 🛠️ Mitigation Strategies
Two primary techniques for dimensionality reduction:

### <a id="selection"></a> ✂️ Feature Selection
- Choose a subset of the most relevant existing features.  
- Keep the best columns, discard the rest.  
- Example: `SelectKBest`, `RFE`, `Lasso`.  

### <a id="extraction"></a> 🔄 Feature Extraction
- Create new, smaller sets of features by combining or transforming original variables.  
- Captures essential information in fewer dimensions.  
- Example: **Principal Component Analysis (PCA)**.  

---

📌 **Takeaway:** Both feature selection and feature extraction combat the curse of dimensionality, but they do so in different ways—selection keeps the best features, while extraction builds new ones.


![alt text](image.png)

# 🎯 Principal Component Analysis (PCA)

---

## <a id="intro"></a> 📘 Introduction
- PCA is an **unsupervised machine learning technique** for **dimensionality reduction** and **feature extraction**.
- Helps simplify high-dimensional datasets while preserving essential behavior.
- 📌 *Video reference: 0:44–4:43*

---

## <a id="goal"></a> 🎯 Goal of PCA
- Reduce high-dimensional datasets into lower dimensions.  
- Preserve the "essence" of the data.  
- Analogy: A photographer capturing a **3D soccer match** in a **2D photograph**.  
- 📌 *Video reference: 2:50–4:35*

---

## <a id="benefits"></a> ⚡ Core Benefits
- **Algorithmic Efficiency**: Fewer features → faster models, less computation.  
- **Visualization**: Project high-dimensional data (e.g., 784 features) into 2D/3D for pattern discovery.  
- 📌 *Video reference: 5:16–7:16*

---

## <a id="fs-vs-fe"></a> 🔍 Feature Selection vs. Feature Extraction
- **Feature Selection**  
  - Choose a subset of existing columns based on importance.  
  - Limitation: fails when features are equally important or correlated.  
  - 📌 *Video reference: 8:41–14:19*

- **Feature Extraction (PCA)**  
  - Creates new features (Principal Components).  
  - Rotates coordinate system to maximize **variance**.  
  - 📌 *Video reference: 17:21–23:45*

---

## <a id="variance"></a> 📊 Role of Variance
- **Variance** = statistical measure of data spread.  
- Maximizing variance ensures lower-dimensional projections preserve distances and relationships.  
- 📌 *Video reference: 25:01–33:51*

---

## <a id="formula"></a> 🧮 Mathematical Formula



\[
Variance = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n}
\]



Where:
- \(x_i\): individual data point  
- \(\bar{x}\): mean of dataset  
- \(n\): total number of data points  

👉 Squared differences are used instead of absolute differences because the function is **differentiable at zero**, which is crucial for optimization in PCA.  
- 📌 *Video reference: 26:53–29:23*

---

📌 **Takeaway:** PCA reduces dimensions by creating new axes (principal components) that maximize variance, making data simpler, faster to process, and easier to visualize.
