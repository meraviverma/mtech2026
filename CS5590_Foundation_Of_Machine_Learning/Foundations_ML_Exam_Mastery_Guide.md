---
title: "Foundations of Machine Learning"
subtitle: "Exam Mastery Workbook | IIT Hyderabad MDS Lecture Companion"
date: "26 October 2026"
toc: true
toc-depth: 2
geometry:
  - margin=23mm
fontsize: 11pt
mainfont: "FreeSerif"
colorlinks: true
linkcolor: "MidnightBlue"
---

# How to use this workbook

This companion synthesizes **your uploaded lectures** into an exam-oriented learning sequence. The source references use **PDF page numbers** (starting at 1), or **PowerPoint slide numbers**. Some course decks have a printed internal slide number one below the PDF page number because the cover is unnumbered; the page references here are deliberately unambiguous. The two copies of *Probabilistic Generative Models* are byte-identical; the two 64-page *Modelling and Parameter Estimation updated* uploads are byte-identical. The earlier 44-page parameter-estimation deck strongly overlaps with the updated 64-page version, so the expanded version is the primary reference.

**Source key**: `INTRO` = 23-page *Introduction to Machine Learning*; `EST` = updated 64-page *Probabilistic Models and Parameter Estimation*; `REG` = 74-page *Supervised Learning: Linear Regression and Logistic Regression*; `DIM` = 64-slide *Dimensionality Reduction* PowerPoint; `KNN` = 30-page *Non-parametric Models: k Nearest Neighbours*; `SVM` = 40-page *Support Vector Machines*; `GEN` = 50-page *Probabilistic Generative Models*; `MID` = 9-page handwritten probability midterm.

**Labels used throughout**:

- **[SLIDE]** means the material is directly taught or illustrated in the uploaded lectures.
- **[BACKGROUND]** means a prerequisite, explanation, worked derivation, or example added to make the material intelligible; it is not a direct quotation from the instructor.
- **Exam priority** is a study recommendation inferred from centrality and derivational importance, **not** a claim about actual examination frequencies or marks. High priority = 5/5.
- **Must memorize** = formulas / definitions to reproduce. **Must understand** = reasoning and application. **Nice to know** = context beyond the central exam objective.

The **Slide Coverage Register** at the end indexes every page/slide of the seven principal teaching decks and the nine scanned midterm pages. Closely related animation-build slides and successive derivation slides are taught together in the chapters. This is a comprehensive **topic-level companion and slide index**, not a claim that each slide has a separate essay.

# 0. Mathematical prerequisites: learn these first

**Source:** [BACKGROUND] Added from standard probability, calculus and linear algebra theory. These ideas are used repeatedly but are not all taught from first principles in the lectures.

## 0.1 Random variables, PMF, PDF, CDF

A **random variable** converts an uncertain outcome into a number. A Bernoulli variable $X\in\{0,1\}$ can represent whether a customer clicks. A continuous variable $Y$ could represent a student's weight.

For **discrete** $X$, a probability mass function assigns $P(X=x)$ and must satisfy $\sum_x P(X=x)=1$. For **continuous** $Y$, a density $f_Y(y)$ satisfies $f_Y(y)\ge0$, $\int_{-\infty}^{\infty} f_Y(y)\,dy=1$; probability comes from **area**:

$$P(a\le Y\le b)=\int_a^b f_Y(y)\,dy.$$

A continuous PDF value can exceed 1; the integral over the full support is still 1. A continuous variable has $P(Y=a)=0$ at any specified point if it has a PDF. The **CDF** works for both kinds of variables:

$$F_X(x)=P(X\le x).$$

If $X$ is continuous and the CDF is differentiable, $f_X(x)=F'_X(x)$; conversely $F_X(x)=\int_{-\infty}^x f_X(t)\,dt$.

**Mean** describes the centre: $E[X]=\sum_x xP(X=x)$ (discrete) or $\int x f_X(x)dx$ (continuous). **Variance** measures spread:

$$\operatorname{Var}(X)=E[(X-E[X])^2]=E[X^2]-E[X]^2.$$

## 0.2 Joint, marginal and conditional distributions

For a joint density $f_{X,Y}(x,y)$, the **marginal** of $X$ removes $Y$ by integration:

$$f_X(x)=\int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy,$$

but **only integrate over the region where the joint density is nonzero**. This is a major reason students lose marks: they use $-\infty$ to $\infty$ symbolically but then substitute the wrong finite bounds. If the support is $0\le y\le x\le1$, then $0\le x\le1$ and $\int_0^x f(x,y)\,dy$ is the correct integral.

Conditioning gives, where $f_X(x)>0$:

$$f_{Y\mid X=x}(y)=\frac{f_{X,Y}(x,y)}{f_X(x)}.$$

**Independence** means $f_{X,Y}(x,y)=f_X(x)f_Y(y)$ on the joint domain, not simply that the variables are uncorrelated. Bayes' theorem in event form is

$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)},\quad P(B)>0.$$

In model form, $p(\theta\mid D)\propto p(D\mid\theta)p(\theta)$: posterior is likelihood times prior, divided by evidence.

## 0.3 Essential distributions table

| Distribution | Model | Mean | Variance | Typical use |
|---|---|---|---|---|
| Bernoulli | $P(X=x)=p^x(1-p)^{1-x}$, $x\in\{0,1\}$ | $p$ | $p(1-p)$ | Yes/no, click/not click |
| Binomial | $P(X=k)={n\choose k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Count successes in $n$ trials |
| Poisson | $P(X=k)=e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | Calls or events per interval |
| Gaussian | $f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ | Continuous, approximately symmetric measurements |
| Beta | $p(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$ | $\alpha/(\alpha+\beta)$ | $\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$ | Prior for Bernoulli success probability |

**Do not confuse** Poisson $\lambda$ (mean count per interval), Gaussian $\sigma$ (standard deviation), and a regression regularization parameter also conventionally written $\lambda$.

## 0.4 Derivatives, logs and optimization

An **estimator** is often found by solving an optimization problem. You need these identities:

$$\frac{d}{dx}\ln x=\frac1x,\qquad \frac{d}{dx}x^2=2x,\qquad \ln(ab)=\ln a+\ln b.$$

A log changes products of independent probabilities into sums, preserving the maximizer because log is strictly increasing. To minimize a differentiable loss $J(w)$, the first-order condition is $\nabla J(w)=0$. **Gradient descent** updates

$$w^{(t+1)}=w^{(t)}-\eta\nabla J(w^{(t)}),$$

where $\eta>0$ is the step size. A zero gradient is not enough by itself to guarantee a minimum; convexity or a suitable second-order condition is relevant.

## 0.5 Vectors, matrices, eigenvalues and SVD

A feature vector $x=(x_1,\ldots,x_d)^T$ contains $d$ numeric inputs. A linear prediction is $w^Tx+b$. The dot product $w^Tx=\sum_jw_jx_j$; the Euclidean length is $\|w\|_2=\sqrt{w^Tw}$.

For an $n\times d$ design matrix $X$, $X^TX$ is $d\times d$. An **eigenvector** $v\ne0$ of square $A$ satisfies $Av=\lambda v$; its eigenvalue $\lambda$ tells how much $A$ stretches in that direction. The **singular value decomposition**, which exists for any real $m\times n$ matrix, is

$$A=U\Sigma V^T.$$

$U$ and $V$ contain orthonormal singular-vector directions; $\Sigma$ contains nonnegative singular values. Retaining only the first $k$ large singular values constructs a rank-$k$ approximation $A_k=U_k\Sigma_kV_k^T$. We return to this in the dimensionality chapter.

### Prerequisite exam check

1. The support is $0\le y\le x\le1$. What are the inner integration bounds for finding $f_X(x)$? **Answer:** $y=0$ to $y=x$.
2. Why is $\log\prod_i p(x_i\mid\theta)=\sum_i\log p(x_i\mid\theta)$ useful? **Answer:** simpler differentiation and greater numerical stability.
3. What are the shapes of $X^TX$ and $X^Ty$ when $X\in\mathbb R^{n\times d}$? **Answer:** $d\times d$ and $d\times1$.

# 1. Introduction to Machine Learning

**Slide references:** `INTRO` pp. 2–3 "Machine learning" and "Why machine learning: Data every where!"; p. 4 "Machine learning is interdisciplinary"; pp. 5–9 history; pp. 10–19 types of learning, model families, generalization; pp. 20–23 syllabus, reading and evaluation. **Exam importance: 4/5.** Must understand generalization, learning types, train/test and overfitting. Must memorize definitions and contrasts. History is nice to know unless specifically taught as examinable.

## 1.1 What ML actually learns

**[SLIDE]** The initial lecture uses Kevin Murphy's formulation: automatically detect patterns in data and use them for prediction or decisions under uncertainty (`INTRO` p. 3). It explicitly emphasizes **generalization performance**, meaning success on unseen examples (`INTRO` pp. 2, 14–15).

**[BACKGROUND]** Suppose we want to predict whether an incoming transaction is fraudulent. Hard-coded logic might say "any transaction greater than ₹50,000 is fraud"; it will miss many attacks and reject legitimate users. ML instead learns a function $f_\theta(x)$ from examples $(x_i,y_i)$, where $x$ is a set of transaction features, $y$ a label, and $\theta$ learned parameters. Training should minimize a loss, but the final purpose is low error on new, unseen transactions.

A typical process is **define target** $\to$ **collect and clean data** $\to$ **split train/validation/test** $\to$ **fit parameters on train** $\to$ **choose hyperparameters on validation** $\to$ **evaluate once on test** $\to$ **deploy and monitor drift**. The last steps are helpful background that extends the lecture's train/test principle.

## 1.2 Classification, regression and unsupervised learning

**[SLIDE]** The slide on supervised learning (`INTRO` p. 11) uses labelled input/output pairs. Classification (`INTRO` p. 12) predicts a discrete class. Regression (`INTRO` p. 13) predicts a real number. The later comparison of supervised and unsupervised, probabilistic and non-probabilistic model families appears on pp. 17–19.

- **Regression:** input home size and location, output home price, e.g. $\hat y=75.4$ lakh.
- **Classification:** input a message, output spam/not spam, perhaps with posterior $P(\mathrm{spam}\mid x)=0.92$.
- **Unsupervised learning:** inputs without a target, used for clusters, latent representations, or density estimation.
- **Probabilistic predictor:** represents uncertainty via a probability or density. **Non-probabilistic predictor:** returns a direct prediction or score without a fully specified predictive probability distribution.

**Common trap:** logistic *regression* is a **classification** algorithm when it models binary class probabilities.

## 1.3 Visual interpretation: underfit, good fit and overfit

**[SLIDE]** The regression plot in `INTRO` p. 13 shows inputs and real-valued outputs; pp. 14–15 focus on training, testing and overfitting. **Read the figure as**: dots are training data, the learned curve is the model, and the task is not just to pass through existing dots, but to predict the trend of new dots.

**[BACKGROUND]** A straight-line model for a curved pattern may **underfit**: it is too rigid, so it incurs systematic error on train and test. A degree-12 polynomial through a handful of noisy examples may **overfit**: it reproduces accidental noise and test error rises. A good model balances complexity and generalization. More training accuracy never by itself proves a better model.

## 1.4 Why the history slides matter

**[SLIDE]** `INTRO` pp. 5–9 trace foundations, Rosenblatt's perceptron, the critique of single-layer networks, AI winters, backpropagation, statistical learning and modern deep learning. The conceptual moral is that learning architecture, optimization method, computational resources and data availability collectively influence what is possible. It is **not** that a single-layer perceptron can solve every nonlinear classification problem.

### Model exam question and full-mark points

**Q: Define generalization and distinguish underfitting from overfitting.**

**Model answer:** Generalization is performance on previously unseen data drawn from the intended target distribution. Underfitting occurs when the hypothesis class or training procedure cannot capture meaningful patterns and commonly exhibits large training as well as test error. Overfitting occurs when a model learns noise or accidental specifics of training examples, producing a large train–test performance gap. Remedies depend on the cause: richer features/capacity for underfitting; more suitable regularization, better validation, improved data, or less capacity for overfitting.

**Exam pitfall:** Don't equate overfitting solely with a large number of parameters; data amount, noise, inductive bias and regularization matter too.

**Understanding check**: (1) Can a model have 100% training accuracy and bad test performance? (2) Is clustering supervised? (3) Why does a held-out test set matter? **Answers:** yes; no; it provides a less biased assessment when untouched during model selection.

# 2. Probabilistic Modelling and Parameter Estimation

**Slide references:** `EST` pp. 2–11 "Sampling and Estimation" / "Parameter Estimation"; pp. 12–27 likelihood and Bernoulli/Poisson/Gaussian examples; pp. 28–30 density estimation; pp. 31–48 Bayesian inference, MLE/MAP; pp. 49–64 multivariate modelling and regression. **Exam importance: 5/5.** This unit supports almost every subsequent chapter.

## 2.1 Sampling versus estimation; model versus parameter

**[SLIDE]** `EST` p. 2 explicitly asks the two reverse questions: if a distribution and parameters are known, **how do we sample possible observations?** If observations are known, **how do we estimate unknown parameters?** `EST` pp. 4–5 show distributions of student weights and rainy/not-rainy days.

**[BACKGROUND]** For $X\sim\mathcal N(\mu,\sigma^2)$, the *distribution family* is Gaussian, the *parameters* are $\mu$ and $\sigma$, and each measured weight is a *sample*. Estimating $\mu$ and $\sigma$ from observed weights is the inverse problem. With rainfall, a Bernoulli model has an unknown rain probability $p$; $p$ is not the observed rain indicator itself.

## 2.2 Likelihood is a function of the parameter

**[SLIDE]** `EST` p. 11 defines the maximum-likelihood estimate as $\arg\max_\theta p(D\mid\theta)$ and notes that $L(\theta)$ need not be a probability distribution over $\theta$.

For i.i.d. data $D=(x_1,\ldots,x_n)$,

$$L(\theta;D)=\prod_{i=1}^n p(x_i\mid\theta),\quad \hat\theta_{ML}=\arg\max_\theta L(\theta;D).$$

**Difference:** $p(x\mid\theta)$ is a normalized density / PMF over **data** for fixed $\theta$. Holding data fixed and varying $\theta$ creates a **likelihood curve**, not a normalized posterior distribution. Log-likelihood $\ell(\theta)=\sum_i\log p(x_i\mid\theta)$ has the same maximizer.

## 2.3 Derivation: Bernoulli maximum-likelihood estimator

**[SLIDE]** `EST` pp. 12–17 derive success-rate MLE; p. 21 gives the example of 921 acceptable chips among 1,000 tested.

**Question:** $x_i\in\{0,1\}$ are independent Bernoulli$(p)$; derive $\hat p$.

1. One observation has $P(X_i=x_i)=p^{x_i}(1-p)^{1-x_i}$.
2. Multiply: $L(p)=p^{\sum_i x_i}(1-p)^{n-\sum_i x_i}$.
3. Log: $\ell(p)=s\log p+(n-s)\log(1-p)$ where $s=\sum_i x_i$.
4. Differentiate: $\ell'(p)=s/p-(n-s)/(1-p)$.
5. Set to zero and solve: $s(1-p)=(n-s)p$, giving

$$\boxed{\hat p_{ML}=\frac{s}{n}=\bar x.}$$

The second derivative $-s/p^2-(n-s)/(1-p)^2$ is negative for an interior solution. For $s=0$ or $s=n$, check boundary optima $p=0$ or $p=1$.

**Numerical:** 921 of 1000 acceptable RAM chips $\Rightarrow\hat p=0.921$, i.e. 92.1%. That is a point estimate, not proof that the next 1,000 will have exactly 921 acceptable chips.

## 2.4 Derivation: Poisson MLE

**[SLIDE]** `EST` pp. 22–25 use calls received per minute and Poisson counts. For observed $x_i\in\{0,1,2,\ldots\}$,

$$L(\lambda)=\prod_i\frac{e^{-\lambda}\lambda^{x_i}}{x_i!},\quad
\ell(\lambda)=-n\lambda+\left(\sum_i x_i\right)\log\lambda-\sum_i\log(x_i!).$$

Differentiate: $\ell'(\lambda)=-n+\sum_i x_i/\lambda=0$; therefore

$$\boxed{\hat\lambda_{ML}=\bar x.}$$

For the slide's calls data $3,5,2,4,6,3,4,5$, sum $=32$, $n=8$, $\hat\lambda=4$ calls/minute. The prediction for the **expected** count in the ninth minute is 4, while the actual count remains random. **Trap:** Poisson models event counts, not a binary call/no-call outcome.

## 2.5 Derivation: Gaussian mean and variance MLE

**[SLIDE]** `EST` pp. 26–27 present weight measurements and a Normal model. With $X_i\sim\mathcal N(\mu,\sigma^2)$ i.i.d.,

$$\ell(\mu,\sigma^2)=-\frac n2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_i(x_i-\mu)^2.$$

Set $\partial\ell/\partial\mu=0$ to obtain $\hat\mu=\frac1n\sum_i x_i$. Substituting this back and differentiating in $\sigma^2$ yields

$$\boxed{\hat\sigma^2_{ML}=\frac1n\sum_i(x_i-\bar x)^2.}$$

**Important distinction:** the *unbiased sample variance* replaces $n$ by $n-1$; the **Gaussian MLE** uses $n$. Be explicit which the exam is asking for. Under a fitted Gaussian, a query such as $P(55<X<60)$ becomes $\Phi((60-\hat\mu)/\hat\sigma)-\Phi((55-\hat\mu)/\hat\sigma)$.

## 2.6 Why log-likelihood becomes a loss

**[SLIDE]** `EST` pp. 19–20 link maximum likelihood to minimizing negative log-likelihood and to KL divergence. **[BACKGROUND]** Maximizing $\ell$ is algebraically the same as minimizing $-\ell$. For Bernoulli targets,

$$-\ell(w)=-\sum_i\{y_i\log p_i+(1-y_i)\log(1-p_i)\},$$

which is **binary cross-entropy**. For fixed data-generating $P$, minimizing the expected negative log-likelihood of model $Q_\theta$ is equivalent to minimizing $D_{KL}(P\|Q_\theta)$ because $H(P)$ is constant:

$$E_P[-\log Q_\theta(X)]=H(P)+D_{KL}(P\|Q_\theta).$$

**Trap:** empirical negative log-likelihood is a sample approximation; do not claim that it equals the unknowable population expectation exactly.

## 2.7 Histogram versus kernel density estimation

**[SLIDE]** `EST` pp. 28–30 compare a fixed Gaussian with a bimodal true pattern, a histogram at different bin widths, and a smooth kernel density estimate. The **visual lesson**: an incorrectly chosen parametric family cannot represent two modes, regardless of extra data; too-small histogram bins give jagged estimates; too-wide bins merge real structure.

For $n$ samples in one dimension, a histogram estimates density in bin $j$ by $\hat p(x)=k_j/(nh)$, where $h$ is bin width. A kernel density estimator (KDE) is

$$\hat p_h(x)=\frac1{nh^d}\sum_{i=1}^n K\left(\frac{x-x_i}{h}\right).$$

Here $d$ is feature dimension and $K$ integrates to 1. **Bandwidth $h$ is the dominant smoothing choice**: small $h$ risks high variance; large $h$ risks high bias. No parameterized density shape such as a single Gaussian is assumed by KDE.

## 2.8 Bayes, MLE, MAP, and full Bayesian prediction

**[SLIDE]** `EST` pp. 31–35 compare maximum likelihood with maximum a posteriori estimation; pp. 36–41 develop a voting example; pp. 45–48 distinguish choosing one parameter value from averaging uncertainty.

$$\boxed{p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}}$$

- **MLE:** $\hat\theta_{ML}=\arg\max_\theta p(D\mid\theta)$; uses data likelihood alone.
- **MAP:** $\hat\theta_{MAP}=\arg\max_\theta p(D\mid\theta)p(\theta)$; incorporates a prior and still returns **one point**.
- **Full Bayes:** uses the **whole posterior** and integrates over parameter uncertainty in predictions.

**Derivation, Beta-Bernoulli MAP.** Take $p\sim\mathrm{Beta}(\alpha,\beta)$ and observe $s$ successes and $f=n-s$ failures. The posterior is Beta$(\alpha+s,\beta+f)$. When both posterior shape parameters exceed 1,

$$\boxed{\hat p_{MAP}=\frac{s+\alpha-1}{n+\alpha+\beta-2}.}$$

The posterior mean is $(s+\alpha)/(n+\alpha+\beta)$ and is **not generally equal** to MAP. The formula for an interior mode must not be blindly used at boundaries.

**Slide election example (`EST` pp. 36–40):** $s=12,n=20$ and $\alpha=\beta=5$: MLE $=12/20=0.6$; MAP $=(12+4)/(20+10-2)=16/28=4/7\approx0.5714$. The prior's mode is 0.5, so MAP moves the answer toward 0.5. With uniform Beta$(1,1)$ prior, interior MAP coincides with MLE.

**A second slide challenge:** $s=3,n=5,\alpha=\beta=5$ gives MLE $=0.6$ and MAP $=(3+4)/(5+10-2)=7/13\approx0.5385$.

**One dry day (`EST` pp. 46–48):** Let daily rain chance $q\sim\mathrm{Beta}(1,1)$ and observe $X=0$. Posterior $q\mid X=0\sim\mathrm{Beta}(1,2)$ with density $2(1-q)$. MLE $=0$, MAP $=0$, but posterior predictive probability of rain tomorrow is

$$P(X_{\rm tomorrow}=1\mid X_{\rm today}=0)=E[q\mid D]=\int_0^1 q\cdot 2(1-q)\,dq=\frac13.$$

**Intuition:** a single dry day is weak evidence; zero observed rainy days does not prove rain impossible. Here the Bayesian prediction preserves uncertainty.

## 2.9 Gaussian prior, Gaussian mean: why MAP shrinks

**[SLIDE]** `EST` pp. 42–44 derive MAP for a Gaussian mean. If $x_i\sim\mathcal N(\mu,\sigma^2)$ with known observation variance and prior $\mu\sim\mathcal N(\mu_0,\tau^2)$, posterior precision is $n/\sigma^2+1/\tau^2$. Its mean (also MAP for a Gaussian posterior) is

$$\boxed{\hat\mu_{MAP}=\frac{(n/\sigma^2)\bar x+(1/\tau^2)\mu_0}{n/\sigma^2+1/\tau^2}.}$$

It is a **precision-weighted average** of the observed mean and prior mean. More observations $n$ increase the data's influence; a narrow prior (small $\tau^2$) increases prior influence.

## 2.10 Multivariate Gaussian and bridge to regression

**[SLIDE]** `EST` pp. 49–55 use height and weight and a two-dimensional scatter diagram. Each point is one person's pair $(\mathrm{height},\mathrm{weight})$. The slanted elliptical cloud illustrates **positive correlation**; contour ellipses encode covariance, not merely separate univariate spreads. For $d$ dimensions,

$$x\sim\mathcal N(\mu,\Sigma),\quad p(x)=\frac{\exp\{-\tfrac12(x-\mu)^T\Sigma^{-1}(x-\mu)\}}{(2\pi)^{d/2}|\Sigma|^{1/2}}.$$

Parameter count for unrestricted covariance: $d$ mean entries and $d(d+1)/2$ unique covariance entries. In two dimensions, this is $2+3=5$.

For a jointly Gaussian $(X,Y)$ the conditional expectation is

$$E[Y\mid X=x]=\mu_y+\Sigma_{yx}\Sigma_{xx}^{-1}(x-\mu_x),$$

and conditional variance is $\Sigma_{yy}-\Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}$. `EST` pp. 56–57 demonstrate that the conditional mean is affine: a regression-like relationship emerges.

**[SLIDE]** `EST` pp. 58–64 show why modelling $p(x)$ may be unwieldy when features include booleans, one-hot categories or basis expansions. Instead model $p(y\mid x)$ and let its **conditional mean depend on input**: $\mu(x)=w^Tx+b$. A Gaussian conditional leads to least squares, providing the transition to the next lecture.

### Exam questions and answer checkpoints

1. **Derive Bernoulli MLE.** Include the full likelihood, log, derivative, stationary equation and boundary cases.
2. **Derive Poisson MLE for independent equal-rate counts.** Answer $\bar x$; note count-unit interpretation.
3. **Derive Gaussian mean and variance MLE.** Distinguish $1/n$ (MLE) from $1/(n-1)$ (unbiased variance).
4. **Why does a Beta prior act as smoothing?** It supplies pseudo-counts, moving an extreme small-sample estimate toward the prior.
5. **Why does maximizing log-likelihood correspond to minimizing cross-entropy?** Negative logarithm changes a likelihood maximization into a differentiable loss minimization; Bernoulli conditional likelihood yields binary cross-entropy.
6. **What is the conceptual difference between MAP and Bayesian prediction?** MAP selects a single posterior mode, while the predictive probability integrates over all plausible parameter values.
7. **Must memorize:** $L(\theta)$, Bayes rule, Bernoulli/Poisson/Gaussian MLEs, Beta-Bernoulli MAP, multivariate Gaussian structure. **Must understand:** choice of support, independence assumptions, why log is used, how a prior affects estimates.

# 3. Supervised Learning: Linear, Polynomial and Logistic Regression

**Slide references:** `REG` pp. 2–15 examples, regression and basis functions; pp. 16–25 least squares and geometry; pp. 26–43 polynomial overfitting, MAP, ridge/lasso and validation; pp. 44–54 GLMs, bias–variance and Bayesian linear regression; pp. 55–74 classification, logistic regression, regularized likelihood, multiclass and metrics. **Exam importance: 5/5.**

## 3.1 How a conditional distribution becomes a predictor

**[SLIDE]** `REG` pp. 2–8 show growth versus age, coffee sales and housing prices, followed by an air-quality example predicting temperature from solar radiation, ozone and wind. The scatter plots are **observations**; the line or curve is the **learned approximation**. The many outcomes for similar feature values represent noise and omitted influences, not automatic model failure.

**[BACKGROUND]** A simple linear regression model is

$$y_i=w^Tx_i+b+\varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,\sigma^2).$$

Equivalently $Y_i\mid X_i=x_i\sim\mathcal N(w^Tx_i+b,\sigma^2)$. We usually absorb $b$ into $w$ by adding an all-ones column to $X$. The word **linear** refers to the adjustable coefficients $w$, not necessarily a straight line in raw input: the predictor $w_0+w_1x+w_2x^2$ is polynomial in $x$ but linear in the parameter vector $(w_0,w_1,w_2)$.

## 3.2 Least squares from Gaussian maximum likelihood

**[SLIDE]** `REG` pp. 16–23 state and derive least squares; p. 19 works through its gradient. **[BACKGROUND]** For a design matrix $X\in\mathbb R^{n\times d}$, target $y\in\mathbb R^n$ and parameter $w\in\mathbb R^d$,

$$J(w)=\frac12\|Xw-y\|_2^2=\frac12(Xw-y)^T(Xw-y).$$

Expand $J(w)=\tfrac12w^TX^TXw-w^TX^Ty+\tfrac12y^Ty$. Differentiate:

$$\nabla_wJ(w)=X^TXw-X^Ty=X^T(Xw-y).$$

Set to zero to get the **normal equations** $X^TXw=X^Ty$. If $X^TX$ is invertible,

$$\boxed{\hat w=(X^TX)^{-1}X^Ty.}$$

When $X^TX$ is singular, use the Moore–Penrose pseudoinverse $X^+y$ or a regularized estimator, **not** an illegal inverse. Numerically one generally prefers QR/SVD-based solvers to explicitly computing $(X^TX)^{-1}$ because the latter can exacerbate numerical instability.

**Gradient descent alternative:** $w\gets w-\eta X^T(Xw-y)$ (for the loss above). Scaling the objective by $1/n$ scales the gradient accordingly. **Common mistakes:** dropping the transpose, wrong dimensions, forgetting the intercept, confusing a stationary point with an arbitrary global optimum. Here the Hessian $X^TX$ is positive semidefinite, so any stationary point is a global minimizer; uniqueness requires full column rank.

## 3.3 The geometry of least squares

**[SLIDE]** `REG` p. 23 shows the column space of the design matrix as a plane, a target point $y$, and a projection $\hat y$. The right-angle visual means that the residual $r=y-X\hat w$ is orthogonal to every column of $X$:

$$X^Tr=0.$$

Because $\hat y=X\hat w$ is the closest representable vector in the column space of $X$, least squares is a **projection problem**. This geometric fact is another route to the normal equations. **Exam importance: 5/5** for matrix-based derivations.

## 3.4 Polynomial regression: why exact fit is not the goal

**[SLIDE]** `REG` pp. 14–15 introduce polynomial bases; pp. 26–30 display increasing degree $M$ and the familiar overfitting curve. In the 4-panel image on p. 26, the low-degree model misses the wave, intermediate degree follows it, and the high-degree curve oscillates dramatically between/near examples. The key message is a **training fit vs generalization** trade-off.

With basis vector $\phi(x)=[1,x,x^2,\ldots,x^M]^T$, predict $\hat y=w^T\phi(x)$. With only 10 distinct points and polynomial degree 9, a square interpolation system may fit all training targets nearly perfectly if nonsingular. However, closely spaced inputs and high-degree monomials can produce an ill-conditioned Vandermonde matrix; the coefficients become extremely sensitive to noise. **A huge coefficient is not automatically proof of optimization failure**—the issue is instability, model complexity and overfitting.

## 3.5 Ridge, lasso, Gaussian priors and MAP

**[SLIDE]** `REG` pp. 31–38 explicitly connect a prior on $w$ with an added regularization penalty. The comparison on p. 33 of two degree-12 fits shows MLE interpolating noise aggressively while Gaussian-prior MAP gives a smoother prediction. The penalty plots around p. 38 use circular $L_2$ and diamond-shaped $L_1$ constraint geometry.

**Ridge** minimizes

$$J_{\rm ridge}(w)=\frac12\|Xw-y\|^2+\frac\lambda2\|w\|_2^2,$$

and, for $\lambda>0$, has solution

$$\boxed{\hat w_{\rm ridge}=(X^TX+\lambda I)^{-1}X^Ty.}$$

A prior $w\sim\mathcal N(0,\tau^2I)$ with Gaussian observation noise gives exactly such an $L_2$ penalty under MAP; its regularization strength depends on the ratio of observation variance to prior variance. Ridge shrinks weights **continuously**, usually not exactly to zero.

**Lasso** uses $J_{\rm lasso}=\tfrac12\|Xw-y\|^2+\lambda\|w\|_1$. Because the $L_1$ norm has sharp corners, lasso can set some weights exactly to zero, enabling feature selection. A Laplace prior provides a corresponding MAP view. In both cases, an intercept is commonly left unpenalized after centring/standardizing variables.

**Exam traps:** Saying ridge forces exact sparsity is usually false; saying lasso always keeps the better correlated feature is too strong; saying regularization eliminates bias is false. Regularization typically **increases bias and reduces variance**. The numerical definition of $\lambda$ varies with whether the likelihood sums or averages losses, so show your objective before quoting the closed form.

## 3.6 Train, validation, test and cross-validation

**[SLIDE]** `REG` pp. 39–43 discuss model selection and unbiased evaluation. **Training data** fit parameters. **Validation data** choose polynomial order, regularization $\lambda$, early stopping or other hyperparameters. **Test data** estimate performance after choices are fixed. In $k$-fold cross-validation, split the training pool into $k$ folds, fit on $k-1$ and validate on the held-out fold, then average scores.

**[BACKGROUND]** Do not standardize features using mean and variance computed from the complete train+test dataset: that leaks test information. Preprocessing must be fit within each training fold and applied to its corresponding validation fold.

## 3.7 Bias–variance decomposition

**[SLIDE]** `REG` pp. 48–54 show many fitted curves on different samples, their mean, squared bias, variance and total error. If $y=f(x)+\varepsilon$ with $E[\varepsilon]=0$ and variance $\sigma^2$ independent of training data, then at a fixed $x$,

$$E_{D,\varepsilon}[(y-\hat f_D(x))^2]
=\underbrace{(E_D\hat f_D(x)-f(x))^2}_{\text{squared bias}}
+\underbrace{E_D[(\hat f_D(x)-E_D\hat f_D(x))^2]}_{\text{variance}}
+\underbrace{\sigma^2}_{\text{irreducible noise}}.$$

**Interpret the slide:** a rigid learner has overlapping, similarly shaped curves across datasets (low variance but possibly wrong centre); a flexible learner has curves that differ dramatically across datasets (high variance). The total error need not be minimized by either extreme. A data set with more examples often allows greater flexibility without overfitting, but not under all possible conditions.

## 3.8 Other conditional models: Poisson regression

**[SLIDE]** `REG` pp. 45–47 model event counts as Poisson. Because a Poisson mean must be positive, set $\lambda(x)=\exp(w^Tx+b)$ (a **log link**), so $Y\mid x\sim\operatorname{Poisson}(\lambda(x))$. Maximize conditional likelihood or minimize negative log-likelihood; unlike ordinary Gaussian least squares, the target-dependent error function is not simply squared error.

**When to choose:** call arrivals per hour or defect counts. Check whether the Poisson equal mean/variance assumption is plausible; severe overdispersion can motivate a more flexible count model.

## 3.9 Logistic regression from Bernoulli likelihood

**[SLIDE]** `REG` pp. 55–67 move from continuous targets to binary classification and present the sigmoid, conditional likelihood, gradients and regularization.

For $y_i\in\{0,1\}$ define a score $z_i=w^Tx_i+b$, sigmoid $\sigma(z)=1/(1+e^{-z})$, and $p_i=P(Y_i=1\mid x_i)=\sigma(z_i)$. Thus

$$\log\frac{p_i}{1-p_i}=w^Tx_i+b.$$

**Meaning of log-odds:** a unit increase in feature $x_j$ (others fixed) adds $w_j$ to the log-odds, and multiplies the odds by $e^{w_j}$. A model is **linear in the log-odds**, not in the probability itself.

For independent observations, conditional likelihood and loss are

$$L(w,b)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i},\qquad
J(w,b)=-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].$$

**Gradient derivation:** $\sigma'(z)=\sigma(z)(1-\sigma(z))$. For one observation, $\partial J_i/\partial z_i=p_i-y_i$. By the chain rule, $\partial z_i/\partial w=x_i$, so

$$\boxed{\nabla_wJ=X^T(p-y),\qquad \frac{\partial J}{\partial b}=\sum_i(p_i-y_i).}$$

A full-batch gradient step uses $w\gets w-\eta X^T(p-y)$ and $b\gets b-\eta\sum_i(p_i-y_i)$. In contrast to least squares, logistic regression has no ordinary closed-form analytic optimum; iterative optimization is typical.

**Decision boundary:** at threshold $p=0.5$, $z=0$, so $w^Tx+b=0$. This is a line in 2D, a plane in 3D and a hyperplane in $d$ dimensions. The probability curve itself is nonlinear in $x$; the boundary is linear in the original features unless a nonlinear feature map is used.

## 3.10 Separable data, multiclass and classification metrics

**[BACKGROUND]** With unregularized logistic regression and **strictly linearly separable** data, the likelihood's supremum is approached as coefficient norms grow without bound, usually along a separating direction. There need not be a finite MLE; gradient descent can keep increasing coefficients while the classification boundary stabilizes. Adding $L_2$ regularization controls their size. This is an exam-relevant consequence of conditional log-likelihood even if not emphasized at length in the slides.

**[SLIDE]** `REG` p. 68 covers one-vs-rest multiclass classification; p. 73 introduces evaluation. With $K$ classes, one-vs-rest trains $K$ binary classifiers. Multinomial logistic regression instead uses softmax

$$P(Y=k\mid x)=\frac{e^{w_k^Tx+b_k}}{\sum_{j=1}^K e^{w_j^Tx+b_j}}.$$

Binary classification definitions: $\mathrm{accuracy}=(TP+TN)/(TP+TN+FP+FN)$; $\mathrm{precision}=TP/(TP+FP)$; $\mathrm{recall}=TP/(TP+FN)$; $F_1=2PR/(P+R)$ for precision $P$ and recall $R$. In imbalanced problems high accuracy may hide a poor minority-class recall.

### Exam questions and models

1. **Derive normal equations.** Start from $\tfrac12\|Xw-y\|^2$, expand, differentiate, set equal to zero. Explain full-rank invertibility and the pseudoinverse fallback.
2. **Why is a degree-9 polynomial dangerous for 10 points?** A very flexible interpolator may have unstable coefficients and low training error but high prediction variance; the Vandermonde matrix can be ill-conditioned.
3. **Ridge vs lasso.** Ridge $L_2$ shrinks all coefficients; lasso $L_1$ can set some to zero; connect Gaussian and Laplace priors respectively.
4. **Derive logistic cross-entropy and its gradient.** Show Bernoulli likelihood and use sigmoid derivative to obtain $X^T(p-y)$.
5. **What if classes are perfectly linearly separable?** Unregularized logistic weights tend to unbounded norm as loss approaches its infimum.
6. **State bias–variance decomposition.** Include squared bias, variance and irreducible noise with necessary assumptions.
7. **Must memorize:** regression and logistic model, normal equations, ridge solution, sigmoid, binary cross-entropy, gradient, metrics. **Must understand:** projection, regularization/prior connection, separation and model selection.

# 4. Dimensionality Reduction, Eigenvalues and SVD

**Slide references:** `DIM` slides 1–27 motivation, costs, curse of dimensionality and methods; slides 28–31 feature selection; slides 32–44 information retrieval, latent semantic analysis and rank; slides 45–58 eigenvalues, SVD and low-rank approximation; slides 59–64 latent representations and references. **Exam importance: 5/5** for linear-algebra derivations; 4/5 for motivating applications.

## 4.1 What dimension is and why it costs

**[SLIDE]** Slides 2–3 contrast naive nearest-neighbour computation as both number of records $N$ and number of features $d$ grow. Distances between one test example and all training examples cost roughly $O(Nd)$ operations. More dimensions also require more memory and complicate visualization (`DIM` slides 19, 23–25).

**[SLIDE]** Slide 20 illustrates **sparsity**. If one dimension is divided into $m$ bins, then $d$ dimensions contain $m^d$ joint bins. With 10 bins/axis: 1D has 10, 2D has 100, 3D has 1,000, 10D has $10^{10}$ cells. Holding sample size fixed as $d$ grows leaves ever more cells empty. This is one manifestation of the **curse of dimensionality**. It helps explain why naive density estimation and distance-based learning degrade in high-dimensional spaces.

**[BACKGROUND]** There is no general guarantee that every extra feature causes overfitting. Useful informative features can improve prediction; features that are irrelevant, redundant or noisy may hurt or drive up sample/computational requirements. The aim is to preserve information relevant to the task while reducing representation complexity.

## 4.2 Feature selection versus feature extraction

**[SLIDE]** `DIM` slide 27 lists feature selection (filter, wrapper) and extraction (SVD, PCA, LDA, autoencoders); slides 29–31 show wrapper procedures.

- **Feature selection** keeps a subset of the original variables, e.g. choose age, balance and transaction frequency out of 300 original features. It preserves original meaning.
- **Feature extraction** generates new variables, e.g. $z_1=0.4x_1-0.2x_2+0.8x_3$, that summarize original variables and may be harder to interpret.
- **Filter method:** rank features with statistical criteria, e.g. correlation, variance or chi-square, without repeatedly fitting a chosen predictive model.
- **Wrapper method:** train/validate a particular model on candidate feature sets; generally more expensive and susceptible to overfitting selection to a small validation set.
- **Embedded method:** select/regularize within model fitting (e.g. lasso).

**Sequential forward selection (SFS):** begin with no features; at each iteration try each not-yet-selected feature, choose the addition with best validation improvement, and stop when the chosen criterion is reached. **Sequential backward selection (SBS):** start with all features, remove the feature whose removal gives the best validation performance, and repeat. Exhaustive subset selection would examine $2^d$ subsets, generally impractical for large $d$.

## 4.3 Why matrix rank and latent meaning enter text retrieval

**[SLIDE]** Slides 33–38 use document retrieval and synonyms such as *car* versus *automobile*: a raw term-frequency vector-space model may miss semantic similarity when equivalent concepts use different words. Slides 39–42 discuss matrix rank and factorization. The idea is that a term-by-document matrix has noisy or redundant directions, while a smaller set of **latent concepts** can represent recurring patterns across terms/documents.

**[BACKGROUND]** If $A$ is an $m\times n$ term-document matrix, a low-rank factorization $A\approx BC$ can represent documents in $k$ dimensions with $k\ll\min(m,n)$. The original matrix might have many zero entries; latent directions reveal associations between terms that occur in similar documents. This is **latent semantic analysis (LSA/LSI)**. It is not a language model and does not fully solve ambiguity or synonymy.

## 4.4 Eigenvalues and singular value decomposition

**[SLIDE]** `DIM` slides 45–48 move from eigenvalue decomposition to SVD. **Eigenvector:** $Av=\lambda v$ (for square $A$). Not every matrix has a useful eigenvector basis in real numbers; SVD is broader and numerically fundamental:

$$\boxed{A=U\Sigma V^T.}$$

For $A\in\mathbb R^{m\times n}$, a full SVD may use orthogonal $U\in\mathbb R^{m\times m}$, $V\in\mathbb R^{n\times n}$, and diagonal-shaped $\Sigma\in\mathbb R^{m\times n}$. Nonzero singular values $\sigma_1\ge\sigma_2\ge\cdots\ge0$ are square roots of nonzero eigenvalues of $A^TA$. Columns of $V$ are right singular vectors (input directions); columns of $U$ are left singular vectors (output directions).

**Mental model:** (1) $V^T$ rotates/reflection-changes coordinates, (2) $\Sigma$ stretches or compresses along special axes, (3) $U$ rotates/reflection-changes again. If $\sigma_j$ is tiny, very little of $A$ varies along that singular direction.

## 4.5 Best rank-$k$ approximation

**[SLIDE]** `DIM` slides 52–58 build truncated SVD and a low-rank reconstruction. Retain only largest $k$ singular values:

$$\boxed{A_k=U_k\Sigma_kV_k^T.}$$

The **Eckart–Young–Mirsky theorem** says that $A_k$ minimizes approximation error among matrices of rank at most $k$, for either Frobenius or spectral norm:

$$\|A-A_k\|_F^2=\sum_{j>k}\sigma_j^2,\qquad \|A-A_k\|_2=\sigma_{k+1}.$$

**[BACKGROUND]** Singular values quantify how much structured variation each rank-one component contributes. If singular values are $(5,3,1)$ and we keep $k=2$, squared Frobenius error is $1^2=1$, so Frobenius error is 1. Total squared Frobenius norm was $25+9+1=35$, and the retained squared-energy proportion is $34/35\approx97.1\%$. This is a matrix approximation fact; do not confuse it with supervised prediction accuracy.

**Connection to PCA:** centre observation data before PCA. Principal-component directions are right singular vectors of the centred data matrix. The sample covariance is proportional to $X_c^TX_c$, and explained variances are proportional to squared singular values. LSA often applies SVD to term-document matrices (possibly weighted), whereas PCA focuses on variance in a centred data matrix. The distinction in preprocessing matters.

## 4.6 Visual interpretation of the lecturer's factorization

**[SLIDE]** `DIM` slides 60–62 draw **term-document matrix = term-to-concept matrix × concept strength matrix × concept-to-document matrix**. In SVD symbols, this is $U\times\Sigma\times V^T$. Each column in $U_k$ describes how terms load on a latent concept; $\Sigma_k$ tells the strength of each retained direction; columns of $\Sigma_kV_k^T$ describe each document in the reduced concept space. Slide 59 labels this "Hidden or Latent Concepts." The professor expects students to connect matrix algebra to meaningful feature reduction—not to memorize only three matrix letters.

### Exam questions and answers

1. **Why does the number of histogram cells grow exponentially with $d$?** Because independent bin choices on $d$ axes create $m^d$ Cartesian combinations.
2. **Contrast feature selection with extraction.** Selection retains original variables; extraction creates new combinations/representations.
3. **Explain forward selection.** Enumerate train/validate-add-best-repeat-stop, and why it is greedy rather than globally exhaustive.
4. **State SVD and dimensions.** $A=U\Sigma V^T$; dimensions above.
5. **Derive best rank-$k$ approximation and its error.** Retain leading $k$ components; squared Frobenius residual is sum of squares of discarded singular values.
6. **Interpret the term-document SVD diagram.** Term associations, concept strengths, reduced document vectors.

# 5. Non-Parametric Models and k-Nearest Neighbours

**Slide references:** `KNN` pp. 2–10 parametric/non-parametric contrast; pp. 11–15 density estimation and classification; pp. 16–23 kNN rule and choice of $K$; pp. 24–29 curse, trade-offs and improvements; p. 30 readings. **Exam importance: 5/5** for decision rule, bandwidth/neighbour selection and bias–variance.

## 5.1 Parametric versus non-parametric

**[SLIDE]** `KNN` pp. 3–4 show data with two peaks: a **single Gaussian** produces one broad hump whereas a KDE can reproduce two. Slides 5–8 ask the important question: does the **effective complexity of the fitted representation** stay fixed, or can it grow with sample size? A model is not called non-parametric because it has literally zero parameters. A fixed-architecture neural network is parametric even if very large; kNN and KDE are standard non-parametric methods whose effective representation grows with available data.

**Visual interpretation:** dotted two-peak shape is the observed distribution; parametric Gaussian (one bell curve) cannot express two modes; the KDE sum of small kernels follows both peaks. This is a statement about **model-family bias**, not that KDE always outperforms Gaussian fitting on finite data.

## 5.2 kNN as local evidence

**[SLIDE]** `KNN` pp. 13–15 derive the intuition from local density estimates: in a small region of volume $V$, the class-conditional density approximately equals local class count divided by class sample count and volume. When priors reflect class frequencies and the neighbourhood is shared, the Bayes class comparison reduces to which class occurs most frequently among nearby observations.

**kNN classifier:** To predict at $x$, compute distances $d(x,x_i)$, pick the $K$ smallest, and return the majority class:

$$\hat y(x)=\operatorname{mode}\{y_i:i\in N_K(x)\}.$$

A local empirical class probability is $\hat P(Y=c\mid x)=\frac1K\sum_{i\in N_K(x)}\mathbf1(y_i=c)$. This is not necessarily a perfectly calibrated probability estimate. For regression, predict the mean of neighbours' numeric targets, optionally distance-weighted.

## 5.3 Distance metrics and feature scaling

**[SLIDE]** `KNN` pp. 20–21 demonstrate that "near" depends on the metric. Euclidean distance $\sqrt{\sum_j(x_j-z_j)^2}$, Manhattan distance $\sum_j|x_j-z_j|$, Minkowski $p$-norm, or cosine similarity for directional text embeddings may fit different use cases.

**[BACKGROUND]** Suppose two people differ by 1 year in age and 50,000 rupees in income. Without scaling, income can dominate Euclidean distance just because of units. Standardization $z_j=(x_j-\mu_j)/s_j$ or min-max scaling can correct gross scale imbalance. Fit scaler only on training data; apply the same scaler to validation/test.

## 5.4 Choosing K is choosing smoothness

**[SLIDE]** `KNN` pp. 22–23 are especially important: when $K=1$, training error may be nearly zero but predictions are sensitive to noise (**low bias/high variance**). Very large $K$ averages across different classes and misses local boundaries (**high bias/low variance**). Use a validation split or cross-validation to select $K$; never choose it solely by minimizing training error.

**Worked numerical:** neighbours' labels sorted by distance are $[A,B,A,A,B]$. With $K=1$, predict $A$. With $K=3$, labels $[A,B,A]$ give $A$. With $K=5$, $A$ occurs 3 times versus $B$ twice, so predict $A$ with local frequency $3/5=0.6$.

## 5.5 Strengths, limitations and efficiency

**[SLIDE]** `KNN` pp. 24–28 cover distance concentration in high dimensions, $O(Nd)$ naive query cost, distance weighting, scaling, neighbour-search structures and storage reduction. kNN has little explicit "training" computation, but naïvely stores all points and is expensive at inference. KD-trees/Ball-trees can help in moderate dimensions, whereas approximate nearest-neighbour search and representation learning become useful at scale.

**Common errors:** assuming non-parametric means there are no hyperparameters; thinking $K=1$ is always optimal; allowing target labels into the distance computation; forgetting normalization; using the classification majority vote to predict a continuous target.

### Exam questions

1. **Parametric vs non-parametric?** Fixed-size parameterization versus capacity/representation that can grow with data; both have choices and assumptions.
2. **Explain the bias–variance effect of $K$.** Small $K$: flexible/noisy; large $K$: smooth/possibly underfit.
3. **Cost for $N=10^6$, $d=10^4$?** A single brute-force query compares roughly $10^{10}$ feature differences, ignoring constant factors and indexing.
4. **What if every feature is binary?** Hamming distance may suit discrete features better than raw Euclidean distance.
5. **What are the roles of bandwidth $h$ and neighbourhood count $K$?** Both control the locality/smoothness of non-parametric estimates.

# 6. Support Vector Machines: Geometry, Primal, Dual and Kernels

**Slide references:** `SVM` pp. 2–11 history and linear separation; pp. 12–22 margin geometry and hard-margin formulation; pp. 23–34 soft margin, Lagrangian, dual and KKT; pp. 35–39 multiclass and hinge loss; p. 40 readings. **Exam importance: 5/5**, especially derivation of margin and primal/dual optimization.

## 6.1 What separates points, and why a large margin helps

**[SLIDE]** `SVM` pp. 4–11 show two point clouds labelled $+1$ and $-1$ with multiple possible separating lines. Many lines give zero training error; the SVM asks which one leaves the **widest cushion** between classes. The dots touching that cushion are the **support vectors**.

A linear classifier is $\hat y=\operatorname{sign}(w^Tx+b)$ with labels $y_i\in\{-1,+1\}$. The decision boundary is $w^Tx+b=0$. Its normal vector is $w$, so the absolute perpendicular distance of a point $x$ from the boundary is

$$\boxed{d(x,\mathrm{boundary})=\frac{|w^Tx+b|}{\|w\|_2}.}$$

**Proof intuition [BACKGROUND]:** a unit vector normal to the line is $w/\|w\|$; moving a point a distance $t$ in that direction changes $w^Tx+b$ by $t\|w\|$. To land on score zero requires $t=|w^Tx+b|/\|w\|$.

## 6.2 Scale invariance and the margin formula

**[SLIDE]** `SVM` pp. 12–19 display a central separator $w^Tx+b=0$ and two parallel supporting boundaries $w^Tx+b=+1$ and $-1$. Multiplying $(w,b)$ by a positive constant does not change the predicted sign, so the scale is fixed by requiring the nearest training points to have functional margin 1:

$$y_i(w^Tx_i+b)\ge1.$$

The distance between the $+1$ and $-1$ boundaries is $2/\|w\|$. Therefore maximizing width is equivalent to minimizing $\|w\|$, and conveniently $\tfrac12\|w\|^2$. **Hard-margin SVM:**

$$\boxed{\min_{w,b}\ \frac12\|w\|^2\quad\text{subject to } y_i(w^Tx_i+b)\ge1\ \forall i.}$$

This is a convex quadratic optimization problem with linear inequalities. A feasible solution requires linear separability (and the scaling normalization as written).

## 6.3 Soft margin: noisy and nonseparable points

**[SLIDE]** `SVM` pp. 23–25 introduce noisy data, slack $\xi_i$ and a penalty controlled by $C$. Soft-margin primal:

$$\boxed{\min_{w,b,\xi}\ \frac12\|w\|^2+C\sum_{i=1}^n\xi_i\quad
\text{s.t. }y_i(w^Tx_i+b)\ge1-\xi_i,\quad\xi_i\ge0.}$$

**Interpretation:** $\xi_i=0$ when the point is correctly classified on or outside its class margin. $0<\xi_i<1$ means on the correct side of the decision boundary but inside the margin. $\xi_i>1$ means misclassified (with $\xi_i=1$ on the boundary). Larger $C$ penalizes margin violations more heavily, often producing a tighter fit; smaller $C$ allows a wider margin with more violations. This is a trade-off, not a guarantee that any one value generalizes best.

## 6.4 Hinge loss equivalence

**[SLIDE]** `SVM` pp. 38–39 compare hinge, logistic, squared and 0–1 losses. For fixed $w,b$, the minimum feasible slack is $\xi_i=\max(0,1-y_i(w^Tx_i+b))$. Substitute into the primal:

$$\boxed{\min_{w,b}\ \frac12\|w\|^2+C\sum_i\max\{0,1-y_i(w^Tx_i+b)\}.}$$

Let $m=y_i(w^Tx_i+b)$ denote signed functional margin. Hinge loss is zero for $m\ge1$, linear for $m<1$, and does not directly provide a calibrated probability. Logistic loss $\log(1+e^{-m})$ is smooth and always positive at finite $m$. The plotted curves on slide 39 compare how each penalizes low-confidence or incorrect predictions.

## 6.5 Deriving the Lagrangian dual

**[SLIDE]** `SVM` pp. 26–33 introduce Lagrange multipliers, weak/strong duality, the stationarity conditions and complementary slackness. For the standard **half-squared-norm** convention written in the soft-margin objective above, define

$$\mathcal L=\tfrac12\|w\|^2+C\sum_i\xi_i-\sum_i\alpha_i[y_i(w^Tx_i+b)-1+\xi_i]-\sum_i\mu_i\xi_i,$$

where $\alpha_i,\mu_i\ge0$. Stationarity gives

$$\frac{\partial\mathcal L}{\partial w}=w-\sum_i\alpha_i y_i x_i=0
\ \Rightarrow\ \boxed{w=\sum_i\alpha_i y_i x_i},$$

$$\frac{\partial\mathcal L}{\partial b}=-\sum_i\alpha_i y_i=0
\ \Rightarrow\ \boxed{\sum_i\alpha_i y_i=0},$$

$$\frac{\partial\mathcal L}{\partial\xi_i}=C-\alpha_i-\mu_i=0
\ \Rightarrow\ \boxed{0\le\alpha_i\le C}.$$

Substitute stationary solutions into the Lagrangian. The **dual problem** is

$$\boxed{\max_{\alpha}\ \sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j(x_i^Tx_j)}$$

subject to $0\le\alpha_i\le C$ and $\sum_i\alpha_i y_i=0$. A point with $\alpha_i>0$ influences $w$ and is a support vector. The prediction becomes $\operatorname{sign}(\sum_i\alpha_i y_i x_i^Tx+b)$.

**Important slide-convention warning:** `SVM` p. 29 shows an intermediate expression $w=\tfrac12\sum_i\alpha_i y_i x_i$ and a dual quadratic coefficient $\tfrac14$; those follow if one begins with the primal objective **$\|w\|^2$ without its conventional factor $1/2$**. Elsewhere the slides use $\tfrac12\|w\|^2$. These are equivalent optimization formulations only with the multipliers rescaled accordingly; **do not mix the two formula conventions in a single derivation**. For exams, write your primal objective first and maintain its normalization consistently.

## 6.6 KKT conditions and point types

**[SLIDE]** `SVM` pp. 28 and 33–34 use complementary slackness. With $f_i=w^Tx_i+b$:

$$\alpha_i[y_if_i-1+\xi_i]=0,\qquad (C-\alpha_i)\xi_i=0.$$

- Correct and strictly outside the margin: $y_if_i>1$, typically $\alpha_i=0$ (not a support vector).
- Exactly on margin, $\xi_i=0$, often $0<\alpha_i<C$.
- Margin violation with $\xi_i>0$: $\alpha_i=C$.

At degenerate solutions, exact boundary values may require qualification; the complementary slackness equations are authoritative.

## 6.7 Kernel extension and multiclass

**[SLIDE]** Early SVM slides refer to kernel methods; the main algebra in the uploaded deck is the linear dual. **[BACKGROUND]** The dual depends on data through inner products $x_i^Tx_j$. Replacing them with a valid positive-semidefinite kernel $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$ gives a nonlinear boundary in original input space while a linear separator exists in the feature space:

$$f(x)=\operatorname{sign}\left(\sum_i\alpha_i y_i K(x_i,x)+b\right).$$

Examples: linear $K(x,z)=x^Tz$; polynomial $K(x,z)=(\gamma x^Tz+r)^q$ for suitable choices; RBF $K(x,z)=\exp(-\gamma\|x-z\|^2)$. A kernel must satisfy mathematical requirements for the standard convex kernel SVM; arbitrary similarity scores are not guaranteed valid.

**[SLIDE]** `SVM` pp. 35–37 outline one-vs-rest and one-vs-one multiclass schemes: a binary SVM can be extended via multiple binary classifiers. For one-vs-rest, train $K$ classifiers, one class against the rest. For one-vs-one, train $K(K-1)/2$ pairwise classifiers and aggregate predictions.

### Exam questions and answers

1. **Derive margin width.** Distances to the supporting planes $w^Tx+b=\pm1$ are $1/\|w\|$ from centre; total margin $2/\|w\|$.
2. **Write hard- and soft-margin primal.** Include every constraint, signs of $y_i$, $\xi_i\ge0$, and role of $C$.
3. **Derive dual.** Construct Lagrangian; stationary $w,b,\xi$; show bounds $0\le\alpha_i\le C$; substitute.
4. **Define support vector using KKT.** Nonzero dual weight for an active margin condition, subject to degeneracy details.
5. **Explain kernel trick.** Replace input dot products with feature-space inner products without explicitly constructing $\phi(x)$.
6. **Compare logistic and hinge loss.** Smooth likelihood-based versus piecewise-linear margin-based loss.

# 7. Probabilistic Generative Classifiers: Bayes, LDA, QDA, Naive Bayes

**Slide references:** `GEN` pp. 2–11 roadmap, classifier types and posteriors; pp. 12–23 Gaussian class-conditional models, LDA/QDA; pp. 24–28 estimation and high-dimensional covariance issues; pp. 29–43 naive Bayes, smoothing and worked spam example; pp. 44–50 model trade-offs, decision theory, summary and references. **Exam importance: 5/5.**

## 7.1 Three routes to a classifier

**[SLIDE]** The diagram on `GEN` p. 4 compares: (1) a **discriminant function** directly returns a label/score; (2) a **discriminative probabilistic model** directly estimates $P(C_k\mid x)$; (3) a **generative classifier** estimates $p(x\mid C_k)$ and $P(C_k)$ then obtains the posterior by Bayes' rule. The third route models the joint $p(x,C_k)=p(x\mid C_k)P(C_k)$.

$$\boxed{P(C_k\mid x)=\frac{p(x\mid C_k)P(C_k)}{\sum_jp(x\mid C_j)P(C_j)}.}$$

**[SLIDE]** `GEN` p. 6 says the model is "generative" because you can sample a class from its prior and a feature vector from that class's density. The extra model of $p(x)$ enables anomaly scores, handling missing features by marginalization, and synthetic sampling—though actual practicality depends on model quality.

## 7.2 Why the posterior of two classes is sigmoid

**[SLIDE]** `GEN` pp. 9–11 derive and visualize the posterior. Let $a(x)=\log[p(x\mid C_1)P(C_1)]-\log[p(x\mid C_2)P(C_2)]$, the **log posterior odds**. Then

$$\frac{P(C_1\mid x)}{P(C_2\mid x)}=e^{a(x)}.$$

Because the two posterior probabilities sum to 1, rearrange to obtain

$$\boxed{P(C_1\mid x)=\frac1{1+e^{-a(x)}}=\sigma(a(x)).}$$

**Visual:** the two weighted density curves overlap. Where they cross, the posterior of either class is $1/2$, defining the equal-cost, equal-error decision threshold. If class priors differ, the crossing of raw unweighted densities need not be the decision boundary. For $K$ classes, exponentiating class log-joint scores and normalizing gives **softmax**.

## 7.3 Gaussian class-conditionals: LDA vs QDA

**[SLIDE]** `GEN` pp. 12–23 compare Gaussian densities and covariance assumptions. Suppose $x\mid C_k\sim\mathcal N(\mu_k,\Sigma_k)$ with class prior $\pi_k$.

**Linear discriminant analysis (LDA):** assume **shared covariance** $\Sigma_1=\cdots=\Sigma_K=\Sigma$. The per-class log-joint score, after dropping terms common to all classes, is

$$\boxed{\delta_k(x)=x^T\Sigma^{-1}\mu_k-\tfrac12\mu_k^T\Sigma^{-1}\mu_k+\log\pi_k.}$$

**Derivation:** expand $(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)$ inside the Gaussian log-density. The shared quadratic term $-\tfrac12x^T\Sigma^{-1}x$ cancels when comparing classes. Therefore the decision boundary between any pair is **linear in $x$**.

For two classes, $w=\Sigma^{-1}(\mu_1-\mu_2)$ and the intercept follows from the remaining terms and priors. The `GEN` p. 17 **rotated-mean-difference** visual shows that correlated covariance can rotate the decision normal relative to $\mu_1-\mu_2$: it is a **Mahalanobis-weighted difference**, not simply a Euclidean line joining class centres.

**Quadratic discriminant analysis (QDA):** allow each class its own covariance $\Sigma_k$. Discriminant:

$$\boxed{\delta_k(x)=-\tfrac12\log|\Sigma_k|-\tfrac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\log\pi_k.}$$

Quadratic terms in $x$ do not generally cancel, so boundaries are **quadratic**. A curved boundary may fit differently shaped class clouds, but it estimates more parameters and can suffer high variance with small per-class samples. `GEN` p. 22 illustrates the bias–variance trade-off: LDA has more structural assumption, QDA more flexibility. These are about **Gaussian generative classification**, not Fisher's different-but-related LDA projection objective.

## 7.4 MLE for Gaussian generative classifiers

**[SLIDE]** `GEN` pp. 25–27 show that fitting the class model decomposes into counts and averages. With $N_k$ observations of class $k$ among $N$:

$$\hat\pi_k=\frac{N_k}{N},\quad
\hat\mu_k=\frac1{N_k}\sum_{i:y_i=k}x_i,\quad
\hat\Sigma_k=\frac1{N_k}\sum_{i:y_i=k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^T.$$

For LDA, estimate a common covariance by pooling within-class scatter; standard conventions for denominators differ depending on ML versus unbiased sample-covariance objectives. Note the convention explicitly if asked to compute a numeric estimate.

In $d$ dimensions, full covariance needs $d(d+1)/2$ parameters **per class** in QDA. With $d\ge N_k$, empirical covariance estimated after centring is necessarily singular or at best rank deficient; density inversion becomes unreliable. `GEN` p. 28 presents remedies: shared covariances, diagonal matrices, regularization/shrinkage. This ties directly back to the curse of dimensionality.

## 7.5 Naive Bayes conditional independence assumption

**[SLIDE]** `GEN` pp. 29–37 introduce naive Bayes. The condition is not that all features are unconditionally independent; it is that they are **independent given the class**:

$$\boxed{p(x_1,\ldots,x_d\mid C_k)=\prod_{j=1}^d p(x_j\mid C_k).}$$

Prediction compares

$$\hat k=\arg\max_k\left\{\log\pi_k+\sum_{j=1}^d\log p(x_j\mid C_k)\right\}.$$

For binary features, each $p(x_j=1\mid C_k)=\theta_{jk}$ is learned from frequencies. An unrestricted binary feature vector needs up to $2^d-1$ parameters **per class** to describe its full joint distribution; conditional independence reduces this to $d$ Bernoulli parameters per class. This enormous simplification is why naive Bayes can work with little data, although its class probability estimates may be poorly calibrated when dependence is strong.

**Numerical trap:** calculating a product of hundreds of numbers near zero can underflow on a computer. Compute log probabilities and add them. Comparing log scores gives the same class argmax as comparing products because log is increasing.

## 7.6 Laplace smoothing and the uploaded spam example

**[SLIDE]** `GEN` pp. 38–43 work through five training documents with binary indicators for `good`, `bad`, `very`. The classes and counts are:

| Quantity | HAM (2 docs) | SPAM (3 docs) |
|---|---:|---:|
| Prior $\pi_k$ | $2/5$ | $3/5$ |
| `good` presence count | 2 | 0 |
| `bad` presence count | 0 | 3 |
| `very` presence count | 1 | 2 |

Test document has **all three words present**. Plain Bernoulli MLE assigns $P(\mathrm{good}\mid\mathrm{spam})=0$ and $P(\mathrm{bad}\mid\mathrm{ham})=0$, so both likelihood products collapse to zero. The issue is *zero probability due solely to unseen combinations in tiny samples*.

With Beta$(1,1)$ prior on each binary feature parameter, **posterior mean** smoothing estimates

$$\boxed{\hat\theta_{jk}=\frac{N_{jk}+1}{N_k+2}.}$$

This is **not** the MAP mode under Beta$(1,1)$; it is the posterior mean/predictive estimate. More generally with Beta$(\alpha,\beta)$, the posterior mean is $(N_{jk}+\alpha)/(N_k+\alpha+\beta)$.

Smoothed probabilities:

| Word | HAM | SPAM |
|---|---:|---:|
| good | $3/4$ | $1/5$ |
| bad | $1/4$ | $4/5$ |
| very | $1/2$ | $3/5$ |

Unnormalized scores for a document containing all three:

$$S_{ham}=\frac25\cdot\frac34\cdot\frac14\cdot\frac12=0.0375,$$

$$S_{spam}=\frac35\cdot\frac15\cdot\frac45\cdot\frac35=0.0576.$$

Normalize: $P(\mathrm{spam}\mid x)=0.0576/(0.0576+0.0375)\approx0.6057$; $P(\mathrm{ham}\mid x)\approx0.3943$. **Verdict: SPAM.** The slide's 60.6%/39.4% result follows directly. **Do not conflate Bernoulli document presence with multinomial token-frequency modelling**: their denominators and likelihood expressions differ.

## 7.7 When generative versus discriminative modelling helps

**[SLIDE]** `GEN` pp. 44–48 explain trade-offs. A generative classifier models $p(x\mid y)$ plus priors, so it can sample from its model and—in principle—marginalize missing features. But density estimation in high-dimensional $x$ is hard, and a wrong density family creates persistent bias. Discriminative models such as logistic regression can focus directly on $P(y\mid x)$ without estimating the full feature distribution; they may achieve better predictive results when class probabilities are the only goal.

**Never make an absolute claim:** "generative always better with less data" or "discriminative always better given enough data" has exceptions. The right choice depends on assumptions, model capacity, target, sample size and the metrics that matter.

### Exam questions and answers

1. **Derive the sigmoid using Bayes for two classes.** Define log posterior odds $a$, apply algebra, obtain $\sigma(a)$.
2. **Why LDA linear and QDA quadratic?** Expand Gaussian log-densities; shared covariance cancels $x^T\Sigma^{-1}x$, class-specific covariance does not.
3. **Count covariance parameters.** $d(d+1)/2$ per unrestricted covariance; explain sample-size consequences.
4. **Conditional independence vs unconditional independence?** Independence is assumed only after conditioning on the class.
5. **Solve the 5-document spam example.** Estimate priors and smoothed class-conditional probabilities, multiply scores, normalize to 60.6% SPAM.
6. **When does Laplace smoothing address zero-count collapse?** By assigning positive posterior predictive mass to unseen outcomes under a suitable prior.

# 8. Probability Midterm: handwritten worked solutions

**Source:** `MID`, nine scanned handwritten pages. This is a **separate probability review**, not part of the ML slides. Read the hand-drawn support regions as carefully as the equations. **Exam importance: 5/5** for probability integration and transformations.

## 8.1 Problem 1 (pages 1–3): CDF and PDF of a sum

**[SLIDE/SCAN]** The scan states $X,Y$ jointly continuous with

$$f_{X,Y}(x,y)=\frac1x,\qquad 0\le y\le x\le1,$$

and asks about $Z=X+Y$. Outside the triangle, the density is zero. The triangle's sloping upper boundary is $y=x$ and its right boundary is $x=1$.

**Step 1: verify normalization.**

$$\int_0^1\int_0^x\frac1x\,dy\,dx=\int_0^1 1\,dx=1.$$

**Step 2: identify possible $Z$.** Because $0\le y\le x\le1$, smallest sum 0 and largest sum 2, hence $0\le Z\le2$.

**Step 3: write geometric event.** $F_Z(z)=P(X+Y\le z)=P(Y\le z-X)$. Intersect the region **below** the line $y=z-x$ with the triangle $0\le y\le x\le1$. For $0<z<1$, split at $x=z/2$, the intersection of $y=x$ and $y=z-x$:

$$F_Z(z)=\underbrace{\int_0^{z/2}\int_0^x\frac1x\,dy\,dx}_{\text{whole vertical triangle}}
+\underbrace{\int_{z/2}^z\int_0^{z-x}\frac1x\,dy\,dx}_{\text{slice capped by sum line}}.$$

Evaluate first piece $=z/2$. Evaluate second:

$$\int_{z/2}^{z}\frac{z-x}{x}\,dx
=\left[z\ln x-x\right]_{z/2}^{z}
=z\ln2-\frac z2.$$

Therefore $\boxed{F_Z(z)=z\ln2}$ for $0<z\le1$.

**Step 4: for $1<z<2$, the slanted line exits through $x=1$.** Now the first part ends at $x=z/2$ and the remainder extends to the triangle's right edge:

$$F_Z(z)=\int_0^{z/2}\int_0^x\frac1x\,dy\,dx
+\int_{z/2}^1\int_0^{z-x}\frac1x\,dy\,dx.$$

The first term is $z/2$; the second is $-z\ln(z/2)-1+z/2$. Thus $F_Z(z)=z-1+z\ln(2/z)$ for $1<z<2$. Piecewise answer:

$$\boxed{F_Z(z)=\begin{cases}
0,&z\le0,\\
z\ln2,&0<z\le1,\\
z-1+z\ln(2/z),&1<z<2,\\
1,&z\ge2.
\end{cases}}$$

Differentiate to get PDF:

$$\boxed{f_Z(z)=\begin{cases}
\ln2,&0<z<1,\\
\ln(2/z),&1<z<2,\\
0,&\text{otherwise.}
\end{cases}}$$

At $z=1$, the two expressions agree: $\ln2$. Check $\int_0^2f_Z(z)dz=1$. **High-scoring presentation:** sketch support, label intersection $(z/2,z/2)$, split $z$ into two cases, show each integral, differentiate once, check endpoint values.

## 8.2 Problem 2 (pages 4–5): two exponential variables and a transform

**[SCAN]** The pages consider independent Exponential$(\lambda)$ variables, their sum and a ratio-type transformation. In the standard version, let $Z=X+Y$ and $W=X/(X+Y)$. The following is the complete **[BACKGROUND] derivation** that supplies the missing details in the handwritten working.

For $x,y\ge0$, $f_{X,Y}(x,y)=\lambda^2e^{-\lambda(x+y)}$. Invert: $x=wz$, $y=(1-w)z$, so $z>0$ and $0<w<1$. Jacobian:

$$\left|\frac{\partial(x,y)}{\partial(z,w)}\right|
=\left|\begin{matrix}w&z\\1-w&-z\end{matrix}\right|=z.$$

Thus $f_{Z,W}(z,w)=\lambda^2z e^{-\lambda z}$ on $z>0,0<w<1$. It factors into $(\lambda^2z e^{-\lambda z})\cdot1$, so $Z$ and $W$ are independent, $Z\sim\operatorname{Gamma}(\text{shape}=2,\text{rate}=\lambda)$ and $W\sim\operatorname{Uniform}(0,1)$. The key exam skills are inverting a transformation, identifying the new support and computing an absolute Jacobian determinant.

## 8.3 Problem 5 (pages 6–7): normalization and CDF

The scanned notes show a random variable $Y$ with $f_Y(y)=K/y^3$ on $y\ge2$. Normalize:

$$1=\int_2^\infty\frac K{y^3}\,dy=K\left[\frac{-1}{2y^2}\right]_2^\infty=\frac K8,$$

so $K=8$. For $y\ge2$,

$$F_Y(y)=\int_2^y\frac8{t^3}\,dt=1-\frac4{y^2}.$$

Including the domain: $F_Y(y)=0$ for $y<2$, and $1-4/y^2$ for $y\ge2$. Check continuity at 2 and limit 1 at infinity. **Common mistake:** integrating from zero even though the density is zero below 2.

## 8.4 Problem 6 (pages 8–9): shifting inequalities and symmetry

The scan considers a symmetric continuous density $f_X(x)=f_X(-x)$ and an integral of a shifted interval, of the form $P(|X-t|\le2)$. First translate the absolute inequality:

$$|X-t|\le2 \quad\Longleftrightarrow\quad t-2\le X\le t+2.$$

Then $P(|X-t|\le2)=\int_{t-2}^{t+2}f_X(x)\,dx=F_X(t+2)-F_X(t-2)$. If $X$ is symmetric about zero, $F_X(-u)=1-F_X(u)$ where the CDF is continuous. Use this only after checking the appropriate signs of bounds; don't apply it to an arbitrary nonsymmetric distribution.

# 9. Concept map and cross-chapter connections

**The course is a network, not independent algorithms:**

```text
Probability (PMF/PDF, independence, conditional probability)
  |
  +--> Likelihood and maximum likelihood (MLE)
  |      |
  |      +--> Bernoulli likelihood --> Logistic regression --> Cross-entropy
  |      +--> Gaussian likelihood --> Linear regression --> Least squares
  |      +--> Poisson likelihood --> Poisson regression --> Log link
  |
  +--> Bayes theorem --> Prior + likelihood --> Posterior
         |                         |
         |                         +--> MAP --> regularization/ridge
         +--> Generative classifiers
                 |
                 +--> Gaussian p(x|class) --> LDA / QDA
                 +--> conditional independence --> Naive Bayes

Generalization --> model capacity --> bias / variance --> validation and CV
  |                                    |
  +--> dimensionality reduction ------+
  |      +--> feature selection
  |      +--> SVD and latent semantics / PCA
  |
  +--> non-parametric learning --> KDE --> kNN, choose K
  |
  +--> discriminant functions --> SVM --> margin --> dual / kernels
```

**Six connections worth explaining orally:**

1. **MLE to least squares:** Gaussian observation noise makes negative log-likelihood proportional to squared residual error.
2. **MLE to logistic loss:** Bernoulli output probability with sigmoid makes negative conditional log-likelihood equal binary cross-entropy.
3. **MAP to ridge:** a zero-centred Gaussian prior on coefficients adds an $L_2$ penalty to the optimization objective.
4. **LDA to logistic shape:** Gaussian class-conditional densities with a shared covariance yield **linear log posterior odds**, hence a sigmoid for two classes.
5. **Curse of dimensionality to kNN:** fewer informative local neighbours and costly distances make high-dimensional nearest-neighbour learning difficult; scaling and feature extraction help.
6. **SVD to LSA:** truncated singular directions compress a term-document matrix into latent components that can align related word usage.

# 10. One-day final revision plan

| Time | Focus | Required output without looking at notes |
|---|---|---|
| 45 min | PMF/PDF/CDF, joint/marginal/conditional, Bayes | Draw support and evaluate at least one integral |
| 75 min | Bernoulli, Poisson, Gaussian MLE; MAP | Derive three MLEs and Beta-Bernoulli MAP |
| 60 min | Linear regression, ridge/lasso | Derive normal equations and ridge objective |
| 45 min | Logistic regression | Derive sigmoid loss and gradient |
| 45 min | Overfitting and model selection | Explain bias–variance and train/val/test |
| 60 min | SVD and dimensionality | Explain $A=U\Sigma V^T$, rank-$k$ error, forward selection |
| 45 min | kNN | Compute a neighbour example, choose K and metric |
| 75 min | SVM | Derive margin, primal, stationarity and dual |
| 60 min | Generative models | Derive LDA boundary, QDA distinction, spam smoothing |
| 45 min | Probability midterm and mixed numerical practice | Derive $F_{X+Y}$ by drawing the support |

This is about **9 hours 15 minutes of active work** excluding breaks; condense the first pass if only a few hours remain. Prioritize derivations and problems you cannot reproduce unaided.

# 11. Formula cheat sheet

**Probability**

$$F_X(x)=P(X\le x),\quad f_X(x)=F'_X(x),\quad f_X(x)=\int f_{X,Y}(x,y)dy,\quad p(\theta\mid D)\propto p(D\mid\theta)p(\theta).$$

**MLE / MAP**

$$\hat\theta_{ML}=\arg\max_\theta\sum_i\log p(x_i\mid\theta),\quad
\hat\theta_{MAP}=\arg\max_\theta\big(\log p(D\mid\theta)+\log p(\theta)\big).$$

$$\hat p_{Bern}=\frac sn,\quad\hat\lambda_{Pois}=\bar x,\quad\hat\mu_{Gauss}=\bar x,\quad\hat\sigma^2_{Gauss,ML}=\frac1n\sum_i(x_i-\bar x)^2.$$

$$\hat p_{Beta-Bern,MAP}=\frac{s+\alpha-1}{n+\alpha+\beta-2}\ \text{(interior mode only)}.$$

**Linear / logistic / regularization**

$$\hat w_{OLS}=(X^TX)^{-1}X^Ty,\quad \hat w_{ridge}=(X^TX+\lambda I)^{-1}X^Ty.$$

$$\sigma(z)=\frac1{1+e^{-z}},\quad \log\frac p{1-p}=w^Tx+b,\quad \nabla_wJ_{logistic}=X^T(p-y).$$

$$J_{ridge}=\frac12\|Xw-y\|^2+\frac\lambda2\|w\|_2^2,\quad J_{lasso}=\frac12\|Xw-y\|^2+\lambda\|w\|_1.$$

**Dimensionality and neighbours**

$$\|x-z\|_2=\sqrt{\sum_j(x_j-z_j)^2},\quad A=U\Sigma V^T,\quad A_k=U_k\Sigma_kV_k^T,\quad\|A-A_k\|_F^2=\sum_{j>k}\sigma_j^2.$$

**SVM**

$$f(x)=\operatorname{sign}(w^Tx+b),\quad\text{margin width}=2/\|w\|,\quad \min_{w,b,\xi}\ \frac12\|w\|^2+C\sum_i\xi_i.$$

$$y_i(w^Tx_i+b)\ge1-\xi_i,\quad\xi_i\ge0,\quad 0\le\alpha_i\le C,\quad\sum_i\alpha_i y_i=0.$$

$$\max_\alpha\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j).$$

**Generative**

$$P(C_k\mid x)\propto p(x\mid C_k)P(C_k),\quad p(x\mid C_k)=\prod_jp(x_j\mid C_k)\ \text{(naive Bayes)}.$$

$$\delta_k^{LDA}=x^T\Sigma^{-1}\mu_k-\frac12\mu_k^T\Sigma^{-1}\mu_k+\log\pi_k.$$

$$\delta_k^{QDA}=-\frac12\log|\Sigma_k|-\frac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\log\pi_k.$$

# 12. Confidence test: 15 mixed examination prompts

Attempt all questions before reading the key. Here, **the exam importance ratings express a reasoned revision priority**, not verified past-paper frequency.

1. Define generalization and provide an example of overfitting.
2. Write the difference between a PMF and PDF, including how each gives interval probability.
3. Derive MLE for Bernoulli parameter from first principles.
4. A Poisson process produced $2,3,4,6,5$ calls per minute. Estimate its rate and the expected next count.
5. Derive Beta-Bernoulli MAP when $n=10,s=2,\alpha=\beta=3$.
6. Given Gaussian noise, derive least squares from negative log-likelihood.
7. Derive $\nabla_w\frac12\|Xw-y\|^2$ and state the normal equations.
8. Explain why ridge is MAP with a Gaussian prior.
9. Why may a degree-9 fit to 10 points give huge coefficients?
10. Derive the logistic sigmoid gradient from binary cross-entropy.
11. Describe how cross-validation chooses $K$ for kNN without leaking test information.
12. State SVD and explain why truncating small singular values yields the optimal rank-$k$ approximation in Frobenius norm.
13. Derive the SVM geometric margin $2/\|w\|$; formulate soft-margin SVM.
14. Show why LDA is linear but QDA is generally quadratic.
15. Solve the smoothed `good/bad/very` spam classification from the lecture.

## Answer key

1. Unseen-data performance; degree-12 model memorizes noise.
2. PMF assigns point probability to discrete values; PDF integrates area for continuous probabilities.
3. $\ell=s\log p+(n-s)\log(1-p)$, derivative zero gives $s/n$ (boundary cases separately).
4. Mean $(2+3+4+6+5)/5=4$ calls/minute; expected next count 4.
5. $(s+\alpha-1)/(n+\alpha+\beta-2)=(2+3-1)/(10+3+3-2)=4/14=2/7$.
6. $-\log p(y\mid X,w)=\mathrm{SSE}/(2\sigma^2)+\mathrm{const}$.
7. $X^T(Xw-y)$; set $X^TXw=X^Ty$.
8. Negative log Gaussian prior contributes a term proportional to $\|w\|_2^2$.
9. Ill-conditioned Vandermonde/interpolation, sensitivity to noise, high complexity.
10. $\partial J_i/\partial z_i=p_i-y_i$, then $\nabla_wJ=X^T(p-y)$.
11. Search candidate $K$ using validation folds within training set; test only once afterwards.
12. $A=U\Sigma V^T$; $A_k=U_k\Sigma_kV_k^T$; residual squared error is $\sum_{j>k}\sigma_j^2$.
13. Parallel supporting planes have distance $2/\|w\|$; minimize $\tfrac12\|w\|^2+C\sum\xi_i$ with margin and nonnegative slack constraints.
14. Common covariance cancels quadratic log-density term; distinct covariance matrices leave quadratic terms.
15. Smoothed scores 0.0576 spam, 0.0375 ham; posterior spam $=60.6\%$.


# 13. Quick-check MCQs with solutions

These are **newly generated practice questions**, not claimed to be past IIT Hyderabad exam questions.

**Q1.** In a continuous distribution, $P(X=2)$ is usually: (A) $f_X(2)$ (B) 1 (C) 0 (D) $F_X(2)$. **Answer C** for an absolutely continuous distribution; density height is not point probability.

**Q2.** Which value is the Bernoulli MLE if seven successes occur in ten independent trials? (A) 0.5 (B) 0.7 (C) 7 (D) 0.3. **Answer B**; sample proportion.

**Q3.** A Beta prior with $\alpha=\beta=1$ gives, for an interior Bernoulli mode: (A) always 0.5 (B) MAP equals MLE (C) MAP equals posterior mean (D) no posterior. **Answer B**; mode equals likelihood maximum under a constant prior (boundaries considered separately).

**Q4.** Gaussian regression with fixed variance and identity mean link gives which loss up to constants? (A) hinge (B) square error (C) 0-1 (D) absolute error. **Answer B**.

**Q5.** A zero-centred Gaussian prior on $w$ produces which standard MAP penalty? (A) $L_0$ (B) $L_1$ (C) $L_2$ squared (D) entropy. **Answer C**.

**Q6.** If unregularized logistic classification data are strictly linearly separable, what can happen? (A) weights diverge in norm (B) test accuracy guaranteed 100% (C) sigmoid vanishes (D) weights must become exactly zero. **Answer A**.

**Q7.** In a $d$-dimensional grid with 8 bins/axis, the number of cells is (A) $8d$ (B) $8^d$ (C) $d^8$ (D) $d/8$. **Answer B**.

**Q8.** $A=U\Sigma V^T$ and $A_k$ keeps leading $k$ singular values. Which is correct? (A) $A_k$ always equals $A$ (B) $A_k$ is a best rank-$k$ approximation in Frobenius norm (C) $k$ must equal sample count (D) singular values may be negative. **Answer B**.

**Q9.** Increasing $K$ in kNN generally causes: (A) more jagged predictions (B) smoother predictions (C) lower training error always (D) automatic feature scaling. **Answer B**.

**Q10.** In standard SVM primal, the full gap between canonical support hyperplanes is (A) $\|w\|$ (B) $1/\|w\|$ (C) $2/\|w\|$ (D) $2\|w\|$. **Answer C**.

**Q11.** Gaussian class-conditional classifiers with shared covariance matrices lead to: (A) necessarily quadratic decision boundaries (B) linear boundaries (C) no posterior (D) only kNN. **Answer B**.

**Q12.** What assumption makes Naive Bayes "naive"? (A) classes equally likely (B) features independent unconditionally (C) features conditionally independent given class (D) classes normally distributed. **Answer C**.

**Scoring interpretation:** 10–12 correct: strong recognition, now practise derivations. 7–9 correct: review 2–3 weakest chapters. Below 7: return to mathematical prerequisites and worked examples. MCQs check recognition, not mastery of proofs.

# Appendix A. Slide coverage register (every teaching page)

This register provides a **full slide/page locator**. The preceding chapters expand related pages as topic clusters; animation-build variants may repeat the same slide title. Source titles are lightly shortened where extraction includes decorative text. `PDF p.` and `PPT slide` use actual file positions, not printed footer numbering.

## INTRO source page index

- **PDF p. 01:** Introduction to Machine Learning / Dr. Srijith P K
- **PDF p. 02:** Machine learning / • Machine learning has become
- **PDF p. 03:** Why machine learning : Data every where ! / • This deluge of data calls for automated methods of data analysis, which is what
- **PDF p. 04:** Machine learning is interdisciplinary
- **PDF p. 05:** A Brief History of Machine Learning
- **PDF p. 06:** Machine Learning, Neural Networks : / The Beginning
- **PDF p. 07:** AI over optimism and Winter / The overpromise
- **PDF p. 08:** Return of the Neural Networks and Learning / Why it mattered
- **PDF p. 09:** Machine Learning Dominance / 1990-present : Rise of Machine Learning
- **PDF p. 10:** Overview of / Machine learning
- **PDF p. 11:** Supervised / learning
- **PDF p. 12:** Supervised learning (Classification )
- **PDF p. 13:** Input and output y a real value. Learn a function / Input
- **PDF p. 14:** Training and Testing ML models / ML models should be able to predict well (generalization ability)
- **PDF p. 15:** Generalization ability and / overfitting
- **PDF p. 16:** ML in practice
- **PDF p. 17:** Machine learning / algorithms
- **PDF p. 18:** Supervised Learning Models / PROBABILISTIC vs NON-PROBABILISTIC
- **PDF p. 19:** Unsupervised Learning Models / PROBABILISTIC vs NON-PROBABILISTIC
- **PDF p. 20:** Brief Syllabus : Foundations of ML / • Overview of machine learning
- **PDF p. 21:** Foundations of Machine learning / Main references
- **PDF p. 22:** Foundations of ML / • Assume mathematical and
- **PDF p. 23:** Foundations of ML : Evaluation / Tentative Evaluation scheme

## EST source page index

- **PDF p. 01:** Foundations of ML / Probabilistic Models
- **PDF p. 02:** Sampling and Estimation / • What you have learnt : X is a random
- **PDF p. 03:** Sampling and Estimation / • Given data, Statistics/ML aim to learn parameters of the underlying distribution from data
- **PDF p. 04:** Parameter Estimation / • Data points representing the
- **PDF p. 05:** Parameter Estimation / • Data points representing the
- **PDF p. 06:** Modelling and Estimation / • Given Data (x1, . . . , xn)
- **PDF p. 07:** Parameter Estimation / • Any statistic used to estimate the value of an unknown parameter θ is called an
- **PDF p. 08:** Parameter Estimation / • Any statistic used to estimate the value of an unknown parameter θ is called an
- **PDF p. 09:** which of the following would maximize the probability of observing the data / Mean = 100, SD = 10
- **PDF p. 10:** Parameter estimation / which of the following would maximize the probability of observing the data
- **PDF p. 11:** Parameter Estimation / Maximum likelihood estimator
- **PDF p. 12:** Bernoulli Distribution / • Data points representing the
- **PDF p. 13:** Maximum likelihood estimation / • (Maximum Likelihood Estimator of a Bernoulli Parameter) Suppose you
- **PDF p. 14:** Maximum likelihood estimation / • (Maximum Likelihood Estimator of a Bernoulli Parameter) Suppose you
- **PDF p. 15:** Maximum likelihood estimation / • (Maximum Likelihood Estimator of a Bernoulli Parameter) Suppose you
- **PDF p. 16:** Maximum likelihood estimation / • (Maximum Likelihood Estimator of a Bernoulli Parameter) Suppose you
- **PDF p. 17:** Maximum likelihood estimation / • (Maximum Likelihood Estimator of a Bernoulli Parameter) Suppose you
- **PDF p. 18:** Bernoulli Distribution / • Data points representing the if it
- **PDF p. 19:** Maximum likelihood estimation and / loss minimization
- **PDF p. 20:** Information Theory / • Maximizing the log-likelihood is equivalent to
- **PDF p. 21:** Parameter Estimation / Suppose that each RAM (random access memory) chip produced by
- **PDF p. 22:** Modelling and Estimation / Suppose a call center logs the number of calls received in each of 8
- **PDF p. 23:** Maximum likelihood estimation / (Maximum Likelihood Estimator of a Poisson Parameter) Suppose X1, . . . ,
- **PDF p. 24:** Maximum likelihood estimation / (Maximum Likelihood Estimator of a Poisson Parameter) Suppose X1, . . . ,
- **PDF p. 25:** Maximum likelihood estimation / (Maximum Likelihood Estimator of a Poisson Parameter) Suppose X1, . . . ,
- **PDF p. 26:** Modelling and Estimation / The data: weights (kg) of 15 students in a class:
- **PDF p. 27:** Maximum likelihood / estimation
- **PDF p. 28:** Beyond Parametric Density Estimation / Two ways to answer the same question: given samples x₁ … xₙ, what is p(x)?
- **PDF p. 29:** The Histogram — the Original Non-parametric / Estimator
- **PDF p. 30:** Kernel Density Estimation (Parzen Windows) / Stop asking which bin a point falls in. Centre a small bump on every point instead.
- **PDF p. 31:** Bayes Theorem / Bayes Theorem
- **PDF p. 32:** Bayes Theorem in ML
- **PDF p. 33:** Seek that value for θ which maximizes the posterior prob(θ / X) / Maximum Likelihood Estimation
- **PDF p. 34:** Seek that value for θ which maximizes the posterior prob(θ / X) / Maximum Likelihood Estimation
- **PDF p. 35:** What Does the MAP Estimate Get / Us That the ML Estimate Does NOT
- **PDF p. 36:** MAP Estimation Example / [Election]
- **PDF p. 37:** MAP Estimate : Prior Belief on p / • The prior for p must be zero outside the [0, 1] interval.
- **PDF p. 38:** MAP Estimate : Prior Belief on p / • The prior for p must be zero outside the [0, 1] interval.
- **PDF p. 39:** Example [Indian Election] / • Due to some emerging situations, lets assume that voters
- **PDF p. 40:** MAP Estimation [Indian Election] / With N = 20 and with 12 of the 20 previously
- **PDF p. 41:** Seek that value for θ which maximizes the posterior prob(θ / X) / Maximum A-Posteriori (MAP) Estimation
- **PDF p. 42:** MAP Estimation of a Gaussian Mean / SETUP
- **PDF p. 43:** MAP Estimation of a Gaussian Mean / SETUP
- **PDF p. 44:** MAP Estimation of a Gaussian Mean / SETUP
- **PDF p. 45:** Bayesian Inference / Treat parameters (θ) as a random variable and define a distribution over it.
- **PDF p. 46:** Bayesian Inference: One Dry Day / It did not rain today. What should we now believe about q, the chance of rain on any day?
- **PDF p. 47:** Bayesian Inference: One Dry Day / It did not rain today. What should we now believe about q, the chance of rain on any day?
- **PDF p. 48:** Three Ways to Answer — Only One Keeps the / Doubt
- **PDF p. 49:** Modelling Multivariate observations
- **PDF p. 50:** height_c / weight
- **PDF p. 51:** A Multivariate Gaussian on Height and Weight / THE MODEL
- **PDF p. 52:** A Multivariate Gaussian on Height and Weight / THE MODEL
- **PDF p. 53:** The Data and Its Distributions / One picture holds the two marginals and the joint — and already hints that something is off.
- **PDF p. 54:** The Data and Its Distributions / One picture holds the two marginals and the joint — and already hints that something is off.
- **PDF p. 55:** What Can You Ask of This Model? / PROBLEM
- **PDF p. 56:** The Same Trick for Regression: Joint Modelling or / Generative Modelling
- **PDF p. 57:** The Same Trick for Regression: Joint Modelling or / Generative Modelling
- **PDF p. 58:** Relook at the Regression Problem / Can I improve my prediction by considering more features ?
- **PDF p. 59:** Real Covariates Have No Sensible Gaussian / This is the decisive practical objection, and it is not a matter of approximation quality.
- **PDF p. 60:** Reasons to discard joint Modelling (Generative Modelling) / NON-LINEAR MEANS
- **PDF p. 61:** One Move: Let the Parameter Depend on the Covariates / We have been estimating a single parameter for the whole sample. But no two hous
- **PDF p. 62:** One Move: Let the Parameter Depend on the / Covariates
- **PDF p. 63:** Gaussian + Identity Link = Linear Regression / Predicting the price of a house: y is real-valued, so a Gaussian on y is the natural cho
- **PDF p. 64:** Gaussian + Identity Link = Linear Regression / Predicting the price of a house: y is real-valued, so a Gaussian on y is the natural cho

## REG source page index

- **PDF p. 01:** Supervised Learning : Linear Regression and / Logistic Regression
- **PDF p. 02:** Supervised learning / https://www.datacamp.com/community/tutorials/linear-regression-R
- **PDF p. 03:** Supervised learning / https://medium.com/@heyozramos/regression-vs-classification-86d73c281c5e
- **PDF p. 04:** Supervised learning : / Regression
- **PDF p. 05:** A historical look at / regression
- **PDF p. 06:** Airquality data. / • Data set has various air quality parameters in New York city.
- **PDF p. 07:** Airquality data / https://www.edvancer.in/step-step-guide-to-execute-linear-regression-r/
- **PDF p. 08:** Linear regression / • Temp=w1.Solar.R +w2.Ozone + w3.Wind + error.
- **PDF p. 09:** One Move: Let the Parameter Depend on the / Covariates
- **PDF p. 10:** Linear Regression / • Observations need not lie on a line
- **PDF p. 11:** Linear regression / Learn a function which maps input to output f : X -> Y
- **PDF p. 12:** Linear regression / Learn a function which maps input to output f : X -> Y
- **PDF p. 13:** From a line to a hyperplane / • A hyperplane is a flat surface with
- **PDF p. 14:** Polynomial Regression - / Example
- **PDF p. 15:** Polynomial Regression - / curve fitting
- **PDF p. 16:** Linear Regression : Parameter Estimation / THE MODEL
- **PDF p. 17:** Linear Regression - Learning Parameters / Learn the function which minimises
- **PDF p. 18:** Linear Regression - Learning Parameters / Learn the function which passes
- **PDF p. 19:** Least squares: deriving the gradient step by step / Start from the objective
- **PDF p. 20:** Linear Regression - Learning Parameters / Learn the function which passes
- **PDF p. 21:** Linear Regression - Learning Parameters / Learn the function which passes through as many points as possible
- **PDF p. 22:** Polynomial Regression - Parameter / estimation
- **PDF p. 23:** Least squares: geometric interpretation / Φ =
- **PDF p. 24:** Question / write down three equations for the line y = mx+c to go through y =
- **PDF p. 25:** Question / write down three equations for the line y = mx+c to go through
- **PDF p. 26:** Polynomial regression – Model Selection / • Model selection (choosing M) : higher order polynomial (M = 9),
- **PDF p. 27:** Polynomial regression Overfitting / • Model selection (choosing M) : higher order polynomial (M = 9),
- **PDF p. 28:** Polynomial Regression - Parameter / estimation
- **PDF p. 29:** Polynomial regression - Model Selection / • Generalization performance : root mean square error on test data
- **PDF p. 30:** Polynomial regression - Model Selection / • Given model complexity, the over-fitting problem become less
- **PDF p. 31:** Maximum Aposteriori Estimatation / ●Compute Maximum aposteriori (MAP) estimate
- **PDF p. 32:** Maximum Aposteriori Estimatation / ●Compute Maximum aposteriori (MAP) estimate
- **PDF p. 33:** EMPIRICAL DEMONSTRATION / Same Data, Same Model — Different Objective
- **PDF p. 34:** MLE vs. MAP at a Glance / MLE
- **PDF p. 35:** Polynomial regression - regularization / • Add a penalty term to the error function (1.2) in order to
- **PDF p. 36:** Polynomial regression - regularization
- **PDF p. 37:** Regularized Least Squares / • Parameter shrinkage, weight decay
- **PDF p. 38:** Regularized Least Squares / • Parameter shrinkage, weight decay
- **PDF p. 39:** Model Selection / Limiting the number of basis functions in order to avoid over-fitting
- **PDF p. 40:** 1. Training set is a set of examples used for learning a model parameters / (e.g., weight vector w in linear regression
- **PDF p. 41:** Estimating Generalization Error / • Testing Error
- **PDF p. 42:** Use of Validation and Test Sets / • If we want unbiased estimates of accuracy during the learning process:
- **PDF p. 43:** Cross Validation / https://scikit-learn.org/stable/modules/cross_validation.html
- **PDF p. 44:** Linear Regression / • Parameter Estimation
- **PDF p. 45:** Supervised learning : / Regression
- **PDF p. 46:** Poisson Regression / • Poisson distribution : Model number of events occurring in a fixed
- **PDF p. 47:** Poisson Regression : / Learning parameters
- **PDF p. 48:** Bias Variance Decomposition / Given the actual conditional distribution p(t / x), and p(x,t) = p(t / x)
- **PDF p. 49:** Bias- Variance Decomposition / Bias Error
- **PDF p. 50:** Bias- Variance Decomposition / Variance Error
- **PDF p. 51:** Bias Variance Decomposition / L = 100 data sets, each having N = 25 data points,
- **PDF p. 52:** Bias Variance Decomposition / L = 100 data sets,
- **PDF p. 53:** Bayesian Linear Regression
- **PDF p. 54:** Bias Variance Tradeoff / Dimensionality reduction and feature selection can decrease variance by
- **PDF p. 55:** Supervised Learning : Classification / • Binary classification : y = {0,1}
- **PDF p. 56:** Classification : Linear Models
- **PDF p. 57:** Linear regression to Logistic regression / • Y takes value 0 or 1
- **PDF p. 58:** Least squares classification / • Loss function
- **PDF p. 59:** Logistic Regression / • Let X be the data instance, and Y be the class label {0,1} :
- **PDF p. 60:** Logistic Regression / • Let X be the data instance, and Y be the class label:
- **PDF p. 61:** Logistic Regression / P(y / X)
- **PDF p. 62:** two points XA and XB both of which lie on the / decision surface.
- **PDF p. 63:** • In logistic regression, we learn the conditional distribution P(y / x) / • Let py(x;w) be our estimate of P(y / x), where w is a vect
- **PDF p. 64:** • Conditional data likelihood - Probability of observed Y values in / the training data, conditioned on corresponding X values.
- **PDF p. 65:** • Equivalently, we can work with log of conditional likelihood: / • Conditional data log likelihood, l(W), can be written as
- **PDF p. 66:** • We need to estimate: / • Equivalently, we can minimize negative log likelihood using gradient
- **PDF p. 67:** • Overﬁtting can arise especially when data has very high / dimensions and is sparse.
- **PDF p. 68:** Multi Class Classification / • One vs Rest
- **PDF p. 69:** • LR: Functional form of P(Y / X), no assumption on P(X / Y) / • LR is a linear classifier
- **PDF p. 70:** Probabilistic Generative Models
- **PDF p. 71:** Probabilistic Generative Models
- **PDF p. 72:** Probabilistic Generative Models
- **PDF p. 73:** Classification : Evaluation metrics / Accuracy :
- **PDF p. 74:** Thank you ! / Reference

## DIM source page index

- **PPT slide 01:** Dimensionality Reduction
- **PPT slide 02:** Supervised Learning: Some Issues
- **PPT slide 03:** Motivating Example
- **PPT slide 04:** Supervised Learning
- **PPT slide 05:** Classification
- **PPT slide 06:** [visual / animation continuation]
- **PPT slide 07:** [visual / animation continuation]
- **PPT slide 08:** [visual / animation continuation]
- **PPT slide 09:** f(humidity, pressure) f(93, 999.7) = Rain f(49, 1015.5) = No Rain f(79, 1031.1) = No Rain h(humidity,pressure)
- **PPT slide 10:** [visual / animation continuation]
- **PPT slide 11:** nearest-neighbor classification algorithm that, given an input, chooses the class of the nearest data point to that input
- **PPT slide 12:** [visual / animation continuation]
- **PPT slide 13:** [visual / animation continuation]
- **PPT slide 14:** [visual / animation continuation]
- **PPT slide 15:** k-nearest-neighbor classification algorithm that, given an input, chooses the most common class out of the k nearest data points to tha
- **PPT slide 16:** [visual / animation continuation]
- **PPT slide 17:** [visual / animation continuation]
- **PPT slide 18:** [visual / animation continuation]
- **PPT slide 19:** Supervised Learning: Some Issues
- **PPT slide 20:** Curse of Dimensionality: Sparsity
- **PPT slide 21:** Curse of Dimensionality: Sparsity
- **PPT slide 22:** … … / Reduce Data from 2D to 1D
- **PPT slide 23:** Data Visualization
- **PPT slide 24:** Data Visualization
- **PPT slide 25:** Data Visualization
- **PPT slide 26:** Dimensionality: Overfitting
- **PPT slide 27:** Dimensionality Reduction Methods
- **PPT slide 28:** Feature Selection
- **PPT slide 29:** Subset Selection
- **PPT slide 30:** Forward Selection (Wrapper Method)
- **PPT slide 31:** Backward Selection
- **PPT slide 32:** Feature Extraction Methods
- **PPT slide 33:** Use Case: Information Retrieval
- **PPT slide 34:** Problems with VSM
- **PPT slide 35:** The Problem
- **PPT slide 36:** Motivating Example
- **PPT slide 37:** Motivating Example
- **PPT slide 38:** LSA: Motivating Example
- **PPT slide 39:** Bit of Linear Algebra: Rank
- **PPT slide 40:** Bit of Linear Algebra: Rank
- **PPT slide 41:** Rank Leads to Dimensionality
- **PPT slide 42:** Matrix Factorization
- **PPT slide 43:** First Step towards LSA
- **PPT slide 44:** Criteria for Approximation?
- **PPT slide 45:** Eigenvectors and Eigenvalues
- **PPT slide 46:** Eigenvalue and Eigenvector
- **PPT slide 47:** Eigenvector Decomposition
- **PPT slide 48:** Singular Value Decomposition (SVD)
- **PPT slide 49:** Singular Value Decomposition
- **PPT slide 50:** Singular Value Decomposition (SVD)
- **PPT slide 51:** Singular Value Decomposition (SVD)
- **PPT slide 52:** Dimensionality Reduction
- **PPT slide 53:** Dimensionality Reduction
- **PPT slide 54:** Singular Value Decomposition (SVD)
- **PPT slide 55:** Singular Value Decomposition (SVD)
- **PPT slide 56:** Singular Value Decomposition (SVD)
- **PPT slide 57:** Low Rank Approximation Problem
- **PPT slide 58:** Low Rank Approximation from SVD
- **PPT slide 59:** What did we do?
- **PPT slide 60:** Term-Document Matrix
- **PPT slide 61:** Singular Value Decomposition
- **PPT slide 62:** Singular Vector Decomposition
- **PPT slide 63:** SVD: Summary
- **PPT slide 64:** Introduction to Information Retrieval Christopher D. Manning Prabhakar Raghavan Hinrich Schütze Chapter 6 and 18

## KNN source page index

- **PDF p. 01:** Non-parametric Models
- **PDF p. 02:** Parametric or not?
- **PDF p. 03:** One dataset, two philosophies
- **PDF p. 04:** One dataset, two philosophies
- **PDF p. 05:** Parametric models
- **PDF p. 06:** Non-parametric models
- **PDF p. 07:** What changes when more data arrives
- **PDF p. 08:** A name that misleads
- **PDF p. 09:** Where the complexity knob sits
- **PDF p. 10:** The two families, side by side
- **PDF p. 11:** Density estimation / Estimating p(x) from a sample without assuming the shape of p
- **PDF p. 12:** Smooth kernels
- **PDF p. 13:** k-nearest-neighbour density estimation
- **PDF p. 14:** From density estimation to classification
- **PDF p. 15:** From density estimation to classification
- **PDF p. 16:** k nearest neighbours / The classifier that falls out of the density estimate
- **PDF p. 17:** The idea
- **PDF p. 18:** Majority vote
- **PDF p. 19:** The same rule, three uses
- **PDF p. 20:** Measuring nearness
- **PDF p. 21:** When Euclidean distance is the wrong question
- **PDF p. 22:** Choosing K
- **PDF p. 23:** CROS S-VA L IDA T ION , N OT IN TU IT ION
- **PDF p. 24:** The curse of dimensionality : Distances stop
- **PDF p. 25:** Strengths and weaknesses
- **PDF p. 26:** Improving kNN · 1
- **PDF p. 27:** Improving kNN · 2
- **PDF p. 28:** Improving kNN · 3
- **PDF p. 29:** The lecture in one picture
- **PDF p. 30:** Readings

## SVM source page index

- **PDF p. 01:** Support Vector Machines
- **PDF p. 02:** SVM: Overview and History / • A discriminant function to do classification
- **PDF p. 03:** SVM: Overview and History / • Associated key words
- **PDF p. 04:** Linear Classifiers / denotes +1
- **PDF p. 05:** Linear Classifiers / denotes +1
- **PDF p. 06:** Linear Classifiers / denotes +1
- **PDF p. 07:** Linear Classifiers / denotes +1
- **PDF p. 08:** Linear Classifiers / denotes +1
- **PDF p. 09:** Linear Classifiers / denotes +1
- **PDF p. 10:** Maximum Margin Classifier / denotes +1
- **PDF p. 11:** Why Maximum Margin? / denotes +1
- **PDF p. 12:** Estimating the Margin / • What is the distance expression for a point x to a line wx+b= 0?
- **PDF p. 13:** Estimating the Margin / • Distance from example to the separator is
- **PDF p. 14:** Maximum Margin Formulation 1 / • Distance from example to the separator is
- **PDF p. 15:** Maximum Margin Formulation 1 / • Distance from example to the separator is
- **PDF p. 16:** Maximum Margin Formulation 1 / • Distance from example to the separator is
- **PDF p. 17:** Maximum Margin Formulation 1 / • Distance from example to the separator is
- **PDF p. 18:** Fixing the Scale of w and b / 2/‖w‖
- **PDF p. 19:** Maximum Margin Formulation 2 / • Let us choose normalization such that
- **PDF p. 20:** Maximizing the Margin / wT x + b = 0
- **PDF p. 21:** Hard Margin Linear support vector machines / wT x + b = 0
- **PDF p. 22:** Hard Margin Linear support vector machines
- **PDF p. 23:** Non-separable Data / denotes +1
- **PDF p. 24:** SVM for Noisy Data / Class 1
- **PDF p. 25:** Soft-Margin SVM : SVM for Noisy Data / Balance the trade off between margin and
- **PDF p. 26:** SVM for Noisy Data / • Use the Lagrangian formulation for the optimization problem.
- **PDF p. 27:** Min–Max View of the Lagrangian / The Lagrangian 𝐿𝑝(𝑤, 𝑏, 𝜀, 𝛼, 𝛽) folds both constraint sets into the objective, with multipliers 𝛼𝑖≥ 0
- **PDF p. 28:** SVM for Noisy Data / Karush-Kuhn-Tucker Conditions
- **PDF p. 29:** Deriving the Dual Formulation / 1. Expand 𝐿𝑝and group terms
- **PDF p. 30:** SVM Lagrangian Dual / Maximize
- **PDF p. 31:** Back to / SVM
- **PDF p. 32:** SVM Lagrangian Dual / Maximize
- **PDF p. 33:** Why αn takes these values: the KKT conditions / Soft-margin primal: minimise ½ /  / w /  / ² + C Σ ξn
- **PDF p. 34:** SVMs and Dual variables
- **PDF p. 35:** Multi-class Classification with SVMs / • SVMs can only handle two-class outputs.
- **PDF p. 36:** Multi-class Classification using SVM / One- versus-all
- **PDF p. 37:** [image or diagram only]
- **PDF p. 38:** Soft Margin SVMs and Hinge Loss
- **PDF p. 39:** Hinge Loss vs other losses / • Blue : hinge loss
- **PDF p. 40:** Readings / • PRML, Bishop, Chapter 7 (7.1-7.3)

## GEN source page index

- **PDF p. 01:** Probabilistic
- **PDF p. 02:** What we will cover
- **PDF p. 03:** Setting the scene / Three ways to build a classifier — and what makes one of them “generative”
- **PDF p. 04:** Classification so far
- **PDF p. 05:** Generative classifiers
- **PDF p. 06:** Why “generative”?
- **PDF p. 07:** What modelling p(x) buys you
- **PDF p. 08:** From densities to posteriors / What happens when the class-conditional densities are Gaussian
- **PDF p. 09:** Two classes: the posterior is a sigmoid
- **PDF p. 10:** Seeing it: densities → posterior
- **PDF p. 11:** Two classes: the posterior is a sigmoid
- **PDF p. 12:** Gaussian class-conditional densities
- **PDF p. 13:** Gaussian class-conditional densities
- **PDF p. 14:** Shared covariance ⇒linear boundary
- **PDF p. 15:** Shared covariance ⇒linear boundary
- **PDF p. 16:** Shared covariance ⇒linear boundary
- **PDF p. 17:** Rotated mean difference
- **PDF p. 18:** K classes: the posterior is a softmax
- **PDF p. 19:** Discriminant analysis: LDA and QDA / One decision — tied or per-class covariance — sets the shape of the boundary
- **PDF p. 20:** PER-CLASS COVARIANCE
- **PDF p. 21:** Linear discriminant analysis (LDA)
- **PDF p. 22:** BIAS–VARIANCE IN ONE PICTURE
- **PDF p. 23:** Covariance choices: a spectrum of models
- **PDF p. 24:** Estimating the parameters / Maximum likelihood, where it breaks, and what to do about it
- **PDF p. 25:** Maximum likelihood: the objective splits
- **PDF p. 26:** Closed-form maximum likelihood estimates
- **PDF p. 27:** Maximum likelihood overfits in high dimensions
- **PDF p. 28:** Three ways to tame the covariance
- **PDF p. 29:** The naive Bayes classifier / One strong assumption, a huge reduction in parameters, surprisingly good accuracy
- **PDF p. 30:** The naive Bayes assumption
- **PDF p. 31:** How many parameters does that save?
- **PDF p. 32:** How many parameters does that save?
- **PDF p. 33:** “Naive” — and yet it works
- **PDF p. 34:** Choosing the one-dimensional densities
- **PDF p. 35:** Training a naive Bayes classifier
- **PDF p. 36:** The algorithm, in full
- **PDF p. 37:** Prediction — and why you work in logs
- **PDF p. 38:** Worked example: spam or ham? / Five short documents, three words, and one instructive failure
- **PDF p. 39:** The training data
- **PDF p. 40:** Maximum likelihood estimates
- **PDF p. 41:** The zero-count problem
- **PDF p. 42:** The fix: Use a prior distribution over parameters
- **PDF p. 43:** Redo the calculation — and the verdict
- **PDF p. 44:** Generative vs. discriminative / What modelling the joint costs you, and what it pays back
- **PDF p. 45:** The cost of modelling the joint
- **PDF p. 46:** When the generative route wins
- **PDF p. 47:** When the discriminative route wins
- **PDF p. 48:** Two stages: inference, then decision
- **PDF p. 49:** Key takeaways
- **PDF p. 50:** Sources and further reading

## MID scanned solution pages

- **MID p. 01:** Joint density, triangular support and range of Z=X+Y (visual reading of handwritten page)
- **MID p. 02:** CDF region for Z=X+Y, split at x=z/2 (visual reading of handwritten page)
- **MID p. 03:** Integration for first CDF interval (visual reading of handwritten page)
- **MID p. 04:** Independent exponential samples and sum transformation (visual reading of handwritten page)
- **MID p. 05:** Ratio-type transform and Jacobian notes (visual reading of handwritten page)
- **MID p. 06:** Normalize second density and transform example (visual reading of handwritten page)
- **MID p. 07:** Normalize K/y^3 and find CDF (visual reading of handwritten page)
- **MID p. 08:** Symmetric density and shifted interval inequality (visual reading of handwritten page)
- **MID p. 09:** Evaluate shifted interval probability using symmetry (visual reading of handwritten page)

**Coverage count:** 345 substantive teaching pages/slides plus 9 midterm scanned pages = 354 indexed items. (An empty trailing page in the SVM PDF is omitted.) The earlier 44-page parameter-estimation slides are covered through the expanded EST reference; exact duplicate uploads are not counted again.
