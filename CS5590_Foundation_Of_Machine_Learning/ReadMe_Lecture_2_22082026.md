# Lecture 2: Likelihood and Bayesian Inference

## Table of Contents

- [Lecture 2: Likelihood and Bayesian Inference](#lecture-2-likelihood-and-bayesian-inference)
  - [Table of Contents](#table-of-contents)
  - [Likelihood](#likelihood)
    - [The Key Difference](#the-key-difference)
    - [Likelihood in Bayesian Inference](#likelihood-in-bayesian-inference)
  - [Prior Distribution](#prior-distribution)
    - [Formal Definition](#formal-definition)
    - [Prior Intuition](#prior-intuition)
    - [Alpha and Beta Example](#alpha-and-beta-example)
    - [Role in Bayesian Inference](#role-in-bayesian-inference)
  - [Posterior](#posterior)
    - [🔹 Formal Definition](#-formal-definition)
    - [🔹 Intuition](#-intuition)
    - [🔹 Example](#-example)
  - [Posterior distribution in Bayesian inference](#posterior-distribution-in-bayesian-inference)
    - [Posterior Formula](#posterior-formula)
    - [MAP Estimate](#map-estimate)
    - [Dirac Delta Notation](#dirac-delta-notation)
    - [Putting It Together](#putting-it-together)
  - [MAP](#map)
    - [Contrast with MLE](#contrast-with-mle)
    - [Example: Coin Toss](#example-coin-toss)
    - [Why MAP Is Useful](#why-map-is-useful)
  - [Bayesian Formulation in Statistical Modeling](#bayesian-formulation-in-statistical-modeling)
    - [The Equation](#the-equation)
    - [Meaning of Each Term](#meaning-of-each-term)
    - [Intuition](#intuition)
      - [Example: Coin Toss](#example-coin-toss-1)
    - [Beta Distribution as a Prior](#beta-distribution-as-a-prior)
    - [Example: Coin Toss with a Prior](#example-coin-toss-with-a-prior)
    - [Posterior Distribution](#posterior-distribution)
    - [MAP Estimate](#map-estimate-1)
    - [MLE vs MAP vs Full Posterior](#mle-vs-map-vs-full-posterior)
    - [Key Insight](#key-insight)
    - [What Alpha and Beta Represent](#what-alpha-and-beta-represent)
    - [Intuitive Meaning](#intuitive-meaning)
    - [Example](#example)
    - [Analogy](#analogy)

## Likelihood

In statistics, **likelihood tells us how well a particular parameter value explains the data we observed.**

Suppose we have a coin with an unknown probability $\theta$ of landing heads.

You toss it 10 times and observe:

$$
D = 7H, 3T
$$

For a particular value of $\theta$, the probability of getting this data is:

$$
P(D \mid \theta) = \binom{10}{7}\theta^7(1-\theta)^3
$$

When we **treat the data as fixed and vary $\theta$**, this same expression is called the **likelihood**:

$$
\boxed{L(\theta \mid D) = P(D \mid \theta)}
$$

### The Key Difference

This is subtle but very important:

- **Probability:** Assume $\theta$ is known and ask how likely the data is.
- **Likelihood:** The data is already observed and we ask which values of $\theta$ make that data more plausible.

For example, if you observed 7 heads out of 10:

$$
L(\theta \mid D) \propto \theta^7(1-\theta)^3
$$

The likelihood is largest around:

$$
\theta = 0.7
$$

So, the data provides evidence that the coin's probability of heads is around $0.7$.

### Likelihood in Bayesian Inference

Likelihood is one of the three pieces in Bayes' theorem:

$$
\boxed{p(\theta \mid D) \propto p(D \mid \theta)p(\theta)}
$$

Equivalently:

$$
\boxed{\text{Posterior} \propto \text{Likelihood} \times \text{Prior}}
$$

Think of it this way:

- **Prior:** What did I believe about $\theta$ before seeing the data?
- **Likelihood:** How compatible is the observed data with each possible $\theta$?
- **Posterior:** What do I believe about $\theta$ after combining the prior and data?

The posterior can then be used to form a posterior predictive distribution, which lets us predict **new data**.

> **One-sentence intuition:** Likelihood is a score telling us how well different parameter values explain the data we have observed.

## Prior Distribution

The **prior** is your initial belief about a parameter or hypothesis *before* you see any data. It is the starting point in Bayesian inference.

### Formal Definition

$$
P(\theta)
$$

- $\theta$: The parameter or hypothesis.
- $P(\theta)$: The prior distribution, representing your belief about possible values of $\theta$ before observing data.

### Prior Intuition

Think of the prior as your **baseline assumption**.

- If you have no strong opinion, you might choose a non-informative prior, such as a uniform distribution.
- If you already have knowledge from past experiments, you can encode that knowledge in an informative prior.

### Alpha and Beta Example

For a coin toss:

- Before tossing, you believe the coin is fair.
- Prior: $P(p = 0.5) = 1$, with all belief concentrated at $0.5$.
- If you are less certain, you might use a Beta distribution prior, such as $\text{Beta}(2, 2)$, which says the coin is roughly fair while allowing uncertainty.

### Role in Bayesian Inference

Bayes' theorem combines:

- **Prior:** Belief before seeing data.
- **Likelihood:** How well the data fits each hypothesis.
- **Posterior:** Updated belief after seeing the data.

In short, the **prior** is your **starting belief** about parameters, before evidence reshapes it.

## Posterior

The **posterior** is the updated probability distribution of a parameter (or hypothesis) after you’ve seen data. It’s the cornerstone of Bayesian inference.

---

### 🔹 Formal Definition
If:
- Prior belief: \(P(\theta)\)  
- Likelihood: \(P(D \mid \theta)\)  
- Evidence (normalizing constant): \(P(D)\)  

Then the **posterior** is:

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}
$$

- \(P(\theta \mid D)\): posterior — what you believe about \(\theta\) after seeing data \(D\).  
- \(P(\theta)\): prior — what you believed before seeing data.  
- \(P(D \mid \theta)\): likelihood — how well the data fits each possible \(\theta\).  
- \(P(D)\): evidence — ensures probabilities sum to 1.

---

### 🔹 Intuition
Think of it as **belief updating**:
- **Prior**: your initial guess.  
- **Likelihood**: how consistent the data is with that guess.  
- **Posterior**: your refined belief after combining both.

---

### 🔹 Example
Suppose you think a coin is fair (\(p=0.5\)) before tossing it.  
You toss 10 times and see 7 heads.  
- Prior: belief coin is fair.  
- Likelihood: probability of 7 heads given different values of \(p\).  
- Posterior: updated distribution over \(p\), now shifted toward values closer to 0.7.

---

## Posterior distribution in Bayesian inference

👉 In short: the **posterior** is your **new belief** about parameters or hypotheses after incorporating evidence.  

Perfect — the page you shared is diving into the **posterior distribution** in Bayesian inference, and it’s showing how the math formalizes the idea of “belief updating.” Let me unpack what’s going on there in plain language:

---

###  Posterior Formula
The general form is:

$$
p(h|\mathcal{D}) = \frac{p(\mathcal{D}|h)\,p(h)}{\sum_{h' \in \mathcal{H}} p(\mathcal{D}|h')\,p(h')}
$$

- \(h\): a hypothesis (or parameter value).  
- \(\mathcal{D}\): observed data.  
- \(p(h)\): prior belief about the hypothesis.  
- \(p(\mathcal{D}|h)\): likelihood — how well the hypothesis explains the data.  
- Denominator: normalizing constant (ensures probabilities sum to 1 across all hypotheses).  

So the posterior is **prior × likelihood, normalized**.

---

###  MAP Estimate
The text then shows:

$$
\hat{h}^{MAP} = \arg\max_h p(h|\mathcal{D})
$$

This is the **Maximum A Posteriori (MAP) estimate** — the single hypothesis that has the highest posterior probability.  
- If you want one “best guess,” MAP is what you pick.  
- If you want the full uncertainty picture, you keep the entire posterior distribution.

---

###  Dirac Delta Notation
$$
p(h|\mathcal{D}) \to \delta_{h_{MAP}}(h)
$$

This means: as the amount of data grows very large, the posterior distribution tends to concentrate all its mass on the MAP hypothesis. In other words, with infinite data, uncertainty vanishes and the posterior collapses to a single point.

---

###  Putting It Together
- **Prior**: what you believed before data.  
- **Likelihood**: how consistent each hypothesis is with the data.  
- **Posterior**: updated belief after combining both.  
- **MAP**: the most probable hypothesis under the posterior.  
- **Delta collapse**: with enough data, the posterior becomes sharply peaked at the MAP.

---

##  MAP
MAP is the **best guess for a parameter after seeing data**, but it also respects what you believed before (the prior).  

Formula:
$$
\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D)
$$

Using Bayes’ theorem:
$$
P(\theta|D) \propto P(D|\theta) \cdot P(\theta)
$$

So MAP = **Likelihood × Prior**, then pick the peak.

---

###  Contrast with MLE
- **MLE (Maximum Likelihood Estimate):**  
  Only looks at data.  
  $$
  \hat{\theta}_{MLE} = \arg\max_{\theta} P(D|\theta)
  $$

- **MAP:**  
  Looks at data **and** prior belief.  
  $$
  \hat{\theta}_{MAP} = \arg\max_{\theta} P(D|\theta) \cdot P(\theta)
  $$

👉 If prior is uniform (no preference), MAP = MLE.  
👉 With strong prior, MAP shifts toward the prior.  
👉 With lots of data, MAP ≈ MLE (data dominates).

---

###  Example: Coin Toss
You toss a coin 10 times, get 7 heads.  
- **MLE:** Bias estimate = \(0.7\).  
- **MAP:** If your prior belief is “coin is fair” (\(p=0.5\)), the posterior balances prior + data. MAP might be around \(0.65\).  

So MAP tempers extreme estimates when data is limited.

---

###  Why MAP Is Useful
- Prevents overfitting when data is scarce.  
- Encodes domain knowledge (priors).  
- Used in Bayesian machine learning, NLP, healthcare models.  
- Bridges Bayesian inference (posterior) with practical point estimates.

---

✅ In short:  
- **MLE** = purely data-driven.  
- **MAP** = data-driven + prior knowledge.  
- MAP is the **most probable hypothesis after updating with data**.


## Bayesian Formulation in Statistical Modeling

### The Equation

$$
P(\text{Parameters} \mid \text{Data}, \text{Model}) =
\frac{P(\text{Data} \mid \text{Model}, \text{Parameters}) \cdot P(\text{Parameters} \mid \text{Model})}
{P(\text{Data} \mid \text{Model})}
$$

### Meaning of Each Term

- **Posterior:** $P(\text{Parameters} \mid \text{Data}, \text{Model})$. The probability distribution of parameters after seeing the data.
- **Likelihood:** $P(\text{Data} \mid \text{Model}, \text{Parameters})$. How probable the observed data is, given the parameters.
- **Prior:** $P(\text{Parameters} \mid \text{Model})$. What we believed about the parameters before seeing the data.
- **Evidence (marginal likelihood):** $P(\text{Data} \mid \text{Model})$. The normalizing constant that ensures probabilities sum to 1.

### Intuition

Bayes' theorem says:

1. Start with a prior belief about parameters.
2. Update it using the likelihood of observed data.
3. Normalize with the evidence to get the posterior.

This is the backbone of Bayesian inference in machine learning, statistics, and AI.

#### Example: Coin Toss

Suppose you are estimating the bias of a coin:

- **Prior:** You believe it is fair, $p = 0.5$.
- **Likelihood:** You toss it 10 times and see 7 heads.
- **Posterior:** Bayes' theorem updates your belief, suggesting the coin is probably biased toward heads.

### Beta Distribution as a Prior

The Beta distribution is useful for modeling prior beliefs about a probability $p$, such as the bias of a coin.

$$
\text{Beta}(\alpha, \beta) \quad \Rightarrow \quad
f(p) = \frac{1}{B(\alpha, \beta)}p^{\alpha-1}(1-p)^{\beta-1}
$$

- $\alpha$: Prior successes, such as heads.
- $\beta$: Prior failures, such as tails.
- Mode, when $\alpha, \beta > 1$:

$$
\frac{\alpha - 1}{\alpha + \beta - 2}
$$

### Example: Coin Toss with a Prior

Suppose:

- Prior belief: The coin is fair, so $\alpha = 2$ and $\beta = 2$.
- Data: 10 tosses, with 7 heads and 3 tails.

### Posterior Distribution

For a Beta prior and a Binomial likelihood, the posterior is also Beta:

$$
\text{Posterior} = \text{Beta}(\alpha + \text{heads}, \beta + \text{tails})
$$

$$
= \text{Beta}(2 + 7, 2 + 3) = \text{Beta}(9, 5)
$$

### MAP Estimate

MAP is the mode of the posterior:

$$
\hat{\theta}_{\text{MAP}} =
\frac{\alpha + \text{heads} - 1}{\alpha + \beta + n - 2}
$$

$$
= \frac{2 + 7 - 1}{2 + 2 + 10 - 2} = \frac{8}{12} = 0.667
$$

Therefore, the MAP estimate is **0.667**.

### MLE vs MAP vs Full Posterior

- **MLE:** Uses only the data, giving $0.7$.
- **MAP:** Balances the prior and data, giving $0.667$.
- **Posterior distribution:** Represents a full curve, $\text{Beta}(9, 5)$, showing uncertainty around the estimate.

### Key Insight

The Beta distribution lets us encode prior beliefs. Bayes' theorem updates those beliefs with data. MAP gives a point estimate, while the full posterior gives a distribution of possible values and therefore richer information about uncertainty.

![Prior and posterior Beta distributions](images/PriorPosterior.png)

The visualization compares the prior and posterior Beta distributions:

- **Blue curve (Prior: Beta(2, 2)):** The initial belief, symmetric around 0.5, suggesting the coin was likely fair.
- **Red curve (Posterior: Beta(9, 5)):** The updated belief after observing 7 heads and 3 tails, with the peak shifted toward 0.67.

### What Alpha and Beta Represent

In the Beta distribution,

$$
\text{Beta}(\alpha, \beta)
$$

$\alpha$ and $\beta$ shape the curve that represents your belief about a probability $p$.

- **Alpha ($\alpha$):** Prior count of successes, such as heads.
- **Beta ($\beta$):** Prior count of failures, such as tails.

They do not have to be whole numbers. They are pseudo-counts expressing how confident you are before seeing data.

### Intuitive Meaning

| Parameter | Interpretation | Example shape |
| --- | --- | --- |
| $\alpha = \beta = 1$ | Uniform (flat), with no prior preference | Equal chance for all $p$ |
| $\alpha > \beta$ | Success is believed to be more likely | Curve peaks near 1 |
| $\alpha < \beta$ | Failure is believed to be more likely | Curve peaks near 0 |
| Both $\alpha$ and $\beta$ are large | Strong prior confidence | Narrow curve around the prior mean |

### Example

If you start with $\alpha = 2$ and $\beta = 2$, your prior belief is that the coin is roughly fair, with mean $0.5$. After seeing 7 heads and 3 tails, your posterior becomes $\alpha = 9$ and $\beta = 5$, shifting the curve toward $0.67$.

### Analogy

Think of $\alpha$ and $\beta$ as the memory of past experiences:

- $\alpha$ represents how many times you have seen success before.
- $\beta$ represents how many times you have seen failure before.

Bayes' theorem updates these counts with new evidence.
