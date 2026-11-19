---
title: "Model Comparison: Which description of my data is the correct one?"
date: 2126-04-21
permalink: /posts/2126/11/2126-11-model-comparison/
tags:
  - Introductory
header-includes:
  - \usepackage{amsmath}
---



UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION
---


## Resources


- [ANOVA - Crash Course Statistics](https://youtu.be/oOuu8IBd-yo)
- [ANOVA Part 2 -Crash Course Statistics](https://youtu.be/wo1xlefg5KI)
- [ Fitting Models Is like Tetris - Crash Course Statistics  ](https://youtu.be/QdmG6uUtB4o)
- [T-tests - Crash Course Statistics](https://youtu.be/AGh66ZPpOSQ)
- [](https://youtube.com/playlist?list=PLWMBgIqOUbrnGulJZarEgvbywR5X20Grq&si=RdKUbKFQk5DlPEOj) - 3.4 + 3.5
- Bishop, Pattern Recognition and Machine Learning (2006), §3.4–3.5: Bayesian model comparison and the evidence approximation (likely your “3.4 + 3.5”)
- MacKay (1992), Bayesian Interpolation, Neural Computation 4(3)
- Gull (1989), Developments in maximum entropy data analysis
- Vehtari, Gelman & Gabry (2017), Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC
- https://arxiv.org/abs/0907.4728
- [The Relative Instability of Model Comparison with Cross-validation](https://arxiv.org/abs/2508.04409)
- [Comparison of Bayesian predictive methods for model selection](https://arxiv.org/abs/1503.08650)
- [Bayesian Evidence and Model Selection](https://arxiv.org/abs/1411.3013v2)
- [Bayesian Model Selection, the Marginal Likelihood, and Generalization](https://arxiv.org/abs/2202.11678)
- [Statistical Inference for Generative Model Comparison](https://arxiv.org/abs/2501.18897)



## Table of Contents

- Motivation: from "is B different from A?" to "which model explains the data best?"
- One framework: tests as linear models
    - t-tests and one-way ANOVA as special cases of linear regression (+)
    - Two-way / factorial ANOVA and interactions (intersectional groups)
    - ANCOVA (and why CUPED is a cousin)
    - Repeated measures, and a pointer to mixed-effects models (+)
    - Generalised linear models: logistic regression for conversion data
- Frequentist model comparison
    - Nested models: the F-test and the likelihood-ratio test (+)
    - Wilks' theorem and chi-squared / dof (+)
- Information-theoretic comparison
    - KL divergence as the foundation
    - AIC: an estimate of expected KL divergence
    - BIC: a large-sample approximation to the evidence
    - Why AIC and BIC disagree: prediction vs identification
- Bayesian model comparison
    - The evidence and Bayes factors
    - The Occam factor: why the evidence penalises complexity automatically (+)
    - The Jeffreys–Lindley paradox: when p-values and Bayes factors disagree (+)
- The evidence approximation (Gull 1989; MacKay 1992)
    - Maximising the evidence for linear basis-function models
    - The effective number of parameters, γ
- Predictive alternatives: cross-validation, LOO and WAIC (brief) (+)
- Case study: revisiting the A/B data, comparing models with and without segment interactions (+)
- Rules of thumb