---
title: "But what are Gibbs and slice sampling?"
date: 2027-01-30
permalink: /posts/2027/01/2027-01-30-RCFM/
tags:
- Variational Inference
- Simulation-Based Inference
- SBI
- VI
- Flow Matching
- Normalising Flows
header-includes:
  - \usepackage{amsmath}

---

In this post I'm going to go through Gibbs and slice sampling, you've probably seen them used everywhere if you're a statistician, but have you ever looked into _why_ they work in detail? 
(UNDER CONSTRUCTION)


# UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION UNDER CONSTRUCTION


---

## Resources



## Table of Contents
1. Introduction

    - 1.1 Beyond Metropolis-Hastings
    - 1.2 When specialized samplers shine

2. Gibbs Sampling: Coordinate-Wise Updates
    - 2.1 The Core Idea
        - Breaking joint distributions into conditionals
        - The alternating update scheme
    - 2.2 Why It Works
        - Detailed balance via conditional distributions
        - Connection to MH with acceptance probability 1
    - 2.3 A Simple Example: Bivariate Normal
        - Conditional distributions
        - Implementation and visualization
        - Convergence behavior
    - 2.4 When Gibbs Sampling Excels
        - Conjugate models (Beta-Binomial, Gamma-Poisson)
        - Hierarchical models
        - Missing data problems
    - 2.5 When Gibbs Sampling Struggles
        - High correlation between parameters
        - The banana distribution problem
        - Visualizing slow mixing
    - 2.6 Practical Tips
        - Block Gibbs sampling
        - Collapsed Gibbs sampling
        - Rao-Blackwellization for variance reduction

3. Slice Sampling: Sampling Under the Curve
    - 3.1 The Core Idea

        - Augmenting with a uniform auxiliary variable
        - The "slice" of probability mass

    - 3.2 The Algorithm

        - Step 1: Sample vertical position
        - Step 2: Sample horizontal position (the tricky part)
        - Stepping out and shrinkage procedures

    - 3.3 A Simple Example: Univariate Distribution

        - Implementing basic slice sampling
        - Visualization of the slice
        - Comparison with rejection sampling

    - 3.4 Why Slice Sampling is Remarkable

        - No tuning required (no proposal distribution!)
        - Automatically adapts to local scale
        - Handles multimodality gracefully

    - 3.5 Extensions and Variants

        - Multivariate slice sampling (coordinate-wise)
        - Elliptical slice sampling (for Gaussian priors)
        - Shrinking rank slice sampling

    - 3.6 When to Use Slice Sampling

        - Difficult univariate conditionals
        - Unknown or complex target distributions
        - When you want a "set and forget" sampler

4. Comparison and Practical Guidance
    - 4.1 Gibbs vs. Slice vs. Metropolis-Hastings

        - Computational cost comparison
        - Ease of implementation
        - Mixing properties

    - 4.2 Decision Guide

        - When to use Gibbs
        - When to use slice sampling
        - When to stick with MH or HMC

    - 4.3 Combining Samplers

        - Within-Gibbs slice sampling
        - Hybrid MCMC schemes

5. Code Examples
    - 5.1 Gibbs Sampling: Complete Implementation

        - Bivariate normal example
        - Convergence diagnostics

    - 5.2 Slice Sampling: Complete Implementation

        - Univariate example with stepping out
        - Visualization utilities

6. Common Pitfalls and Debugging
    - 6.1 Gibbs Sampling Issues

        - Detecting poor mixing
        - The scan order matters (sometimes)
        - Random vs. systematic scan

    - 6.2 Slice Sampling Issues

        - Choosing initial interval width
        - Computational cost with complex slices
        - Numerical precision problems

7. Conclusion


