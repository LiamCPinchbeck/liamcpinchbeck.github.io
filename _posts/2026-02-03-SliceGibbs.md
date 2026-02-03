---
title: "But what are Gibbs and slice sampling?"
date: 2026-02-03
permalink: /posts/2026/02/2026-02-03-RCFM/
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

### Gibbs Sampling

- ["The Gibbs Sampler Revisited from the  Perspective of Conditional Modeling" - Kuo & Wang](https://arxiv.org/abs/2410.09559)
- ["Explaining the Gibbs Sampler" - Casella & George](https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Basic/CasellaGeorge1992.pdf)
- ["Gibbs Sampling" - Wikipedia](https://en.wikipedia.org/wiki/Gibbs_sampling)

### Slice Sampling

- ["Slice Sampling" - Wikipedia](https://en.wikipedia.org/wiki/Slice_sampling)



--- 

## Table of Contents
1. [Gibbs Sampling](#1-gibbs-sampling)


2. [Slice Sampling](#2-slice-sampling)



3. [Examples](#3-examples)


4. [Common Pitfalls and Debugging](#4-common-pitfalls-and-debugging)


5. [Conclusion](#5-conclusion)

6. [HMC and slice sampling](#a-hmc-and-slice-sampling)


--- 

# 1. Gibbs Sampling

I'm not gonna bore you with motivations for why Gibbs sampling is important, you're either here because you're either interested or being forced to be interested for whatever reason. Let's just jump in.

## 1.1 The Core Idea

The core idea of Gibbs sampling is that you have access to conditional distributions of your data. Such that if the density you're trying to explore is $$p(x_1, x_2)$$ for example ($$x_1$$ and $$x_2$$ can be multi-dimensional if you want), then you have access to both $$p(x_1 \vert x_2)$$ and $$p(x_2 \vert x_1)$$ and with some initial condition, can sample $$p(x_1, x_2)$$ by sampling the conditionals back and forth. An actual algorithm (however simple it may be) is given below.


>
#### Gibbs Sampling (2D / 2 Blocks)
1. Initialise: 
    - Have a distribution you want to sample from up to some normalisation constant (duh) $$f(x_1, x_2)$$, and it's conditionals $$p(x_1 \vert x_2)$$ and $$p(x_2 \vert x_1)$$
        - You only need to be able to evaluate the conditionals, you don't need to evaluate $$f(x_1, x_2)$$
    - manually create a starting point for the algorithm $$X^0 = (x_1^0, x_2^0) $$,
    - pick the number of samples you can be bothered waiting for $$N$$
2. For each iteration $$n$$/Repeat $$N$$ times
    1. Sample $$x_1^{n+1} \sim p(x_1^{n} \vert x_2^{n})$$
    2. Sample $$x_2^{n+1} \sim p(x_2^{n} \vert x_1^{n+1})$$

Yep that's it. An example of it in action is shown in the GIF below.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2026-01-LMC/gibbs_sampler.gif" 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>



## 1.2 Detailed balance



## 1.3 A Simple Example: Bivariate Normal

## 1.4 When Gibbs Sampling Excels

- Conjugate models (Beta-Binomial, Gamma-Poisson)
- Hierarchical models
- Missing data problems

## 1.5 When Gibbs Sampling Struggles

- High correlation between parameters
- The banana distribution problem

## 1.6 Practical Tips

- Block Gibbs sampling
- Collapsed Gibbs sampling
- Rao-Blackwellization for variance reduction



# 2. Slice Sampling

## 2.1 The Core Idea


## 2.2 The Algorithm


## 2.3 A Simple Example: Univariate Distribution

    - Implementing basic slice sampling
    - Visualization of the slice
    - Comparison with rejection sampling

## 2.4 Why Slice Sampling is cool as f*!$

    - No tuning required
    - Automatically adapts to local scale
    - Handles multimodality nicely

## 2.5 Extensions and Variants

    - Multivariate slice sampling (coordinate-wise)
    - Elliptical slice sampling (for Gaussian priors)
    - Shrinking rank slice sampling


# 3. Examples

## 3.1 Plain ol' gaussian example

## 3.2 Bivariate normal example

## 3.3 Ba-nana-nana-nana-nana Banan




# 4. Common Pitfalls and Debugging


## 4.1 Gibbs Sampling Issues

    - Detecting poor mixing
    - The scan order matters (sometimes)
    - Random vs. systematic scan

## 4.2 Slice Sampling Issues

    - Choosing initial interval width
    - Computational cost with complex slices
    - Numerical precision problems

# 5. Conclusion


# A. HMC and slice sampling