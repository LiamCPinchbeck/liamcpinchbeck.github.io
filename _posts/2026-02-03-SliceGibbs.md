---
title: "But what are Gibbs and slice sampling?"
date: 2026-02-03
permalink: /posts/2026/02/2026-02-03-SliceGibbs/
tags:
  - MCMC
  - Introductory
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
    - This is good, but I think the way they describe Gibbs sampling in general sounds more like specifically [collapsed Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling#:~:text=backward%20algorithm.-,Collapsed%20Gibbs%20sampler,-%5Bedit%5D)
- ["Gibbs Sampling" - Wikipedia](https://en.wikipedia.org/wiki/Gibbs_sampling)
- ["Gibbs Sampling, Conjugate Priors and Coupling" - Diaconis, Khare and Coste](https://www.stat.berkeley.edu/~aldous/206-RWG/RWGpapers/diaconis_gibbs.pdf)

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

Gibbs sampling is evidently a Markov Chain, as each new set of points only depends on the previous, and is ergodic (because the conditional distributions should allow any coordinate in parameter space to be sampled), so to be a valid MCMC method we only have to show that the sampling satifies detailed balance (or the weaker condition of [_Global Balance_](https://en.wikipedia.org/wiki/Balance_equation)).

In Gibbs sampling, we move along one "axis" (one variable) at a time. Thus, to understand the transition kernel let's focus on the update for the $$i$$-th variable, $$x_i$$ for a target distribution $$\pi$$. You can just treat $$x_i$$ as the first block in the GIF above and every other variable $$\mathbf{x}_{-i}$$ as the second.

Let our current state be $$\mathbf{x} = (x_i, \mathbf{x}_{-i})$$ and the proposed next state be $$\mathbf{x}' = (x_i', \mathbf{x}_{-i})$$. 

Note that only the $$i$$-th component changes. The transition kernel (the probability of moving from $$\mathbf{x}$$ to $$\mathbf{x}'$$) for this step is:
$$\begin{align}
P_i(\mathbf{x} \to \mathbf{x}') = \pi(x_i' \mid \mathbf{x}_{-i})
\end{align}$$

Now, we check if $$\pi(\mathbf{x}) P_i(\mathbf{x} \to \mathbf{x}') = \pi(\mathbf{x}') P_i(\mathbf{x}' \to \mathbf{x})$$: 

#### Left-Hand Side (Forward Flow):

Using the chain rule $$\pi(A, B) = \pi(A \mid B)\pi(B)$$:

$$\pi(x_i, \mathbf{x}_{-i}) \cdot \pi(x_i' \mid \mathbf{x}_{-i}) = \left[ \pi(x_i \mid \mathbf{x}_{-i}) \pi(\mathbf{x}_{-i}) \right] \cdot \pi(x_i' \mid \mathbf{x}_{-i})$$

#### Right-Hand Side (Backward Flow):

$$\pi(x_i', \mathbf{x}_{-i}) \cdot \pi(x_i \mid \mathbf{x}_{-i}) = \left[ \pi(x_i' \mid \mathbf{x}_{-i}) \pi(\mathbf{x}_{-i}) \right] \cdot \pi(x_i \mid \mathbf{x}_{-i})$$

#### Together

If you look at the final expressions for both sides you find:

$$\pi(\mathbf{x}_{-i}) \cdot \pi(x_i \mid \mathbf{x}_{-i}) \cdot \pi(x_i' \mid \mathbf{x}_{-i})$$.

Since the flow is equal in both directions, the $$i$$-th update satisfies detailed balance with respect to $$\pi$$.

### Systematic vs. Random Scan

There is a subtle distinction in how you apply these updates that affects whether the entire algorithm satisfies detailed balance.

#### Random Scan Gibbs: 

If you pick a variable $$i$$ at random to update in each step, the entire algorithm satisfies detailed balance. It is reversible.

#### Systematic Scan Gibbs: 

If you always update $$x_1, x_2, \dots, x_d$$ in a fixed order, the full cycle does not actually satisfy detailed balance (because you can't just reverse the order and get the same transition probability).

However, even if the full cycle doesn't satisfy detailed balance, it still satisfies [_Global Balance_](https://en.wikipedia.org/wiki/Balance_equation). Because each individual step $$P_i$$ leaves $$\pi$$ invariant ($$\pi P_i = \pi$$), their composition also leaves $$\pi$$ invariant:

$$\pi (P_1 P_2 \dots P_d) = \pi$$

This is why systematic Gibbs sampling is still a valid MCMC algorithm, even though it is technically "non-reversible". You could also define the transition kernel as the whole process of sampling with the combination of kernels proposed above (going from one vector of $$\vec{x}$$ to one where every element has been sampled from a given conditional $$\vec{x}'$$) and that would satisfy detail balance. But that's pretty much the same as the global balance condition in this case.


#### But it doesn't have an accept-reject step?

The typical point for the accept-reject step in MCMC is to correct for the fact your proposal distribution or kernel (sans accept-reject) does not lead to the right target distribution. In this case, the kernel above does this already, and thus doesn't need an accept-reject. i.e. The acceptance probability is 1. This is regardless of dimensionality or complexity, if you have the conditional distributions. This does not mean that the mixing (convergence) time is fantastic, as we'll see below, highly correlated distributions can be tricky for Gibbs sampling.


## 1.4 When Gibbs Sampling Struggles

#### Separated modes

I'm just gonna let the GIF speak for itself here.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2026-02-SliceGibbs/separated_modes_gibbs_sampler.gif" 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>

#### High correlation between parameters

Again, I'm just gonna let the GIF speak for itself here.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2026-02-SliceGibbs/highly_correlated_gibbs_sampler.gif" 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>

#### The banana distribution problem




## 1.5 Practical Tips

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