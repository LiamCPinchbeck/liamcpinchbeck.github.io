---
title: "But what are Gibbs and slice sampling?"
date: 2026-02-03
permalink: /posts/2026/02/2026-02-03-SliceGibbs/
tags:
  - MCMC
  - Introductory
  - Gibbs Sampling
  - Slice Sampling
header-includes:
  - \usepackage{amsmath}
manual_prev_url: /posts/2025/02/2025-02-04-mcmc-diagnostics/
manual_prev_title: "Markov Chain Monte Carlo convergence diagnostics"
manual_next_url: /posts/2026/01/2026-01-18-LMC/
manual_next_title: "Speeding up MCMC with Langevin and Hamiltonian dynamics and stochastic gradient estimates"

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


4. [Conclusion](#5-conclusion)


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


And if you want to get fancy with it, you can generalise to arbitrary dimensions/number of blocks by scanning or sampling through the dimensions.

Firstly we look at the one where scan through the dimensions. This is the most common version, where you visit every dimension in a fixed order (usually $1$ to $D$) during every iteration.

>
#### Gibbs Sampling (N-D, scan through dimensions)
1. Initialise:
    - Have a target distribution $$f(\mathbf{x})$$ where $$\mathbf{x} = (x_1, x_2, \dots, x_D)$$, and its full conditionals $$p(x_i \vert \mathbf{x}_{-i})$$ for all $$i \in \{1, \dots, D\}$$.
        - Note: $$\mathbf{x}_{-i}$$ denotes all variables except $$x_i$$. 
    - Manually create a starting point $$X^0 = (x_1^0, x_2^0, \dots, x_D^0)$$.
    - Pick the number of samples $$N$$.
2. For each iteration $$n$$ (Repeat $$N$$ times):
>
    1. Sample $$x_1^{n+1} \sim p(x_1 \vert x_2^{n}, x_3^{n}, \dots, x_D^{n})$$
>
    2. Sample $$x_2^{n+1} \sim p(x_2 \vert x_1^{n+1}, x_3^{n}, \dots, x_D^{n})$$
>
    3. Sample $$x_3^{n+1} \sim p(x_3 \vert x_1^{n+1}, x_2^{n+1}, \dots, x_D^{n})$$
>
    ...
>
    D. Sample $$x_D^{n+1} \sim p(x_D \vert x_1^{n+1}, x_2^{n+1}, \dots, x_{D-1}^{n+1})$$
        - Note: As soon as a variable is updated, its new value is used for all subsequent conditional samples within that same iteration.

Instead of updating every dimension, you pick one dimension at random to update during each step. This is often used in theoretical analysis and physics-based models.

>
#### Gibbs Sampling (N-D, Random Scan)
1. Initialise:
    - Have a target distribution $$f(\mathbf{x})$$ and its full conditionals $$p(x_i \vert \mathbf{x}_{-i})$$.
    - Manually create a starting point $$X^0 = (x_1^0, x_2^0, \dots, x_D^0)$$.
    - Pick the number of steps $$N$$.
2. For each step $$n$$ (Repeat $$N$$ times):
    1. Pick a dimension $$i$$ uniformly at random from $$\{1, 2, \dots, D\}$$.
    2. Update that dimension: $$x_i^{n+1} \sim p(x_i \vert x_1^{n}, \dots, x_{i-1}^{n}, x_{i+1}^{n}, \dots, x_D^{n})$$
    3. For all other dimensions $$j \neq i$$, keep the value the same: $$x_j^{n+1} = x_j^{n}$$

General tips:
- Systematic Scan (the first one) is generally more efficient for computers because it ensures every variable gets updated regularly, which usually leads to faster "mixing" through the space.
- Random Scan is useful if you have a massive number of dimensions and updating all of them in one "iteration" is computationally too expensive, or if you want to avoid periodic patterns in your sampler.


## 1.2 Detailed balance

Gibbs sampling is evidently a Markov Chain, as each new set of points only depends on the previous, and is ergodic (because the conditional distributions should allow any coordinate in parameter space to be sampled), so to be a valid MCMC method we only have to show that the sampling satifies detailed balance (or the weaker condition of [_Global Balance_](https://en.wikipedia.org/wiki/Balance_equation)).

In Gibbs sampling, we move along one "axis" (one variable) at a time. Thus, to understand the transition kernel let's focus on the update for the $$i$$-th variable, $$x_i$$ for a target distribution $$\pi$$. 

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

This is why systematic Gibbs sampling is still a valid MCMC algorithm, even though it is technically "non-reversible". You could also define the transition kernel as the whole process of sampling with the combination of kernels proposed above (going from one vector of $$\vec{x}$$ to one where every element has been sampled from a given conditional $$\vec{x}'$$, basically how I implemented in the algorithm for the systematic scan) and that would satisfy detail balance. But that's pretty much the same as the global balance condition in this case.


#### But it doesn't have an accept-reject step?

The typical point for the accept-reject step in MCMC is to correct for the fact your proposal distribution or kernel (sans accept-reject) does not lead to the right target distribution. In this case, the kernel above does this already, and thus doesn't need an accept-reject. i.e. The acceptance probability is 1. This is regardless of dimensionality or complexity, if you have the conditional distributions. This does not mean that the mixing (convergence) time is fantastic, as we'll see below, highly correlated distributions can be tricky for Gibbs sampling.


## 1.4 When Gibbs Sampling Struggles

#### Separated modes

I'm just gonna let the GIF speak for itself here.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2026-02-SliceGibbs/Gibbs/separated_modes_gibbs_sampler.gif" 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>

#### High correlation between parameters

Again, I'm just gonna let the GIF speak for itself here.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2026-02-SliceGibbs/Gibbs/highly_correlated_gibbs_sampler.gif" 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>



# 2. Slice Sampling

Slice sampling is an important technique that along with being a standard standalone sampler, is also used as part of HMC-NUTS and is implemented as part of other sampling schemes such as nested sampling. To me, the [introductory paper by Neal](https://arxiv.org/abs/physics/0009028) although exhaustive, was a little hard to follow imo.

## 2.1 The Core Idea

Slice sampling, to me, is basically a smart combination of Gibbs (the same as above) and [Rejection sampling](https://liamcpinchbeck.github.io/posts/2025/01/2025-01-28-rejection-sampling/). 

As a quick refresher for the second method, _rejection sampling_, samples a target distribution by uniformly sampling the volume under the density. This process if demonstrated below, using GIFs from my introductory post on the method.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/norm_dist.gif" 
      alt="GIF showing animation of rejection sampling principle with a normal distribution." 
      title="GIF showing animation of rejection sampling principle with a normal distribution." 
      style="width: 75%; height: auto; border-radius: 8px;">

<img 
    src="/files/BlogPostData/2025-01-28/argus_dist.gif" 
    alt="GIF showing animation of rejection sampling principle with an ARGUS distribution." 
    title="GIF showing animation of rejection sampling principle with an ARGUS distribution." 
    style="width: 75%; height: auto; border-radius: 8px;">

<img 
    src="/files/BlogPostData/2025-01-28/powerlaw_dist.gif" 
    alt="GIF showing animation of rejection sampling principle with an power law distribution (a=2)." 
    title="GIF showing animation of rejection sampling principle with an power law distribution (a=2)." 
    style="width: 75%; height: auto; border-radius: 8px;">

</div>


Throwing away real-world limitations for a sec, the fundamental method behind slice sampling is where the user pretends that the density values themselves $$f(x)$$ is a specific value of a variable $$y$$ (like in rejection sampling), and then a two part Gibbs sampling where you sample a $$y$$ value from between $$0$$ and $$f(x)$$, then you sample $$x$$ by uniformly sampling $$x \sim U(S)$$ where $$S = \{x: f(x) < y\}$$. This implies a joint density $$p(x, y)$$ that once one marginalises over $$y$$ (or in practice _throws out_ the samples of $$y$$) you attain $$p(x)$$ (or samples representative of it, the normalised $$f(x)$$).

This may or may not immediately seem like rejection sampling, but the key point is that we don't have an envelope distribution, in this theoretical setup there are no _"rejected"_ samples! 

Real world slice sampling is then an approximation to the above that is asymptotically correct. Below are some GIFs showing the exact case, as it's pretty easy to figure out the interval $$S = \{x: f(x) < y\}$$ in the case of the normal distribution. For the second plot I only show 100 of the coordinates at a time.

<div style="text-align: center;">
  <img 
    src="/files/BlogPostData/2026-02-SliceGibbs/Slice/slice_sampling.gif" 
    style="width: 99%; height: auto; border-radius: 8px;">
<img 
    src="/files/BlogPostData/2026-02-SliceGibbs/Slice/slice_sampling_long.gif" 
    style="width: 99%; height: auto; border-radius: 8px;">
</div>


## 2.2 The Practical Algorithm

Now, in almost all interesting cases we do not have enough information to efficiently derive (or derive at all) $$S = \{x: f(x) < y\}$$. So the question is how do we uniformly sample from $$S$$ without explicitly know what $$S$$ is?

Well [Neal](https://arxiv.org/abs/physics/0009028) suggests 4 options (end of page 8 if you wanna double check):

1. Git good and just figure it out (although they suggest this may not be feasible)
2. If there is a hard bound on what values $$x_p$$ can take, i.e. it can only be between 0 and 3, then we just sample that whole interval and reject points where $$y>f(x_p)$$. But, similar to rejection sampling, this could be very inefficient. E.G. A beta distribution between 0 and 1, with an effective standard deviation of 0.01... and also doesn't apply if the distribution isn't bounded...
3. Estimate a width scale for $$S$$, $$w$$, randomly picking an initial interval of size $$w$$, containing $$x_0$$ (previous accepted proposal), and then expand it via a ‘stepping out’ procedure. Similar to 2 we can reject points where $$y>f(x_p)$$ and possibly use that to expand our estimate of $$S$$ as a stopping criterion.
4. Similar to 3, we can randomly pick an initial interval of size $$w$$, and then expand it by a doubling procedure and have a similar stopping criterion.


Now, it would seem that the doubling procedure may be better in most cases as it can expand to larger sizes more quickly, making up for a potentially underestimated $$w$$. However, it requires a rejection test, that scheme 3 doesn't need/have, to ensure the transition is reversible. This added overhead means that more often than not the "step-out" scheme is used unless you are dealing with an extremely heavy-tailed distributions where a linear search would be slow enough to justify the added overhead of the doubling procedure.

On top of some rules for expanding the intervals, both schemes are made more efficient by a shrinkage step that uses rejected samples after the initial growth to, you guessed it, shrink the intervals to focus in on more relevant areas. 

The general process for both schemes is detailed in Figures 1 and 2 in [Neal's paper](https://arxiv.org/abs/physics/0009028) that I've copy-pasted below (not including the captions, head over to the paper on page 9 if you want those).

<figure style="text-align: center; margin: 20px 0;">
<div style="text-align: center;">
    <img 
    src="/files/BlogPostData/2026-02-SliceGibbs/Slice/Neal_Fig12_dist.png" 
    alt="Neal Figure 1" 
    style="width: 100%; height: auto; border-radius: 8px;">
    <p style="font-size: 0.9em; margin-top: 5px; color: #555;">Figure 1/2 distribution</p>
</div>
  <div style="display: flex; justify-content: space-between; gap: 10px; margin-bottom: 10px;">
    <div style="flex: 1;">
      <img 
        src="/files/BlogPostData/2026-02-SliceGibbs/Slice/Neal_Fig1_Steps.png" 
        alt="Neal Figure 1" 
        style="width: 100%; height: auto; border-radius: 8px;">
      <p style="font-size: 0.9em; margin-top: 5px; color: #555;">Figure 1</p>
    </div>
    <div style="flex: 1;">
      <img 
        src="/files/BlogPostData/2026-02-SliceGibbs/Slice/Neal_Fig2.png" 
        alt="Neal Figure 2" 
        style="width: 100%; height: auto; border-radius: 8px;">
      <p style="font-size: 0.9em; margin-top: 5px; color: #555;">Figure 2</p>
    </div>
  </div>
  <figcaption style="font-style: italic; font-size: 0.95em; color: #333; line-height: 1.5;">
    Figures 1 and 2 from Neal (2000) (adapted) depicting the step-out and doubling schemes respectively. 
    In step a the estimated interval is expanded, and in b the interval is shrunk 
    and an accepted proposal is shown.
  </figcaption>
</figure>


If this doesn't make 100% sense at the moment that's fine, we'll first go through the algorithms and then a couple GIFs to see how they work in detail.

### 2.2.1 The "Stepping-Out" Algorithm

The stepping-out scheme is the most commonly used scheme for slice sampling because it is robust and preserves detailed balance without extra checks. 

The process is as follows:
- Find the Slice: Given the current $$x_0$$, pick a vertical height $$y \sim U(0, f(x_0))$$. (not changes to above)
- Expand the Interval: Randomly position an interval $$(L, R)$$ of width $$w$$ around $$x_0$$. 
    - If $$f(L) > y$$, move $$L$$ left by $$w$$. If $$f(R) > y$$, move $$R$$ right by $$w$$. 
    - Repeat until both endpoints are outside the slice (i.e., $$f(L) < y$$ and $$f(R) < y$$).
- Shrink and Sample: Pick $$x_{prop} \sim U(L, R)$$. 
    - If $$f(x_{prop}) > y$$, accept it. 
    - If not, shrink the interval by setting $$L$$ or $$R$$ to $$x_{prop}$$ (whichever side $$x_{prop}$$ was on relative to $$x_0$$) and try again.
    

Here's the algorithm for this.

>
#### Slice Sampling: Stepping-Out Scheme
1. Initialize:
    - Have a target density $$f(x)$$, 
    - Propose an initial point $$x_0$$, a
    - Propose an estimated scale width $$w$$.
    - Figure out how many iterations of the algorithm you can be bothered waiting around for $$N$$.
2. For each iteration $$n$$ from $$1$$ to $$N$$:
    - Slice: Pick a vertical height $$y \sim U(0, f(x_n))$$.
    - Stepping-Out:
        - Create an initial interval $$I = (L, R)$$ by picking $$U \sim U(0, 1)$$ and setting $$L = x_n - w \cdot U$$ and $$R = L + w$$.
        - While $$y < f(L)$$, subtract $$w$$ from $$L$$.
        - While $$y < f(R)$$, add $$w$$ to $$R$$.
        - Shrinkage (The Proposal): Repeat until a point is accepted:
            - Sample a proposal $$x_{prop} \sim U(L, R)$$.
            - If $$f(x_{prop}) > y$$, Accept: $$x_{n+1} = x_{prop}$$ and break.
            - Else, Shrink: If $$x_{prop} < x_n$$, set $$L = x_{prop}$$. Else, set $$R = x_{prop}$$.


<div style="text-align: center;">
  <img 
    src="/files/BlogPostData/2026-02-SliceGibbs/Slice/Slice_StepOut.gif" 
    style="width: 99%; height: auto; border-radius: 8px;">
</div>


### 2.2.2 The "Doubling" Algorithm

The doubling scheme follows a similar logic but attempts to find the boundaries of the slice exponentially faster:
- Find the Slice: Same as above ($$y \sim U(0, f(x_0))$$).
- Double the Interval: Double the size of the interval by randomly adding the current width $$w$$ to either the left or the right side.
- Acceptance Test: Because doubling can "jump" over large regions of low density and land in a separate mode, you must perform a reversibility test. 
    - When a point $$x_{prop}$$ is proposed, you have to verify that if you had started at $$x_{prop}$$, the doubling procedure _could_ have produced the exact same interval you are currently using. 
    - Particularly, you don't want to find that the procedure would stop before getting to the previously accepted point.
    - If this test fails, the point is rejected to prevent biasing the sampler toward wider regions of the distribution.
    - In [Neal's paper](https://arxiv.org/abs/physics/0009028) this test is shown in Figure 6.

And the algorithm...


>
#### Slice Sampling: Doubling Scheme
1. Initialize:
    - Have a target density $$f(x)$$, 
    - Propose an initial point $$x_0$$, a
    - Propose an estimated scale width $$w$$.
    - Figure out how many iterations of the algorithm you can be bothered waiting around for $$N$$.
    - Figure out a limit on the number of doublings $$K$$.
2. For each iteration $$n$$ from $$1$$ to $$N$$:
    - Slice: Pick a vertical height $$y \sim U(0, f(x_n))$$.
    - Doubling:
        - Create an initial interval $$I = (L, R)$$ as above.
        - Repeat $$K$$ times (or until both $$f(L) < y$$ and $$f(R) < y$$):
            - Flip a coin. 
                - If Heads, move $$L$$ left by $$(R-L)$$. 
                - If Tails, move $$R$$ right by $$(R-L)$$.
    - Shrinkage with Reversibility Check: Repeat until a point is accepted:
        - Sample $$x_{\text{prop}} \sim U(L, R)$$.
        - If $$f(x_{\text{prop}}) > y$$ AND the `AcceptCheck(x_prop, x_n, y, L, R, w)` is true: Accept: $$x_{n+1} = x_{\text{prop}}$$ and break.
        - Else, Shrink: If $$x_{\text{prop}} < x_n$$, set $$L = x_{\text{prop}}$$. Else, set $$R = x_{\text{prop}}$$.
>
#### Func AcceptCheck($$x_\text{prop}$$, $$x_n$$, $$y$$, $$L$$, $$R$$, $$w$$):
1. Initialize:
    - Let $$(\hat{L}, \hat{R}) \leftarrow (L, R) $$ .
    - Set rejected = False.
2. While $$\hat{R} - \hat{L} > 1.1w$$:
    - M $$\leftarrow$$ $$(\hat{R} - \hat{L})/2$$
    - if $$x_n < M$$ and $$x_{\text{prop}} \geq M$$ or $$x_n \geq M$$ and $$x_{\text{prop}} \lt M$$ then rejected = True
    - if $$x_\text{prop} < M$$ then $$\hat{R} \leftarrow M$$ else $$\hat{L} \leftarrow M$$
    - if rejected and $$y \geq f (\hat{L})$$ and $$y \geq f (\hat{R})$$ --> new point is not acceptable
3. Return not(rejected)

<div style="text-align: center;">
  <img 
    src="/files/BlogPostData/2026-02-SliceGibbs/Slice/doubling_slice_sampling.gif" 
    style="width: 99%; height: auto; border-radius: 8px;">
</div>



### 2.2.3 Final thoughts on the algorithms

Notice that in step 3 of both algorithms, if we pick a point outside the slice, we don't just throw it away and stay at $$x_0$$ (like in Metropolis-Hastings). We use that "failed" point to define a new, smaller boundary. 

This means the sampler learns the shape of the slice during every single iteration. If $$w$$ was way too large, the shrinkage step quickly brings it down to size. 

This makes slice sampling much more stable than standard Metropolis-Hastings, where a bad proposal width $$w$$ can lead to an acceptance rate of 0% and a stalled chain. And I think is one reason why slice sampling was later used as part of HMC-NUTS.

## 2.3 Are they valid MCMC algorithms?

### Ergocity


### Detailed Balance

## 2.4 Extensions and Variants

    - Multivariate slice sampling (coordinate-wise)
    - Elliptical slice sampling (for Gaussian priors)
    - Shrinking rank slice sampling


# 3. Examples

## 3.1 Plain ol' gaussian example

## 3.2 Bivariate normal example

## 3.3 Ba-nana-nana-nana-nana Banan



<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2026-02-SliceGibbs/Gibbs/banana_Gibbs_sampler.gif" 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>




# 4. Conclusion



<hr style="margin-top: 40px; margin-bottom: 20px; border: 0; border-top: 1px solid #eee;">

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  
  <div style="width: 48%; text-align: left;">
    {% if page.manual_prev_url %}
      <div style="font-weight: bold; font-size: 0.9em; margin-bottom: 5px;">
        &larr; Previous post
      </div>
      <a href="{{ page.manual_prev_url }}" style="text-decoration: underline;">
        {{ page.manual_prev_title }}
      </a>
    {% endif %}
  </div>

  <div style="width: 48%; text-align: right;">
    {% if page.manual_next_url %}
      <div style="font-weight: bold; font-size: 0.9em; margin-bottom: 5px;">
        Next post &rarr;
      </div>
      <a href="{{ page.manual_next_url }}" style="text-decoration: underline;">
        {{ page.manual_next_title }}
      </a>
    {% endif %}
  </div>

</div>

# Footnotes