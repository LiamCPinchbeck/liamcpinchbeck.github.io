---
title: 'Rejection Sampling'
date: 2025-01-28
permalink: /posts/2025/01/2025-01-28-rejection-sampling/
tags:
  - Introductory
  - Sampling Methods
#   - category2
---

In this post I'm going to introduce rejection sampling as a way to generate samples from an unnormalised pdf as further background to MCMC.

Like my posts so far, I take heavy inspiration from a few resources. In this case these in particular are:
1. [Accept-Reject Sampling : Data Science Concepts](https://youtu.be/OXDqjdVVePY) - ritvikmath
2. [An introduction to rejection sampling](https://youtu.be/kYWHfgkRc9s) - Ben Lambert
    - Although this one is icky because he uses [mathematica](https://dictionary.cambridge.org/dictionary/english/horrible)

---

## Table of Contents
- [Intuition Introduction](#intuition-introduction)
    - [Further Examples](#further-examples)
- [Math Intro](#more-mathematical-introduction)
- [Next Steps](#next-steps)

## Intuition Introduction

Let's begin with the same kind of function as in my previous post on Inverse Transform Sampling (IVS).


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/exp_func_plot.png" 
      alt="Plot showing how exp(-3x) looks." 
      title="Plot showing how exp(-3x) looks." 
      style="width: 75%; height: auto; border-radius: 8px;">

</div>

Similar to IVS, presuming that we can generate uniform samples between any two finite bounds, how can we produce an algorithm to give us samples of chosen PDF?

One thing you could do for the case above is produce samples between 0 and 2 for $$x$$ and 0 to 1 for $$y$$, producing the below scatter plot.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/initial_uniform_samples.png" 
      alt="2D Uniform samples in X ~U(0, 2) and Y~U(0,1)." 
      title="2D Uniform samples in X ~U(0, 2) and Y~U(0,1)." 
      style="width: 75%; height: auto; border-radius: 8px;">

</div>

We then overlay our pdf and the samples.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/uniform_samples_pdf_overlay.png" 
      alt="2D Uniform samples in X ~U(0, 2) and Y~U(0,1) with exp(-3x) overlaid." 
      title="2D Uniform samples in X ~U(0, 2) and Y~U(0,1) with exp(-3x) overlaid." 
      style="width: 75%; height: auto; border-radius: 8px;">

</div>

Then because $$y$$ is uniformly distributed, you can think of what we have as a [dot plot](https://en.wikipedia.org/wiki/Dot_plot_(statistics)), and throw out samples above our pdf.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/accepted_samples_exp_-3x.png" 
      alt="Figure of blue samples that fall below PDF curve and are thus accepted." 
      title="Figure of blue samples that fall below PDF curve and are thus accepted" 
      style="width: 75%; height: auto; border-radius: 8px;">

</div>

Converting these samples into a histogram we then find the following.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/exp_accepted_samples_histogram_1000_samples.png" 
      alt="Figure of blue samples that fall below PDF curve and are thus accepted." 
      title="Figure of blue samples that fall below PDF curve and are thus accepted" 
      style="width: 75%; height: auto; border-radius: 8px;">

</div>

Which you can kind of see is following the right curve but let's increase the number of samples to be sure.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/rejection_sampling_process_10000_samples.png" 
      alt="Figure of blue samples that fall below PDF curve and are thus accepted with 10000 samples." 
      title="Figure of blue samples that fall below PDF curve and are thus accepted with 10000 samples" 
      style="width: 100%; height: auto; border-radius: 1px;">

</div>

And that's the basic idea of [Rejection Sampling](https://en.wikipedia.org/wiki/Rejection_sampling). 

This base form is obviously extremely inefficient as producing exact representative samples unlike Inverse Transform Sampling. However, it is easier to implement for multi-dimensional distributions and when you can't rigorously normalise the PDF (either due to dimensionality or stability). The only requirement is that you have some $$M$$ such that $$PDF(x)<M$$ for all $$x$$. In the previous case $$M=1$$, but we could also have used 2 or anything higher if we wanted, it would just be less efficient as we would be wasting more samples.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-28/rejection_sampling_process_10000_samples_M_2.png" 
      alt="Figure of blue samples that fall below PDF curve and are thus accepted with 10000 samples and M=2." 
      title="Figure of blue samples that fall below PDF curve and are thus accepted with 10000 samples and M=2" 
      style="width: 100%; height: auto; border-radius: 1px;">

</div>

Here's the code to produce the plots for $$M=1$$.

```python
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform

lambda_val = 3

# Choose the number of samples
num_samples = int(1e4)

# Sample X uniformly between 0 and 2
uniform_x_samples = uniform(loc=0, scale=2).rvs(num_samples)

# Sample Y uniformly between 0 and 1
uniform_y_samples = uniform(0, 1).rvs(num_samples)



fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

ax = np.array(ax).flatten()

## Producing plot with overlay of uniform samples and PDF
ax[0].scatter(uniform_x_samples, uniform_y_samples, s=1)
ax[0].set_ylabel("y")
ax[0].set_xlabel("x")
ax[0].grid(ls='--', c='grey', alpha=0.2)


ax[0].plot(x_inputs, exp_func(x_inputs, lambda_val), label=r"$p(x) = exp(-$"+f"{lambda_val}"+r"$x)$", c='k')


## Producing plot with rejected samples
ax[1].plot(x_inputs, exp_func(x_inputs, lambda_val), label=r"$p(x) = exp(-$"+f"{lambda_val}"+r"$x)$", c='k')

# Find indices of samples where y_i < PDF(x_i)
good_point_indices = np.where(uniform_y_samples<exp_func(uniform_x_samples, lambda_val))[0]

# Look at samples that don't satisfy criterion and reduce their opacity
ax[1].scatter(uniform_x_samples[~good_point_indices], uniform_y_samples[~good_point_indices], s=1, label='bad samples', alpha=0.2, c='tab:orange')

# Look at samples that do satisfy criterion 
ax[1].scatter(uniform_x_samples[good_point_indices], uniform_y_samples[good_point_indices], s=1, label='good samples', c='tab:blue')


ax[1].set_ylabel("y")
ax[1].set_xlabel("x")
ax[1].grid(ls='--', c='grey', alpha=0.2)



## Producing the histogram
good_point_indices = np.where(uniform_y_samples<exp_func(uniform_x_samples, lambda_val))[0]

hist_output = ax[2].hist(uniform_x_samples[good_point_indices], label='good samples', density=True, bins=48)

func_vals = exp_func(x_inputs, lambda_val)
ax[2].plot(x_inputs, np.max(hist_output[0][:-5])*func_vals/np.max(func_vals[:-5]), label=r"$p(x) = exp(-$"+f"{lambda_val}"+r"$x)$", c='k')

ax[2].set_ylabel("y")
ax[2].set_xlabel("x")
ax[2].grid(ls='--', c='grey', alpha=0.2)
plt.tight_layout()
plt.show()
```

### Further Examples


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



## More mathematical introduction

## Next Steps
