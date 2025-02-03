---
title: 'Markov Chain Monte Carlo convergence diagnostics'
date: 2225-02-04
permalink: /posts/2025/02/2025-02-04-mcmc-diagnostics/
tags:
  - MCMC
  - Introductory
header-includes:
   - \usepackage{amsmath}

---

In this post I will detail popular diagnostic tests to quantify how well/if your MCMC sampling has converged.

---

Before I jumpy into the many wonderful and interesting different MCMC algorithms (e.g. [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) or [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)) I think it's important to understand whether the algorithm has actually given you what you want or whether it has converged at all in the first place.


And as usual, if you don't like/understand the way I explain these concepts, it's not because you're too dumb, it is probably just that my writing style doesn't match your learning style. So, here are some other nice places to learn about these concepts.

- [Convergence diagnostics for Markov chain Monte Carlo](https://arxiv.org/abs/1909.11827) - [Vivekananda Roy](https://vroys.github.io/)
    - I use a couple similar examples to this
- [MCMC Diagnostics](https://youtu.be/68rdSWT9HMo) - [Stephen Woodcock](https://www.youtube.com/@stephenwoodcock9107)
    - Good, but the sound quality is utterly atrocious
- [Evaluating MCMC samplers](https://statmodeling.stat.columbia.edu/2024/04/27/evaluating-mcmc-samplers/) - [Bob Carpenter](https://bob-carpenter.github.io/)
- [DATA ANALYSIS RECIPES:USING MARKOV CHAIN MONTE CARLO](https://arxiv.org/pdf/1710.06068) - David W. Hogg, and Daniel Foreman-Mackey
- [Convergence tests for MCMC](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/astrophysics/public/icic/data-analysis-workshop/2018/Convergence-Tests.pdf) - [Elan Sellentin](https://www.universiteitleiden.nl/en/staffmembers/elena-sellentin#tab-1)
- [Florida State University Bayesian Workshop - Applied Bayesian Analysis for the Social Sciences - Day 2: MCMC Convergence](https://spia.uga.edu/faculty_pages/rbakker/bayes/Day2/Day2_Convergence.pdf) - [Ryan Bakker](https://spia.uga.edu/faculty_pages/rbakker/)
- [Markov Chain Monte Carlo (MCMC) diagnostics](https://www.statlect.com/fundamentals-of-statistics/Markov-Chain-Monte-Carlo-diagnostics) by [Marco Taboga](https://sites.google.com/site/marcotabogaspersonalwebpage/)
    - Marco Taboga has a pretty [in-depth website on statistics](https://www.statlect.com/fundamentals-of-statistics/) in general that I would recommend a read if you haven't found it already

--- 

## Table of Contents
- [Example MCMC Issues](#example-mcmc-issues)
- [Traceplots - ___DO THIS ANYWAY----DO IIIIIT___](#traceplots)
- [Running mean](#running-mean)
- [Effective Sample Size](#effective-sampling-size)
- [Autocorrelation is not your friend](#autocorrelation-is-not-your-friend)
- [Integrated Autocorrelation Time](#integrated-autocorrelation-time)
- [Gelman-Rubin - The dynamic duo](#gelman-rubin---the-diagnostic-dynamic-duo)
- [Next Steps](#next-steps)



---

## Example MCMC Issues

For as many questions that you are trying to solve with MCMC there are many more _problems_ with MCMC that you could face. In this post I'm going to focus on 5 broad issues relating to MCMC algorithms, but I can tell you that many _many_ times I've gone down this road of diagnostics, and the real problem was that my question was ill-posed or I hadn't actually set up the fundamental statistical method properly. So, absolutely go through these, but if you're doing this because you're getting incorrect results you may want to just double check some base outputs first if you haven't already. 

I would also say that I use these more in practice to make my results _more certain_, i.e. a double check that I didn't miraculously get a false positive result. Enough prologue though, what should we look out for?

1. Convergence (duh)
    - We need to know that the samples that we are using for inference are actually samples from the target distribution and not some biased result because we didn't let the algorithm run for long enough
2. Mixing (time)
    - When statisticians refer to "good" or "bad" mixing when it comes to MCMC they are more accurately referring to the mixing time of the Markov chain.
    - This basically refers to how long/how many iterations it takes for the sampler to start properly sampling from the target distribution
        - Or in a more fancy way how long it takes for a chain to forget where it came from/become independent of it's initialisation
3. Sample Independence
    - As discussed in [another post](/_posts/2025-01-29-practical-MCMC-intro.md) by its very nature MCMC produces correlated samples as each one is dependent on the last
    - When we extract samples from the eventual result we want to know that the sample we take out are approximately independent or independent enough as this is required for if we were to sample the distribution correctly
4. Stability of estimates
    - If I run the sampler with a different starting position, slightly different number of samples, etc, do I get similar results? Or do they fly out the window?
5. Adequacy of the chosen burn-in period
    - We want to the maximise the number of accurate samples we produce so we want to make the burn-in as small as possible without contaminating our results with non-representative samples

## Traceplots


## Running mean


## Effective Sampling Size


## Autocorrelation is not your friend


## Integrated Autocorrelation Time


## Gelman-Rubin - The (diagnostic) dynamic duo




## Next Steps

Next I'll try and detail some more commonly used MCMC methods starting with [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) then [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), [Metropolis-Adjusted Langevin Algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) then finally [Slice Sampling](https://en.wikipedia.org/wiki/Slice_sampling). 