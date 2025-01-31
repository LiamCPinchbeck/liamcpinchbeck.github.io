---
title: 'Markov Chain Monte Carlo: Understanding the Foundations'
date: 2225-01-31
permalink: /posts/2025/01/2025-01-31-mcmc-guide/
tags:
  - MCMC
  - Introductory
---

In this post I'll go through what is MCMC? How is it useful for statistical inference? And the conditions under which it is stable.

As usual, here are some other resources if you don't like mine.

- [The algorithm that (eventually) revolutionized statistics](https://www.youtube.com/watch?v=Jr1GdNI3Vfo&t=307s&pp=ygUQVmVyeSBOb3JtYWwgTUNNQw%3D%3D)
    - Focuses on Metropolis-Hastings algorithm but briefly dives into MCMC and detailed balance
- [Markov Chain Monte Carlo (MCMC): Data Science Concepts](https://www.youtube.com/watch?v=yApmR-c_hKU&t=277s&pp=ygUQVmVyeSBOb3JtYWwgTUNNQw%3D%3D)
- 

--- 

## Table of Contents
- [What is MCMC?](#what-is-mcmc)
- [Detailed Balance and Stationarity](#detailed-balance-and-stationarity)
- [How Metropolis-Hastings satisfies detailed balance](#how-metropolis-hastings-satisfies-detailed-balance)
- [Practical considerations for MCMC](#practical-considerations-for-mcmc)
- [Next Steps](#next-steps)


---



## What is MCMC?

As I'm writing this, I'm trying to be the silliest boi that ever was. The boingiest. Juiciest. Jesterous. Meandering to the pointiest[^1]. But I'm finding it quite hard to come up with a nice intro to go into MCMC algorithms. I could tell you that Markov Chain Monte Carlo methods is actually two methods that have been beautifully smashed together to give us a sampling algorithm that given enough time wil start drawing exactly representative samples of whatever distribution we wish[^2]. But that seems a bit dry. So I'm going to detail the [^3] time I was climbing a mountain blind folded, wearing a pair of stelletos, and altimeter.

Now, as a clumsy person, I have quite a difficulty wearing high heels[^4].

[^1]: Wait no, what was that middle one.
[^2]: Albeit "infinite time" can sometimes be the "enough" time
[^3]: definitely real
[^4]: however great they make my legs look
---



## Detailed Balance and Stationarity

Detailed balance is a condition that ensures the Markov Chain converges to the target distribution over time. In this section, I’ll explain:
- What detailed balance means conceptually.
- Why it's necessary for unbiased sampling.
- How it relates to equilibrium in a Markov process.



---



## How Metropolis-Hastings satisfies detailed balance

Metropolis-Hastings (M-H) achieves detailed balance through its acceptance ratio, which compares the relative probability densities of the current and proposed states. Here, I'll walk through the key steps:
1. Proposal sampling.
2. Calculating the acceptance probability.
3. Maintaining detailed balance and ensuring convergence.



---



## Practical considerations for MCMC

When using MCMC, there are several practical factors to consider:
- **Burn-in and thinning**: How many initial samples should be discarded, and whether thinning is necessary.
- **Proposal distribution**: Choosing a proposal distribution that balances exploration and convergence.
- **Diagnostics** (brief overview): Assessing whether the chain has converged and whether samples are representative of the target distribution.



---



## Next Steps

In my next post, we’ll dive into diagnostics for MCMC chains, focusing on algorithms like NUTS, HMC, and Gibbs sampling. We’ll cover:
- Tools to assess convergence (e.g., trace plots, Gelman-Rubin statistic).
- Effective sample size (ESS) and why it matters.


---