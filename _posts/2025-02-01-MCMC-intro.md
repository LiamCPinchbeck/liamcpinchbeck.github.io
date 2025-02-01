---
title: 'Markov Chain (+) Monte Carlo methods'
date: 2025-02-01
permalink: /posts/2025/02/2025-02-01-mcmc-guide/
tags:
  - MCMC
  - Introductory
---


In this post I'll go through what is MCMC? How is it useful for statistical inference? And the conditions under which it is stable.


# PAGE UNDER CONSTRUCTION - read at your own risk

As usual, here are some other resources if you don't like mine.

- [The algorithm that (eventually) revolutionized statistics](https://www.youtube.com/watch?v=Jr1GdNI3Vfo&t=307s&pp=ygUQVmVyeSBOb3JtYWwgTUNNQw%3D%3D)
    - Focuses on Metropolis-Hastings algorithm but briefly dives into MCMC and detailed balance
- [Markov Chain Monte Carlo (MCMC): Data Science Concepts](https://www.youtube.com/watch?v=yApmR-c_hKU&t=277s&pp=ygUQVmVyeSBOb3JtYWwgTUNNQw%3D%3D)
- [Markov Chains Clearly Explained! Part - 1](https://youtu.be/i3AkTO9HLXo)
- [Markov Chains Clearly Explained! Part - 2](https://youtu.be/VNHeFp6zXKU)
- [Markov Chains: n-step Transition Matrix \| Part - 3](https://youtu.be/Zo3ieESzr4E)
- [Markov Chains: Data Science Basics](https://youtu.be/prZMpThbU3E)
- [Monte Carlo Simulations: Data Science Basics](https://youtu.be/EaR3C4e600k)
- [A Conceptual Introduction to Markov Chain Monte Carlo Methods](https://arxiv.org/abs/1909.12313) - [Joshua S. Speagle](https://arxiv.org/search/stat?searchtype=author&query=Speagle,+J+S)
- [An effective introduction to the Markov Chain Monte Carlo method](https://arxiv.org/abs/2204.10145) - [Wenlong Wang](https://arxiv.org/search/physics?searchtype=author&query=Wang,+W)


--- 

## Table of Contents
- [What is MCMC?](#what-is-mcmc)
    - [Markov Chains](#)
- [Detailed Balance and Stationarity](#detailed-balance-and-stationarity)
- [How Metropolis-Hastings satisfies detailed balance](#how-metropolis-hastings-satisfies-detailed-balance)
- [Practical considerations for MCMC](#practical-considerations-for-mcmc)
- [Next Steps](#next-steps)


---



## What is MCMC?

<!-- As I'm writing this, I'm trying to be the silliest boi that ever was. The boingiest. Juiciest. Jesterous. Meandering to the pointiest[^1]. But I'm finding it quite hard to come up with a nice intro to go into MCMC algorithms. I could tell you that Markov Chain Monte Carlo methods is actually two methods that have been beautifully smashed together to give us a sampling algorithm that given enough time wil start drawing exactly representative samples of whatever distribution we wish[^2]. But that seems a bit dry. So I'm going to detail the [^3] time I was climbing a mountain blind folded, wearing a pair of stelletos, and altimeter.

Now, as a clumsy person, I have quite a difficulty wearing high heels[^4].

[^1]: Wait no, what was that middle one.
[^2]: Albeit "infinite time" can sometimes be the "enough" time
[^3]: definitely real
[^4]: however great they make my legs look -->

MCMC stands for "Markov Chain Monte Carlo" which actually details the combination of two separate (but often than not combined) ideas of ["Markov Chains"](https://en.wikipedia.org/wiki/Markov_chain) and ["Monte Carlo methods"](https://en.wikipedia.org/wiki/Monte_Carlo_method) that simulate data to approximate targets of interest. In the case of Markov Chains we are generally interested in the "equilibrium distribution" or "equilibrium state" and for Monte Carlo methods is a broad category of methods that use random sampling to obtain results of interest[^1]. I'll attempt to introduce both separately but then cover some important notes on the "MCMC" circling back to the Metropolis-Hastings algorithm for a bit as well.

[^1]: I know this is a bit vague, but this in part because of the wide range of outputs you can get from Monte Carlo methods. Plus, I'll detail some concrete examples below.

---

### Markov Chains

To introduce this I'm going to closely follow an example from one of the [(VCE) General Mathematics](https://www.vcaa.vic.edu.au/curriculum/vce/vce-study-designs/generalmathematics/Pages/Index.aspx)[^2] textbooks that I use with students.

[^2]: I do love that this relatively abstract concept that has broad implications not only for day-to-day life but for advanced statistical methods and many concepts in the natural sciences is taught in the base maths curriculum in Victoria but not the more advanced maths curricula?? Good job VCAA. 

Let's say that you own a rental car company that originated in the town of Bendigo and have now opened a new branch in the town of Colac (trust me we'll get back to probability distributions in a minute). 

All your cars are currently at the Bendigo branch, but as time goes on you expect drivers from the Bendigo branch to drop off their cars in Colac some percentage of the time.

Specifically you expect that 40% of the cars in Colac will be dropped off in Bendigo because the number of people that live around Bendigo is larger, and similarly only 10% of cars from Bendigo are dropped off at Colac.

From this we can see that 60% of the cars in Colac will also be dropped off at Colac, and 90% of the cars in Bendigo will stay in Bendigo.

This is a lot simpler to see if from the below (very sophisticated) diagram (that I definitely didn't screenshot from PowerPoint).


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-01/MarkovChain_example.png" 
    alt="2D Brute forced posterior on our gradient and intercept" 
    title="2D Brute forced posterior on our gradient and intercept" 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>


### Monte Carlo (biscuit) methods



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