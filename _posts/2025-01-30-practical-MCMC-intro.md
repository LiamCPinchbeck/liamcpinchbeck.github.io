---
title: 'Practical MCMC Intro/Fitting a line II'
date: 2025-01-30
permalink: /posts/2025/01/2025-01-30-practical-MCMC-intro/
tags:
  - Bayesian Analysis
  - MCMC
  - Introductory
#   - category2
---

In this post I'm going to try to give an intuitive intro into MCMC methods without getting bogged down in much of the math to show the utility of these methods.

MCMC is one of the most successful analytical methods that statisticians have ever used and is the bench mark for all future analysis methods that we will explore. I am basing this tutorial on various sources such as:
- [A Conceptual Introduction to Markov Chain Monte Carlo Methods](https://arxiv.org/abs/1909.12313) - Joshua S. Speagle
- [Markov chain Monte Carlo sampling](https://astrowizici.st/teaching/phs5000/5/) - Andy Casey
- The wikipedia page on this topic is also very good [MCMC](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)

## Table of Contents
- [Motivation](#motivation)
  - [Brute force? In this economy?](#brute-force-in-this-economy)
- [What do we actually get from our analysis?](#what-do-we-actually-get-from-our-analysis)
    - [Samples!?](#samples)
- [How do we sample our posterior?](#how-do-we-sample-our-posterior)
    - [MCMC! An intuitive intro](#mcmc-an-intuitive-intro)
    - [MCMC! Slightly more mathy intro](#mcmc-slightly-more-mathy-intro)
- [What packages can we use?](#what-packages-can-we-use)
- [Coding it up](#coding-it-up)
- [Well... now what?](#well-now-what)

## Motivation
### Brute force? In this economy?

In the last post we brute forced our analysis but scanning the whole entire parameter space for areas of high posterior probability. Here's a quick recape.

1. We were given some data that I told you was from a straight line with some added gaussian noise

    <div style="text-align: center;">
    <img 
        src="/files/BlogPostData/2025-01-30/initial_data.png" 
        alt="Straight Line Data" 
        title="Initial data distributed about a straight line of unknown parameters." 
        style="width: 75%; height: auto; border-radius: 8px;">
    </div>

2. We then quantified a likelihood and prior based on this information and produced this something like this colormap of the posterior probabilities of the input parameters

    <div style="text-align: center;">
    <img 
        src="/files/BlogPostData/2025-01-30/brute-force-unnormed-posterior.png" 
        alt="2D Brute forced posterior on our gradient and intercept" 
        title="2D Brute forced posterior on our gradient and intercept" 
        style="width: 75%; height: auto; border-radius: 8px;">
    </div>



In most scenarios this is infeasible (or _at least_ __expensive__) and we need something more sophisticated to explore regions of high probabilty and somehow get some sort of representation of our posterior. So what could we do instead.

### What do we actually get from our analysis?

If our goal is to get something representative of the posterior we generally wantto do 1 or more of the following:
1. Guess where the most optimal set of parameter values are based on our data is (not unique to posterior inference, can just be done with optimisation)
2. Generate further values after taking into account parameter uncertainties
3. __Quantify uncertainty__
4. Compare models via evidence values (normalisation for the posterior)

So whatever other method we use to generate something representative of the posterior we need to keep these in mind.

You are already familiar with another way that we often represent the results of analysis, that is through samples. For example, sure I could tell you that for model X parameter Y follows a normal distribution with mean of 1 and standard deviation of 0.5. (First of all this only works if our result has a functional representation and we presume our audience even knows what a normal distribution is...you'd be surprised.) Or, we could show them a histogram of our results like the following.

```python
from scipy.stats import norm
from matplotlib import pyplot as plt
import numpy as np


samples = norm(scale=0.5, loc=1).rvs(100000)


fig, ax = plt.subplots(1,1, figsize=(4,4), dpi=200)
ax.hist(samples, bins=48, label="Y samples", density=True)
ax.set_xlabel("Y")
ax.legend()
plt.show()
```
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-30/Y_samples.png" 
    alt="Samples of a theoretical Y parameter" 
    title="Samples of a theoretical Y parameter" 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>

### Samples!?

Now if you tell anyone off the street that the parameter follows this distribution, they would be able glean most of the key information (although I wouldn't recommend pulling people of the street and asking them random statistics questions btw...you often don't get the best responses. Unless the question is how many times can someone wear in the span of ~10 seconds, in which it's extremely informative...). 
1. You can see where the mode of the distribution is
2. We could generate further samples of variables by using the samples in this distribution and see the result
3. You can see the spread of the distribution on the parameter to understand our uncertainty on it. (We can also construct [credible intervals](https://en.wikipedia.org/wiki/Credible_interval) through various methods to get quantitative values for our uncertainties )

And from these samples we've also shown essentially the same information a our first brute scan colormap. What we require are a set of samples representative of the same probability density! This both reduces the initial computation cost as we don't have to explore regions of the parameter space where the probabilities are extremely small and in the storage of the result, as we will likely only require $$\lesssim 100,000$$ samples/numbers to show this.

We don't explicitly have a way to get evidence values for model comparison from this, but we can tackle this later if/when we look at [Nested Sampling](https://en.wikipedia.org/wiki/Nested_sampling_algorithm).


### How do we sample our posterior?

Now, unlike the above normal distribution, our function isn't normalised, we have the top part of the fraction that makes up Bayes' theorem.

$$\begin{align}
p(\vec{\theta}\mid\vec{d}) = \frac{\mathcal{L}(\vec{d}\mid\vec{\theta})\pi(\theta)}{\mathcal{Z}(\vec{d})}
\end{align}$$

But from our perspective, $$\vec{d}$$ is a constant, so you can say,

$$\begin{align}
p(\vec{\theta}\mid\vec{d}) \propto \mathcal{L}(\vec{d}\mid\vec{\theta})\pi(\theta).
\end{align}$$

Now you might want to calculate $$\mathcal{Z}(\vec{d})$$ directly but you would run into the same issues (and worse) when we tried to directly scan the posterior to begin with (this is further explore in the [Section 4](https://arxiv.org/abs/1909.12313) of Speagle's introduction). So we need some method where we can sample something _proportional_ to a probability density.


## MCMC! An intuitive intro

One answer to this is [Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo). 

Now in advance, this part of the tutorial is ___extremely___ influenced by the YouTuber [ritvikmath's](https://www.youtube.com/@ritvikmath) videos [Accept-Reject Sampling : Data Science Concepts](https://www.youtube.com/watch?v=OXDqjdVVePY), [Markov Chain Monte Carlo (MCMC) : Data Science Concepts](https://www.youtube.com/watch?v=yApmR-c_hKU), [Metropolis - Hastings : Data Science Concepts](https://www.youtube.com/watch?v=yCv2N7wGDCw). His videos are a great intro to these methods with very little foundational knowledge.



### Accept-Reject Sampling

### Basic Markov Chain Monte Carlo

### Accept-Reject Sampling



## What packages can we use?


## Coding it up


## Well... now what?