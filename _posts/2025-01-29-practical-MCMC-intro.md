---
title: 'Practical Intro to the Metropolis-Hastings Algorithm/Fitting a line II'
date: 2025-01-29
permalink: /posts/2025/01/2025-01-29-practical-MHA-intro/
tags:
  - Bayesian Analysis
  - MCMC
  - Introductory
#   - category2
---

In this post I'm going to try to give an intuitive intro into MCMC methods without getting bogged down in much of the math to show the utility of these methods.

MCMC is one of the most successful analytical methods that statisticians have ever used and is the bench mark for all future analysis methods that we will explore. I am basing this tutorial on various sources such as:
- 📝 [A Conceptual Introduction to Markov Chain Monte Carlo Methods](https://arxiv.org/abs/1909.12313) - Joshua S. Speagle
- 🌐 [Markov chain Monte Carlo sampling](https://astrowizici.st/teaching/phs5000/5/) - Andy Casey
- 🌐 The wikipedia page on this topic is also very good [Metropolis–Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)
- ▶️ [Understanding Metropolis-Hastings algorithm](https://youtu.be/0lpT-yveuIA) - [Machine Learning TV](https://www.youtube.com/@MachineLearningTV)
- ▶️ [The algorithm that (eventually) revolutionized statistics - #SoMEpi](https://youtu.be/Jr1GdNI3Vfo) - [Very Normal](https://www.youtube.com/@very-normal)
    - I particularly meshed with this one, so if this post seems too similar... sorry 😬
    - Although I was able to get my own algorithm to work 😬 (although I have no idea why my man went straight to 12 parameters...)
- ▶️ [Metropolis-Hastings - VISUALLY EXPLAINED!](https://youtu.be/oX2wIGSn4jY) - [Kapil Sachdeva](https://www.youtube.com/@KapilSachdeva)
- 🌐 [The Metropolis-Hastings algorithm](https://blog.djnavarro.net/posts/2023-04-12_metropolis-hastings/) - [Danielle Navarro](https://djnavarro.net/)
- 🌐 [Why Metropolis–Hastings Works](https://gregorygundersen.com/blog/2019/11/02/metropolis-hastings/) - [Gregory Gundersen](https://gregorygundersen.com/)

If you don't like/mesh well with how I explain the concepts below absolutely head over to any of these resources. They do widely vary in rigor and presentation method but are all great.

And the style of my GIFs is inspired by this video [PDF Sampling: MCMC - Metropolis-Hastings algorithm](https://youtu.be/zL2lg_Nfi80) - [Ridlo W. Wibowo](https://www.youtube.com/@ridlowibowo08)



## Table of Contents
- [Motivation](#motivation)
  - [Brute force? In this economy?](#brute-force-in-this-economy)
- [What do we actually get from our analysis?](#what-do-we-actually-get-from-our-analysis)
    - [Samples!?](#samples)
    - [How do we sample our posterior?](#how-do-we-sample-our-posterior)
- [Metropolis-Hastings algorithm - Dessert before dinner](#metropolis-hastings-intro)
    - [The algorithm](#the-algorithm)
- [Coding it up...again](#coding-it-up--again)
- [Well... now what?](#well-now-what)

## Motivation
### Brute force? In this economy?

In the last post we brute forced our analysis but scanning the whole entire parameter space for areas of high posterior probability. Here's a quick recape.

1. We were given some data that I told you was from a straight line with some added gaussian noise

    <div style="text-align: center;">
    <img 
        src="/files/BlogPostData/2025-01-29/initial_data.png" 
        alt="Straight Line Data" 
        title="Initial data distributed about a straight line of unknown parameters." 
        style="width: 75%; height: auto; border-radius: 8px;">
    </div>

2. We then quantified a likelihood and prior based on this information and produced this something like this colormap of the posterior probabilities of the input parameters

    <div style="text-align: center;">
    <img 
        src="/files/BlogPostData/2025-01-29/brute-force-unnormed-posterior.png" 
        alt="2D Brute forced posterior on our gradient and intercept" 
        title="2D Brute forced posterior on our gradient and intercept" 
        style="width: 75%; height: auto; border-radius: 8px;">
    </div>



In most scenarios this is infeasible (or _at least_ __expensive__) and we need something more sophisticated to explore regions of high probabilty and somehow get some sort of representation of our posterior. So what could we do instead?

### What do we actually get from our analysis?

If our goal is to get something representative of the posterior we generally wantto do 1 or more of the following:
1. Guess where the most optimal set of parameter values are based on our data is (not unique to posterior inference, can just be done with optimisation)
2. Generate further values after taking into account parameter uncertainties
3. __Quantify uncertainty__
4. Compare models via evidence values (normalisation for the posterior)

So whatever other method we use to generate something representative of the posterior we need to keep these in mind[^1].

[^1]: And if you don't need evidence values or rigorous understanding of uncertainties... just optimise my guy. In the approximate works of Andy Casey, don't burn down forests just so you can look cool using an MCMC sampler.

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
    src="/files/BlogPostData/2025-01-29/Y_samples.png" 
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

Now you might want to calculate $$\mathcal{Z}(\vec{d})$$ directly but you would run into the same issues (and worse) when we tried to directly scan the posterior to begin with (this is further explored in the [Section 4](https://arxiv.org/abs/1909.12313) of Speagle's introduction). So we need some method where we can sample something _proportional_ to a probability density.


## Metropolis-Hastings! - Having dessert before dinner

One answer to this is the [Metropolis-Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm). And like the heading I'm going to show you the end result so you can first see why what we're doing is so cool.

Using the unnormalised posterior function we used in the last post on line fitting, I'm going to create a pretty small function that will almost magically generate sample our distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

# Metropolis-Hastings algorithm for 2D
def metropolis_hastings_function(posterior, data, num_samples, proposal_cov, start_point):
    samples = []
    current = np.array(start_point)
    for _ in range(num_samples):
        # Propose a new point using a 2D Gaussian
        proposal = np.random.multivariate_normal(current, proposal_cov)

        # Calculate acceptance probability
        acceptance_ratio = np.exp(posterior(*data, *proposal) -  posterior(*data, *current))

        # Cap at 1, gotta be a probability and probabilities can't be more than 1
            # Probability _densities_ can but those aren't _probabilities_ until you integrate them
        acceptance_ratio = min(1, acceptance_ratio)  

        # Accept or reject
        if np.random.rand() < acceptance_ratio:
            current = proposal

        samples.append(current)

    return np.array(samples)

# Parameters
num_samples = 50000
proposal_cov = [[0.1, 0], [0, 0.1]]  # Proposal covariance matrix
start_point = [m_true, c_true]  # Starting point for the chain

# Run Metropolis-Hastings
samples = metropolis_hastings_function(unnormalised_log_posterior_better, (y, X_true), num_samples, proposal_cov, start_point)
```
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-29/Base_Metropolis_Hastings_2D.png" 
    alt="Comparison of brute force scan vs sampling with the Metropolis-Hastings algorithm" 
    title="Comparison of brute force scan vs sampling with the Metropolis-Hastings algorithm" 
    style="width: 100%; height: auto; border-radius: 1px;">
</div>

Magic! Now it isn't the prettiest thing, and to be fair to it this algorithm is ([recently, as of 2020](https://ideas.repec.org/a/oup/biomet/v107y2020i1p1-23..html)) over 50 years old! And since then there have been a lot of improvements, but isn't it amazing that with so little code we are able to do something like this? Additionally, with the samples there's a handy package called [corner](https://corner.readthedocs.io/en/latest/). That allows us to have a closer look at our samples.

```python
from corner import corner
default_kwargs = {'smooth': 0.9,
    'label_kwargs': {'fontsize': 16},
    'title_kwargs': {'fontsize': 16},
    'color': 'tab:blue',
    'truth_color': 'k',
    'levels': (0.3934693402873666,
    0.8646647167633873,
    0.9888910034617577,
    0.9996645373720975,
    0.999996273346828),
    'plot_density': True,
    'plot_datapoints': False,
    'fill_contours': True,
    'max_n_ticks': 4,
    'hist_kwargs':{'density':True},
    "smooth":0.9,
}

fig = plt.figure(figsize=(6, 6), dpi=200)
corner(samples, 
    fig=fig,
    bins=36,
    truths=[m_true, c_true,],
    titles=[r"$m$", r"$c$"],
    show_titles=True,
    **default_kwargs)
plt.show()
```
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-29/Base_Metropolis_Hastings_corner_plot.png" 
    alt="Corner plot showing our samples of the unnormalised posterior function using the Metropolis-Hastings algorithm" 
    title="Corner plot showing our samples of the unnormalised posterior function using the Metropolis-Hastings algorithm" 
    style="width: 100%; height: auto; border-radius: 1px;">
</div>

The top left and bottom right plots in the above show the [_marginal distributions_](https://en.wikipedia.org/wiki/Marginal_distribution) of the relevant parameters. These plots effectively show distributions once you take the explicit dependence of the other variable(s) out[^2]. This is extremely useful as when summarising our results by text or by word we can't show the plot, so we typically summarise our findings to one variable at a time, and further simplify it by asking what the width of the [highest density region that contains the same area as a normal distribution](https://en.wikipedia.org/wiki/Credible_interval) between $$1\sigma$$, $$2\sigma$$, $$3\sigma$$ (68%, 95%, 99.7%) or etc.

[^2]: There is kind of still a dependence but it's through the model itself, and importantly, it's not dependent on any _particular_ value of the other variable(s) (as stated). You can also think of it as taking the average of distributions of the relevant variable over the other parameter(s). 

### The algorithm

So here are the steps in plain-ish english.

>
#### Metropolis Algorithm
1. Initialise: 
    1. Have a distribution you want to sample from (duh)
    2. $$f(x)$$, manually create a starting point for the algorithm,
    3. pick a symmetric distribution $$g(x\mid y)$$ to sample from 
        - like a gaussian with a fixed covariance matrix such that $$g(x\mid y)=g(y\mid x)$$
    4. pick the number of samples you can be bothered waiting for $$N$$
2. For each iteration $$n$$/Repeat $$N$$ times
    1. Sample a new _proposal point_ $$x^*$$ from the syymetric distribution centred at the previous sample 
    2. Calculate the _acceptance probability_ $$\alpha$$ given as $$\alpha = f(x^*)/f(x_n)$$
        - And if the acceptance probability is more than 1, cap it at 1.
    3. Generate a number, $$u$$, from a uniform distribution between 0 and 1
    4. Accept or reject[^3]
        1. Accept if: $$u\leq\alpha$$, s.t. $$x_{n+1} = x^*$$
        2. Reject if: $$u>\alpha$$, s.t. $$x_{n+1} = x_n$$

[^3]: The process of comparing the acceptance probability to the uniform sample is to simulate a random process where the probability of accepting the proposal is $$\alpha$$

The first thing to notice is that we don't require the normalisation of our function is as it just depends on the ratios, and the normalisation cancels itself out. And the second is that it doesn't have many steps despite being able to do quite a lot.

Now this is the point that I want to show you an animation of the process, but I don't want to do that with the 2D distribution as it's slow enough as it is and it would be annoying figuring out how to nicely visualise accepting or rejecting a sample. So forgive me for doing this for a single dimensional gaussian that I made up on the spot.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-29/metropolis_gif.gif" 
    alt="GIF showing the process of a Metropolis algorithm" 
    title="GIF showing the process of a Metropolis algorithm" 
    style="width: 100%; height: auto; border-radius: 1px;">
</div>

Slowing it down and having a look at the accepting conditions.
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-29/metropolis_slow_gif.gif" 
    alt="GIF showing the process of a Metropolis algorithm" 
    title="GIF showing the process of a Metropolis algorithm" 
    style="width: 100%; height: auto; border-radius: 1px;">
</div>
You will notice that in this one there are some samples off to the side that don't seem to fit the distribution. We refer to this as the "burn-in phase", it is a sequence of samples at the beginning of MCMC sampling (not specific to Metropolis or the Metropolis-Hastings) where the chain of samples hasn't reached the key part of the distribution yet. When doing your own MCMC sampling you should be sure to throw away a few samples at the beginning[^4].

[^4]: There is no hard and fast rule for this that will work every time but if you're new to MCMC I would start with ~10% of your samples and then wiggle that percentage around for each problem. You want to maximise the number of samples you have in your distribution but you don't want bad ones.

[Very Normal](https://www.youtube.com/@very-normal) had a great analogy for this process, because it may not be immediately intuitive why simply asking the ratio of two probabilities at a time allows us to construct the full probability distribution. 

Let's say you wanted to undertake the average distribution of activities that [Melburnian's](https://en.wiktionary.org/wiki/Melburnian) undertake every week. 
- You go an activity around Melbourne that you think Melburnian's undertake (initialisation)
- You do the activity
- At the end of the activity you ask one of the natives whether they are going to go to a different activity of stay at this one (tomorrow inclusive in both stay and next) with a probability based around how much the Melburnian wants to change activities
    - If they stay, you stay, and then you ask a different person next time when an activity ends
    - If they are going to another activity, follow them (ask them if it's okay first though)
- Repeat

And eventually, even if you didn't pick an activity that was very good, you will eventually be lead to the "good" activities (equilibrium distribution) and start to do the same activities as typical Melburnian's do despite only ever comparing two choices at a time "stay" or "next". However, if this decision process only allows transitions between certain activities (e.g., people who go to cafes only talk to others at cafes), then some activities might be overrepresented while others remain underexplored[^5].

This is an issue for the _Metropolis_ algorithm[^6], which is what I detailed above, but not the generalisation of the algorithm by [Wilfred Keith Hastings](https://en.wikipedia.org/wiki/W._K._Hastings) the _Metropolis Hastings_ algorithm. 


[^5]: Trying to essentially have a common sense explanation of [detailed balance](https://en.wikipedia.org/wiki/Detailed_balance). Please feel free to suggest another short and plain English way to explain this.
[^6]: by [Nicholas Metropolis](https://en.wikipedia.org/wiki/Nicholas_Metropolis)(one of the best names ever btw),  [Arianna W. Rosenbluth](https://en.wikipedia.org/wiki/Arianna_W._Rosenbluth), [Marshall Rosenbluth](https://en.wikipedia.org/wiki/Marshall_Rosenbluth), [Augusta H. Teller](https://en.wikipedia.org/wiki/Augusta_H._Teller) and [Edward Teller](https://en.wikipedia.org/wiki/Edward_Teller)

For the _Metropolis_ algorithm to work we presume that the proposal distribution, $$g$$, is symmetric, i.e. $$g(x\mid y)=g(y\mid x)$$, but this can be restrictive. Some distributions that may produce faster convergence or better fit a particular posterior setup may not be symmetric. So Hastings generalised the result to allow this, by modifying the acceptance probability from $$\alpha = f(x^*)/f(x_n)$$ to $$\alpha = \frac{f(x^*)g(x_n\mid x^*)}{f(x_n)g(x^*\mid x_n)}$$ which accounts for any assymetry in $$g$$ (I'll go into more detail in a later post).

>
#### Metropolis-Hastings Algorithm
1. Initialise: 
    1. Have a distribution you want to sample from (duh)
    2. $$f(x)$$, manually create a starting point for the algorithm,
    3. pick a symmetric distribution $$g(x\mid y)$$ to sample from 
        - like a gaussian with a fixed covariance matrix such that $$g(x\mid y)=g(y\mid x)$$
    4. pick the number of samples you can be bothered waiting for $$N$$
2. For each iteration $$n$$/Repeat $$N$$ times
    1. Sample a new _proposal point_ $$x^*$$ from the syymetric distribution centred at the previous sample 
    2. Calculate the _acceptance probability_ $$\alpha$$ given as $$\alpha = \frac{f(x^*)g(x_n\mid x^*)}{f(x_n)g(x^*\mid x_n)}$$ <small>(Here's the change!)</small>
        - And if the acceptance probability is more than 1, cap it at 1.
    3. Generate a number, $$u$$, from a uniform distribution between 0 and 1
    4. Accept or reject[^3]
        1. Accept if: $$u\leq\alpha$$, s.t. $$x_{n+1} = x^*$$
        2. Reject if: $$u>\alpha$$, s.t. $$x_{n+1} = x_n$$


Now we'll see when an asymmetric proposal distribution can do better than a symmetric one, we'll look at the two algorithms side-by-side[^7].

[^7]: Although special note, they aren't really "two" algorithms it's just that the Metropolis is a _specific case_ of the Metropolis-Hastings


For the distribution that we're trying to model we'll use the [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) with $$\alpha=2$$ and our two proposal distributions will be a normal distribution with a standard deviation of 0.5 for our symmetric distribution and a [log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution) with $$\sigma=0.5$$ for our asymmetric distribution. 

Additionally, through some voodoo magic that I don't understand, [arviz](https://python.arviz.org/en/stable/) can estimate the number of _effective_ samples based on just giving it a set of 'em, but it seems to support my point (more so than I would have thought) so I'll leave it in.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-29/assymetric_proposal.gif" 
    alt="GIF showing a comparison between sampling a gamma distribution using MCMC with a symmetric vs asymmetric proposal distribution" 
    title="GIF showing a comparison between sampling a gamma distribution using MCMC with a symmetric vs asymmetric proposal distribution" 
    style="width: 90%; height: auto; border-radius: 1px;">
</div>

You can see that the asymmetric proposal distribution is able to get the core shape of the distribution quicker than the symmetric distribution. Especially if you look at the left edge of the distribution and the tail passed 7 or so. 

Viva le estadistica[^8]!... (looks up noun gender of statistics in Spanish)... Viva la estadistica[^9]!

[^8]: statistics
[^9]: statistics with correct grammar

## Coding it up ... again

Just for completeness I'll copy-paste my implementation of the Metropolis-Hastings algorithm here as well.

```python
def metropolis_hastings(
        target_pdf, 
        proposal_sampler, 
        proposal_pdf, 
        num_samples=5000, 
        x_init=1.0):

    samples = [x_init]

    x = x_init

    for _ in range(num_samples):

        # Propose a new sample
        x_prime = proposal_sampler(x) 

        # Just for the gamma distribution to ensure positivity. 
            # I couldn't get it to work nicely without this
            # If _you_ wanna use this for something else, I would take this 
            # bad boi out or replace it with a relevant constraint for the 
            # distribution of interest
        if x_prime <= 0:  
            continue

        # Compute acceptance ratio \alpha = f(x^*)/f(x_n) * g(x_n|x^*)/g(x_^*|x_n)
        p_accept = (target_pdf(x_prime) / target_pdf(x)) \
        * proposal_pdf(x, scale=x_prime)/proposal_pdf(x_prime, scale=x)

        # Cap it, bop it, twist it
        p_accept = min(1, p_accept)

        # Simulate random process of accepting the proposed sample with probability p_accept
        if np.random.rand() < p_accept:
            x = x_prime
        # else: x = x

        samples.append(x)

    return np.array(samples)
```

## Well... now what?

Next I'm going to attempt to explain what MCMC in general is and why [detailed balance](https://en.wikipedia.org/wiki/Detailed_balance) as a property of our MCMC algorithms is important and later to maybe give a general intro into the now much more commonly used NUTS algorithm that most MCMC python packages like [emcee](https://emcee.readthedocs.io/en/stable/), [pyMC](https://www.pymc.io/welcome.html) and many others use.

---