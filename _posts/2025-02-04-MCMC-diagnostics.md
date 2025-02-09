---
title: 'Markov Chain Monte Carlo convergence diagnostics'
date: 2025-02-04
permalink: /posts/2025/02/2025-02-04-mcmc-diagnostics/
tags:
  - MCMC
  - Introductory
header-includes:
   - \usepackage{amsmath}

---

In this post I will detail popular diagnostic tests to quantify how well/if your MCMC sampling has converged. ***UNDER CONSTRUCTION - read at your own risk***

---
# UNDER CONSTRUCTION - read at your own risk

Before I jump into the many wonderful and interesting different MCMC algorithms (e.g. [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) or [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)) I think it's important to understand whether the algorithm has actually given you what you want or whether it has converged at all in the first place.


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
- [Some Diagnostics for MCMC Simulation](https://bookdown.org/kevin_davisross/bayesian-reasoning-and-methods/diagnostics.html) - [Kevin Ross](https://statistics.calpoly.edu/content/kevin-ross)
- [Autocorrelation: What It Is, How It Works, Tests](https://www.investopedia.com/terms/a/autocorrelation.asp) - [Tim Smith](https://www.investopedia.com/contributors/54411/)


--- 

## Table of Contents
- [Example MCMC Issues](#example-mcmc-issues)
- [Traceplots - ___DO THIS ANYWAY---DO IIIIIT___](#traceplots)
- [Running mean](#running-mean)
- [Effective Sample Size (ESS)](#effective-sampling-size)
- [Autocorrelation is not your friend](#autocorrelation-is-not-your-friend)
- [Integrated Autocorrelation Time](#integrated-autocorrelation-time)
- [Gelman-Rubin - The dynamic duo](#gelman-rubin---the-diagnostic-dynamic-duo)
- [Toy Models - You've got a friend](#toy-modelling)
- [Summary](#summary)
- [Next Steps](#next-steps)



---

## Example MCMC Issues

For as many questions that you are trying to solve with MCMC there are many more _problems_ with MCMC that you could face. In this post I'm going to focus on 5 broad issues relating to MCMC algorithms, but I can tell you that many _many_ times I've gone down this road of diagnostics, and the real problem was that my question was ill-posed or I hadn't actually set up the fundamental statistical method properly. So, absolutely go through these, but if you're doing this because you're getting incorrect results you may want to just double check some base outputs first if you haven't already. 

I would also say that I use these more in practice to make my results _more certain_, i.e. a double check that I didn't miraculously get a false positive result. Enough prologue though, what should we look out for?

1. Convergence (duh)
    - We need to know that the samples that we are using for inference are actually samples from the target distribution and not some biased result because we didn't let the algorithm run for long enough
2. Mixing (time)
    - When statisticians refer to "good" or "bad" mixing when it comes to MCMC they are more accurately referring to the [mixing time of the Markov chain](https://en.wikipedia.org/wiki/Markov_chain_mixing_time).
    - This basically refers to how long/how many iterations it takes for the sampler to start properly sampling from the target distribution
        - Or in a more fancy way how long it takes for a chain to forget where it came from/become independent of it's initialization
3. Sample Independence
    - As discussed in [another post](/_posts/2025-01-29-practical-MCMC-intro.md) by its very nature MCMC produces autocorrelated samples as each one is dependent on the last
    - When we extract samples from the eventual result we want to know that the sample we take out are approximately independent or independent enough as this is required for if we were to sample the distribution correctly
4. Stability of estimates
    - If I run the sampler with a different starting position, slightly different number of samples, etc, do I get similar results? Or do they fly out the window?
5. Adequacy of the chosen burn-in period
    - We want to maximise the number of accurate samples we produce so we want to make the burn-in as small as possible without contaminating our results with non-representative samples



## Traceplots

Traceplots as in the name simply _trace_ the progression of a chain in your MCMC algorithm. For example, let's cook up a very basic example where we sample a 1d distribution of two normal distributions stacked on top of each other.

I am using the python package [emcee](https://emcee.readthedocs.io/en/stable/) to do the sampling that uses [Affine-Invariant Ensemble Sampling](https://arxiv.org/abs/1202.3665) where you initialise multiple chains, often called 'walkers', that proposes new samples not only based on it's current position but the positions of the other walkers. For this post you can just think of it as Metropolis-Hastings where the chains talk to each to propose more useful proposal samples[^1]. 

[^1]: One thing to note here is that I'm using a small number of walkers so later plots are easier to look at, but in practice you should be using more if you can.

```python
import numpy as np
import emcee

def log_prob(x, ivar):
    return np.logaddexp(-0.5 * np.sum((x-1) ** 2/ivar**2), -0.5 * np.sum((x+1) ** 2/ivar**2))

ndim, nwalkers = 1, 10
ivar = np.random.rand(ndim)
p0 = np.random.randn(nwalkers, ndim)
print(p0.shape)
p0 = np.random.normal(loc=np.zeros((ndim,)), scale = np.ones((ndim,)), size=(nwalkers, ndim))
print(p0.shape)


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[ivar])
sampler_results = sampler.run_mcmc(p0, 10000)
```

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Basic_Two_Mode_Posterior_Corner.png" 
    alt="Simple bimodal probability density distribution" 
    title="Simple bimodal probability density distribution" 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>

Just as promised, a simple bimodal probability density distribution. Because we know exactly what we put in we know that these samples match what we expected, but we basically never now the exact form of our posterior density, otherwise why are we using MCMC? So if we didn't know _if_ this was what this was meant to look like we could try and figure out if the sampler seems to have converged or not. We can possibly do this by analysing the trace. 

I'm using the [ArviZ](https://python.arviz.org/en/stable/index.html) python package to do this analysis.

```python
import arviz as az
idata = az.from_emcee(sampler, var_names=[''])


# Plot the trace of the samples
ax = az.plot_trace(idata, figsize=(15,5))
ax = np.array(ax).flatten()
ax[0].set_xlabel("x", size=18)
ax[1].set_xlabel("iterations", size=18)
plt.tight_layout()
plt.savefig("converged_bimodal_trace_plot.png")
plt.show()

```
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/converged_bimodal_trace_plot.png" 
    alt="ArviZ traceplot of bimodal gaussians distribution" 
    title="ArviZ traceplot of bimodal gaussians distribution" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

The key takeways here are that 
- if you look at the left plot, which shows the sample distributions for each of the chains/walkers, that they all look roughly the same,
- if you look at th right plot, which shows the positions of the chains as a function of the iteration number of the sampler, that they are centred about the same values

I then wanted to show what a very obviously unconverged trace looks like by giving it progressively more pathological examples but it kept doing it so quickly that I couldn't really show the problematic behaviour properly. So I'm going to re-use a slightly different implementation of the Metropolis-Hastings algorithm that I used in a [previous post](/_posts/2025-01-29-practical-MCMC-intro.md) that I've put in the drop down below but you can skip that as it's not the important bit.

<details>
    <summary>Metropolis-Hastings code</summary>

    <br>

    <pre><code class="language-python">
import numpy as np
import scipy.stats as stats

# Metropolis-Hastings Sampler
def metropolis_hastings(target_pdf, proposal_sampler, proposal_logpdf, num_samples=5000, x_init=1.0):
    samples = [x_init]
    x = x_init
    accept_count = 0

    for _ in range(num_samples):
        x_prime = proposal_sampler(x)  # Propose a new sample
        if x_prime <= 0:  # Ensure positivity for Gamma distribution
            continue

        # Compute acceptance ratio
        p_accept = np.exp((target_logpdf(x_prime) - target_logpdf(x)) + (proposal_logpdf(x, scale=x_prime)-proposal_logpdf(x_prime, scale=x)))
        p_accept = min(1, p_accept)

        if np.random.rand() < p_accept:
            x = x_prime
            accept_count += 1

        samples.append(x)

    return np.array(samples), accept_count / num_samples  # Return samples and acceptance rate

# Symmetric Proposal: Normal Distribution centered at x
def symmetric_proposal_sampler(x, sigma=0.1):
    return np.random.normal(loc=x, scale=sigma)


def symmetric_proposal_logpdf(x, sigma=0.1, scale = 1):
    return stats.norm.logpdf(x, scale=sigma, loc=scale) 


    </code></pre>
</details>

```python
import numpy as np
import scipy.stats as stats

def target_logpdf(x):
    ivar = 1/10
    return (np.logaddexp(-0.5 * np.sum((x-100) ** 2/ivar**2), -0.5 * np.sum((x+100) ** 2/ivar**2)))


num_samples = 10000

symmetric_samples, sym_accept_rate = metropolis_hastings(
    target_logpdf, 
    lambda x: symmetric_proposal_sampler(x, sigma=0.01),
    lambda x, scale: symmetric_proposal_logpdf(x, sigma=0.01, scale=scale),
    num_samples=num_samples,
    x_init=20,

)
```
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Unconverged_Two_Mode_Posterior_Corner.png" 
    alt="corner plot of unconverged MCMC" 
    title="corner plot of unconverged MCMC" 
    style="width: 66%; height: auto; border-radius: 8px;">
</div>

So, I've increased the distance between the peaks, made it so that the proposal sampler was fairly narrowed meaning that it's pretty slow to explore the parameter space and made the initialization pretty far away from either of the modes which has left us with the above. But if we didn't know the form of our posterior, we may think that maybe our data was completely uninformative on our data and the posterior would subsequently be uniform and we'd get the above. However, if we look at the trace we'll see that this is definitely not the case.


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Unconverged_MH_bimodal_gauss.png" 
    alt="trace plot of unconverged MCMC" 
    title="trace plot of unconverged MCMC" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>


So unfortunately I had to switch to a non-ensembler sampler so we can't use the left plot for much, but the right is extremely informative. If the distribution was in fact uniform within two bounds then we would expected the trace to show fluctuations within those bounds, but what we have is the sample values seem to be increasing at every iteration i.e. they haven't started fluctuating about a single set of values (e.g. like the two gaussian centres in the traceplot above) so it hasn't converged. This would indicate that either we should broaden our proposal distribution and/or run the code for longer (in practice the proposal distribution is handled by the given package though). If we do both of these we find the following.

<details>
    <summary>Code</summary>

    <br>

    <pre><code class="language-python">
num_samples = 20000

symmetric_samples, sym_accept_rate = metropolis_hastings(
    target_logpdf, 
    lambda x: symmetric_proposal_sampler(x, sigma=1),
    lambda x, scale: symmetric_proposal_logpdf(x, sigma=1, scale=scale),
    num_samples=num_samples,
    x_init=20,

)
    </code></pre>
</details>

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Converged_MH_bimodal_gauss_including_burn.png" 
    alt="trace plot of seemingly converged MCMC including burn-in samples" 
    title="trace plot of seemingly converged MCMC including burn-in samples" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

You can see that the sampler converges much faster and that the samples eventually start fluctuating about one point. I'll get rid of the burn-in samples so that this is easier to see.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Seemingly_converged_MH_bimodal_gauss_no_burn.png" 
    alt="trace plot of seemingly converged MCMC not including burn-in samples" 
    title="trace plot of seemingly converged MCMC not including burn-in samples" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

Now this looks pretty good, but, we know that the posterior should be centred about two values. Just from the trace we're not able to see anything wrong, so keep this in mind if you usually use the trace diagnostic.

And here's an example of a more typical case where the trace hasn't converged, I'll go through the full example with another diagnostic.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Metropolis_Trace.png" 
    alt="trace plot of not converged MCMC on Leman density" 
    title="trace plot of not converged MCMC on Leman density" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>


## Running mean / statistical moment

A distribution as a whole always has a single mean, it might have multiple modes, but it will always have a single mean. What we do with a running mean is further use the traceplots described above, but as in the name, take a running mean of the samples. If you're sampler has converged then this mean should start to converge on a single value. That's it! Let's look at a couple of examples.

Let's look at the trace of the nice example of the two gaussians first.
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/converged_bimodal_trace_plot.png" 
    alt="ArviZ traceplot of bimodal gaussians distribution" 
    title="ArviZ traceplot of bimodal gaussians distribution" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

We can modify the samples from this to take a running mean like so.
```python
import arviz as az
idata = az.from_dict(posterior={"": [np.mean(samples[:i]) for i in range(1, len(samples))]})


# Plot the trace of the samples
ax = az.plot_trace(idata, figsize=(15,5))
ax = np.array(ax).flatten()
ax[0].set_xlabel("running mean", size=18)
ax[1].set_xlabel("iterations", size=18)
plt.tight_layout()
plt.savefig("converged_bimodal_running_mean_plot.png")
plt.show()
```
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/converged_bimodal_running_mean_plot.png" 
    alt="ArviZ traceplot of bimodal gaussians distribution's running mean" 
    title="ArviZ traceplot of bimodal gaussians distribution's running mean" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>
So this is the expected behaviour, but now let's go back to that messed up example I hinted at before. The mathematical definition for the density is,

$$\begin{align}
p(x, y) = \exp\left(-\frac{x^2}{2}\right)\exp\left(-\frac{-(csc(y)^5-x)^2}{2}\right) +10^{-10}, \quad\quad \forall  x, y \in [-10, 10].
\end{align}$$

The $$10^{-10}$$ is because I was getting numerical instability issues. For reference, this is what the density is meant to look like.
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Exact_Leman_Density.png" 
    alt="Directly calculated Leman density" 
    title="Directly calculated Leman density" 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>

Now I'll throw my Metropolis-Hastings sampler at it and look at the corner plot of the samples.
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Bad_Corner.png" 
    alt="Unconverged Leman density corner" 
    title="Unconverged Leman density corner" 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>

Then the trace.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Metropolis_Trace.png" 
    alt="Unconverged Leman density trace plot" 
    title="Unconverged Leman density trace plot" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

And now the running mean.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Metropolis_Running_Mean_Trace.png" 
    alt="Unconverged Leman density running mean trace plot" 
    title="Unconverged Leman density running mean trace plot" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

Now if we didn't know exactly what this density looked like, maybe we could convince ourselves that the corner plot looked alright. The trace plot then indicated that maybe it didn't fully converge, and now the running means really show that the average hasn't converged. If I let the algorithm go for longer I can then show what this was meant to look like[^1] for this density.

[^1]: I had to fine tune the parameters to get something reasonably bad but not outright horrible for these demonstrations. The most common way that you will get pathological posteriors/target densities is through high dimensionality, but as I show above this _doesn't mean you can't get them in low dimensions_.

The corner.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Good_Corner.png" 
    alt="Converged Leman density corner" 
    title="Converged Leman density corner" 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>



The trace.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Metropolis_Good_Trace.png" 
    alt="Converged Leman density trace plot" 
    title="Converged Leman density trace plot" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>



And the running mean.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/Leman_Metropolis_Running_Mean_Good_Trace.png" 
    alt="Converged Leman density running mean trace plot" 
    title="Converged Leman density running mean trace plot" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

Now we can see by comparing the corner plot to the plot I calculated directly that this MCMC has pretty much converged. From the trace this might not be clear, seeing as the y variable seems to be switching between modes, so it's hard to tell if it's actually centring around modes or mode switching (similar to [label switching](https://mc-stan.org/docs/2_20/stan-users-guide/label-switching-problematic-section.html)). However, with the running mean, you can see that we have converged to a singular value, despite the multi-modality of the problem!

Another key thing to note here is that I've done this for the mean, but you could just as well do this for any statistical measure e.g. running variance, running median, or the running mean of any function of your samples that outputs a singular value.


## Autocorrelation is not your friend

MCMC samples are autocorrelated[^2], but how autocorrelated are they? If not a lot, then great! If quite a lot, not great. Here we detail quantitative ways to estimate how autocorrelated samples from a sampler's chains are.

[^2]: "Autocorrelation" might sound a bit strange if you are unfamiliar. The difference between "correlation" and "autocorrelation" is that _correlation_ refers to the dependencies between two __separate__ variables and [_autocorrelation_](https://en.wikipedia.org/wiki/Autocorrelation) refers to dependencies to itself.


Autocorrelation in simple terms is how related a variable's current value with it's past values.

The calculation of autocorrelation is nicely expressed as a combination of the _autocovariance_ and _variance_.

In case it's been a while since you looked at the formula for _covariance_ here it is again.

$$\begin{align}
\text{cov}(X, Y) = \frac{1}{n} \sum_{i=1}^n (x_i - \mathbb{E}(X))(y_i - \mathbb{E}(Y))
\end{align}$$

Where $$n$$ is the number of datapoints, in our case this will specifically be the number of samples. The _autocovariance_ with _lag_ of $$k$$ is then,

$$\begin{align}
\text{K}(k) = \frac{1}{n} \sum_{i=k+1}^n (x_i - \mathbb{E}(X))(x_{\left(i-k \right)} - \mathbb{E}(X)),
\end{align}$$

when $$k=0$$ then you just get the _variance_, $$\text{var}(X)$$. And then finally we get to the _autocorrelation_ with lag $$k$$,

$$\begin{align}
\text{R}(k) = \frac{\text{K}(k)}{\text{K}(0)}.
\end{align}$$

From this we can see that the magnitude of autocorrelation is less than or equal to 1, the top has to be less than the bottom. Autocorrelation isn't specific to MCMC but has a wide range of applications, particularly in signal processing (so I apologize for those in the field if I don't describe all this at 100%). Just to give as much of an intuitive feel to this as possible, we'll look at a maximal and minimal example of autocorrelation.

Starting with the minimal, 0 autocorrelation relates to completely random values or independent samples (our goal with the MCMC samples). If I generate random samples from a normal distribution, (i.e. each sample is independent from the next) then with enough samples we should see the values approach 0 (as you can see in the GIF below).

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-02-04/normal_random_samples_autocorrelation.gif" 
    alt="GIF showing the running autocorrelations of random independent samples" 
    title="GIF showing the running autocorrelations of random independent samples" 
    style="width: 100%; height: auto; border-radius: 8px;">
</div>

The lines in the gif show the range of samples that are used for the calculation for the different lags on the right. Different lags look for behaviours over shorter ranges for shorter lags and larger ranges for larger lags. When analysing MCMC samples we generally prioritise shorters lags as if we presume that the "good" chains were they have found the target density and are approximately independent ("well mixed") the autocorrelation should decline rapidly. However, long lags may tell you about long term behaviour such as getting trapped in certain areas of the parameter space. 

Best thing to do is similar to the above, look at the autocorrelation for multiple lag values. Let's do this for the 


## Effective Sampling Size

You might think that because you have 100,000 samples in your MCMC chain that you should have enough samples. Well buddy, absolutely not. For various reasons your mixing rate may be abysmally slow or you initialized your chain in what happens to be a local maxima and the sampler got confused or various other reasons. If the main goal of using MCMC is to generate samples representative of the posterior (which it generally is) then we should check that we actually have a reasonable number of them.

Effective sample size (ESS) is an attempt to estimate the amount of independent information or samples in a (autocorrelated) chain. 






## Integrated Autocorrelation Time

When investigating autocorrelation in the previous section we specifically looked at what the correlation between samples in chain look like at various _lags_. Another useful diagnostic is looking at the average lag, this is close to what Integrated Autocorrelation Time is.



## Gelman-Rubin Diagnostic - The dynamic duo

If you run multiple chains, do they give you similar results? Or is it that every time you run a chain it gives you something different. Well the Gelman-Rubin diagnostic helps us quantify this phenomenon.


## MSE


## Toy modelling - You've got a friend

One of the key skills that I've developed as a statistician/physicist is being able to create perfectly representative values of how I believe the data is generated so that I have a clean dataset to test my framework on.



## Summary

*Insert summary table here with each diagnostic and what it can show you*

And one final note. I've tried to show that all of these methods are incomplete, they show you some things but not others per se. So, when doing any kind of diagnostics, more is better. This includes diagnostics but also just the number of samples. If you are unsure of whether a sampler has converged, just run it for longer if you can.

## Next Steps

Next I'll try and detail some more commonly used MCMC methods starting with [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) then [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling), [Metropolis-Adjusted Langevin Algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) then finally [Slice Sampling](https://en.wikipedia.org/wiki/Slice_sampling). 