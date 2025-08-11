---
title: 'An introduction to Simulation-Based Inference with NPE and NLE'
date: 2025-08-11
permalink: /posts/2025/08/2025-08-11-SBI-w-NPE-NLE/
tags:
  - PyTorch
  - Variational Inference
  - Simulation Based Inference
  - NPE
  - NLE
header-includes:
   - \usepackage{amsmath}
---


In this post, I’ll attempt to give an introduction to simulation-based inference specifically delving into the methods NPE and NLE including rudimentary implementations. ***UNDER CONSTRUCTION***

---

## Resources

As usual, here are some of the resources I’m using as references for this post. Feel free to explore them directly if you want more information or if my explanations don’t quite click for you. I will highly recommend it for this particular post as I'm using it as motivation to learn about these methods myself in more detail.

- [A robust neural determination of the source-count distribution of the Fermi-LAT sky at high latitudes](https://arxiv.org/abs/2505.02906) by [Eckner](https://arxiv.org/search/astro-ph?searchtype=author&query=Eckner,+C) et al.
- [The frontier of simulation-based inference](https://arxiv.org/pdf/1911.01429) by [Kyle Cranmer](https://theoryandpractice.org/), [Johann Brehmer](https://johannbrehmer.github.io/) and [Gilles Louppe](https://glouppe.github.io/)
- [Recent Advances in Simulation-based Inference for Gravitational Wave Data Analysis](https://arxiv.org/abs/2507.11192) by [Bo Liang](https://www.researchgate.net/profile/Bo-Liang-34) and [He Wang](https://iphysresearch.github.io/-he.wang/author/he-wang-%E7%8E%8B%E8%B5%AB/)
    - Really recommend giving this a read, it's hard to find papers that discuss the general topics without getting into the weeds of the specific implementation that they are trying to advocate for or simply too vague.
- [Consistency Models for Scalable and Fast Simulation-Based Inference](https://proceedings.neurips.cc/paper_files/paper/2024/file/e58026e2b2929108e1bd24cbfa1c8e4b-Paper-Conference.pdf)

---

## Table of Contents


---


# Motivation

The TLDR of simulation-based-inference (SBI)[^lfi] is that you have a prior on some parameters $$\vec{\theta}$$ and a simulator $$g$$, which can give you realistic data $$\vec{x}=g(\vec{\theta})$$, and you utilise advances in machine learning to learn the likelihood or posterior for use in analysis without having to actually specify the likelihood directly[^lfi]. 

[^lfi]: Also equivalently known likelihood-free-inference (LFI), but I prefer the use of SBI as the analysis isn't "likelihood-free" per say but that you _learn_ the likelihood instead of providing it from the get-go.

The benefits of SBI include but are not limited to:
1. The _ability to handle large numbers of nuisance parameters_ (see above)
2. The user _does not have to specify the likelihood_ and allows direct inference if a realistic simulator already exists (e.g. climate modelling)
3. There have been a few works showing that _SBI methods can better handle highly non-gaussian and [highly-multi-modal]() relationships_ within probability distributions
4. _Amortised inference_, you can train a model to approximate the probabilities for a dataset and then re-use for other observations relatively trivially
5. Through the use of the simulators and neural networks involved, _SBI is generally easier to parallelise_
6. _Efficient exploration of parameter space_, through the fact that the simulator will often only output realistic data, the algorithms don't have to waste time in regions of the parameter space that don't lead to realistic data.

The ability to handle a large number of nuisance parameters is actually what sparked my interest in SBI through the paper [A robust neural determination of the source-count distribution of the Fermi-LAT sky at high latitudes](https://arxiv.org/abs/2505.02906) by [Eckner](https://arxiv.org/search/astro-ph?searchtype=author&query=Eckner,+C) et al. who used Nested Ratio Estimation (NRE, which I'll discuss later) to analyse data with a huge number of nuisance parameters introduced by an unknown source distribution in the gamma-ray sky.

I would recommend looking at [The frontier of simulation-based inference](https://arxiv.org/pdf/1911.01429) by [Kyle Cranmer](https://theoryandpractice.org/), [Johann Brehmer](https://johannbrehmer.github.io/) and [Gilles Louppe](https://glouppe.github.io/) and the references therein to check these claims out for yourself if you want.


And more recently, I came across this great paper by [Bo Liang](https://www.researchgate.net/profile/Bo-Liang-34) and [He Wang](https://iphysresearch.github.io/-he.wang/author/he-wang-%E7%8E%8B%E8%B5%AB/) called [Recent Advances in Simulation-based Inference for Gravitational Wave Data Analysis](https://arxiv.org/abs/2507.11192) that discusses the use of SBI within gravitational wave data analysis (in the title I know) but it also discusses some of the popular SBI methods in use as of writing. So, I thought I would try and touch on how each of them work in a little more detail than the paper allowed and try to make it a little more general, additionally showing some rudimentary implementations of some of them, with the end goal really being understanding the below figure (Fig. 1 from the paper).

<p>
    <div style="text-align: center;">
    <img 
        src="/files/BlogPostData/2025-08-sbi/model.png" 
        alt="Figure 1. from 'Recent Advances in Simulation-based Inference for Gravitational Wave Data Analysis' detailing various SBI methods"
        title="Figure 1. from 'Recent Advances in Simulation-based Inference for Gravitational Wave Data Analysis' detailing various SBI methods"
        style="width: 90%; height: auto; border-radius: 32px;">
        <figcaption> Fig.1 caption from Liang and Wang's paper - <em>"Overview of five SBI methods—NPE, NRE, NLE, FMPE, and CMPE—designed for efficient Bayesian parameter
estimation. Each method includes distinct training and inference stages. NPE trains a neural network to directly approximate the
posterior from simulated data. NRE and NLE estimate the likelihood ratio and likelihood function, respectively, and integrate with
MCMC for posterior sampling. FMPE uses an ODE solver guided by a neural network to characterize the parameter posterior.
CMPE fits a probability flow with a neural network to sample from posterior distributions. These approaches leverage neural
networks to approximate complex posteriors, providing a computationally efficient and flexible alternative to traditional Bayesian
inference methods. </em></figcaption>
    </div>
</p>

In this post I will go through Neural Posterior Estimation and Neural Likelihood Estimation and in later posts Neural Ratio Estimation, Classifer-based Mutual Posterior Estimation and finally Flow Matching Posterior Estimation (rough order of how hard it will be to make rudimentary implementations).


# Core Idea


First we assume that one has priors for the set of hyperparameters that theoretically influence the data of a given system. e.g.

$$\begin{align}
\vec{\theta}\sim \pi(\vec{\theta}),
\end{align}$$

where $$\vec{\theta}$$ is the set of hyperparameters we are interested in. And further assume (for now) that either: 
- the set of nuisance parameters $$\vec{\eta}$$ can be further sampled based on these values, 
- or that the two sets are independent.

Taking the stronger assumption of independence as it is often not restricting in practice,

$$\begin{align}
\vec{\theta}, \vec{\eta} \sim \pi(\vec{\theta})\pi(\vec{\eta}).
\end{align}$$

Denoting the simulator that takes in these values and outputs possible realisations of the data as $$g$$ then,

$$\begin{align}
\vec{x} \sim g(\vec{\theta}, \vec{\eta}).
\end{align}$$

This is in effect samples from the likelihood and with this we have samples from the joint probability distribution through Bayes' theorem with marginalisation over the nuisance parameters ,

$$\begin{align}
\vec{x}, \vec{\theta}, \vec{\eta} &\sim \mathcal{L}(\vec{x}\vert \vec{\theta}, \vec{\eta}) \pi(\vec{\theta})\pi(\vec{\eta}) \\
&= p(\vec{x}, \vec{\eta}, \vec{\theta} ),
\end{align}$$

assuming that we can robustly sample over the space of nuisance parameters, we can imagine simultaneously marginalising them out[^m] when generating the samples such that[^exp],

[^m]: in practice this just comes to throwing the samples of the nuisance parameters out
[^exp]: If you're unfamiliar with the notation $$\mathbb{E}_{\vec{\eta}\sim \pi(\vec{\eta}) }$$ denote the average over $$\vec{\eta}$$ using the probability distribution $$ \pi(\vec{\eta})$$ in the continuous case, which is most often assumed for these problems, $$\mathbb{E}_{\vec{\eta}\sim \pi(\vec{\eta}) }\left[f(\vec{\eta}) \right] = \int_{\vec{\eta}} d\left(\vec{\eta}\right) \pi(\vec{\eta}) f(\vec{\eta}) $$

$$\begin{align}
\vec{x}, \vec{\theta} &\sim \mathbb{E}_{\vec{\eta}\sim \pi(\vec{\eta}) } \left[\mathcal{L}(\vec{x}\vert \vec{\theta}, \vec{\eta}) \pi(\vec{\theta})\pi(\vec{\eta})\right] \\
&= \mathcal{L}(\vec{x} \vert \vec{\theta} )\pi(\vec{\theta}) \\
&= p(\vec{x}, \vec{\theta} ).
\end{align}$$

Now because we have these samples, we can try and approximate the various densities that are behind them, using variational approximations such as normalising flows, variational autoencoders, etc. And that's SBI, the different methods differ in specifically how they choose to model these densities (e.g. flow vs VAE) and importantly which densities they are actually trying to approximate. e.g. Neural Posterior Estimation directly models the posterior density $$p(\vec{\theta}\vert\vec{x})$$, while Neural Likelihood Estimation tries to model the likelihood $$\mathcal{L}(\vec{x}\vert \vec{\theta})$$ and then you use something like MCMC to obtain the posterior density $$p(\vec{\theta}\vert\vec{x})$$. Arguably Neural Posterior Estimation is easier to implement, so we'll start with that.

# Neural Posterior Estimation (NPE)


Continuing off from where we left the math, we can then try and estimate the density $$p(\vec{\theta}\vert\vec{x})$$ more simply known as the posterior with some variational approximation $$q(\vec{\theta}\vert \vec{x})$$ such as conditional normalising flows through the forward [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).


$$\begin{align}
\textrm{KL}(p\vert\vert q) &= \mathbb{E}_{\pi(\vec{\theta})} \left[\mathbb{E}_{\vec{x} \sim \mathcal{L}(\vec{x}\vert \vec{\theta})} \left[ \log p(\vec{\theta}\vert \vec{x}) - \log q(\vec{\theta}\vert \vec{x}) \right]\right]
\end{align}$$

We train the variational approximation to the distribution by optimising over the parameters that dictate the shape of said approximation, e.g. $$\vec{\varphi}$$, that are separate to the parameters of the actual problem we are trying to solve. Meaning, that our KL divergence looks more like, 

$$\begin{align}
\textrm{KL}(p\vert\vert q ; \vec{\varphi}) &= \mathbb{E}_{\pi(\vec{\theta})} \left[ \mathbb{E}_{\vec{x} \sim \mathcal{L}(\vec{x}\vert \vec{\theta})} \left[ \log p(\vec{\theta}\vert \vec{x}) - \log q(\vec{\theta}\vert \vec{x} ; \vec{\varphi}) \right]\right],
\end{align}$$

where I use the symbol "$$;$$" to specifically highlight the dependence through the variational approximation and not the conditional dependencies in the density that we are trying to model. 

During training, all that we trying to do is minimise this divergence with respect to the parameters $$\vec{\varphi}$$. Hence from this perspective, the first term inside the divergence is a constant, and plays no part in the _loss function_ we are trying to optimise. So the final form of the _loss_ that we are trying to minimise is[^eqn4],

[^eqn4]: Equation 4 in [Recent Advances in Simulation-based Inference for Gravitational Wave Data Analysis](https://arxiv.org/pdf/2507.11192) if you're following along there.

$$\begin{align}
\textrm{L}(\vec{\varphi}) &= \mathbb{E}_{\pi(\vec{\theta})}\left[\mathbb{E}_{\vec{x} \sim \mathcal{L}(\vec{x}\vert \vec{\theta})} \left[ - \log q(\vec{\theta}\vert \vec{x} ; \vec{\varphi}) \right]\right] \\
&= - \mathbb{E}_{\pi(\vec{\theta})} \left[ \mathbb{E}_{\vec{x} \sim \mathcal{L}(\vec{x}\vert \vec{\theta})} \left[\log q(\vec{\theta}\vert \vec{x} ; \vec{\varphi}) \right]\right] \\
\end{align}$$


