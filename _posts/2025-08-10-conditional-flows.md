---
title: 'A RealNVP conditional normalising flow (from scratch?)'
date: 2025-08-09
permalink: /posts/2025/08/2025-08-09-CondNF/
tags:
  - Pyro
  - PyTorch
  - Normalising Flows
  - Variational Inference
  - Conditional Density Estimation
header-includes:
   - \usepackage{amsmath}
---

In this post I will attempt to give an introduction to _conditional normalising flows_, not to be confused with _continuous_ normalising flows, modelling both $$\vec{\theta}$$ and $$\vec{x}$$ in the conditional distribution $$p(\vec{\theta}\vert\vec{x})$$. I was nicely surprised at how simple it is to implement compared to unconditional normalising flows so I thought I'd show this in a straightforward way. Assumes you've read my post on [Building a normalising flow from scratch using PyTorch](https://liamcpinchbeck.github.io/posts/2025/08/2025-08-04-flow-from-scratch/). ***UNDER CONSTRUCTION***

---

## Resources

- [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)
    - Specifically section 3.7
    - This is literally a single paragraph but it just expressed the concept so simply that when I read it a lot of things slotted into place in my head.
- [Learning Likelihooods with Conditional Normalising Flows](https://arxiv.org/abs/1912.00042)
    - Specifically section 3.1
- [Recent Advances in Simulation-based Inference for Gravitational Wave Data Analysis](https://arxiv.org/abs/2507.11192)
    - The general discussion and what they did have for NPE (effectively conditional flows) was really helpful when figuring out how to structure this
    - Also from what I've read of the references they are also of similar quality


## Table of Contents




# Motivation

If you've clicked on this blog post you're likely already interested in conditional flows and/or conditional density estimation but just for the non-believers out there, I'll still lay out the use cases for conditional flows.

The essence of the method is that instead of just learning the probability distribution for a set of parameters $$p(\vec{\theta})$$ you can learn the _conditional_ probability distribution $$p(\vec{\theta}|\vec{x})$$ which allows you thing including but not limited to:
1. Pre-train a conditional density based on possible realisations of the data and when you want to apply it in real life, it's just a question of plugging the data in. And then if you get more data, you can just plug that in practically without having to redo the analysis. i.e. ___Amortised Inference___
    - e.g. [Dingo](https://dingo-gw.readthedocs.io/en/latest/) is a gravitational wave analysis tool that in part utilises it for this purpose
2. Predict future states based on past states, i.e. forecasting
    - If my state _was_ $$x_i$$ what is the probability that the state $$x_{i+1}$$ will be...
3. Conditional generation of data/parameters/variables
    - e.g. generating high resolution images from low resolution ones. For example [SRFlow: Learning the Super-Resolution Space with Normalizing Flow](https://arxiv.org/abs/2006.14200)

Otherwise, I'll try and keep this short and just move on to how they work.


# Core Idea

If you have read my other post on [Building a normalising flow from scratch using PyTorch](https://liamcpinchbeck.github.io/posts/2025/08/2025-08-04-flow-from-scratch/) or are already familiar with how RealNVP architectures/normalising flows are constructed then a conditional flow is really not that much more complicated.

The unconditional flow setup is that it transforms some base distribution variable $$\vec{u}$$ that follows some simple analytical distribution $$p_\vec{u}$$ the we learn to transform into the density that we wish to investigate $$p_\vec{\theta}(\vec{\theta})$$,

$$
\begin{align}
p_\mathbf{\vec{\theta}}(\vec{\theta}) &= p_\vec{u}(\vec{u}) \vert J_T(\vec{u})\vert^{-1} \\
&= p_\vec{u}(T^{-1}(\vec{\theta})) \vert J_{T^{-1}}(\vec{\theta}) \vert .
\end{align}
$$

For RealNVP, the transformation is setup with a affine coupling block structure with $$s$$ and $$t$$ being neural networks, for intermediary variable $$\vec{z}^i$$ for the $$i^{\textrm{th}}$$ layer with $$\vec{z}^0 = \vec{u}$$ and for N layers, $$\vec{z}^N = \vec{\theta}$$,

$$\begin{align}
z^{i}_{1:d} &= z^{i-1}_{1:d} \\
z^{i}_{d+1:D} &= z^{i-1}_{d+1:D} \odot \exp(s(z^{i-1}_{1:d})) + t(z^{i-1}_{1:d}).
\end{align}$$

This means that the jacobian for the $$i^{\textrm{th}}$$ layer of transformations $${T_i}$$ (and subsequently the total jacobian) doesn't require any derivatives of the $$s$$ and $$t$$ and looks like the following,

$$
\begin{align}
J_{T_i} &= \left[ \begin{matrix}
\mathbb{I}_d & \vec{\mathbf{0}} \\
\frac{\partial z^{i}_{d+1:D}}{\partial z^{i-1}_{1:d}} & \textrm{diag} \left(\exp\left[ s(z^{i-1}_{1:d}) \right] \right)\\
\end{matrix} \right].
\end{align}
$$

The bulk of this does not change a lick, except that we need to put the dependence on $$\vec{x}$$ somewhere. No deep abstraction here, we just need to put $$\vec{x}$$ somewhere where the neural networks can learn how to transform $$\vec{u}$$ to $$\vec{\theta}$$ using information on $$\vec{x}$$. 

Because of the construction of RealNVP (and flows in general) we can make the neural networks involved as complicated as we like pretty much as the setup doesn't require derivatives over them. So the easiest thing to do, and what is commonly just done, is to just include $$\vec{x}$$ into the inputs of the neural networks,

$$\begin{align}
z^{i}_{1:d} &= z^{i-1}_{1:d} \\
z^{i}_{d+1:D} &= z^{i-1}_{d+1:D} \odot \exp(s(z^{i-1}_{1:d}, \vec{x})) + t(z^{i-1}_{1:d}, \vec{x}).
\end{align}$$



# Mathematical Setup




# Practical Implementation




# Example Training




# Conclusion