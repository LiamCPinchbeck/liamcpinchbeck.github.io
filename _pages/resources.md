---
# permalink: /resources/
title: "Resources"
redirect_from: 
  - /resources/
  - /resources.html
---

A series of websites, papers and packages that I've found particularly helpful for the highlighted topics.


## Gamma-Ray Analysis

- 🖥️ [Gammapy](https://gammapy.org/)
    - Gammapy is an open-source Python package of data analysis for gamma-ray astronomy.
- 🖥️ [GammaBayes](https://gammabayes.readthedocs.io/en/latest/index.html/)
    - GammaBayes is an open-source Python package of Dayesian data analysis in gamma-ray astronomy. (#selfplug)
- 🖥️ [GALPROP](https://galprop.stanford.edu/)
    - GALPROP is a numerical code for calculating the propagation of relativistic charged particles and the diffuse emissions produced during their propagation.


## Dark matter codes

- 🖥️ [MicrOMEGAs](https://lapth.cnrs.fr/micromegas/)
    - A code for the calculation of Dark Matter Properties including the relic density, direct and indirect rates in a general supersymmetric model and other models of New Physics
- 🖥️ [DarkSUSY](https://darksusy.hepforge.org/)
    - DarkSUSY is a flexible and modular Fortran package to calculate observables for a variety of dark matter candidates. 



## Stochastic Sampling Methods

### MC Sampling

- 🖥️ [emcee](https://emcee.readthedocs.io/en/stable/)
    - emcee is an MIT licensed pure-Python implementation of Goodman & Weare’s Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler 

- 🖥️ [pyMC](https://www.pymc.io/welcome.html)
    - PyMC is a probabilistic programming library for Python that allows users to build Bayesian models with a simple Python API and fit them using Markov chain Monte Carlo (MCMC) methods.

- 📚 [A Conceptual Introduction to Markov Chain Monte Carlo Methods](https://arxiv.org/abs/1909.12313) by Joshua S. Speagle
    - Speagle outlines from a pretty low foundational knowledge how MCMC methods work

### Nested Sampling

- 📚 [Nested sampling for physical scientists](https://arxiv.org/abs/2205.15570) by Greg Ashton et al.
    - Nice intro to nested sampling, particularly how prior volume is related to likelihood density values

- 🖥️ [dynesty](https://dynesty.readthedocs.io/en/stable/) by Joshua S. Speagle
    - Easy and efficient python code to use nested sampling

- 📚 [dynesty: A Dynamic Nested Sampling Package for Estimating Bayesian Posteriors and Evidences](https://arxiv.org/abs/1904.02180) by Joshua S. Speagle
    - A denser introduction to nested sampling than the above paper, but does introduce the algorithms in more depth and outlines the features of the dynesty package



## Variational Inference

- 📚 [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) by David M. Blei, Alp Kucukelbir, Jon D. McAuliffe
    - As in the title it's a nice review of variational inference and has a particularly nice intro into normalising flows

### Normalising Flows


- 📚 [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257) by Ivan Kobyzev, Simon J.D. Prince, Marcus A. Brubaker
    - Has a great walkthrough for simple to more complex normalising flow methods


- 🖥️ [FlowJAX](https://danielward27.github.io/flowjax/index.html) by Daniel Ward, Tennessee Hickling, Matthew Mould, Gilad Turok
    - Wicked fast normalising flows, in particular the ability to perform variational inference with them


## Misc

- 🌐 [Academic Pages GitHub](https://github.com/academicpages/academicpages.github.io)
    - What I used to make this website

- 📚 [Andy Casey's Introduction to Data Analysis](https://astrowizici.st/teaching/phs5000/)
    - Best place I've ever found for an introduction to Bayesian analysis methods

- 🖥️ [JAX](https://jax.readthedocs.io/en/latest/quickstart.html)
    - JAX is a library for array-oriented numerical computation (à la NumPy), with automatic differentiation and JIT compilation to enable high-performance machine learning research.

- 🖥️ [CuPy](https://cupy.dev/)
    - NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python

