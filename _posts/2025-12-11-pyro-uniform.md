---
title: "Understanding AutoIAFNormal's Domain: Why It Struggles with Uniform Targets"
date: 2025-12-11
permalink: /posts/2025/12/2025-12-11-pyro-uniform/
tags:
  - PyTorch
  - Variational Inference
  - Deep Learning
  - Normalising Flows
header-includes:
  - \usepackage{amsmath}
  - \usepackage{algpseudocode}

---

`Pyro` is a fantastic probabilistic programming language built on top of PyTorch. 
It has a plethora of extremely useful functionalities and intuitive class structures that allow modelling pretty complex distributions relatively easily.
However, in this post I'm going to explain and try and debug what one would think is a relatively easy problem that `Pyro`'s `AutoIAFNormal` guide seems to fail on while other packages like `Zuko` seem to do fine?

---

## Table of Contents

- [Normalising Flows: A quick recap](#normalising-flows-a-quick-recap)
- [The Problem At Hand](#the-problem-at-hand-training-with-energy)
    - [Testing AutoMultivariateNormal Guide](#testing-automultivariatenormal-guide)
    - [Testing AutoIAFNormal Guide](#testing-autoiafnormal-guide)
    - [Testing Zuko](#testing-zuko)
- [The Problem Kind Of At Hand But Not (training with samples)](#the-problem-kind-of-at-hand-but-not-training-with-samples)
- [Key Takeaways](#key-takeaways)


## Resources

-- 

# Normalising Flows: A quick recap



# `Pyro` Quickstart



# The Problem At Hand (Training with Energy)




## Testing AutoMultivariateNormal Guide




## Testing AutoIAFNormal Guide 





## Testing Zuko




# The Problem Kind Of At Hand But Not (training with samples)


# Key Takeaways





