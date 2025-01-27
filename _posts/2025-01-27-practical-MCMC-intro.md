---
title: 'Practical MCMC Intro/Fitting a line II'
date: 2024-01-27
permalink: /posts/2025/01/2025-01-27-practical-MCMC-intro/
tags:
  - Bayesian Analysis
  - MCMC
  - Introductory
#   - category2
---

In this post I'm going to try to give an intuitive intro into MCMC methods without getting bogged down in much of the math to show the utility of these methods.

MCMC is one of the most successful analytical methods that statisticians have ever used and is the bench mark for all future analysis methods that we will explore.



## Motivation

In the last post we brute forced our analysis but scanning the whole entire parameter space for areas of high posterior probability. In most scenarios this is infeasible and we need something more sophisticated to explore regions of high probabilty and somehow get some sort of representation of our posterior.