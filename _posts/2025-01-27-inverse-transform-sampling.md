---
title: 'Inverse Transform Sampling'
date: 2025-01-27
permalink: /posts/2025/01/2025-01-27-inverse-transform-sampling/
tags:
  - Introductory
  - Sampling Methods
#   - category2
---


Introduction into inverse transform sampling for continuous and discrete probability distributions.


### Introduction
The end goal of the next few posts is to build up to an intuitive understanding of how Markov Chain Monte Carlo (MCMC) methods work following a similar method as the YouTuber [ritvikmath's](https://www.youtube.com/@ritvikmath) videos on the topic. 

At the heart of MCMC is a sampling algorithm for probability distribution with an unknown normalising constant. In this tutorial we will see how one can sample a probability density/mass distribution when you have the explicit form of the distribution.


## Table of Contents
- [Introduction](#introduction)
- [The End Result](#the-end-result)
- [The Math](#the-math)
- [Coding our own sampler](#coding-up-our-own-sampler)
- [Continuous Case](#continuous-case)
- [Discrete Case](#discrete-case)
- [Discretised Continous Case](#discretised-continous-case)
- [Next Steps](#next-steps)

## The End Result

For some this will be the only section that you wish to look into. A general understanding of this topic will allow us to move on to future topics in Rejection Sampling and MCMC methods. 

Let's say you have a normal probability distribution function (pdf) and it's cumulative distribution function (cdf). From the below gif you can see if we invert the cdf and plug in some uniform samples from 0 to 1 then we get samples representative of the normal probability distribution.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-27/normal_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with a normal distribution." 
      title="GIF showing animation of inverse transform sampling principle with a normal distribution." 
      style="width: 50%; height: auto; border-radius: 8px;">
</div>

And we can do this with any analytic continuous distribution. For example below I have an example with the power law distribution and gamma distribution in scipy (the explicit functions aren't important just that you can see that the samples gradually mimic the pdf of the relevant distribution).


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-27/power_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with a power law distribution (a=2)." 
      title="GIF showing animation of inverse transform sampling principle with a power law distribution (a=2)." 
      style="width: 32%; height: auto; border-radius: 8px;">

  <img 
      src="/files/BlogPostData/2025-01-27/gamma_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with a gamma distribution (alpha=2)." 
      title="GIF showing animation of inverse transform sampling principle with a gamma distribution (alpha=2)." 
      style="width: 32%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-01-27/argus_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with a ARGUS distribution (chi=2.5)." 
      title="GIF showing animation of inverse transform sampling principle with a ARGUS distribution (chi=2.5)." 
      style="width: 32%; height: auto; border-radius: 8px;">
</div>

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-27/combined_1_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with a bimodal distribution." 
      title="GIF showing animation of inverse transform sampling principle with a bimodal distribution." 
      style="width: 32%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-01-27/combined_2_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with another bimodal distribution." 
      title="GIF showing animation of inverse transform sampling principle with another bimodal distribution." 
      style="width: 32%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-01-27/combined_3_dist.gif" 
      alt="GIF showing animation of inverse transform sampling principle with a trimodal distribution (coz why not)." 
      title="GIF showing animation of inverse transform sampling principle with a trimodal distribution (coz why not)." 
      style="width: 32%; height: auto; border-radius: 8px;">
</div>

## The Math



## Coding up our own sampler



## Continuous Case



## Discrete Case



## Discretised Continous Case



## Next Steps