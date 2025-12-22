---
title: "Intuitive introduction to Fisher information, Jeffrey's priors and lower bounds on the variance of unbiased estimators (Cramer-Rao bound)"
date: 2025-12-20
permalink: /posts/2025/12/2025-12-20-FisherInfo/
tags:
  - Introductory
header-includes:
  - \usepackage{amsmath}
  - \usepackage{algpseudocode}
---

In this post I'm going to attempt to give an intuitive introduction to Fisher Information, (very briefly) Jeffrey priors, and the lower bounds on the variance of unbiased estimators i.e. the Cramer-Rao bound. 
Hopefully this post will be shorter than my last couple...
UNDER CONSTRUCTION

---
## Resources

- [Fisher Information & Efficiency - Robert L. Wolpert](https://www2.stat.duke.edu/courses/Spring16/sta532/lec/fish.pdf)
- [Fisher Information - Wikipedia](https://en.wikipedia.org/wiki/Fisher_information)
- [The Fisher Information (video)](https://www.youtube.com/watch?v=pneluWj-U-o) - [Mutual Information (YouTube Channel)](https://www.youtube.com/@Mutual_Information)
- [A Tutorial on Fisher Information](https://arxiv.org/pdf/1705.01064) - Alexander Ly, Maarten Marsman, Josine Verhagen, Raoul
Grasman and Eric-Jan Wagenmakers (All University of Amsterdam)
- [Stat 5102 Notes: Fisher Information and Confidence Intervals Using Maximum Likelihood - Charles J. Geyer](https://www.stat.umn.edu/geyer/s06/5102/notes/fish.pdf)
- [The Fisher Information - Gregory Gunderson](https://gregorygundersen.com/blog/2019/11/21/fisher-information/)
- [De Bruijn's Identity: Theory & Applications - Emergent Mind](https://www.emergentmind.com/topics/de-bruijn-s-identity)
- [Generalization of the de Bruijn’s identity to general φ-entropies and φ-Fisher informations - Irene Valero Toranzo, Steeve Zozor and Jean-Marc Brossier](https://arxiv.org/abs/1611.09400)
- [Information Geometry - Wikipedia](https://en.wikipedia.org/wiki/Information_geometry)
- [Fisher information metric - Wikipedia](https://en.wikipedia.org/wiki/Fisher_information_metric)
- [An Elementary Introduction to Information Geometry - Frank Nielson](https://www.mdpi.com/1099-4300/22/10/1100)
- [Some inequalities satisfied by the quantities of information of Fisher and Shannon - A.J. Stam](https://www.sciencedirect.com/science/article/pii/S0019995859903481)


---

## Table of Contents
- [Information and Fisher Information](#information-and-fisher-information)
- [Example Cases of Fisher Information values](#example-cases-of-fisher-information-values)
- [Uninformative (Jeffrey's) priors and the Fisher information](#uninformative-jeffreys-priors-and-the-fisher-information)
- [Example Cases of Uninformative (Jeffrey's) priors](#example-cases-of-uninformative-jeffreys-priors)
- [Summary/Conclusion](#summaryconclusion)

---


# Definition of the Fisher Information

The Fisher Information is defined as,

$$\begin{align}
\mathcal{I}_X(\theta) &= \mathbb{E}_{x \sim \mathcal{L}(x|\theta)}\left[\left(\frac{\partial}{\partial\theta} \log \mathcal{L}(x|\theta) \right)^2 \right] \\
&=\begin{cases}
\int_X dx \mathcal{L}(x|\theta) \left(\frac{\partial}{\partial\theta} \log \mathcal{L}(x|\theta) \right)^2 & \text{(Continuous)}\\
\sum_{i=1}^{N_X} \mathcal{L}(x_i|\theta) \left(\frac{\partial}{\partial\theta} \log \mathcal{L}(x_i|\theta) \right)^2  & \text{(Discrete)}
\end{cases}
\end{align}$$


But honestly this is just a little to esoteric for me to internalise. So I'm gonna make some pretty pictures.

Below is a GIF with three rows and two columns. 

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/initial_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>

The first column exemplifies what some transformations of normal distribution looks like with respect to the random variable it describes, with a mean of 0 and varying standard deviation values. It specifically show the log, the derivative of the log known as the [___score___](https://en.wikipedia.org/wiki/Informant_(statistics)), and the second derivative of the log-likelihood. 

The right column are histograms of these same values with random samples taken from the given likelihood (kind like Monte Carlo [error propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty) for nonlinear functions).

The definition of the Fisher information, boils down to the variance of the second row second column distribution or equivalently the expected value of the bottom right. Which annoyingly for the normal distribution is always the same value so there isn't really a 'distribution'[^distribution]

[^distribution]: And no you nerd, I don't mean in the strict 'mathematical' sense of the word 'distribution' in which case yes it is. I do love my Dirac Delta probability distribution that comes up all the time...(sarcasm)

Okay, but what does this actually imply about our models? Well notice that as the log-likelihood distribution broadens, the distribution of the score



# Example Cases of Fisher Information values

## Normal

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/initial_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>

## Cauchy

Component,Formula

Log-Likelihood: $$-\ln(\pi) - \ln(\gamma) - \ln\left( 1 + \frac{(x-x_0)^2}{\gamma^2} \right)$$

Score: $$\frac{\partial}{\partial x_0} \ln f = \frac{2(x - x_0)}{\gamma^2 + (x - x_0)^2}$$

Hessian: $$\frac{\partial^2}{\partial x_0^2} \ln f = \frac{2\left( (x - x_0)^2 - \gamma^2 \right)}{\left( \gamma^2 + (x - x_0)^2 \right)^2}$$​

Fisher Info: $$1/2\gamma^2$$


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/cauchy_fisher_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>



## Laplace



<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/laplace_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>



## Poisson


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/poisson_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>



# Uninformative (Jeffrey's) priors and the Fisher information




# Example Cases of Uninformative (Jeffrey's) priors






# Summary/Conclusion



---
## Footnotes