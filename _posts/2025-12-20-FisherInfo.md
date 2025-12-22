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
- [Relations between Kullback-Leibler distance and Fisher information](https://www.ece.rice.edu/~dhj/distance.pdf) - Anand G. Dabak & Don H. Johnson
- [THEORETICAL NEUROSCIENCE I $$\Vert$$ Lecture 16: Fisher information - Prof. Jochen Braun](https://bernstein-network.de/wp-content/uploads/2021/02/16_Lecture-16-Fisher-information.pdf)


---

## Table of Contents
- [Information and Fisher Information](#information-and-fisher-information)
- [Example Cases of Fisher Information values](#example-cases-of-fisher-information-values)
    - [Normal](#normal)
    - [Poisson](#poisson)
    - [Laplace](#laplace)
    - [Cauchy](#cauchy)
- [Uninformative (Jeffrey's) priors and the Fisher information](#uninformative-jeffreys-priors-and-the-fisher-information)
- [Example Cases of Uninformative (Jeffrey's) priors](#example-cases-of-uninformative-jeffreys-priors)
- [Summary/Conclusion](#summaryconclusion)

---


# Definition of the Fisher Information

Fisher information is a fundamental quantity used across statistics to quantify and describe information on parameters given our models and data. 

In more rigorous terms,
> "In mathematical statistics, the Fisher information is a way of measuring the amount of information that an observable random variable X carries about an unknown parameter $$\theta$$ of a distribution that models X." - [Wikipedia](https://en.wikipedia.org/wiki/Fisher_information)


But I'm presuming that if you've clicked on this post either that the above definition doesn't make sense to you, or your a nerd who enjoys learning about statistics, so I'll skip the motivation for this post and jump straight into trying to understand it. 


I'm first going to try and approach this purely by interpretting the definition at face value and then do a more fundamental derivation after we know what's going on. The Fisher Information is defined as,

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

The first column exemplifies what some transformations of normal distribution looks like with respect to the random variable it describes, with a mean of 0 and varying standard deviation values. It specifically show the log, the derivative of the log-likelihood with respect to a parameter of interest (the mean) known as the [___score___](https://en.wikipedia.org/wiki/Informant_(statistics)), and the second derivative of the log-likelihood with respect to this parameter. 

The right column are histograms of these same values with random samples taken from the given likelihood (kind like Monte Carlo [error propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty) for nonlinear functions).

The definition of the Fisher information, boils down to the variance of the second row second column distribution or equivalently the negative of the expected value of the bottom right. Which annoyingly for the normal distribution is always the same value so there isn't really a 'distribution'[^distribution]

[^distribution]: And no you nerd, I don't mean in the strict 'mathematical' sense of the word 'distribution' in which case yes it is. I do love my Dirac Delta probability distribution that comes up all the time...(sarcasm)

Okay, but what does this actually imply about our models? Well notice that as the log-likelihood distribution broadens, the variance in the distribution of the score becomes smaller. Or in a different way of looking at it, the number of extreme derivative values decreases. Or a different way again, it's the expected value of the magnitude of the derivatives squared, more higher magnitude values, more info.

If what we're interested in is information (which makes sense as we're trying to understand Fisher _information_) then this narrowing the score distribution, indicates that our data is becoming less informative about what the true value of the parameter we're interested in is. If we have large gradients, then our likelihood is telling us very strongly what direction the true value is in, while for small gradients it only weakly tells us. And if the variance is small, then more of our values congregate towards zero/smaller values.

I'll now bash out an FAQ style answer to "But why this?" in regards to why we specifically construct the Fisher information in this manner:


- __Question__: By why is expected value for the score is always zero?
    - ___Answer___: See below.

- __Question__: By why the _variance_ in the log-derivative instead of _expected value_?
    - ___Answer___: The expected value for the score is always zero, which doesn't relate different amounts of information for different sets of parameters. Mathematically this can be seen as, for the general case for parameter $$\theta$$ and likelihood/probability distribution $$\mathcal{L}(x\vert\theta)$$,

        $$\begin{align}
        &\mathbb{E}_{x\sim \mathcal{L}(x|\theta)}\left[\frac{\partial}{\partial \theta} \log \mathcal{L}(x|\theta)\right] \\
        &= \int_x \mathcal{L}(x|\theta) \frac{\partial}{\partial \theta} \log \mathcal{L}(x|\theta) dx \\
        &= \int_x \mathcal{L}(x|\theta) \frac{1}{\mathcal{L}(x|\theta)} \frac{\partial}{\partial \theta} \mathcal{L}(x|\theta) dx \\
        &= \int_x \frac{\partial}{\partial \theta} \mathcal{L}(x|\theta) dx \\
        &= \frac{\partial}{\partial \theta} \int_x \mathcal{L}(x|\theta) dx \\
        &= \frac{\partial}{\partial \theta} 1 \\
        &= 0 \\
        \end{align}$$
 
    - ___Answer 2___: The average here will always relate back to in some way the derivative at the mode of the distribution/the true value. Which _should_ have a maximal likelihood value and hence should be at least a local max. At this point we shouldn't get any directional information because we _shouldn't_ move, but this looks equivalent to having no information where _we don't know_ where to move. 

- __Question__: By how is the variance of the derivative equivalent to the negative of the expected value of the second?
    - ___Answer___: Again, a tinsy bit of math (using the above fact to simplify $$\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \mathbb{E}[X^2] $$)

        $$\begin{align}
        &-\mathbb{E}\left[\frac{\partial^2}{\partial \theta^2} \log \mathcal{L}(x|\theta) \right] \\
        &= -\mathbb{E}\left[\frac{\partial}{\partial \theta} \frac{\frac{\partial}{\partial \theta} \mathcal{L}(x|\theta)}{\mathcal{L}(x|\theta)}\right] \\
        &= -\mathbb{E}\left[\frac{\frac{\partial^2}{\partial \theta^2} \mathcal{L}(x|\theta)}{\mathcal{L}(x|\theta)} - \left(\frac{\frac{\partial}{\partial \theta} \mathcal{L}(x|\theta)}{\mathcal{L}(x|\theta)}\right)^2\right] \\
        &= -\mathbb{E}\left[\frac{\frac{\partial^2}{\partial \theta^2} \mathcal{L}(x|\theta)}{\mathcal{L}(x|\theta)}\right] + \mathbb{E}\left[\left(\frac{\frac{\partial}{\partial \theta} \mathcal{L}(x|\theta)}{\mathcal{L}(x|\theta)}\right)^2\right] \\
        &= 0 + \mathbb{E}\left[\left(\log \frac{\partial}{\partial \theta} \mathcal{L}(x|\theta)\right)^2\right] \\
        &= \text{Var}\left[ \log \frac{\partial}{\partial \theta} \mathcal{L}(x|\theta)\right] \\
        \end{align}$$

- __Question__: By why do this with the _log_-likelihood, why not just the _likelihood_?
    - ___Answer___: In part because it's nicer to compute. The derivatives propagate linearly in the case of the log-likelihood, but in the case of the likelihood you would have truncated products for example. Plus some more fundamental reasons that I'll touch on in the more fundamental derivation. The fundamental derivation is just by looking at the problem with different initial conditions though, you probably could come up with some similar alternative formulation using just the likelihood, but then we'd go back to the first point on tractability.



Now, before we jump into what is in my opinion the nicer way to understand the Fisher information (through _entropy_) let's look at how we can actually calculate the Fisher information analytically for some distributions.


# Example Cases of Fisher Information values

For all of these, try and have a go yourself first and I'll be doing it for the same parameters that I do so for in the GIFs (coz I needed to do this to make them anyways).


## Normal

<details close>
<summary>Derivation</summary>

$$\begin{align}
\mathcal{I}(\mu) &= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \log \mathcal{N}(x|\mu, \sigma) \right] \\
&= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \left(-\frac{1}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}\right) \right] \\
&= -\mathbb{E}_x\left[\frac{\partial}{\partial \mu}  \frac{(x-\mu)}{\sigma^2} \right] \\
&= -\mathbb{E}_x\left[ \frac{-1}{\sigma^2} \right] \\
&= \frac{1}{\sigma^2}  \mathbb{E}_x\left[ 1 \right] \\
&= \frac{1}{\sigma^2} \\
\end{align}$$

</details>

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/initial_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>

## Poisson

<details close>
<summary>Derivation</summary>

$$\begin{align}
\mathcal{I}(\lambda) &= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \log \left(\text{Poiss}(x | \lambda)\right) \right] \\
&= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \log \frac{\lambda^x e^{-\lambda}}{x!} \right] \\
&= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \left(x\log(\lambda) - \lambda - \log(x!) \right) \right] \\
&= -\mathbb{E}_x\left[\frac{\partial}{\partial \mu} \left(x \frac{1}{\lambda} - 1 \right) \right] \\
&= \mathbb{E}_x\left[\left(x \frac{1}{\lambda^2} \right) \right] \\
&= \frac{1}{\lambda^2} \mathbb{E}_x\left[ x\right] \\
&= \frac{1}{\lambda^2} \lambda \\
&= \frac{1}{\lambda} \\
\end{align}$$

</details>


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/poisson_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>



## Laplace

For this one we'll be using the first derivative definition as the Laplace doesn't satisfy some required smoothness constraints.

<details close>
<summary>Derivation</summary>

$$\begin{align}
\mathcal{I}(\mu) &= \text{Var}_x\left[\frac{\partial}{\partial \mu} \log \left(\text{Laplace}(x | \mu, b) \right)\right] \\
&= \text{Var}_x\left[\frac{\partial}{\partial \mu} \log\left(\frac{1}{2b} \exp\left(-\frac{|x-\mu|}{b} \right) \right) \right] \\
&= \text{Var}_x\left[\frac{\partial}{\partial \mu} \left( - \log(2b) -\frac{|x-\mu|}{b} \right) \right] \\
&= \text{Var}_x\left[\begin{cases} \frac{1}{b} & x<\mu \\ \frac{-1}{b} & x>\mu \end{cases} \right] \\
&= \mathbb{E}_x\left[\begin{cases} \left(\frac{1}{b}\right)^2 & x<\mu \\ \left(\frac{-1}{b}\right)^2 & x>\mu \end{cases} \right] \\
&= \mathbb{E}_x\left[\begin{cases} \frac{1}{b^2} & x<\mu \\ \frac{1}{b^2} & x>\mu \end{cases} \right] \\
&= \frac{1}{b^2}\\
\end{align}$$

Where on the final line we've done some quick physics math: we presume the integral exists, notice that it's \(1/b^2\) everywhere except at \(\mu\), and that the probability density at \(\mu\) is finite so... \(1/b^2\)!

</details>


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/laplace_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>


## Cauchy

For this one we'll be using the second derivative definition as it's slightly easier to get the integral result. But it is a little harder I think because it's not an exponential family.


<details close>
<summary>Derivation</summary>


$$\begin{align}
\mathcal{I}(\mu) &= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \log \left(\text{Cauchy}(x | \mu, \gamma) \right)\right] \\
&= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \log\left(\frac{\gamma/\pi}{(x-\mu)^2 + \gamma^2} \right)\right] \\
&= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \left(-\ln(\pi) + \ln(\gamma) - \ln\left( (x-\mu)^2 + \gamma^2\right) \right) \right] \\
&= -\mathbb{E}_x\left[\frac{\partial}{\partial \mu} \frac{2(x - \mu)}{\gamma^2 + (x - \mu)^2}\right] \\
&= -\mathbb{E}_x\left[\frac{2\left( (x - \mu)^2 - \gamma^2 \right)}{\left( \gamma^2 + (x - \mu)^2 \right)^2}\right] \\
&=  -\frac{2\gamma}{\pi} \int_x dx \frac{\left( (x - \mu)^2 - \gamma^2 \right)}{\left( \gamma^2 + (x - \mu)^2 \right)^2} \frac{1}{(x-\mu)^2 + \gamma^2} \\
&=  -\frac{2\gamma}{\pi} \int_x dx \frac{(x - \mu)^2 - \gamma^2}{\left( \gamma^2 + (x - \mu)^2 \right)^3}\\
&=  -\frac{2\gamma}{\pi} \int_x dx \frac{ x^2 - \gamma^2}{\left( \gamma^2 + x^2 \right)^3}\\
&=  -\frac{2}{\pi \gamma^4} \int_x dx \frac{ \left(\frac{x}{\gamma}\right)^2 - 1}{\left( 1 + \left(\frac{x}{\gamma}\right)^2 \right)^3}\\
&=  -\frac{2}{\pi \gamma^4}  \int_u du \frac{u^2 - 1}{\left( 1 + u^2 \right)^3}\\
&=   -\frac{2}{\pi \gamma^4}  \int_u du \frac{1}{\left( 1 + u^2 \right)^2} - \frac{2}{\left( 1 + u^2 \right)^3}\\
\end{align}$$

From here we'll use the following.

$$\begin{align}
\int_{-\infty}^\infty \frac{1}{t + u^2} du &= \frac{1}{\sqrt{t}}\text{arctan}(\frac{x}{\sqrt{t}}) {\huge\vert}_{-\infty}^\infty \\
&= \frac{\pi}{\sqrt{t}}\\
\end{align}$$

Hence,

$$\begin{align}
\frac{d}{dt} \int_{-\infty}^\infty \frac{1}{t + u^2} du  &=  \int_{-\infty}^\infty \frac{-1}{(t + u^2)^2} du\\
&= \frac{d}{dt} \frac{\pi}{\sqrt{t}} \\
&= \frac{-\pi}{2} \frac{1}{\sqrt{t^3}}, \\
\end{align}$$

and,


$$\begin{align}
\frac{d^2}{dt^2} \int_{-\infty}^\infty \frac{1}{t + u^2} du  &=  \frac{d}{dt} \int_{-\infty}^\infty \frac{-1}{(t + u^2)^2} du \\
&=  \int_{-\infty}^\infty \frac{-2}{(t + u^2)^3} du \\
&=  \frac{d}{dt} \frac{-\pi}{2} \frac{1}{\sqrt{t^3}}, \\
&=  \frac{3\pi}{4} \frac{1}{\sqrt{t^5}}. \\
\end{align}$$

Just substituting $$t=1$$ we get,



$$\begin{align}
\mathcal{I}(\mu) &= -\mathbb{E}_x\left[\frac{\partial^2}{\partial \mu^2} \log \left(\text{Cauchy}(x | \mu, \gamma) \right)\right] \\
&=  \frac{2\pi}{\gamma^4}  \left(\frac{3\pi}{4} - \frac{\pi}{2} \right)\\
&=  \frac{1}{2 \gamma^4}\\
\end{align}$$

</details>


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-12-FisherInfo/cauchy_fisher_defn_gif.gif" 
      style="width: 89%; height: auto; border-radius: 1px;">
</div>
<br>

# Fisher Information Derivation From Relative Entropy

Now for this very in-depth derivation, I'm going to presume you know what entropy is, and how it works from a statistical stand-point. 
If you are unfamiliar with the Entropy I'd recommend the following resources:
- [Entropy (for data science) Clearly Explained!!!](https://www.youtube.com/watch?v=YtebGVx-Fxw) - [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
- And of course I quite like Steve Mould's "one line" definition of entropy, except he does it in many more than one line... [A better description of entropy](https://youtu.be/w2iTCm0xpDc?t=543) -[Steve Mould](https://www.youtube.com/@SteveMould)
- And if you want something that isn't a video then [Three tutorial lectures on entropy and counting - David Galvin](https://arxiv.org/pdf/1406.7872) is pretty to the point with basic concepts and pretty exhaustive when it comes to it's further properties

If you wanna skip that, or have a quick refresher, to quote Gavlin,
> "..., it is most helpful to (informally) think of entropy as a measure of the expected amount of surprise evinced by a realization of X, or as a measure of the degree of randomness of X." - David Galvin in the work Three tutorial lectures on entropy and counting

With the mathematical definition of (specifically _conditional_) entropy for a random variable $$X$$ can be given by the following.

$$\begin{align}
H(X|\theta) = \begin{cases}
- \int_X dx p(x | \theta) \log p(x|\theta) & \text{(continuous X)} \\
- \sum_{X=x} p(x | \theta) \log p(x|\theta) & \text{(discrete X)} \\
\end{cases}
\end{align}$$

The typical example being flipping a coin. We have max entropy or 'randomness' if it is a fair coin, it is truly random whether we'll get a heads or a tails. You can calculate the entropy of this as $$H(X) \approx 0.693$$. 

While if we have an unfair coin, where the heads is 99 times more probable than the tails, then the result isn't nearly as random and you can be pretty confident in saying that the result of any single toss will be a heads. You can calculate the entropy of _this_ example as $$H(X) \approx 0.056$$.

In the second example we can also imagine that if we did get a tails, then that would add a lot of 'information' in that maybe the coin is more fair than we thought. Or in another way, if our system isn't truly random, we can say certain things about different outcomes or we have more 'potential' information.

e.g. If we are using light to look at a star. If the photons coming at us were truly random, we could never say anything about the star, as they have nothing to do with it, they're random. But if they are completely deterministic, the photons the star emits and we receive are exactly the same, and have no randomness, then we can say quite a lot about the star.

_Relative entropy_, more well known as the [KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), denoted something like $$\text{KL}(P\Vert Q)$$. 

If $$H(P)$$ is the "true" entropy with $$H(P, Q\vert\theta) = -\int_X P(X\vert\theta) \log Q(x\vert\theta) dx$$ is the "cross entropy" (the surprise of $$Q$$ measured by $$P$$) then the KL divergence or relative entropy can be expressed as $$\text{KL}(P \Vert Q) = H(P, Q\vert\theta) - H(P\vert \theta)$$. 

You can think about it as how much extra information would be accessible if one used the 'right' distribution $$P$$ instead of $$Q$$ or equivalently how much information is lost if we incorrectly use $$Q$$ instead of $$P$$.


Using the continuous version of the definition we can find that the Fisher Information is given as the second derivative or _Hessian_ of the KL divergence between the true distribution, assuming the likelihood for a true value of $$\theta$$ and an approximate value given by a sample/datapoint. Informally I'd say it's like asking how sensitive the relative information between the true distribution and a given sample are i.e. the curvature of the KL divergence with respect to the parameter of interest.

This is very easy to see through a taylor expansion of the cross entropy about the difference between the distribution using the true value $$\theta$$ and the approximate value $$\theta^* = \theta + \epsilon$$ .

$$\begin{align}
H(p(x|\theta), p(x|\theta^*)) &= -\int_X p(x|\theta) \log p(x|\theta^*) dx \\
&= -\int_X p(x|\theta) \log p(x|\theta + \epsilon) dx \\
&= -\int_X p(x|\theta) \left[\log p(x|\theta) + \epsilon \frac{\partial}{\partial \theta} \log p(x|\theta) + \epsilon^2 \frac{\partial^2}{\partial \theta^2} \log p(x|\theta) + \sum_{k=3}^\infty \epsilon^k \frac{\partial^k}{\partial \theta^k} \log p(x|\theta) \right]dx \\
\end{align}$$

Moving the first term over to the otherside we can see the KL divergence pop out and that the second term will vanish as it's the average of the score which we established to be 0 above. This leaves us with,

$$\begin{align}
\text{KL}(p(x|\theta)\Vert p(x|\theta^*)) &= \mathcal{I}(\theta) - \sum_{k=3}^\infty \epsilon^k \int_X p(x|\theta) \left[\frac{\partial^k}{\partial \theta^k} \log p(x|\theta) \right]dx. \\
\end{align}$$


i.e. if we take the second derivative term in our taylor expansion, or more simply, take the second derivative (where by first principles we take $$\epsilon \rightarrow 0$$) we find that 

$$\begin{align}
\frac{\partial^2}{\partial \epsilon^2} \text{KL}(p(x|\theta)\Vert p(x|\theta + \epsilon)){\huge \vert}_{\epsilon=0} &= \mathcal{I}(\theta) \\
\end{align}$$


In summary, the Hessian of the infinitesimal form of the KL divergence gives the fisher information. Note that this doesn't equate to the second derivative of the entropy is the Fisher information, although it's frustratingly close. The closest expression for that is the following, and the second term doesn't vanish in general.

$$\begin{align}
\frac{d^2H}{d\theta^2} = - \mathcal{I}(\theta) + \int_x \frac{\partial^2}{\partial \theta^2} p(x|\theta) \log p(x|\theta) dx
\end{align}$$


Although _sometimes_ the Fisher information as the negative hessian of the entropy, such as when we look at the mean of the normal distribution and I think the location parameters for distributions in the [location-scale family](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family). And there's something called [De Bruijn’s identity](https://www.emergentmind.com/topics/de-bruijn-s-identity) which I can't seem to find the assumptions for.


___In summary, you can think of the interpretation of the pull between the true distribution and our approximate as another reason why we have the___ log ___-likelihood in our definition of Fisher information.___

<br>



# Uninformative (Jeffrey's) priors and the Fisher information




# Example Cases of Uninformative (Jeffrey's) priors






# Summary/Conclusion



---
## Footnotes