---
title: 'First Blog Post/Fitting a line'
date: 2024-01-26
permalink: /posts/2025/01/2025-01-26-first-blog-post/
tags:
#   - cool posts
#   - category1
#   - category2
---

First blog post, outlining what I'm going to try and do in the next few posts and some basics on Bayesian analysis.


Statistics is cool trust me
----

My research interests are in astro-particle physics. This involves a lot of data, models and parameters. To get information on all of this requires very careful treatment of the analysis.

Due to this I frequently (always) use Bayesian methods. In my first few posts, or however long it takes, I hope to show why this is.

Like many a statistics blogger, I'm going to make this post pretty similar to [Dan Foreman Mackey's](https://dfm.io/posts/fitting-a-plane/) and then I am particularly further influenced by [Andy Casey's](https://astrowizici.st/) teaching, the relevant posts can be found [here](https://astrowizici.st/teaching/phs5000/). One key difference between what I'm going to do and Dr. Casey's website, is that I'm going to skip the analytical methods at the beginning, as I personally rarely use those methods in my work and figure that this is my blog, so I'm going to show what _I_ do. 


Fitting a line to data. (Gotta start somewhere)
=====================

For this post I'm going to through some data at you where I've generated some random x values, from that created some y values using the formula for a straight line, then added some noise that follows a gaussian distribution. Or more simply,

$$ X \sim \mathcal{U}(0,10)$$

$$y \sim \mathcal{N}(\mu=m\cdot X+ c, \sigma^2=1).$$

(First line says that the probability of getting an $$X$$ value is the same or [uniformly distributed](https://en.wikipedia.org/wiki/Continuous_uniform_distribution) between 0 and 10 and the second that $$y$$ is [normally distributed](https://en.wikipedia.org/wiki/Normal_distribution) about the straight line $$m\cdot X+ c$$ with a standard deviation of $$1$$.)

I won't tell you what $$m$$ and $$c$$ are, that is what we're trying to find. But let's say that we for some reason know that $$ m $$ will be between 0 and 10 and $$c$$ will be between $$-10$$ and $$10$$. 

This gave me some data that looks like the following.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-26/initial_data.png" 
      alt="Straight Line Data" 
      title="Initial data distributed about a straight line of unknown parameters." 
      style="width: 50%; height: auto; border-radius: 8px;">
</div>

Let's start with exploration
---------------------

First let's define a function to use for our straight line.


```python
def line_function(x, m, c):
    return m * x + c
```

The first thing that you may want to do is guess. There's nothing explicitly wrong with this, it may give you an idea of your parameters and is part of normal exploratory analysis where you get a feel for the data. I got the $$m$$ and $$c$$ values by gradually changing them by eye.

```python
# Instantiation some x values to use with our straight line
x_example = np.linspace(-2, 12)
# y values based on function and guessed parameters
y_guess = line_function(x_example, m=1.3 , c=-0.8)


fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=200)

# Plotting actual data
plt.scatter(X_true, y, s=15, marker='x', c='tab:orange', label="data")

# Plotting guesstimates
plt.plot(x_example, y_guess, c="tab:green", label="guess")

ax.set_xlim(-2, 12)
ax.set_ylim(-2, 12)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.legend()
plt.show()
```


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-26/guess_analysis.png" 
      alt="Straight Line Guess" 
      title="Data plotted with straight line with guessetimated parameters." 
      style="width: 50%; height: auto; border-radius: 8px;">
</div>



Let's start with a "stupid" thing
---------------------

You can see the header, but the following is not the stupid bit. We will also be using this later, but we need some sort of function to get some sort of probabilistic interpretation of our parameters. For that we will mimic how I generated the data, we'll investigate the probability that one could generate the data for the given input parameters. 

_Through Maths_

$$ 
\begin{align} \mathcal{L}(\vec{y}|\vec{x}, m, c, \sigma) &= \prod_{i} \mathcal{L}(y_i|x_i, m, c, \sigma) \\
&= \prod_{i} \frac{1}{\sqrt{2\pi\sigma^2}} exp\left(-\frac{(y-f(x, m, c))^2}{2\sigma^2}\right)
\end{align}$$


_Through Code_
```python
def ln_likelihood(d, theta, sigma=1):
    y, x = d
    m, c = theta
    ln_like = -(y[:, np.newaxis]-line_function(x[:, np.newaxis], m[np.newaxis, :], c[np.newaxis, :]))**2/(2*sigma**2) - np.log(np.sqrt(2*np.pi)*sigma)
    return ln_like.sum(axis=0)
```
(We take the log value to avoid issues of [numerical instability](https://en.wikipedia.org/wiki/Numerical_stability))

The stupid thing is to then just brute force possible combinations of $$m$$ and $$c$$ using those assumptions we stated at the beginning.

```python
m_values = np.linspace(0, 10, 101)
c_values = np.linspace(-10, 10, 101)

m_mesh, c_mesh = np.meshgrid(m_values, c_values, indexing='ij')

log_likelihood_values = ln_likelihood((y, X_true), (m_mesh.flatten(), c_mesh.flatten()), sigma=1).reshape(m_mesh.shape)
```

We'll then fit a [Gaussian (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation) to it to get contour curves and directly plot the likelihood values.


```python
# Flatten grid and normalize likelihood
data_points = np.vstack([m_mesh.flatten(), c_mesh.flatten()])
likelihood = np.exp(log_likelihood_values.flatten())
weights = likelihood / np.sum(likelihood)

# Apply Gaussian KDE with weights
kde = stats.gaussian_kde(data_points, weights=weights)

# Evaluate KDE on the grid
pdf_values = kde(np.vstack([m_mesh.ravel(), c_mesh.ravel()])).reshape(m_mesh.shape)

# Define confidence levels (e.g., 1σ, 2σ, ...)
sorted_pdf = np.sort(pdf_values.ravel())[::-1]
cumulative = np.cumsum(sorted_pdf) / np.sum(sorted_pdf)
confidence_levels = [sorted_pdf[np.searchsorted(cumulative, level)] for level in [norm.cdf(val)-norm.cdf(-val) for val in range(1,4)]]  # 1σ, 2σ, 3σ

confidence_levels = np.sort(confidence_levels)

# Plot
plt.figure()
plt.pcolormesh(m_values, c_values, pdf_values.T, shading='auto', cmap='viridis')
plt.colorbar(label=r'$\mathcal{L}(\mathbf{y}|\mathbf{x}, m, c, \sigma)$')
plt.contour(m_values, c_values, pdf_values.T, levels=confidence_levels, colors='white', linestyles='--', linewidths=1.5)
plt.scatter([m_true], [c_true], c='tab:red', marker='*', label='True Value', s=50)
plt.xlabel("m")
plt.ylabel("c")
plt.legend()
plt.title("Confidence Contours with KDE")
plt.show()
```

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-26/brute-force-straight-line-analysis.png" 
      alt="Bruce force likelihood values." 
      title="Brute forced likelihood values." 
      style="width: 75%; height: auto; border-radius: 8px;">
</div>

Now these aren't strictly probability values on $$m$$ and $$c$$. What we have evaluated is the probability of the data points _given_ the parameters. 

In essence, 

$$ p(y\mid x, m, c, \sigma)$$

but what we want is

$$ p(m,c\mid x,y,\sigma).$$

We can convert $$p(y\mid x, m, c, \sigma)$$ to the probability density on the parameters using Bayes' theorem. Generally, 

$$p(\vec{\theta}\mid\vec{d}) = \frac{p(\vec{d}\mid\vec{\theta})p(\vec{\theta})}{p(d)}, $$

where $$\vec{\theta}$$ denote the parameters of interest, $$\vec{d}$$ the data. We then generally actually write the equation with different symbols to denote there function in the equation,

$$p(\vec{\theta}\mid\vec{d}) = \frac{\mathcal{L}(\vec{d}\mid\vec{\theta})\pi(\vec{\theta})}{\mathcal{Z}(\vec{d})}. $$

- $$\mathcal{L}(\vec{d}\mid\vec{\theta})$$ is the _likelihood_ that we previously discussed, 
- $$p(\vec{\theta}\mid\vec{d})$$ is called the _posterior_ and is our goal of our analysis (probability of parameter based on data), 
- $$\pi(\vec{\theta})$$ is called the _prior_ and quantifies our _prior_ assumptions on the parameters, and finally 
- $$\mathcal{Z}(\vec{d})$$ is called the _evidence_ or _fully marginalised likelihood_ and is typically used for model comparison if at all (we'll circle back to this one later).


For our specific case, this looks like,

$$p(m, c\mid \vec{y}, \vec{x}, \sigma) = \frac{\mathcal{L}(\vec{y}\mid \vec{x}, m, c, \sigma)\pi(m, c)}{\mathcal{Z}(\vec{y}\mid \vec{x}, \sigma)}. $$


And this is where the Bayesian analysis officially start.

One of the beautiful things about Bayesian analysis is the explicit quantification of our assumption, you might have noticed that at the beginning that I restricted our $$m$$ and $$c$$ values which kind of sneaked it's way in by the range of values that I tested. In Bayesian analysis we place this information into the prior! In essence,

$$
\begin{align} m &\sim \mathcal{U}(0,10) \\
c &\sim \mathcal{U}(-10,10)\end{align}
$$

such that

$$
\pi(m,c) =\pi(m)\pi(c)= \begin{cases}
    \frac{1}{10-0} & \text{if } m \in [0, 10]\\ % & is your "\tab"-like command (it's a tab alignment character)
    0 & \text{otherwise.}
\end{cases} \cdot \begin{cases}
    \frac{1}{10-(-10)} & \text{if } c \in [-10, 10]\\ % & is your "\tab"-like command (it's a tab alignment character)
    0 & \text{otherwise.}
\end{cases} 
$$


Slightly less stupid thing
-----
 Now we'll brute force the values like before but now we've explicitly included our assumptions _into_ the model.

 ```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define grid
m_values = np.linspace(0, 2, 101)
c_values = np.linspace(-3, 4, 101)
m_mesh, c_mesh = np.meshgrid(m_values, c_values, indexing='ij')

# Assume ln_likelihood is a function you have defined
unnormalised_log_posterior_values = unnormalised_log_posterior(y, X_true, m_mesh.flatten(), c_mesh.flatten(), sigma=1).reshape(m_mesh.shape)

# Flatten grid and normalize likelihood
data_points = np.vstack([m_mesh.flatten(), c_mesh.flatten()])
unnormalised_log_posterior_values = np.exp(unnormalised_log_posterior_values.flatten())
weights = unnormalised_log_posterior_values / np.sum(likelihood)

# Apply Gaussian KDE with weights
kde = stats.gaussian_kde(data_points, weights=weights)

# Evaluate KDE on the grid
pdf_values = kde(np.vstack([m_mesh.ravel(), c_mesh.ravel()])).reshape(m_mesh.shape)

# Define confidence levels (e.g., 1σ, 2σ, ...)
sorted_pdf = np.sort(pdf_values.ravel())[::-1]
cumulative = np.cumsum(sorted_pdf) / np.sum(sorted_pdf)
confidence_levels = [sorted_pdf[np.searchsorted(cumulative, level)] for level in [norm.cdf(val)-norm.cdf(-val) for val in range(1,4)]]  # 1σ, 2σ, 3σ

confidence_levels = np.sort(confidence_levels)

# Plot
plt.figure(dpi=200, figsize=(8,6))
plt.pcolormesh(m_values, c_values, pdf_values.T, shading='auto', cmap='viridis')
plt.colorbar(label=r'$\mathcal{p}(m, c\mid \mathbf{y}, \mathbf{x}, \sigma)$')
plt.contour(m_values, c_values, pdf_values.T, levels=confidence_levels, colors='white', linestyles='--', linewidths=1.5)
plt.scatter([m_true], [c_true], c='tab:red', marker='*', label='True Value', s=50)
plt.xlabel("m")
plt.ylabel("c")
plt.legend()
plt.title("Unnormalised posterior with Gaussian KDE")
plt.show()
```

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-26/brute-force-unnormalised-posterior.png" 
      alt="Bruce force unnormalised posterior values." 
      title="Brute forced unnormalised posterior values." 
      style="width: 75%; height: auto; border-radius: 8px;">
</div>

This looks the same as the previous plot as our priors were uniform thus the overall change is a multiplicative constant which if we normalise our probabilities (to make them a probability) this constant disappears. So we were lucky in that the likelihood and posterior are the same.

Issues
=====

Woo! We have a probability density on our parameters. However, there are a few subtles that I failed to mention.
1. Most interesting models have more than 2 parameters. Let's say you have a 10 parameter model, and wish to look at just 10 values in each dimension, this corresponds to 10 billion evaluations of the function, which by itself is a little rediculous, but if we were to store all those values at the same time, presuming 64bit precision, this is approximately 640GB... not great. Additionally, we were lucky that the posterior wasn't too much smaller than our prior so we could zoom into the relevant region. However, if we had enough data then the posterior could be small enough that the parameter values that we evaluate we wouldn't see anything. I'll get back to this one later.

2. There's a bias in the gradient values towards positive values. Naturally you would presume that the gradient values are uniformly distributed by angles between 90 degrees and -90 degrees. With our assumption, there are a lot more large values than small values. e.g. If we look at the gradients that we could have looked at they would look like the following. (Thanks Dr. Casey for this particular plot.)

```python
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

x = np.linspace(-1, 1)

fig, ax = plt.subplots(figsize=(4, 4))

for m in np.linspace(0, 10, 100):
    ax.plot(x, m*x, "-", c="k", lw=0.5, alpha=0.5)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

ax.xaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(4))

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

fig.tight_layout()
```

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-26/possible-gradients.png" 
      alt="Possible gradient values in our analysis." 
      title="Possible gradient values in our analysis." 
      style="width: 75%; height: auto; border-radius: 8px;">
</div>

You can see that we heavily preferenced larger gradient values. We can instead ask that our gradients are uniformly distributed by the sin of the angle where the angle is given by $$\theta = \tan^{-1}(m)$$ (where $$\theta$$ is angle here not "parameters").

In essence,

$$ \begin{align} \pi(sin(\theta)) = const. \end{align}$$

then,

$$ \begin{align} \pi(m) &= \pi(\theta) \left|\frac{d\theta}{dm}\right| \\
&\propto \left|\frac{d}{dm} \sin(\tan^{-1}(m))\right| \\
&= \left|\frac{d}{dm} \frac{m}{\sqrt{1+m^2}}\right| \\
&= (1+m^2)^{-3/2}. \\

\end{align}
$$

Now I don't know about you but first time I saw this I didn't really get it, but I love to figure it out by doing, so if we do the same trick as before. I can't show how I generated the samples without getting sidetracked but this is what random samples of this distribution look like.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-01-26/good-gradient-samples.png" 
      alt="Correct gradient samples for our analysis." 
      title="Correct gradient samples for our analysis." 
      style="width: 75%; height: auto; border-radius: 8px;">
</div>

Much better!

Now if we use _this_ prior in our analysis, we don't include this bias (spoiler alert, for our small case it doesn't make a huge difference). In the next post I'll try and tackle how we can explore the posterior without trying to scan the whole parameter space.

