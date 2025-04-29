---
title: 'Normalising Flows for variational inference (with FlowJAX)'
date: 2025-04-28
permalink: /posts/2025/04/2025-04-28-normalising-flows/
tags:
  - Variational Inference
header-includes:
   - \usepackage{amsmath}

---

In this post I will attempt to give an introduction to normalising flows from the viewpoint of variational inference.

---

This is currently a standalone post mainly teach myself how to rigorously talk about Normalising flows within the context of variational inference. 
I am not very experienced with it in real world settings (as of 28/04/2025 as least), so just keep that in mind if you are reading this. 
Later, I may create a new post that integrates into my series of introductory posts.

Here are the resources I am using as references for this post. Go to any of these if you need more information or don't mesh well with the way I'm discussing this topic.

- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) - Rezende, 2015
    - I connected best with this reference for general variational inference. Highly recommend to give it a read
- [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257v4) - Kobyzev, 2020
    - Specifically for normalizing flows. I found it a tad confusing due to frequent switching between variable names quite often but the core ideas were pretty clear and well communicated in my opinion.
- ["Normalizing Flows" by Didrik Nielsen](https://youtu.be/bu9WZ0RFG0U) - [Probabilistic AI School](https://www.youtube.com/@probabilisticai)
- ["Density estimation using Real NVP"](https://arxiv.org/abs/1605.08803) - Dinh, 2016
- ["Masked Autoregressive Flow for Density Estimation"](https://arxiv.org/abs/1705.07057) - Papamakarios, 2018

--- 

## Table of Contents
- [Variational Inference](#variational-inference)
- [Normalising Flows](#normalising-flows)

- [Normalising Flows with neural network mediated transforms](#normalising-flows-with-neural-network-mediated-transforms)
- [Examples](#examples-analyses)
    - [Example: 2D with 4D posterior](#example-simple-4d-latent-model-normalising-flows-and-nested-sampling-comparison)
    - [Example: Hierarchical Bayesian Model Analysis with 25D posteror](#example-hierarchical-bayesian-model-analysis)
- [Some Annoying Things...](#some-annoying-things-about-normalising-flows)


---



## Variational Inference

One typical goal of analysis is to develop a posterior distribution for inference with,

$$ \begin{align} \pi(z\midx). \end{align} $$

Bayesian inference methods typically approach this problem using MCMC or Nested Sampling, leveraging the product of the likelihood ($$p(x\mid z)$$) and prior on $$x$$ ($$p(z)$$).

$$ \begin{align} \pi(z\mid x) \propto \mathcal{L}(x\midz) \pi(z) \end{align} $$

With the goal of producing representative samples of the posterior density. MCMC and Nested Samplers are exact in their respective limits. If you run MCMC for long enough (infinite time) or you increase the number of live points (e.g. an infinite number live points) then you should theoretically get exact results (or at least exact enough that the difference doesn't matter).

_Variational inference_ trades this possibility of exact-ness in favour of methods that find the closest "nice" distribution to the posterior distribution. 

In essence, you define a family of probability densities/functions that you want to consider, denoted $$\mathcal{Q}$$, and then minimise the difference between your approximate and target densities. The exact target density _may be_ within the set of functions that you consider, but not necessarily.


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/variational_inference_concept.png" 
    alt="Image showing the broad concept of variational inference not guaranteeing exactness" 
    title="Image showing the broad concept of variational inference not guaranteeing exactness" 
    style="width: 80%; height: auto; border-radius: 16px;">
</div>


You may like the guarantee that your algorithm will find the exact distribution, but the variational inference users will argue that in practice we never achieve "exactness" anyway, so why not settle for a method with non-exactness that gives you a much nicer representation of your posterior?

I would like to agree with this perspective, but during this post keep in mind:
- The family of functions you consider may not have anything that follows the exact distribution very closely at all
    - One may take the Laplace approximation of the posterior but some of you may be dealing with a mixture model, and hence some variables will much more closely follow an uninformative Dirichlet distribution which a normal distribution will have a very hard time approximating for example
- As far as I know, there are also no nice ways to analyse the convergence of a given variational inference algorithm like with MCMC and nested sampling do with autocorrelation and $$\text{dlogz}$$
    - Although special note, I've found that once the log of loss curve for the below examples flattens out, it doesn't notably improve anymore and that seems to have been good enough so far

But again, the end result of all variational algorithms (that at least I've stumbled across) give you a _parametric form_ of your posterior, which is extremely useful. However, the best thing that I've found, is that with this approach, it turns the issue of estimating the posterior from a sampling problem into an _optimisation_ problem. And with that, you (theoretically) get an easier problem and can utilise the many improvements in the field of optimisation in the last few years to further speed up your analysis.

### The exact goal of variational inference

Let's say that your approximation to the target density/posterior is $$q_\Phi(z) \in \mathcal{Q}$$[^1] where I use $$\Phi$$ for the parameters that characterise the approximation, $$\Phi$$ determining where $$q$$ is within $$\mathcal{Q}$$. Then ___the exact goal___ of variational inference is the following,

[^1]: Notice that I've dropped the $$x$$ dependence. This is because this approximation does not take x in as an input. Your final result will still be indirectly dependent on $$x$$ due to the optimisation of the KL divergence.

$$ 
\DeclareMathOperator*{\argmin}{arg\,min}
\begin{align}
q_{\Phi^*}(z) = \argmin_{q_\Phi(z) \in \mathcal{Q}} \;\; KL(q(z) || \pi(z|x)).
\end{align}$$

i.e Find the $$\Phi=\Phi^*$$ and thus $$q_\Phi(z)$$ that minimises the KL divergence (distance) of your approximate density to your target density. With $$q_{\Phi^*}(z)$$ being the closest $$q_\Phi(z)$$ in $$\mathcal{Q}$$ to $$\pi(z\mid x)$$ characterised by the set of parameters $$\Phi^*$$ (where the densities are normalised with respect to $$z$$ __not__ $$\Phi$$). So amazingly, we've turned our sampling problem into a minimization problem.

We can then expand the KL divergence like the following,


$$ 
\begin{align}
 KL(q(z) || \pi(z|x)) &= \mathbb{E}_{q(z)}[\log\left(q(z)\right)] - \mathbb{E}_{q(z)}[\log\left(\pi(z|x)\right)] \\
 &= \mathbb{E}_{q(z)}[\log\left(q(z)\right)] - \mathbb{E}_{q(z)}[\log\left(p(z,x)\right)] + \mathbb{E}_{q(z)}[\log\left(\mathcal{L}(x)\right)] \\
 &= \mathbb{E}_{q(z)}[\log\left(q(z)\right)] - \mathbb{E}_{q(z)}[\log\left(p(z,x)\right)] + \log\left(\mathcal{L}(x)\right) \\
 &= -\text{ELBO}(q) + \log\left(\mathcal{L}(x)\right). \\
\end{align}$$

Where the $$\text{ELBO}$$ stands for the evidence lower bound, denoted as such because by a quick rearrangement of the above.

$$ 
\begin{align}
\log\left(\mathcal{L}(x)\right) = KL(q(z) || \pi(z|x)) + \text{ELBO}(q)\\
\end{align}$$

The KL divergence is always positive, hence you can see that the $$\text{ELBO}$$ serves as a lower bound for $$\log\left(\mathcal{L}(x)\right)$$. Additionally, the $$\text{ELBO}$$ only includes the joint probability density and posterior estimate and not the full posterior. Hence, we can use our standard likelihood-prior product for maximisation of the $$\text{ELBO}$$ to find our $$q_{\Phi^*}(z)$$.

## Normalising Flows

normalising flows in a nutshell are the construction of a variational inference posterior estimate with the use of stacked and transformed simpler probability densities. 

With some random variable $$Y$$ with known and tractable probability distribution, $$p_Y$$, and $$Z = g(Y)$$ or $$Y = f(Z)$$, it's the simply the use of,

$$\begin{align}
p_Z(z) &= p_Y(Y=f(z)) | \det \, \mathbf{Df}(z)| \\
&= p_Y(f(z)) | \det \, \mathbf{Dg}(f(z))|^{-1}, \\
\end{align}$$

where $$ \mathbf{Df}(z) = \frac{\partial f}{\partial z}$$ is the Jacobian (matrix) of $$f$$ w.r.t $$z$$ and similarly $$ \mathbf{Dg}(y) = \frac{\partial g}{\partial y}$$ is the Jacobian of $$g$$ w.r.t $$y$$. $$p_Z(z)$$ in this scheme is referred to as the _pushforward_ of $$p_Y$$ by the function $$g$$ sometimes denoted $$g_* p_Y$$. 

The ___normalising___ in normalising flows actually comes from the inverse function $$f$$ that transforms the presumably complicated variable $$Z$$ to the less complicated $$Y$$, as what we are essentially doing is simplifying or _"normalising"_ our data into something simpler that we can compute/sample/explore.

The above only uses a single transformation $$g$$ to achieve this, but we know that if a function is bijective (one-to-one and surjective) with a given inverse, then a series of these functions as a composite function is still bijective with a defined inverse.

$$g = g_N \circ g_{N-1} \circ \dots \circ g_2 \circ g_1, $$

with inverse,

$$f = f_1 \circ f_{2} \circ \dots \circ f_{N-1} \circ f_N, $$

with the determinant of $$f$$ given by,

$$  \det \, \mathbf{Df}(z) = \prod_{i=1}^N  \det \, \mathbf{Df}_i(s_i) $$

With $$s_i$$ being an intermediate variable described by,

$$ s_i = g_i \circ g_{i-1} \circ \dots \circ g_2 \circ g_1(y) =f_{i+1} \circ f_{i+2} \circ \dots \circ f_{N-1} \circ f_N(z), $$

with $$s_N = z$$. So we can construct a series of transformations, that individually can be quite simple, but combined can allow us to express quite complicated behaviour. An example of this at work is shown in the GIF below, made by Eric Jang for [his own tutorial on normalising flows](https://blog.evjang.com/2019/07/nf-jax.html) focusing on how to actually code this all up yourself in JAX.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/flow_gif.gif" 
    alt="GIF showing how stacking multiple transformations on a simple distribution can describe a more complicated distribution in the same way that a normalising flow does" 
    title="GIF showing how stacking multiple transformations on a simple distribution can describe a more complicated distribution in the same way that a normalising flow does" 
    style="width: 80%; height: auto; border-radius: 16px;">
</div>



Back to the math, because we typically like working in log-space to maintain numerical stability we observe that for $$M$$ values from our distribution $$\mathcal{Z}$$, the flow transformation parameters $$\theta$$ and any parameters of the base distribution $$p_Y$$, $$\phi$$, that[^X],

[^X]: With this setup, the parameters $$\phi$$ and $$\theta$$ together signify $$\Phi$$ i.e. $$\Phi = \{\theta, \phi\}$$

$$\begin{align}
\log \, p(\mathcal{Z}|\theta, \phi) &= \sum_{i=1}^M \log p_Z(z^{(i)}|\theta, \phi) \\
&= \sum_{i=1}^M \log p_Y(f(z^{(i)}|\theta)| \phi) + \log|\det \, \mathbf{Df}(z^{(i)}|\theta)|.
\end{align}$$

$$z^(i)$$ being samples from our flow model. So during training, the primary parameters to optimise over are those dictating the transformations $$\theta$$ and the dependencies of the simple distributions $$\phi$$ (which together represent the $$\Phi$$ above i.e. $$\Phi=\{\theta, \phi\}$$ ) to maximize the $$\text{ELBO}$$. By a slight reparameterization of the above such that $$q(z\mid x, \theta) = p_Z(z\mid\theta)$$ and the corollary of,

$$ \begin{align} \mathbb{E}_{p_Z(z|\theta)} [h(z)] = \mathbb{E}_{p_Y(y)}[h(g(y|\theta))], \end{align}$$

(treating $$\phi$$ as a constant, as any non-trivial dependence can be relegated to $$\theta$$ anyways) with $$h$$ being an arbitrary function, allows calculation of gradients for gradient descent/ascent for optimizing[^ZZZ].

[^ZZZ]: Because we have moved the dependence on $$\theta$$ from the function being integrated over to the argument/integrand.

<!-- $$\begin{align} \mathcal{L}(\theta) = \mathbb{E}_{q(z\mid x, \theta)}\left[\log(p(z, x)) - \log(q(z\mid\theta)) \right]. \end{align}$$

The issue being that we need to calculate gradients w.r.t $$\theta$$, but with the reparameterisation $$\theta$$ goes from being part of the density that we are taking our average with respect to, to within a function that we are taking the average of which is much simpler. -->

### The process 

Something to keep in mind for the following is that the general process follows;

1. Create a sample of z according to the current posterior estimator $$q(z\mid\theta)$$
2. Calculate the probability of this sample under $$q(z\mid\theta)$$ and the joint probability $$\mathcal{L}(x\mid z)\pi(z)$$
3. Calculation of jacobian and log determinant
4. Propose new set of values for optimisation parameters



## Examples of Flow Methods

Your next question should be then, well what are the transformations?? What should I specifically make $$g$$/$$f$$?? Well the class of functions that you wish to consider basically dictates the overall approach that you want to take. I'll show some examples of different methods building up to the ultimate goal of this post which is variational inference with normalising flows with neural network assisted transformations. 


### Linear Flows

Linear flows are normalising flows that utilise transformations of the form,

$$\begin{align}

g(s) = A s +b,

\end{align}$$

where $$A \in \mathbb{R}^{D\times D}$$ and $$b_i\in \mathbb{R}^D$$ and $$D$$ is the dimensionality of $$z$$. These types of transformations are relatively restricted in the complexity of posteriors they can express, but are simpler to implement compared to other methods while still being able to capture many less pathological distributions. The determinant of the jacobian of the transformation is just the determinant of $$A$$ and the inverse transformation just $$A^{-1}$$ (and the total transformation just the product of all of these), but both operations can be expensive in high dimensions with complexities of $$\mathcal{O}(D^3)$$. However, we can restrict the form of $$A$$ to increase the efficiency of these operations.

And then you can layer these transformations (letting $$g=g_i$$ above) to create a more flexible model.

#### Diagonal-Linear Flows

If $$A$$ is a diagonal matrix, the inverse is just the reciprocal of the diagonal elements and the determinant is the product of the elements on the diagonal which are both $$\mathcal{O}(D)$$. However, this transformation turns into an element-wise transformation which makes it impossible to capture the correlation between variables.



#### Triangular-Linear Flows

If we make $$A$$ a lower or upper triangular matrix then we can theoretically capture the correlations between variables because multiple variables from the base distribution can mix, but additionally the matrix determinant is still just the product of the diagonal, and matrix inversion is $$\mathcal{O}(D^2)$$ instead of $$\mathcal{O}(D^3)$$ for general matrix inversion.

## Coupling Flows

Coupling flows are set up such that the transforms are conditioned on other values of the variables in posterior of interest. i.e. If you have a D-dimensional posterior still denoted with $$\vec{z} = \{z_1, z_2, ..., z_d, z_{d+1}, ..., z_D\}$$ and $$y$$ and $$x$$ follow this same dimensionality. A _coupling flow_ in this scheme is constructed as a hybrid function, remembering that $$\vec{s}$$ is our fill in "intermediary variable" between $$y$$ and $$z$$, such that,

$$
\begin{align}

\vec{z}= g(\vec{s}) = \begin{cases} 
      h\left(s_{1:d};\Theta(s_{d+1:D})\right)\\
      s_{d+1:D}
   \end{cases}.

\end{align}
$$

In essence, you leave some parts of your intermediary variable alone, $$s_{d+1:D}$$ , and then transform the rest, $$s_{1:d}$$, based on some _coupling function_ $$h$$ with parameters based on some _conditioner_ $$\Theta$$ that are dependent on the part of the variable we leave alone, $$x_{d+1:D}$$. The assumption being that you would then apply some permutation function inbetween subsequent layers so that you apply a transformation to every value in $$y$$ (so you wouldn't really have $$z$$ on the left above but the next stage of the intermediary variable $$s$$ but that'd be confusing). 

This approach is nice as you get a lot of expressiveness out of it (depending on what you choose for $$h$$) while gives you a block triangular matrix with the blocks being the identity matrix and $$Dh$$ (which is just $$d\times D-d$$ as opposed to $$D\times D$$). The method by which you partition it is then up to you, but it's common to either split it in half or do some alternating pixels if you are looking at image data. 

Additionally, $$\Theta$$ can be as complex as you like and is often represented using a neural network (but not the goal of this post).

The inverse then looks like (presuming that the inverse, $$h^{-1}$$, exists),

$$
\begin{align}

\vec{s} = f(\vec{z}) = \begin{cases} 
      h^{-1}\left(z_{1:d};\Theta(s_{d+1:D})\right)\\
      z_{d+1:D}
   \end{cases}.

\end{align}
$$


## Autoregressive Flows

General autoregressive models outside of just normalising flows describe the probability of a given event or variable as some sort of function on previous events/variables. i.e. If I have a variable $$x=(x_1, x_2, ..., x_d)$$ then an autoregressive model might describe the probability as,

$$ p(x) = p(x_1)\cdot p(x_2\mid x_1)\cdot p(x_3\mid x_1, x_2) \cdot \dots \cdot p(x_d\mid x_1, ..., x_{d-1}) $$

This has a nice extensions to what we're doing with our intermediate variables $$s_i$$.

We generate the samples as 

$$\begin{align}
z_i = h(s_i|\Theta_{i}(s_{1:i-1})),
\end{align}$$

where $$\Theta_1$$ is a constant. This is extremely similar to what we saw in the coupling flows except instead of our transformation is blocks of $$\vec{s}$$ being transformed into blocks of $$\vec{s}$$ they are now of blocks of $$\vec{s}$$ being transformed into specific values of $$\vec{s}$$, $$s_i$$. The functions also retain the same terminology. So the determinant here is given by[^Block],

[^Block]: Additionally [Kobyzev, 2020](https://arxiv.org/abs/1908.09257v4) makes the useful comparison that the autoregressive flow architecture can be seen as a non-linear generalization of the triangular affine transformation above. 

$$\begin{align}
\det(Dg) = \prod_{i=1}^D \frac{\partial x_i}{\partial s_i} .
\end{align}$$

The difficulty for this style of flow is the calculation of the inverse that has to be found using _recursion_ (icky) and inherently sequential as,

$$\begin{align}
s_1 &= h^{-1}(z_1;\Theta_1) (\text{easy?})\newline
s_2 &= h^{-1}(z_2;\Theta_2(s_1)) (\text{depends on }s_1)\newline
s_3 &= h^{-1}(z_3;\Theta_3(s_1, s_2)) (\text{depends on }s_2\text{ (which depends on }s_1\text{) and }s_1)\newline
&  \vdots \newline
s_i &= h^{-1}(z_i;\Theta_i(s_{1:i-1})) (\text{depends on }s_{i-1}\text{ (which depends on }s_{i-2}...,s_1\text{) and }s_{i-2}\text{ (which depends on ...)} )\newline
\end{align}$$

Despite this drawback autoregressive models have been shown to have an extreme level of flexibility at capturing complex distributions (e.g. Papamakarios, 2017). And additionally, there is a related architecture called _"inverse autoregressive flows"_ where $$z_i = h(s_i;\theta(z_{1:i-1}))$$ that has the _inverse_ problem that the forward direction is hard to compute but the inverse is simple, i.e. that it is quicker to sample and is thus better suited to methods within variational inference that require lots of sampling.



### Examples of coupling functions

Well after all that it may seem like I've kicked the can down the road as the question was "what functions should I use for my transformations" and now I've left the answer open because now you're likely asking "well what should I use for my coupling functions!". Well for now I would just look at the titles as they're pretty self-explanatory, and I want to finish this post.




#### Affine Coupling Functions



#### Cubic Splines



#### Rational Quadratic Splines



## Notable mentions of approaches that I did not cover

 - Sequential Neural Posterior estimation (see e.g. [Automatic Posterior Transformation for Likelihood-Free Inference](https://arxiv.org/abs/1905.07488) or [On Contrastive Learning for Likelihood-free Inference](https://arxiv.org/abs/2002.03712))
   - A method where you skip the use of priors and likelihoods and instead estimate the posterior using simulations based on the parameters of interest
 - Unconditional density estmation
    - If you have a samples from the distribution directly and wish to construct a parametric form 
 - [RealNVP](https://arxiv.org/abs/1605.08803)
    - A very successful approach to normalising flows algorithm
 - Continuous Flows
    - Not even sure how this one works but it looks neat
 -  
 - Many others, I'd highly recommend looking at all the resources I've linked to above for more info.



## Normalising Flows with neural network mediated transforms (Neural Autoregressive Flows)

Thank you for sticking around for so long (and welcome those that skipped most of the post just to look at the conclusion). The final cherry on top to all of the above is that although it is a minimisation/maximisation exercise we can leverage recent developments in machine learning, particularly in neural networks, to optimise the parameters of the coupling functions or general parameters of the above. 

There's not much specific theory that I can provide here as all it essentially comes to is that you replace the _conditions_ in the above coupling functions ($$\Theta$$) with neural networks. One may ask why not just use neural networks as the transformations from the get-go and I can't say anything concrete as to why but I believe one issue would be that the transformations would be dense (i.e. in matrix form the matrices would be dense) and possibly hard to invert.

Once again, I would recommend [Eric Jang's tutorial](https://blog.evjang.com/2019/07/nf-jax.html) if you are interested in coding up a normalising flow yourself, but personally I cannot be bothered dealing with all the bugs that I would accidentally introduce in the process so I'd prefer to use an "off-the-shelf" code that will do it for me.

In my case, I've quite liked the capabilities of the [FlowJAX](https://danielward27.github.io/flowjax/index.html) python package but had good experiences using [PyTorch](https://github.com/VincentStimper/normalizing-flows) and [TensorFlow](https://blog.evjang.com/2018/01/nf1.html).

For the rest of the tutorial I'm simply going to walk you through some examples of how to use [FlowJAX for variational inference](https://danielward27.github.io/flowjax/examples/variational_inference.html)

## Examples Analyses


### Example: Simple 4D latent model normalising flows and nested sampling comparison

We'll start off with a simpler example with x data generated from a uniform distribution and y data generated by a mixture of two linear models with intrinsic scatter (model-dependent noise). In math,

$$\begin{align}
X &\sim \mathcal{U}(0, 10)\\
Y_1 &\sim \mathcal{N}(\mu=m_1 \cdot X + c_1, \sigma^2=\sigma_1^2)\\
Y_2 &\sim \mathcal{N}(\mu=m_2 \cdot X + c_2, \sigma^2=\sigma_2^2)\\
Y &\sim \begin{cases}
Y_1 \text{ with probability }w_1\\
Y_2 \text{ with probability }w_2
\end{cases}
\end{align}$$

<div style="text-align: center;">
    <div style="display: inline-block; width: 45%; margin: 0 10px;">
        <img 
            src="/files/BlogPostData/2025-04-28/basic_2d_scatter_true_values.png" 
            alt="2D scatter values coloured by which model they come from" 
            title="2D scatter values coloured by which model they come from" 
            style="width: 100%; height: auto; border-radius: 16px;">
    </div>

    <div style="display: inline-block; width: 45%; margin: 0 10px;">
        <img 
            src="/files/BlogPostData/2025-04-28/basic_2d_scatter_data.png" 
            alt="2D scatter data" 
            title="2D scatter data" 
            style="width: 100%; height: auto; border-radius: 16px;">
    </div>
</div>

Now we want to chuck a normalising flow and nested sampler at this and compare the two. I'll fix the intercept of the first model to 0, so that it's less likely to get identifiability issues and fit for _$$m1$$_, _$$m2$$_, _$$c2$$_, and the two mixture fractions (one of which is exactly determined by the other so we only have ___4___ truly separate variables). The likelihood for a given ponit $$(x_i, y_i)$$ follows,

$$\begin{align}
&\mathcal{L}(x_i, y_i|m_1, m_2, c_2, w_1, w_2)\\ 
&= w_1 \frac{1}{\sqrt{2\pi\sigma_1^2}}\exp\left(-\frac{(y_i-(m_1 x_i+0))^2}{2\sigma_1^2}\right) + w_2 \frac{1}{\sqrt{2\pi\sigma_2^2}}\exp\left(-\frac{(y_i-(m_2 x_i+c_2))^2}{2\sigma_2^2}\right)
\end{align}$$

and then a uniform prior on $$m_1$$, $$m_2$$, $$c_2$$ and then a uninformative Dirichlet prior on the mixture fractions (which in this case equates to a uniform prior from 0 to 1). Chucking this into a nested sampler we find the following.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/basic_example_NS_corner.png" 
    alt="Corner plot of posterior on 2D scatter data using nested sampling" 
    title="Corner plot of posterior on 2D scatter data using nested sampling" 
    style="width: 80%; height: auto; border-radius: 16px;">
</div>

We want to just translate this into JAX.

```python
def jax_event_loglikelihood(yval, xval, mval, sigmaval, cval):
    return jax.scipy.stats.norm.logpdf(yval, loc=mval*xval+cval, scale=sigmaval)


def jax_total_loglikelihood(theta, yvals, xvals):
    m1val, m2val, c2val, w1, w2 = theta
    w2 = 1-w1
    return jnp.sum(jax.scipy.special.logsumexp(
        jnp.array([
            jnp.log(w1)+jax_event_loglikelihood(yvals, xvals, m1val, sigmaval=model_1_config['s'], cval=model_1_config['c']),
            jnp.log(w2)+jax_event_loglikelihood(yvals, xvals, m2val, sigmaval=model_2_config['s'], cval=c2val),

        ]), axis=0))


import flowjax.bijections as bij




def jax_logprior(theta):
    m1val, m2val, c2val, w1, w2 = theta

    m1_logprob = jax.scipy.stats.uniform.logpdf(x=m1val, loc=-10, scale=20)
    m2_logprob = jax.scipy.stats.uniform.logpdf(x=m2val, loc=-10, scale=20)
    c2_logprob = jax.scipy.stats.uniform.logpdf(x=c2val, loc=-50, scale=100)

    mixture_logprob = jax.scipy.stats.dirichlet.logpdf(x=jnp.array([w1, w2]), alpha=jnp.array([1, 1]))


    mixture_logprob = jnp.where(jnp.isnan(mixture_logprob), -jnp.inf, mixture_logprob)
        
    logprob= m1_logprob + m2_logprob + c2_logprob + mixture_logprob


    return logprob


# because I could be bothered writing the SigmoidAffine regulariser for later on to use with constrained domains
def unregulariser(theta_reg):
    m1val_reg, m2val_reg, c2val_reg, w1val_reg = theta_reg

    w1val = w1val_reg
    w2val = 1-w1val_reg

    m1val = m1val_reg*20-10
    m2val = m2val_reg*20-10
    c2val = c2val_reg*100-50

    theta = (m1val, m2val, c2val, w1val, w2val)

    return theta

def unnormalised_posterior(theta):

    theta = unregulariser(theta)
    logprior_val = jax_logprior(theta)

    return  logprior_val+ jax_total_loglikelihood(theta, yvals=y_vals, xvals=x_vals)

```

So far, this is not any different to the usual "make a likelihood and prior and chuck them into favourite thinga-majig" setup. And that's kind of the point, as all you need to do after that is chuck the likelihood and prior into your ___new___ favourite thinga-majig!

```python

from flowjax.bijections import Affine
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_key_based_loss
from flowjax.train.losses import ElboLoss
import jax.random as jr
from paramax import non_trainable
from flowjax.distributions import Normal, Transformed
from flowjax.bijections import RationalQuadraticSpline

key = jr.key(0)



# loss = ElboLoss(unnormalised_posterior, num_samples=10000)
loss = ElboLoss(unnormalised_posterior, num_samples=500)

key, flow_key, train_key, sample_key = jr.split(key, 4)
flow = masked_autoregressive_flow(
    flow_key, base_dist=Normal(loc=jnp.array([0.5, 0.5, 0.5, 0.5]), scale=jnp.array([0.01,0.01,0.01,0.01])), 
    transformer=RationalQuadraticSpline(knots=4, interval=(0,1)),
    invert=False,
)


# Train the flow variationally
flow, losses1 = fit_to_key_based_loss(
    train_key, flow, loss_fn=loss, learning_rate=1e-2, steps=100,
)

# Train the flow variationally
flow, losses2 = fit_to_key_based_loss(
    train_key, flow, loss_fn=loss, learning_rate=1e-3, steps=200,
)


```

Which gives you the following loss curve (remembering that ELBO provides a lower bound on the evidence, so it doesn't make sense to be 0 so be offset the loss by the minimum value).
<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/basic_example_NF_loss_curve.png" 
    alt="Loss curve for normalising flow approximation of 2D scatter data posterior" 
    title="Loss curve for normalising flow approximation of 2D scatter data posterior" 
    style="width: 50%; height: auto; border-radius: 16px;">
</div>

And finally the approximate posterior.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/basic_example_NF_corner.png" 
    alt="Loss curve for normalising flow approximation of 2D scatter data posterior" 
    title="Loss curve for normalising flow approximation of 2D scatter data posterior" 
    style="width: 80%; height: auto; border-radius: 16px;">
</div>


We can then stack the two approximations on top of each other and compare.



<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/basic_example_stacked_corner.png" 
    alt="Corner plot showing the nested sampling and normalising flow approximations of the 2D scatter data posterior" 
    title="Corner plot showing the nested sampling and normalising flow approximations of the 2D scatter data posterior" 
    style="width: 80%; height: auto; border-radius: 16px;">
</div>

You can see that for most of the variables the distributions match quite closely although the normalising flow distribution on the second model's gradient seems more localised about $$m_1$$. Note however that this does not mean the normalising flow is better, in fact, the stability and validity of the nested sampler is much higher than the normalising flow, so really this means that the normalising flow has likely _underestimated_ the uncertainties here (which is quite common in variational inference).


### Example: Hierarchical Bayesian Model Analysis

The only reason that you should be interested in normalising flows should be either some benefit of getting a parametric representation out of your analysis, the dimensionality of your problem is high (and you don't have or can't be bothered finding yourself gradient data) or you've got some pathological posterior that other samplers are having trouble with. For this next example I was hoping to show this to you but could only get as far as 10 dimensions (1 degenerate, so 9 independent) for reasons that I will explain after this example. 

For this example we're going to look at pseudo-particle tracking data where you have a three straight lines with gaussian intrinsic scatter going through $$x$$, $$y$$, $$z$$ with parametric representations based on time $$t$$ which is sampled uniformly from $$-20$$ to $$20$$ and a "deposited energy" variable $$E$$ that follows an exponential relationship with $$t$$ (in hopes of reproducing the general behaviour of a particle exponentially losing energy over time). This data is shown in the first row of GIFs. We then add gaussian noise to the spatial data and log-normal noise to the energy data (independent of time and each other for simplicity) with some background noise that is just uniform in space and log-uniform in energy (again independent of time).

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/rotation_w_energy.gif" 
    alt="Rotating GIF showing the pseudo-track data used for analysis" 
    title="Rotating GIF showing the pseudo-track data used for analysis" 
    style="width: 100%; height: auto; border-radius: 16px;">
</div>



<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/frames_w_noise_and_bkg.gif" 
    alt="Rotating GIF showing the pseudo-track data including noise and background counts used for analysis" 
    title="Rotating GIF showing the pseudo-track data including noise and background counts used for analysis" 
    style="width: 100%; height: auto; border-radius: 16px;">
</div>

In essence, for the true values,

$$\begin{align}
T &\sim \mathcal{U}(-20, 20) \\
(X_i, Y_i, Z_i) &\sim \mathcal{N}\left(
\mu=(a_{i1} T + b_{i1}, a_{i2} T + b_{i2}, a_{i3} T + b_{i3}),\ \Sigma = \sigma_i^2 \mathcal{I}
\right) \\
(X_b, Y_b, Z_b) &\sim \mathcal{U}(-40, 40)^3 \\
E_i &\sim E_{10} \exp(-\phi_{i} T) \\
E_b &\sim \log\mathcal{U}(-3, 3) \\
(X, Y, Z, E) &=
\begin{cases}
(X_1, Y_1, Z_1, E_1) & \text{with probability } w_1 \\
(X_2, Y_2, Z_2, E_2) & \text{with probability } w_2 \\
(X_3, Y_3, Z_3, E_3) & \text{with probability } w_3 \\
(X_b, Y_b, Z_b, E_b) & \text{with probability } w_b
\end{cases}
\end{align}$$

then we add noise to mimic a detector,

$$\begin{align}
(x_{obs}, y_{obs}, z_{obs}) &\sim \mathcal{N}(\mu=(X, Y, Z), \Sigma = \sigma_I^2 \mathcal{I}) \\
e_{obs} &\sim \text{Lognormal}(\mu=E, \sigma^2=\sigma_E^2) \\
t_{obs} &\sim \text{Lognormal}(\mu=T, \sigma^2=\sigma_t^2).
\end{align}$$

So we do the same old song and dance of making a prior (which is made up of either normal or log-normal distributions roughly about where the true values are to I'm less likely to run into instability issues) and likelihood with the addition of this handy class to handle any constrained variables,

```python
import jax
import jax.numpy as jnp
from jax import nn
from typing import ClassVar
from flowjax.bijections import AbstractBijection, Affine
from collections.abc import Callable
from functools import partial
from typing import ClassVar

import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, ArrayLike, Shaped
from paramax import AbstractUnwrappable, Parameterize, unwrap
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array
from jax.scipy.special import logit  # Import logit


class SigmoidAffine(AbstractBijection):
    r"""Sigmoid and affine transformation combined: 
    First, applies Sigmoid(x) and then applies affine transformation y = a * x + b.

    Args:
        loc: Location parameter for the affine transformation. Defaults to 0.
        scale: Scale parameter for the affine transformation. Defaults to 1.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: jax.Array
    scale: jax.Array

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
    ):
        # Convert loc and scale to arrays using arraylike_to_array utility
        self.loc = arraylike_to_array(loc, dtype=float)
        self.scale = arraylike_to_array(scale, dtype=float)

        # Ensure loc and scale are broadcastable
        self.loc, self.scale = jnp.broadcast_arrays(self.loc, self.scale)
        self.shape = self.scale.shape

    def transform_and_log_det(self, x, condition=None):
        # Apply Sigmoid first
        sigmoid_x = nn.sigmoid(x)
        # Apply affine transformation
        y = sigmoid_x * self.scale + self.loc
        log_det_affine = jnp.sum(jnp.log(jnp.abs(self.scale)))

        # Log determinant from the sigmoid transformation
        log_det_sigmoid = jnp.sum(nn.log_sigmoid(x) + nn.log_sigmoid(-x))

        # Total log determinant is the sum of both log determinants
        log_det = log_det_sigmoid + log_det_affine
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        # Apply inverse of affine transformation
        affine_inv_x = (y - self.loc) / self.scale
        log_det_affine = -jnp.sum(jnp.log(jnp.abs(self.scale)))

        # Apply inverse of Sigmoid (logit function)
        x = logit(affine_inv_x)

        # Log determinant from the sigmoid inverse transformation
        log_det_sigmoid = -jnp.sum(nn.log_sigmoid(x) + nn.log_sigmoid(-x))

        # Total log determinant is the sum of both log determinants
        log_det = log_det_sigmoid + log_det_affine
        return x, log_det
```

We can then set up the flow as the following,

```python
from flowjax.bijections import Affine, Sigmoid, Identity, RationalQuadraticSpline
from flowjax.distributions import Normal, StudentT, Transformed
from flowjax.flows import masked_autoregressive_flow, triangular_spline_flow, block_neural_autoregressive_flow
from flowjax.train import fit_to_key_based_loss
from flowjax.train.losses import ElboLoss
import jax.random as jr
from paramax import non_trainable

key = jr.key(0)

def create_bijections():
    # Bijections for each parameter to enforce bounds. But most follow a normal distribution instead of uniform where bounds are required, 
        # so identity is used
def create_bijections():
    # Bijections for each parameter to enforce bounds
    bijections = []

    ## Model 1 Stuff
    # as
    bijections.append(Identity())
    bijections.append(Identity())
    bijections.append(Identity())
    # bs
    bijections.append(Identity())
    bijections.append(Identity())
    # es
    bijections.append(Identity())
    bijections.append(Identity())


    ## Model 2 Stuff
    # as
    bijections.append(Identity())
    bijections.append(Identity())
    bijections.append(Identity())
    # bs
    bijections.append(Identity())
    bijections.append(Identity())
    # es
    bijections.append(Identity())
    bijections.append(Identity())

    ## Model 3 Stuff
    # as
    bijections.append(Identity())
    bijections.append(Identity())
    bijections.append(Identity())
    # bs
    bijections.append(Identity())
    bijections.append(Identity())
    # es
    bijections.append(Identity())
    bijections.append(Identity())




    # For sigma_t (log-transformed, bounded between logt_loc and logt_loc + logt_scale)
    bijections.append(Identity())

    # For sigma_e (log-transformed, bounded between logsigma_e_loc and logsigma_e_loc + logsigma_e_scale)
    bijections.append(Identity())
    # For w1reg (bounded between 0 and 1, use Sigmoid to constrain it)
    bijections.append(SigmoidAffine(loc=0, scale=1))
    # For w2reg (bounded between 0 and , use Sigmoid to constrain it)
    bijections.append(SigmoidAffine(loc=0, scale=1))
    # For w3reg (bounded between 0 and , use Sigmoid to constrain it)
    bijections.append(SigmoidAffine(loc=0, scale=1))

    bijections = bij.Stack(bijections)

    return bijections




loss = ElboLoss(unnormalised_posterior, num_samples=500, stick_the_landing=True)


key, flow_key, train_key, sample_key = jr.split(key, 4)
unbounded_flow = triangular_spline_flow(
    key=flow_key,
    base_dist=Normal(loc=0.*jnp.array(list((true_vals))), 
                    scale=0.1*jnp.ones(len((true_vals)))), 
    knots=4,
    tanh_max_val=3.0,
    invert=False,
    init=None,
    flow_layers=6,
)

flow = Transformed(
    unbounded_flow,
    non_trainable(create_bijections()) # Ensure constraint not trained!
)

schedule = [(1e-2, 50),(4e-3, 50),(1e-3, 50),]  
total_losses = []
for learning_rate, steps in schedule:
    train_key, subkey = jr.split(train_key)

        # Train the flow variationally
    flow, losses = fit_to_key_based_loss(
        train_key, flow, loss_fn=loss, learning_rate=learning_rate, steps=steps,
    )
    total_losses.extend(losses)


```

This gave me the following loss curve. Anecdotally, I've found that if the loss curve follows this kind of decaying sinusoidal looking shape that the posterior flow samples are of better quality. I imagine this is because it's found the correct minimum and is bouncing back and forth across the "well" towards the proper solution.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/high_d_NF_loss.png" 
    alt="Loss curve for normalising flow pseudo-particle track data posterior" 
    title="Loss curve for normalising flow pseudo-particle track data posterior" 
    style="width: 50%; height: auto; border-radius: 16px;">
</div>

And posterior.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-04-28/high_d_corner_dist.png" 
    alt="Normalising flow posterior for pseudo-particle track data" 
    title="Normalising flow posterior for pseudo-particle track data" 
    style="width: 100%; height: auto; border-radius: 16px;">
</div>

Due to needing the results to be stable for this dimensionality this run took significantly longer than the first. The normalising flow for the first example took ~30 seconds to train, while this one took ~25 minutes (although the large time for the second is likely due to my relative inexperience with flows). But hey, that's still relatively quick for the $$10^4$$ datapoints that this uses and now we have a parametric representation of our posterior that we can sample really easily and quickly.


## Some Annoying Things About Normalising Flows


Despite what the benefits that I've tried to get across with all of the above, there are still quite a few aspects of normalising flows that are finicky enough that, personally, would mean that I wouldn't recommend using normalising flows to the general statistician that I will highlight (mostly again) here.

1. Optimisation Instability. Despite the "speed" benefits of optimisation, due to the large number of parameters being investigated among other things, gradients are prone to explode and produce major instabilities during fitting. I've referenced a couple approaches to get around some of these problems below but I've found that at least one of them still leaves things being finicky (and the other ones I haven't implemented yet...), and now with the stability measures, not that fast.
2. Convergence. I've told you that I typically determine whether the flow has converged by looking for when the log-loss curve plateus, but we don't have any "nice" convergence measures like one does with auto-correlation in MCMC. My approach has been that if the setup gives me the same results through normalising flows and other more stable methods such as nested sampling or some flavour of MCMC, then I mostly think it's stable. But once I go to enough dimensions (high-dimensionality being the only reason I even am trying to use normalising flows in the first place) then I can't do this stability testing anymore because the other methods are simply too slow. So I have to resort to doing "pp" tests where I use synethetic data and see if the true values fall within $$1\sigma$$ their respective marginals $$68\%$$ of the time and falls within $$2\sigma$$ $$95\%$$ of the time and so on.
3. Is that the posterior, or some weird artefact from the transformations? I will quite often get a posterior out from a normalising flow with a variable with a strange skew or extension in the posterior, but it is hard to tell whether this comes from the posterior actually having that shape and the flow just allows the freedom to sample enough to see this properly or whether it's some artefact of the transformations that the normalising flow has found and left in there.
4. It's just plain ol' finicky. In my experience, using normalising flows has been a kind of mix of optimisation issues, general machine-learning bugs, alongside a bloody sampler. I think I solve one thing or then another problem pops up (seemingly purely out of mechanical spite). But even worse, is that I fix something that is obviously wrong, and that makes the flow _stop_ working. For the above examples it would have been much much _much_ quicker to use standard methods like HMC or nested sampling rather than flows (possibly out of experience, but the flows really seemed like an uphill battle).

Additionally, I have only read this as I haven't had the need to look at discrete distributions in my posterior (and can't be bothered making synthetic data for yet). According to [Kobyzev, 2020](https://arxiv.org/abs/1908.09257v4), discrete distribution were still an open problem, I'm not sure if this is still the case or not.

So, I would recommend giving normalising flows a try, out of interest and maybe you have a better touch for it than I do, but I would lean on the side of caution if one were unfamiliar with normalising flows and trying to implement in work where uncertainties have to be well understood (and if not, just use a optimiser!). In the cases above I probably pretty comfortably could have used the [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) to get a gaussian representation of my posterior as well.



---