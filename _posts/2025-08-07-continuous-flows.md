---
title: 'An introduction to continuous normalising flows'
date: 2025-08-07
permalink: /posts/2025/08/2025-08-07-ContNF/
tags:
  - Pyro
  - PyTorch
  - Normalising Flows
  - Variational Inference
header-includes:
   - \usepackage{amsmath}
---

In this post I will attempt to give an introduction to _continuous normalising flows_, an evolution of normalising flows that translate the idea of training a discrete set of transformations to approximate a posterior, into training an [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) or vector field to do the same thing. 

---


___UNDER CONSTRUCTION, DO NOT TRUST OR LET WHAT IS WRITTEN REFLECT ME AS A PERSON/RESEARCHER PLEASE___

---

## Table of Contents

- [Motivation](#variational-inference)
- [Core Idea](#core-idea)
- [The Math/How to solve an IDE with the adjoint method](#the-adjoint-method-and-derivatives)
- [Further Reading](#further-reading)
- [Appendices](#appendices)

---


# Motivation

Normalising flows as discussed in my other posts ([intro]() and [making one from scratch]()) tout how wonderful normalising flows are: they have the ability to not only efficiently explore high dimensional distributions and sample them all while generating a functional representation for them. However, if you want to model a really complex distribution with many modes and non-gaussian behaviour you may want a more complex transformation behind your normalising flow, but the calculation of the jacobian can be quite arduous because of the need to calculate the determinant of the transformations' jacobian. Or maybe you want to nicer way to train a conditional normalising flow[^1] which is notoriously hard for traditional normalising flows?

[^1]: a type of flow that can produce $$p(y\vert x)$$. Modelling both $$y$$ and $$x$$

One possible answer to that is to use [Flow Matching](https://arxiv.org/abs/2210.02747) a kind of algorithm between normalising flows and [diffusion models](https://arxiv.org/abs/2404.07771). The basic idea of flow matching is to model the vector field describing how samples from simple base distribution into some more complicated distribution. The basic idea is shown in the GIF below which I stole from ["An introduction to Flow Matching"](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) by [Tor Fjelde](https://retiredparkingguard.com/about.html), [Emile Mathieu](https://mlg.eng.cam.ac.uk/blog/2024/01/20/www.emilemathieu.fr), and [Vincent Dutordoir](https://vdutor.github.io/).


<div style="text-align: center;">
<img 
    src="https://mlg.eng.cam.ac.uk/blog/assets/images/flow-matching/representative.gif" 
    alt="https://mlg.eng.cam.ac.uk/blog/assets/images/flow-matching/representative.gif" 
    title="A GIF showing the evolution of a gaussian base distribution to a bi-modal distribution exemplifying the generation of a vector field that describes this transformation" 
    style="width: 100%; height: auto; border-radius: 16px;">
</div>

<br>

But this post isn't titled an introduction to flow matching, it's on continuous normalising flows. Continous flows can be thought of as a kind of precursor or fundamental component (depending on how you look at it) to Flow Matching. But, they are also very cool in their own right and make some cool looking gifs[^2].

[^2]: the real purpose of any scientific endeavour being the generation of cool gifs of course (joke)


# Core Idea

Following the same logical flow as the original continous normalising flows paper, annoyingly called ["Neural Ordinary Differential Equations"](https://arxiv.org/abs/1806.07366)[^3], we begin by describing [residual normalising flows](https://arxiv.org/abs/1906.02735) or semi-equivalently [residual neural networks](https://en.wikipedia.org/wiki/Residual_neural_network).

[^3]: Not actually annoying because the key concept was to replace traditional neural networks with these kinds of systems.

Residual neural networks are neural networks were sequential layers trying to model additive changes to the output of the previous layer rather than strictly just taking the previous layer as an input and pumping out a new output. e.g.

$$\begin{align}
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h_t}, \mathbf{\theta}_t),
\end{align}$$

where $$t$$ is some discrete value that indicates the depth of the network. Additionally you could view normalising flows such as RealNVP with just additive transformations (e.g. unity scaling) as some form of this,

$$\begin{align}z_{t+1}^i = z_t^i + f(\mathbf{z_t}^{1:d}, \mathbf{\theta}_t),\end{align}$$

for $$i>d$$. The amazing thing that [Chen et al. 2019](https://arxiv.org/pdf/1806.07366) noticed is that this is extremely similar to an Euler discretised solution to some continuous transformation. Taking the number of the steps to infinity/take smaller steps we can image that in the limit we recover some quasi-[ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) of the form,

$$\begin{align}
\frac{d\mathbf{z}(t)}{dt} = f(\mathbf{z}(t), t, \mathbf{\theta}).
\end{align}$$

Applying this directly to our flow, we can image that time $$t=0$$ is the distribution of samples or probability under our base distribution (e.g. normal) and our final time, which we'll just denote as $$t_f$$, as the distribution of samples/probability under our target distribution. Once we have it in this form we can then just chuck our favourite black box [ODE solver](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) at.  

There are many fantastic things about this, the three ones that are of interest to me are:
1. __Memory efficiency__. Unlike a traditional normalising flow, increasing the "complexity" of our transformations does not increase our memory cost, just possible training time. i.e. we get a constant memory cost (not something you get from deep neural networks)
2. __Adaptive computation__. Using an ODE solver to solve our transformation allows adaptive computation of the accuracy and tolerance of our solution with many modern ODE solvers being able to adaptively adjust step-size to manage error.
3. __Normalising flow scalability__. On top of the constant memory cost, reparameterising our problem into this form means that our jacobians/change of variables is easier/quicker to compute and the forward and reverse directions of evaluating our flow become roughly equal in cost unlike methods such as autoregressive models that have a particular direction with faster computation while still having great flexibility.


# The Adjoint Method and Derivatives

The key difficulty that [Chen et al. 2019](https://arxiv.org/pdf/1806.07366) highlight in this method is backpropagating our derivatives through this kind of system.


# Practical Examples


## Example 1: Comparisons with other methods


## Example 2: Comparisons with normalising flows


## Example 3: 


# Further Reading


# Appendices