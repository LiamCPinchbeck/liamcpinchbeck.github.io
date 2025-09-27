---
title: 'Flow matching for multi-modal density estimation and image generation'
date: 2025-09-23
permalink: /posts/2025/09/2025-09-23-FM-ImageGen/
tags:
  - PyTorch
  - Variational Inference
  - Normalising Flows
  - Flow Matching
header-includes:
   - \usepackage{amsmath}
---


In this post, I’ll give an practical introduction to flow matching for the sake of estimating complicated sample distributions and image generation.


## Resources

As usual, here are some of the resources I’m using as references for this post. Feel free to explore them directly if you want more information or if my explanations don’t quite click for you.


## Table of Contents

- [Motivation](#motivationtraditional-autoencoders)
- [Core Idea](#core-idea)
- [Checkerboard density dimensional and modal scaling behaviour](#checkerboard-density-dimensional-and-modal-scaling-behaviour)
- [Generating MNIST-like images](#generating-mnist-like-images)
- [Conclusion](#conclusion)

---


# Motivation


In [a previous post](https://liamcpinchbeck.github.io/posts/2025/08/2025-09-08-VAEs/) I explained the use of Variational Autoencoders and how there probabilistic nature allowed us to sample "new" images from the MNIST dataset. However, this came with a few caveats:

1. We were not able to easily enforce a specific structure on the learnt latent space. The latent space or the values represented in the bottleneck were learnt as part of the training.
2. Similar to the previous point, say we were in a true variational inference context. We may want a specific likelihood and prior for our latent space that was informed from physical parameters. This would not be possible with a variational autoencoder without modifications that wouldn't make it a standard variational autoencoder anymore.
3. The learnt distributions were fixed (gaussian) and failed to capture some details that maybe a more complex distribution would be able to capture. But we couldn't specify this as it seemed like there was no way to form a good distribution without knowing what it was beforehand.

Many of the capabilities of the above can be handled by a relatively recent ([2022](https://arxiv.org/abs/2210.02747)) machine learning architecture/density estimation approach called Flow Matching. Initially introduced by [Lipman et al.](https://arxiv.org/abs/2210.02747)[^FML] as an evolution of [continuous normalising flows](https://liamcpinchbeck.github.io/posts/2025/08/2025-08-07-ContNF/), it retains much of the expressibility of continuous flows with much more stable training and no need to explicitly solve ODEs. 

The training and setup is now so simple I personally wasted a lot of time trying to understand flow matching "better", going into Riemannian space optimal transport for example, just to learn that I had the right idea all along. I am thus only going to introduce the final result here, and not much of the underpinning theory (leaving that for the post on SBI where some extra detail _is_ needed) as I think knowing it will only initially get in the way of developing an intuition.

[^FML]: And recently there was a ___fantastic___ paper released by Meta (Facebook) that goes into much more detail than I will here while also starting from a lower bar of entry. HIGHLY HIGHLY HIGHLY recommend giving it a look [https://arxiv.org/abs/2412.06264](https://arxiv.org/abs/2412.06264).



# Core Idea

Flow matching is inherently a simulation-based approach that requires samples from the target distribution. The first step in developing a flow representation of this target is to investigate the _conditional_ paths of the samples. Where all the samples from the base distribution flow into a single sample in the target. Mathematically, if we assume that our base distribution is a normal distribution with mean $$\mu_0$$ and covariance $$\Sigma_0$$, we can describe the probability of a given point during the transform at time $$t$$, $$x_t$$, for a given point in the target distribution $$x_1$$ as,

$$\begin{align}
p_t(x_t | x_1) = \mathcal{N}(x_t | \mu_0 + t \cdot (x_1 - \mu_0), (1-t)^2 \cdot \Sigma_0).
\end{align}$$

Or more simply we can imagine transforming a given point in the base distribution $$x_0$$,


$$\begin{align}
x_t = x_0 + t \cdot (x_1 - x_0).
\end{align}$$


For a visualisation of this you can look at the below GIFs with the given target sample highlighted in red.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_gifs/points_animation_0.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_gifs/points_animation_100.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_gifs/points_animation_300.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_gifs/points_animation_400.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
</div>


The probability path satisfied the conditions,

$$\begin{align}
p_t(x_t | x_1) =\begin{cases}
			\mathcal{N}(x_t | \mu_0, \Sigma_0), & \text{if }t\text{ = 0} \\
            \delta(x_t - x_1), & \text{if }t\text{ = 1}
		 \end{cases}
\end{align}$$

The underlying vector field $$u_t$$ that is driving this is then just,

$$\begin{align}
u_t(x_t ; t, x_1) = \frac{x_1 - x_t}{1-t}
\end{align}$$

This just means that all the points are following straight lines are more simply given via the transform equation above.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_dynamic_vec_field_gifs/points_animation_0.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_dynamic_vec_field_gifs/points_animation_100.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
</div>
 
<div style="text-align: center; margin-top: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_dynamic_vec_field_gifs/points_animation_300.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_dynamic_vec_field_gifs/points_animation_400.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
</div>

If we directly look at the vector field and normalise the lengths so we can look at the directions, you can see that everywhere is just pointing towards the target distribution sample.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_vec_field_gifs/points_animation_0.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_vec_field_gifs/points_animation_100.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
</div>
 
<div style="text-align: center; margin-top: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_vec_field_gifs/points_animation_300.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/point_convergence_w_vec_field_gifs/points_animation_400.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 0px;">
</div>


This doesn't take into account the samples from the base distribution? The vector field we want is of course $$u_t(x_t)$$ not conditioned with respect to a specific target sample. We can take out this dependence by marginalising it out with respect to the probability path we defined above,

$$\begin{align}
u_t(x_t ; t) &= \int dx_1 u_t(x_t ; t, x_1) p(x_1 | x_t) \\
&= \int dx_1 u_t(x_t ; t, x_1) \frac{p(x_t | x_1)p(x_1)}{p(x_t)} \\
\end{align}$$

Which we can estimate with a couple rounds of monte carlo integration,

$$\begin{align}
u_t(x_t ; t) &= \int dx_1 u_t(x_t ; t, x_1) \frac{p(x_t | x_1)p(x_1)}{p(x_t)} \\
&\approx \frac{1}{N_i} \sum_s^{N_i} u_t(x_t ; t, x_1) \frac{p(x_t | x_1^i)}{p(x_t)}, \\
\end{align}$$

and with the same set of samples from the target distribution,

$$\begin{align}
p(x_t) &= \int dx_1 p(x_t | x_1)p(x_1) \\
&\approx \frac{1}{N_i} \sum_i^{N_i} p(x_t | x_1^i).
\end{align}$$

This gives us the following estimates for the vector field that transforms our base distribution to our target distribution.


<div style="text-align: center; margin-top: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field_no_samples.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>

<div style="text-align: center; margin-top: 8px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field_no_field.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>

<div style="text-align: center; margin-top: 8px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px; margin-bottom: 16px">
</div>


And we can look at how the vector field is directly acting on the points themselves.

<div style="text-align: center; margin-top: 8px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field_follow.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px; margin-bottom: 16px">
</div>

But mathematically the points on the left aren't even in the same space as the right. Although they look that way because of the way that I've put them in the gifs. What we're actually doing under the hood is transforming the space itself. Kind of like we're interested in how a surfer is riding a wave (the samples), that were originally standing on a surfboard (space being transformed), and the wave (vector field) is pushing the board (space the samples inhabit) not exactly the surfer (samples)[^surfer].

[^surfer]: You can tell that I'm a surf dude...(sarcasm)

So we can also look at how the samples actual follow how the space deforming not the other way around as more correctly shown below.

<div style="text-align: center; margin-top: 8px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field_grid.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px; margin-bottom: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field_grid_fine.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px; margin-bottom: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/real_dynamic_vector_field_grid_with_samples.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px; margin-bottom: 16px">
</div>

But this would not be feasible for large dimensions or really pathologically shaped distributions. So instead, we try to represent the vector field with a neural network. And boom, that's flow matching.

And here's one I prepared earlier.

<div style="text-align: center; margin-top: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/simple_mapping.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>

But if we want to avoid the monte carlo estimation, then how do we tell the network how to improve, i.e. what should we make the loss?




# Checkerboard density: Dimensionality and modal scaling behaviour



An example of what this looks like is then below.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/checkerboard/checkerboard_8x8.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/checkerboard/checkerboard_5x5.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
</div>






# Generating MNIST-like images






# Conclusion





