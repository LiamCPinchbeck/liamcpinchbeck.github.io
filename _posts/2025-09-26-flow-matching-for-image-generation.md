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



For a visualisation of this you can look at the below figure.


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


Based on this you can imagine there an underlying vector field which would transport all samples to the given point as shown below.

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

This doesn't take into account the samples from the base distribution however, so if we want to investigate this directly we would image the path that all the samples would have to take to go from the base distribution to the give sample.


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

If we then image that each timestep has a given probability if we imagine the first distribution to be known, let's assume it's a gaussian for now, then we can create what is called a _probability path_... 

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


And here's one I prepared earlier.

<div style="text-align: center; margin-top: 16px">
  <img 
      src="/files/BlogPostData/2025-09-fmfig/simple_mapping_GIFs/simple_mapping.gif" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 0px;">
</div>






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





