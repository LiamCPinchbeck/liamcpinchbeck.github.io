---
title: 'Image Classification and Molecular Property Prediction with Fixed and Mixed Curvature VAEs'
date: 2025-11-27
permalink: /posts/2025/08/2025-11-27-mixed-curvature-VAEs/
tags:
  - PyTorch
  - Variational Inference
  - VAEs
  - Variational Autoencoders
  - Manifold Learning
  - Deep Learning
  - Geometric deep learning
header-includes:
   - \usepackage{amsmath}
---


In this post, I’ll go through Fixed Curvature VAEs and Mixed Curvature VAEs for data image classification and molecular property prediction.


---



## Table of Contents

- [Variational Autoencoder Recap: Coolness and Limitations](#variational-autoencoder-recap-coolness-and-limitations)
- [A sneak peek at Hyperspherical VAEs](#a-sneak-peek-at-hyperspherical-vaes)
- [Differential Geometry Primer](#differential-geometry-primer)
    - [Curvature, Hyperspheres and Hyperboloids](#curvature-hyperspheres-and-hyperboloids)
    - [Charts, Geodesics and Transport Maps](#charts-geodesics-and-transport-maps)
- [Putting the Geometry in the Latent Space](#putting-the-geometry-in-the-latent-space)
    - [Hyperspherical VAEs](#hyperspherical-vaes)
    - [Hyperbolic VAEs](#hyperbolic-vaes)
- [Mixed Curvature VAEs](#mixed-curvature-vaes)
- [Image Classification and Generation with MNIST and CelebA](#image-classification-and-generation-with-mnist-and-celeba)
- [Molecular Property Prediction with QM9](#molecular-property-prediction-with-qm9)
- [Conclusions](#conclusions)

## Prerequisites 

- Strong foundation of multivariable calc, linear algebra, and nerdiness



## General Resources


- [Directional Statistics](https://en.wikipedia.org/wiki/Directional_statistics)
- [Data Compression and generation with Variational Autoencoders](https://liamcpinchbeck.github.io/posts/2025/08/2025-09-08-VAEs/)
- [Hyperspherical VAE - Nicola De Cao Blog Post](https://nicola-decao.github.io/s-vae.html)

Papers

- [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Hyperspherical Variational Auto-Encoders - Davidson et al. 2022](https://arxiv.org/abs/1804.00891)
- [Geometric deep learning: going beyond Euclidean data - Bronstein et al. 2017](https://arxiv.org/abs/1611.08097)
- [Directional Statistics-based Deep Metric Learning for Image Classification and Retrieval](https://arxiv.org/abs/1802.09662)
- [Mixed-Curvature Variational Autoencoders - Skopek et al. 2020](https://arxiv.org/abs/1911.08411)
- [Quantum chemistry structures and properties of 134 kilo molecules - QM9 dataset](https://springernature.figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904/5)
- [ToensorFlow - QM9 Dataset](https://www.tensorflow.org/datasets/catalog/qm9)


Also going to try something different in this post and actually link some sources for some broader statements in a reference section[^pseudoref], which will basically just be the authors, titles, and link to the given source. I'll continue just having a basic link for definitional and to-the-point stuff.

[^pseudoref]: Although it won't be stylised and likely wouldn't pass some basic standards for references but I don't wanna do that by hand and markdown doesn't exactly have a bibtex extension.

# Variational Autoencoder Recap: Coolness and Limitations

(If you're already very familiar with VAEs you can skip this section and just move to the [hyperspherical example](#a-sneak-peek-at-hyperspherical-vaes).)

The key property of variational autoencoders [[1](#ref-vae), [2](#ref-vae-tut)] is that they reduce dimensionality like standard autoencoders but do so in a more regularised manner. Learning a conditional gaussian distribution in the latent space/bottleneck layer rather than single point estimates which lead to enhanced performance and more regularised latent spaces among many other benefits [[3](#ref-vae-tut-doersch)].

One easy and common way to do this is with the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) that contains roughly 70,000 handwritten 28x28 pixelated digits in black and white. Some examples I put below.


<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/MNIST_Images.png" 
      alt="MNIST Digits." 
      title="MNIST Digits." 
      style="width: 89%; height: auto; border-radius: 8px;">
</div>

A VAE (or standard autoencoder) is then constructed to take in this data, learn a conditional gaussian on a latent dimensional space [^compress] (or some point estimate in the lower dimensional space) and then uses this lower dimensional representation to reproduce the data. This process for autoencoders.

[^compress]: The latent dimensional space of the VAE is presumed to have a lower dimension than the inputs. For the MNIST data (and images in general) it is presumed to be 1 dim per pixel or 784 total. Therefore the VAE compresses the data.


First the ___Autoencoder___ takes in the data, $$\vec{x}_i$$, uses an __encoder__ to transform this into some __coordinate__ in a lower dimensional space, $$\vec{z}_i$$, and then the __decoder__ uses $$\vec{z}_i$$ to try and reproduce $$\vec{x}_i$$ denoted here with $$\vec{y}_i$$.

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/autoencoder_diagram.png" 
      alt="Diagram showing the general structure of an autoencoder." 
      title="Diagram showing the general structure of an autoencoder." 
      style="width: 89%; height: auto; border-radius: 8px;">
</div>

<br>

The ___Variational Autoencoder___ takes in the data, $$\vec{x}_i$$, uses an __encoder__ to transform it into a mean and standard deviation describing a normal __distribution__ over $$\vec{z}$$, and then we sample a given $$\vec{z}_i$$ (utilising the [reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick) ) to try and produce a normal __distribution__ over $$\vec{x}$$ with the __decoder__, with a given sample denoted here as $$\vec{y}_i$$ (and it is almost always assumed that $$\vec{y}_i = \vec{\mu}_i$$, the mean of the data space distribution).


<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_diagram.png" 
      alt="Diagram showing the general structure of a variational autoencoder." 
      title="Diagram showing the general structure of a variational autoencoder." 
      style="width: 99%; height: auto; border-radius: 8px;">
</div>
<br>

(If you want more details on VAEs I'd recommend any of the sources I linked above or [my own post that goes through it in much more detail](https://liamcpinchbeck.github.io/posts/2025/08/2025-09-08-VAEs/).)

Using the code from my previous post on VAEs we can put the AE and VAE constructions side-by-side with the real data and also look at where the data is mapping to in the 2D latent space I constructed.

First we'll have a look at the standard autoencoder reconstructions and latent space.


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/2_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/7_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/4_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/6_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
<figcaption>MNIST data (left digits) and Standard Autoencoder MNIST digit reconstructions (right digits) using `afmhot` maptlotlib colour map to make it easier to visualise/distinguish features.</figcaption>
</div>
<br>

First thing we can notice is that it completely messed up the 4, and the digits are typically more blurry than their true counterparts (although this is typical in shallow neural network image construction across different architectures). Then we look at the latent space.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/combined_latent_dim.png" 
      style="width: 89%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/separated_latent_dim.png" 
      style="width: 99%; height: auto; border-radius: 0px;">
<figcaption>Standard Autoencoder MNIST 2D latent dimension combined for all digits (top) and for each digit (bottom)</figcaption>
</div>
<br>

In a word: messy. There's some slightly strange behaviours induced because I enforced the samples to fall between 0 and 1 for plot-ability, but the distribution of 4, 7, and 9 are overlapping and separated. Many of the individual distributions also have weird parts that are spread out across the space for some reason. that doesn't really indicate that the VAE is considering them part of the same label set as we would expect.

If we uniformly grid this space and see what the decoder tries to reconstruct we can also kind of see how the VAE is interpretting the space.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/AE_MNIST_Reconstructions/transformed_latent_dim.png" 
      style="width: 100%; height: auto; border-radius: 0px;">
<figcaption>Uniformly gridded coordinates in AE latent space transformed into the data space by the decoder</figcaption>
</div>
<br>

We can see that some areas are very clearly defined and others not so much. And even more annoyingly, if we tried to interpolate between some pretty reasonable numbers (that look similar) like 8 and 9 then we can cross through regions that are completely uninterpretable and carry zero meaning (lower left corrner). And the transitions between numbers that the VAE decides are similar also contain weird artefacts (like the transition between 4 and 6, or the region between 2, 5 and 8).

Now let's compare the above with VAEs.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/2_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/7_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/4_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/6_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
<figcaption>MNIST data (left digits) and Standard VAE MNIST digit reconstructions (right digits) using `afmhot` maptlotlib colour map to make it easier to visualise/distinguish features.</figcaption>
</div>
<br>

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/combined_latent_dim.png" 
      style="width: 89%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/separated_latent_dim.png" 
      style="width: 99%; height: auto; border-radius: 0px;">
<figcaption>Standard VAE MNIST 2D latent dimension combined for all digits (top) and for each digit (bottom)</figcaption>
</div>
<br>
<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/VAE_MNIST_Reconstructions/transformed_latent_dim.png" 
      style="width: 100%; height: auto; border-radius: 0px;">
<figcaption>Uniformly gridded coordinates in VAE latent space transformed into the data space by the decoder</figcaption>
</div>
<br>

Overall, a little better. The 4 doesn't transform to a 4 this time but to an 8, but at least it's a number. And the groups in the latent space seem to have more intelligible structure to them. And the transformation of the latent space still has a region with some unintelligible digits (the very centre) but the transitions and distributions altogether look better.

The indistinguishability in the centre is actually pretty typical in VAEs as the prior on the distributions is centred at 0 so the groups are pulled in this direction. If you wanted to get rid of this you would need to use some uncentred/uninformative prior, but that only really gives us the uniform distribution, which won't regularise our space. This leads us to try and think of geometries where this effect is mitigated or even removed.

<br>

# A sneak peek at Hyperspherical VAEs

In this section I want to show a little of why we are interested in mixed curavture VAEs by looking at the above MNIST example with a hyperspherical (example of _fixed_ curvature) VAE. 

The basic idea is that the normal distribution priors used in the standard VAE above have a tendency to pull representations towards 0. This is true for all the clusters of datapoints and thus inherently they are pulled together and often overlap quite significantly in this region. 

And although we want a smooth representation, we also would like a clear one. As we move within the latent space there are obvious regions that are a '3', an '8' and a '1' with some boundary that nicely transitions between them.

So if the issue is that they all pulled towards some central region for regularisation through the normal prior centred at 0, what if we eliminate the 'centre' but retain the probabilistic interpretation so we get the best of both worlds, and define the distribution on a sphere. The samples live on this sphere, and so in this space there is no 'centre'[^centre]. This idea motivates the concept of [_hyperspherical VAEs_](https://arxiv.org/pdf/1804.00891).

[^centre]: You might think that the 'centre' for a unit sphere would just be the origin, but that point doesn't lie on the sphere and so doesn't exist within the spherical space.

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/davidson_hyperspherical_VAE_MNIST_example.png" 
      alt="Comparison of VAE on euclidean vs spherical space from Davidson et al. 2022." 
      title="Comparison of VAE on euclidean vs spherical space from Davidson et al. 2022." 
      style="width: 99%; height: auto; border-radius: 8px;">
<figcaption>Latent space visualization of the 10 MNIST digits (each digit has it's own colour) in 2 dimensions of both N -VAE (left, the standard VAE) and S-VAE (right, VAE defined on the sphere) taken from Fig. 2 in Davidson et al. 2022 (https://arxiv.org/pdf/1804.00891). (Best viewed in color) </figcaption>
</div>
<br>

This looks much better! (separation wise)

So we can try to learn a conditional distribution on the latent parameters on the sphere which mitigates the pull of the 'centre' but still regularise the space using distributions defined on the sphere. The next few sections will be dedicated to exploring this idea. 


<br>

# Differential Geometry Primer

(For those already familiar with charts, atlases, Riemannian manifolds you can skip to the [Putting the Geometry in the Latent Space](#putting-the-geometry-in-the-latent-space) section)


So, because we want to learn a distribution on spheres or other surfaces in higher dimensions, we are heading into the area of [_differential geometry_](https://en.wikipedia.org/wiki/Differential_geometry), specifically [_Riemannian Geometry_](https://en.wikipedia.org/wiki/Riemannian_geometry). This sounds fancy, and tbh, sometimes is, but is just the math of smooth (in the literal sense) geometry in higher dimensions.


<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/morning_star_image.png" 
      alt="Morgenstern (middle) - illustration from a book 'Handbuch der Waffenkunde' Das Waffenwesen in seiner historischen Entwicklung vom Beginn des Mittelalters bis zum Ende des 18 Jahrhunderts by Wendelin Boeheim, Leipzig, 1890" 
      title="Morgenstern (middle) - illustration from a book 'Handbuch der Waffenkunde' Das Waffenwesen in seiner historischen Entwicklung vom Beginn des Mittelalters bis zum Ende des 18 Jahrhunderts by Wendelin Boeheim, Leipzig, 1890." 
      style="width: 69%; height: auto; border-radius: 8px;">
<figcaption>Examples of surfaces that are not Riemannian manifolds/not smooth surfaces, for which we don't want to model our distribution on.</figcaption>
</div>
<br>


It might seem like a bit of an abstraction, and cards on the table oh boi it is, but you likely already think somewhat in terms of surfaces and curves already. e.g.

- You're driving on the road, technically the houses on the side of the street are on the surface you reside on, but while you're driving, they _shouldn't_ exist. 
- You exist on a sphere (although idk, if I have any readers that are astronauts hit me up). You technically reside in 3D space however, if a person is flying a plane a very long distance (say Melbourne to Sapporo) you won't tell them the downwards coordinate to go straight through the Earth! (we are thinking in terms of _spherical_ spaces here)
- When playing a video game, e.g. Super Mario, that world to the player exists as a 2D space, although to Mario and co. I'm sure they would think they're going in a straight line.
- You can imagine navigating social media as some weird object. Everyone is friends or follows Obama, so you can think that Obama is close to everyone. But just because two people follow Obama, doesn't mean that they are friends or follow each other at all (this is a behaviour that is common in _hyperbolic_ spaces)
- I'm sure you get the point by now.

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-mixed-curvature-vaes/Sapporo_Melbourne_circle_example.png" 
      alt="Diagram showing different paths between Melbourne and Sapporo" 
      title="Diagram showing different paths between Melbourne and Sapporo" 
      style="width: 79%; height: auto; border-radius: 8px;">
<figcaption>Diagram showing different paths between Melbourne and Sapporo one of them is a little harder in practice.</figcaption>
</div>
<br>

So we already have concepts of what we call _non-euclidean spaces_, what we need to do is translate these ideas into more rigorous mathematical concepts.



<br>

## Charts and Atlases


<br>

## Tagent Spaces


<br>

## The Riemannian metric and manifolds


<br>

## Curvature, Hyperspheres and Hyperboloids





<br>

# Putting the Geometry in the Latent Space


<br>

## Hyperspherical VAEs



<br>

## Hyperbolic VAEs



<br>

# Mixed Curvature VAEs



<br>

# Image Classification and Generation with MNIST and CelebA



<br>

# Molecular Property Prediction with QM9




<br>

# Conclusions






<br>

## References
<a id="ref-vae"></a>
[1] Diederik P. Kingma, Max Welling. (2013). 
_Auto-Encoding Variational Bayes_. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

<a id="ref-vae-tut"></a>
[2] Diederik P. Kingma, Max Welling. (2019). 
_An Introduction to Variational Autoencoders_. [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)

<a id="ref-vae-tut-doersch"></a>
[3] Carl Doersch. (2016). 
_Tutorial on Variational Autoencoders_. [arXiv:1606.05908](https://arxiv.org/abs/1606.05908)

<a id="ref-hyperspherical-vae-github"></a>
[4] Davidson, T. R., Falorsi, L., De Cao, N., Kipf, T.,
and Tomczak, J. M. (2018). Hyperspherical Variational
Auto-Encoders. 34th Conference on Uncertainty in Artificial Intelligence (UAI-18). [[GitHub Link](https://github.com/nicola-decao/s-vae-pytorch?tab=readme-ov-file)]


<br>

---

## Footnotes