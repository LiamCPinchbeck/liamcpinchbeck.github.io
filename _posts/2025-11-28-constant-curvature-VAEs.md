---
title: 'Image Classification and Molecular Property Prediction with Constant Curvature VAEs'
date: 2025-11-27
permalink: /posts/2025/08/2025-11-27-constant-curvature-VAEs/
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
  - \usepackage{algpseudocode}

---


In this post, I’ll go through Constant Curvature VAEs (traditional, hyperspherical, and hyperbolic) for image data classification and molecular property prediction.


---



## Table of Contents

- [Variational Autoencoder Recap: Coolness and Limitations](#variational-autoencoder-recap-coolness-and-limitations)
- [A sneak peek at Hyperspherical VAEs](#a-sneak-peek-at-hyperspherical-vaes)
- [Hyperspherical VAE](#hyperspherical-vae)
    - [von Mises-Fisher distribution](#von-mises-fisher-distribution)
    - [Back to the von Mises-Fisher distribution](#back-to-the-von-mises-fisher-distribution)
    - [Reparameterisation trick with the von Mises-Fisher distribution](#reparameterisation-trick-with-the-von-mises-fisher-distribution)
- [Hyperbolic VAE](#hyperbolic-vae)
- [Image Classification and Generation with MNIST and CelebA](#image-classification-and-generation-with-mnist-and-celeba)
- [Molecular Property Prediction with QM9](#molecular-property-prediction-with-qm9)
- [Conclusions](#conclusions)

## Prerequisites 

- Strong foundation of multivariable calc, linear algebra, and nerdiness

<!-- https://github.com/Gazler/githug -->

## General Resources


- [Directional Statistics](https://en.wikipedia.org/wiki/Directional_statistics)
- [Data Compression and generation with Variational Autoencoders](https://liamcpinchbeck.github.io/posts/2025/08/2025-09-08-VAEs/)
- [Hyperspherical VAE - Nicola De Cao Blog Post](https://nicola-decao.github.io/s-vae.html)
- [GitHub Repo for Hyperspherical VAE implementation](https://github.com/nicola-decao/s-vae-pytorch)

Papers

- [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

- [Hyperspherical Variational Auto-Encoders - Davidson et al. 2022](https://arxiv.org/abs/1804.00891)
    - Main reference for hyperspherical VAEs
- [Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders](https://arxiv.org/pdf/1901.06033)
    - Main reference for hyperbolic VAEs


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
      src="/files/BlogPostData/2025-constant-curvature-vaes/MNIST_Images.png" 
      alt="MNIST Digits." 
      title="MNIST Digits." 
      style="width: 89%; height: auto; border-radius: 8px;">
</div>

A VAE (or standard autoencoder) is then constructed to take in this data, learn a conditional gaussian on a latent dimensional space [^compress] (or some point estimate in the lower dimensional space) and then uses this lower dimensional representation to reproduce the data. This process for autoencoders.

[^compress]: The latent dimensional space of the VAE is presumed to have a lower dimension than the inputs. For the MNIST data (and images in general) it is presumed to be 1 dim per pixel or 784 total. Therefore the VAE compresses the data.


First the ___Autoencoder___ takes in the data, $$\vec{x}_i$$, uses an __encoder__ to transform this into some __coordinate__ in a lower dimensional space, $$\vec{z}_i$$, and then the __decoder__ uses $$\vec{z}_i$$ to try and reproduce $$\vec{x}_i$$ denoted here with $$\vec{y}_i$$.

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/autoencoder_diagram.png" 
      alt="Diagram showing the general structure of an autoencoder." 
      title="Diagram showing the general structure of an autoencoder." 
      style="width: 89%; height: auto; border-radius: 8px;">
</div>

<br>

The ___Variational Autoencoder___ takes in the data, $$\vec{x}_i$$, uses an __encoder__ to transform it into a mean and standard deviation describing a normal __distribution__ over $$\vec{z}$$, and then we sample a given $$\vec{z}_i$$ (utilising the [reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick) ) to try and produce a normal __distribution__ over $$\vec{x}$$ with the __decoder__, with a given sample denoted here as $$\vec{y}_i$$ (and it is almost always assumed that $$\vec{y}_i = \vec{\mu}_i$$, the mean of the data space distribution).


<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_diagram.png" 
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
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/2_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/7_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/4_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/6_ae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
<figcaption>MNIST data (left digits) and Standard Autoencoder MNIST digit reconstructions (right digits) using `afmhot` maptlotlib colour map to make it easier to visualise/distinguish features.</figcaption>
</div>
<br>

First thing we can notice is that it completely messed up the 4, and the digits are typically more blurry than their true counterparts (although this is typical in shallow neural network image construction across different architectures). Then we look at the latent space.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/combined_latent_dim.png" 
      style="width: 89%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/separated_latent_dim.png" 
      style="width: 99%; height: auto; border-radius: 0px;">
<figcaption>Standard Autoencoder MNIST 2D latent dimension combined for all digits (top) and for each digit (bottom)</figcaption>
</div>
<br>

In a word: messy. There's some slightly strange behaviours induced because I enforced the samples to fall between 0 and 1 for plot-ability, but the distribution of 4, 7, and 9 are overlapping and separated. Many of the individual distributions also have weird parts that are spread out across the space for some reason. that doesn't really indicate that the VAE is considering them part of the same label set as we would expect.

If we uniformly grid this space and see what the decoder tries to reconstruct we can also kind of see how the VAE is interpretting the space.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/AE_MNIST_Reconstructions/transformed_latent_dim.png" 
      style="width: 100%; height: auto; border-radius: 0px;">
<figcaption>Uniformly gridded coordinates in AE latent space transformed into the data space by the decoder</figcaption>
</div>
<br>

We can see that some areas are very clearly defined and others not so much. And even more annoyingly, if we tried to interpolate between some pretty reasonable numbers (that look similar) like 8 and 9 then we can cross through regions that are completely uninterpretable and carry zero meaning (lower left corrner). And the transitions between numbers that the VAE decides are similar also contain weird artefacts (like the transition between 4 and 6, or the region between 2, 5 and 8).

Now let's compare the above with VAEs.

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/2_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/7_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/4_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/6_vae_reconstruction.png" 
      style="width: 49%; height: auto; border-radius: 0px;">
<figcaption>MNIST data (left digits) and Standard VAE MNIST digit reconstructions (right digits) using `afmhot` maptlotlib colour map to make it easier to visualise/distinguish features.</figcaption>
</div>
<br>

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/combined_latent_dim.png" 
      style="width: 89%; height: auto; border-radius: 0px;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/separated_latent_dim.png" 
      style="width: 99%; height: auto; border-radius: 0px;">
<figcaption>Standard VAE MNIST 2D latent dimension combined for all digits (top) and for each digit (bottom)</figcaption>
</div>
<br>
<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/VAE_MNIST_Reconstructions/transformed_latent_dim.png" 
      style="width: 100%; height: auto; border-radius: 0px;">
<figcaption>Uniformly gridded coordinates in VAE latent space transformed into the data space by the decoder</figcaption>
</div>
<br>

Overall, a little better. The 4 doesn't transform to a 4 this time but to an 8, but at least it's a number. And the groups in the latent space seem to have more intelligible structure to them. And the transformation of the latent space still has a region with some unintelligible digits (the very centre) but the transitions and distributions altogether look better.

The indistinguishability in the centre is actually pretty typical in low latent dimensional VAEs as the prior on the distributions is centred at 0 so the groups are pulled in this direction. If you wanted to get rid of this you would need to use some uncentred/uninformative prior, but that only really gives us the uniform distribution, which won't regularise our space. 

And it turns out for high dimensional VAEs, samples from the normal distribution really start to look like uniform samples on the sphere (the so called 'soap bubble effect'). e.g. For the two plots below, which one do you think is a set of samples from a high dimensional normal distribution, and which is from a high dimensional sphere? (And I assure you there is in fact one of each here.)


<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/HighDNormal/500D_multivariatenormal_corner.png" 
      style="width: 49%; height: auto; border-radius: 0px;"
      alt="Are you trying to cheat?" 
      title="Are you trying to cheat?" >
  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/HighDNormal/500D_uniform_on_sphere_corner.png" 
      style="width: 49%; height: auto; border-radius: 0px;"
      alt="Are you trying to cheat again?!" 
      title="Are you trying to cheat again?!" >

<figcaption>Comparison of a sample distribution on a uniform sphere and normal distribution in 500-dimensional space (not necessarily in that order)</figcaption>
</div>

Not any easy game is it? 

This leads us to try and think of geometries where the centralising effect is mitigated or even removed, and maybe waste less time sampling a sphere indirectly and instead just sample the sphere itself! And this also motivates looking at other geometrical spaces and whether they can inherently better represent some data structures.

<br>

# A sneak peek at Hyperspherical VAEs

In this section I want to show a little of why we are interested in mixed curavture VAEs by looking at the above MNIST example with a hyperspherical (example of _constant_ curvature) VAE. 

The basic idea is that the normal distribution priors used in the standard VAE above have a tendency to pull representations towards 0. This is true for all the clusters of datapoints and thus inherently they are pulled together and often overlap quite significantly in this region. 

And although we want a smooth representation, we also would like a clear one. As we move within the latent space there are obvious regions that are a '3', an '8' and a '1' with some boundary that nicely transitions between them.

So if the issue is that they all pulled towards some central region for regularisation through the normal prior centred at 0, what if we eliminate the 'centre' but retain the probabilistic interpretation so we get the best of both worlds, and define the distribution on a sphere. The samples live on this sphere, and so in this space there is no 'centre'[^centre]. This idea motivates the concept of [_hyperspherical VAEs_](https://arxiv.org/pdf/1804.00891).

[^centre]: You might think that the 'centre' for a unit sphere would just be the origin, but that point doesn't lie on the sphere and so doesn't exist within the spherical space.

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/davidson_hyperspherical_VAE_MNIST_example.png" 
      alt="Comparison of VAE on euclidean vs spherical space from Davidson et al. 2022." 
      title="Comparison of VAE on euclidean vs spherical space from Davidson et al. 2022." 
      style="width: 99%; height: auto; border-radius: 8px;">
<figcaption>Latent space visualization of the 10 MNIST digits (each digit has it's own colour) in 2 dimensions of both N -VAE (left, the standard VAE) and S-VAE (right, VAE defined on the sphere) taken from Fig. 2 in Davidson et al. 2022 (https://arxiv.org/pdf/1804.00891). (Best viewed in color) </figcaption>
</div>
<br>

This looks much better! (separation wise)

So we can try to learn a conditional distribution on the latent parameters on the sphere which mitigates the pull of the 'centre' but still regularise the space using distributions defined on the sphere. The next few sections will be dedicated to exploring this idea. 


It might seem like a bit of an abstraction, and cards on the table oh boi it is, but you likely already think somewhat in terms of non-euclidean spaces already. e.g.

- To a very close approximation, you exist on a sphere (although idk, if I have any readers that are astronauts hit me up). You technically reside in 3D space however, if a person is flying a plane a very long distance (say Melbourne to Sapporo) you won't tell them the downwards coordinate to go straight through the Earth! (we are thinking in terms of _spherical_ spaces here)
- You can imagine navigating social media as some weird object. Everyone is friends or follows Obama, so you can think that Obama is close to everyone. But just because two people follow Obama, doesn't mean that they are friends or follow each other at all (this is a behaviour that is common in _hyperbolic_ spaces).

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-constant-curvature-vaes/Sapporo_Melbourne_circle_example.png" 
      alt="Diagram showing different paths between Melbourne and Sapporo" 
      title="Diagram showing different paths between Melbourne and Sapporo" 
      style="width: 79%; height: auto; border-radius: 8px;">
<figcaption>Diagram showing different paths between Melbourne and Sapporo one of them is a little harder in practice.</figcaption>
</div>
<br>

And it turns out that some data _is_ better represented in these spaces e.g. as shown in [Bronstein et al. 2017](https://arxiv.org/abs/1611.08097).

# Hyperspherical VAE

## von Mises-Fisher Distribution


The [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) is essentially the replacement distribution in the VAE instead of a traditional normal distribution. It is described as the normal distribution on a sphere and has the following PDF,

$$\begin{align}
f_{vMF}(\vec{x} | \vec{\mu}, \kappa) = \frac{\kappa^{p/2-1}}{(2\pi)^{p/2}I_{p/2-1}(\kappa)} \exp(\kappa \vec{\mu}^T\vec{x}).
\end{align}$$

This to me, makes no sense on the face of things. 


### von Mises Distribution

So let's first look at the simpler case where $$p=2$$, i.e. the circle, called the [von-Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution). It has the following PDF,

$$\begin{align}
f_{vM}(\vec{x} | \vec{\mu}, \kappa) = C(\kappa) \exp(\kappa  \vec{\mu}^T\vec{x}).
\end{align}$$

Where $$C(\kappa)$$ will just be a normalisation factor to us for now. In the case where $$\vec{\mu}$$ and $$\vec{x}$$ have unit length (which we will presume they do as they exist on the unit circle), then the dot product inside the exponential simplifies to the cosine of the angle between the two vectors, $$\theta_{x\mu}$$.

$$\begin{align}
f_{vM}(\vec{x} | \vec{\mu}, \kappa) = C(\kappa) \exp(\kappa  \cos(\theta_{x\mu})).
\end{align}$$

So $$ \cos(\theta_{x\mu})$$ will take a maximal value of 1 when the two vectors are parallel and minimal value of -1 when they are anti-parallel. Because $$\vert \cos(\theta_{x\mu})\vert \leq 1$$ then the exponent will always be less than or equal to 1 as well. So the absolute difference in angle is similar to how the normal distribution is only dependent on the absolute difference between two values, with $$\kappa$$ acting much like the variance in a traditional normal distribution. 

It also turns out that the normalisation factor has a bessel function in it, but for now we don't need to worry about that.

$$\begin{align}
C(\kappa) = \int_{-\pi}^{\pi} \exp(\kappa \cos(x)) dx = 2\pi I_0(\kappa),
\end{align}$$

where $$I_0(\kappa)$$ is the modified [Bessel function](https://en.wikipedia.org/wiki/Bessel_function#Bessel_functions_of_the_first_kind:_J%CE%B1) of the first kind of order 0.


Let's have a look at this distribution as a function of angle as well as on the actual circle.


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0_1_kappa_0.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0_1_kappa_0.1.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0_1_kappa_1.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0_1_kappa_10.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">



<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0.707_0.707_kappa_0.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0.707_0.707_kappa_0.1.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0.707_0.707_kappa_1.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_0.707_0.707_kappa_10.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">

<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_1_0_kappa_0.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_1_0_kappa_0.1.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_1_0_kappa_1.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vM_distribution_figures/mu_1_0_kappa_10.0.png" 
    style="width: 49%; height: auto; border-radius: 0px;">
</div>


## Back to the von Mises-Fisher Distribution

So back to the full distribution, I'll actually explain it in detail now.

$$\begin{align}
f_{vMF}(\vec{x} | \vec{\mu}, \kappa) = \frac{\kappa^{p/2-1}}{(2\pi)^{p/2}I_{p/2-1}(\kappa)} \exp(\kappa \vec{\mu}^T\vec{x}).
\end{align}$$

The $$p$$ denotes the dimension of the sphere +1. So for a sphere in 3D ambient space then $$p=3$$.

Once again, let's have a look at what the distribution implies for different parameters, particularly $$p=3$$ and $$\mu=[1, 0, 0]$$. (i.e. we'll be looking at how the spread parameter $$\kappa$$ works)

<br>

<div style="text-align: center;">
<iframe 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vMF_Figs/1e-05_3d_scatter.html" 
    width="89%" 
    height="500px"
    style="border:none;"
></iframe>
<figcaption> Starting off with \(\kappa=0\), this once again gives us the uniform distribution on the sphere. </figcaption>

</div>


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vMF_Figs/1e-05_corner.png" 
    style="width: 79%; height: auto; border-radius: 0px;">
    <figcaption> Starting off with \(\kappa=0\), this once again gives us the uniform distribution on the sphere. </figcaption>
</div>


<div style="text-align: center;">
<iframe 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vMF_Figs/1.0_3d_scatter.html" 
    width="99%" 
    height="500px"
    style="border:none;"
></iframe>
<figcaption> Then \(\kappa=1.0\) it's a little hard to see on the interactive plot but the samples are converging around \(x=+1\).</figcaption>

</div>

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vMF_Figs/1.0_corner.png" 
    style="width: 79%; height: auto; border-radius: 0px;">
    <figcaption> Then \(\kappa=1.0\) it's a little hard to see on the interactive plot but the samples are converging around \(x=+1\).</figcaption>
</div>

<div style="text-align: center;">
<iframe 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vMF_Figs/100.0_3d_scatter.html" 
    width="89%" 
    height="500px"
    style="border:none;"
></iframe>
<figcaption> And finally \(\kappa=100.0\), where the samples are very obviously centred about the direction vector \(\mu=[1,0,0]\). </figcaption>

</div>

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/vMF_Figs/100.0_corner.png" 
    style="width: 79%; height: auto; border-radius: 0px;">
    <figcaption> And finally \(\kappa=100.0\), where the samples are very obviously centred about the direction vector \(\mu=[1,0,0]\). </figcaption>
</div>
<br>

So broadly the von Mises-Fisher distribution boils down to,

$$\begin{align}
f_{vMF}(\vec{x}\vert\vec{\mu}, \kappa) \propto \kappa^{p/2 - 1} \exp(\kappa \vec{\mu} \cdot \vec{x}),
\end{align}$$

where again we can make the comparisons to the traditional normal distribution. $$\kappa$$ works similar to inverse variance, 
and instead of asking how similar a vector is to the mean by finding the absolute norm squared $$\lVert \vec{x} - \vec{\mu}\rVert^2$$, 
we quantify the similarity through the [_cosine similarity_](https://en.wikipedia.org/wiki/Cosine_similarity) 
or simply the dot product $$\vec{\mu} \cdot \vec{x}$$ when the two vectors have unit magnitude. 
We just get something weird in the normalisation constant because of the form of our distribution.


## Reparameterisation Trick with the von Mises-Fisher distribution

This is all great, well and good now instead of learning a mean vector and standard deviation vector as intermediates in our VAEs latent distributions, we can learn the mean direction vector and $$\kappa$$! However, standard VAEs rely quite heavily on the [_reparameterisation trick_](https://en.wikipedia.org/wiki/Reparameterization_trick) which allows us to "inject" the stochasticity of sampling into the training procedure. 

i.e. We create some noise through the variable $$\vec{\epsilon}$$ that we say comes from a standard normal with the same dimensionality as our latent space, $$\vec{\epsilon} \sim \mathcal{N}(\vec{0}, \vec{1})$$ then we could sample the conditional gaussian in this space by $$z = \vec{\sigma} \odot \vec{\epsilon} + \vec{\mu}$$.

This meant when we were optimising/taking derivatives with respect to $$\vec{\sigma} $$ and $$\vec{\mu}$$ we would get pretty simple answers, either $$\vec{\epsilon}$$ or $$\vec{1}$$. Or in more mathematically rigorous way, our loss can be represented on some level as an expectation of some function.

$$\begin{align}
L(\vec{\mu}, \vec{\sigma}, \vec{\phi}) = \mathbb{E}_{\vec{z} \sim q(\vec{z}|\kappa, \vec{\mu})}\left[f_\vec{\phi}(z) \right],
\end{align}$$

when optimising we then need to taking derivatives with respect to the parameters we are optimising... which are within the expectation which is hard or at least the estimator, called the [REINFORCE estimator](https://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/) you get out has quite high variance. But if we can perform the reparameterisation trick then we can pull the derivative _inside_ the expectation. For example with respect to $$\vec{\mu}$$ we can calculate the derivatives as,

$$\begin{align}
\nabla_{\vec{\mu}} L(\vec{\mu}, \vec{\sigma}, \vec{\phi}) &= \nabla_{\vec{\mu}} \left( \mathbb{E}_{\vec{\epsilon} \sim \mathcal{N}(\vec{0}, \vec{1})}\left[f_\vec{\phi}(z = \vec{\mu} + \vec{\sigma} \odot \vec{\epsilon}) \right]\right) \\
&= \mathbb{E}_{\vec{\epsilon} \sim \mathcal{N}(\vec{0}, \vec{1})}\left[ \nabla_{\vec{\mu}} f_\vec{\phi}(z = \vec{\mu} + \vec{\sigma} \odot \vec{\epsilon}) \right]
\end{align}$$

But how do we do this for the von Mises-Fisher distribution...? We can't use the same trick, sampling $$\vec{\epsilon}$$ uniformly on the sphere for example, as we don't (currently) have a way to taking a mean vector and $$\kappa$$ and stay on the sphere. So what do we do??


One of the main results of the paper [Hyperspherical Variational Auto-Encoders by Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) was basically this trick for the hypersphere. 

It basically comes down to using the symmetry of the vMF (von Mises-Fisher) distribution. If you just rotate the sphere (/change your perspective) you can always make it so that the samples around the mean vector are centred around the vector in the first dimension $$\vec{e}_1 = [1,0,0,...,0]$$. We can then sample some 1D [rejection sampling](https://liamcpinchbeck.github.io/posts/2025/01/2025-01-28-rejection-sampling/) distribution about this direction to encode the spread.


The basic idea of the sampling is shown in the below figure which I took from [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891), which shouldn't make complete sense yet.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/DavidsonEtAl_Diagram.png" 
    style="width: 99%; height: auto; border-radius: 0px;">
    <figcaption>Fig. 4 from Davidson et al. (2018) with caption - "Overview of von Mises-Fisher sampling procedure. Note that as ω is a scalar, the procedure does not suffer
from the curse of dimensionality." </figcaption>

</div>

If you're alright with the picture so far, that we have a spherical distribution, and we have some method to do something similar to the reparameterisation trick with this distribution, you can move on to the next section. For the rest of _this_ section we'll try and learn how this trick actually works.


<div style="float: right; width: 50%; margin-left: 20px;">
    <img 
        src="/files/BlogPostData/2025-constant-curvature-vaes/DavidsonAlgorithm1.png" 
        style="width: 100%; height: auto; border-radius: 0px;">
    <figcaption style="text-align: center;">Algorithm 1 from Davidson et al. (2018)</figcaption>
</div>

### The deets

To sample the vMF distribution with something like the reparameterisation trick, there are only four steps (plain english version of Algorithm 1 from [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891)).

1. Generate samples on the unit sphere 1-dimension less than the dimension you want (we'll refer to this as $$\vec{v} \sim U(\mathcal{S}^{m-2})$$)
2. Sample the spread due to $$\kappa$$ (where we sample the equivalent $$\vec{\epsilon}$$) in the first dimension from some weird distribution $$g(\omega, \kappa)$$
3. Scale the previously uniform samples to add the first dimension of samples. Now getting samples on the full sphere $$z' = (\omega;(\sqrt{1-\omega^2})\vec{v}^T)^T$$
4. Rotate the distribution so that the samples are centred around the mean vector $$\vec{\mu}$$ (using the [Householder transform](https://en.wikipedia.org/wiki/Householder_transformation))


This is encapsulated in the below function `sample_vMF`.

```python
from scipy.stats import uniform_direction
import numpy as np


def sample_vMF(mean_vec, k, num=10):

    m = len(mean_vec)

    e1vec = np.zeros(m)
    e1vec[0] = 1.

    # Step 1
    uniform_sphere_samples = uniform_direction(dim=m-1).rvs(num)
    #-----

    # Step 2
    W = sample_vMF_mixture(k, m, num=num)
    #-----

    # Step 3
    adjusted_u_sphere_samples = (np.sqrt(1-W**2) * uniform_sphere_samples.T).T
    zprime = np.concatenate((W[:, None], adjusted_u_sphere_samples), axis=1)
    #-----

    # Step 4
    U = householder(mean_vec, e1vec)
    z = (zprime @ U)
    #-----

    return z
```

The first step, getting $$\vec{v}\sim U(\mathcal{S}^{m-2})$$, is pretty easy actually. You can sample any rotationally symmetric distribution and scale the samples to be on the unit sphere. 
e.g. The multivariate normal distribution with 0 mean and covariance. For us we'll just use the `scipy.stats.uniform_direction` distribution for simplicity.

The second step was by far the hardest to wrap my head around, so I'll cover the other two first.

Presuming that you have samples about the first dimension that represent the spread from $$\kappa$$, we need to modify the other uniform samples on the sphere such that together, everything is still on the sphere.
By some simple algebra you can see that 

$$\begin{align}
z' \cdot z' &= \omega^2 + (1-\omega^2) \vec{v} \cdot \vec{v} \\
&=  \omega^2 + (1-\omega^2)  \\
&= 1 \text{  }.
\end{align}$$


The final step that constructs $$U$$ that transforms our samples about $$\vec{e}_1$$ to $$\vec{\mu}$$ is done by the [householder transform or reflection](https://en.wikipedia.org/wiki/Householder_transformation). 
Essentially creates a plane between the direction vector $$\vec{\mu}$$ and $$\vec{e}_1$$ such that if you reflect about it, $$\vec{e}_1$$ turns into $$\vec{\mu}$$. The actual transform is given as,

$$\begin{align}
U = I - \hat{u} \, \hat{u}^T,
\end{align}$$

where $$\hat{u}$$ is the unit vector in the direction of $$\vec{e}_1 - \vec{\mu}$$.

There's a tonne of useful videos and lectures notes on this that I'd recommend for a more rigorous take on this operation 
([this YouTube video](https://www.youtube.com/watch?v=pOiOH3yESPM) seemed pretty good but I'd put them on 2x speed) but I include a little diagram for some quick visual intuition below.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/HouseHolderDiagram/Householder1.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/HouseHolderDiagram/Householder2.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/HouseHolderDiagram/Householder3.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/HouseHolderDiagram/Householder4.png" 
    style="width: 48%; height: auto; border-radius: 1px;">

<figcaption>Should be a diagram showing example of Householder transform construction. Top Left: Example setup, green is the mean vector and purple the e1 vector. Top Right: Construction of e1-mean vector. Bottom Left: Construction of reflection plane with normal vector e1-mean. Bottom Right: Example transformation of vector reflected about the reflection plane.</figcaption>
</div>


<br>


The operation is encoded in the `householder` function below (pretty much the same as Algorithm 3 from [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891)).

```python
def householder(dirvec, e1vec):

    uprime = e1vec - dirvec

    uprime /= np.linalg.norm(uprime)

    U = np.eye(len(e1vec)) - 2*uprime[:, None]*uprime[None, :]

    return U
```

### Sampling the mixture distribution

Now let's circle back to the second step. For this we need to motivate two things:

1. Splitting the distribution such that the we can sample the 1D mixture distribution $$g(\omega\vert \kappa, m)$$ and 
2. basically how to efficiently sample this function in a 'reparameterisation trick'-y way.


#### 1. Splitting the distribution

Splitting the distribution into two is actually fairly intuitive when observing the problem from a specific frame. 

I'll try and walk through this with a 3D example. We have the mean vector and the sphere.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/MeanVec.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
</div>
<br>


Let's just presume that we can generate some samples for now.


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/Initial_vMF_Samples.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
</div>
<br>


What we're really looking at here is a 3D plot but that's hard to get right in matplotlib due to it's bad ordering behaviours.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/Initial_vMF_Samples_3D.png" 
    style="width: 88%; height: auto; border-radius: 1px;">
</div>
<br>



Back to the 2D side-on view, because the samples are symmetric about the mean vector we can kind of see that the spread due to $$\kappa$$ can be mostly attributed along this direction, 
and then the samples in the other directions are uniform in direction.

<div style="text-align: center;">

<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/Initial_vMF_Samples_projected_onto_mean.png" 
    style="width: 48%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/Initial_vMF_Samples_projected_onto_not_mean.png" 
    style="width: 48%; height: auto; border-radius: 0px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/Initial_vMF_Samples_projected_onto_not_mean_top_view.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
</div>
<br>

Again, just presuming that this will work for now, let's scale the samples in the other dimensions such that they fall on the sphere 1-dimension lower than our original one (in this case the _circle_).

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/top_view_scaling_transformation.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/top_view_scaled_samples.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
</div>
<br>

Adding a few more samples we can then look at the angular distribution confirming that it's uniform on the sphere because it's uniform in angle.


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/top_view_scaled_samples_lotsa.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/angular_dist_of_top_view_scaled_samples_lotsa.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
</div>
<br>


Going back to the lower number of samples,

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/top_view_scaled_samples.png" 
    style="width: 48%; height: auto; border-radius: 1px;">
</div>
<br>

we can then pretend that we sampled the first dimension via some complicated distribution accounting for the spread (getting $$\omega$$), 
and scale our uniform samples on $$\vec{v} \sim S^{1}$$ back to $$S^{2}$$ via $$(\omega, \sqrt{1-\omega^2} \vec{v}^T)$$ 
(but in reality just take the projected values from the beginning) to see that we recover the exact same distribution.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SplitExplanation/SideOn_initialvMFSamples_and_reproduced_samples.png" 
    style="width: 99%; height: auto; border-radius: 1px;">
</div>
<br>

So this really implies that we can uniformly sample the sphere 1-dimension lower than the one we want, sample some 1D distribution, 
and then scale our samples with the 1D variable as a new dimension as a way to sample the vMF.

If you want a more rigorous explanation, then I'm going to have to refer to the same papers that [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) refer to called 
["Computer Generation of Distributions on the _M_-sphere" (1984) by Gary Ulrich](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2347441) 
and the paper that Ulrich cites called ["A Family of Distributions on the m-Sphere and Some Hypothesis Tests" (1978) by John G. Saw](https://www.jstor.org/stable/2335278).


#### 2. Sampling the 1D Mixture distribution in a 'reparameterisation trick'-y way

<br>

So hopefully we now understand why we can split the distribution in two where $$\vec{z} = (\omega, (\sqrt{1-\omega^2})v^T)$$ with $$f(\vec{z}\vert\kappa, m) = g(\omega \vert \kappa, m) \cdot g_2(\vec{v} \vert m-1)$$. 
We can sample $$\vec{v}$$ by scaling samples from the multivariate standard normal distribution (as the samples are directionally uniform/symmetric). 
But what is $$g(\omega \vert \kappa, m)$$, and how do we sample it in a 'reparameterisation trick'-y way?

The general method will be:
1. We take samples from a distribution that we can _already_ sample efficiently that has a similarish form to $$g(\omega \vert \kappa, m)$$
2. We perform rejection sampling with these samples, where the probability of accepting the values is the ratio of $$g(\omega \vert \kappa, m)$$ and our proposal distribution
3. Figure out how this gives us derivatives of the loss that doesn't involve a derivative of the distribution over which we are taking an average

To save some time (for both you and me) we're just going to propose that,

$$\begin{align}
g(\omega \vert \kappa, m) &\propto \exp(\kappa \vec{e}_1 \cdot x) (1-\omega^2)^{(m-3)/2} \\
&=\exp(\kappa \omega) (1-\omega^2)^{(m-3)/2},
\end{align}$$

which in essence is the von Mises-Fisher distribution with a geometrical jacobian factor for what would be a surface integral. 
Accounting for the surface area of lower dimensional sphere with the dimensions perpendicular to $$\vec{e}_1$$ ((where we get the uniform samples) ), $$\text{Area} \propto R^{m-2} = (1-\omega^2)^{\frac{m-2}{2}}$$, 
and the conversion between $$\omega$$ as a z-coordinate and $$\phi$$ in angular coordinates (as any integration has to be done on the sphere),

$$\begin{align}
d\omega &= d(\cos(\phi)) = -\sin(\phi) d\phi \\
d\phi &= d\omega/\sqrt{1-\cos^2(\phi)} = d\omega/\sqrt{1-\omega^2}\\
\implies J(\omega) &= (1/\sqrt{1-\omega^2}) \cdot (1-\omega^2)^{\frac{m-2}{2}} \\
J(\omega) &= (1-\omega^2)^{\frac{m-3}{2}}. \\
\end{align}$$

So that gives us the form of $$g(\omega \vert \kappa, m)$$ (up to some multiplicative constant with respect to $$\omega$$). Next, we are going to guess that we can sample our noise from the following distribution 
(this similar to the bit in the reparameterisation trick where we sample from a distribution independent of our parameters to get the stochastic component), 

$$\begin{align}
\epsilon \sim s(\epsilon) = \text{Beta}(\frac{m-1}{2}, \frac{m-1}{2}),
\end{align}$$

and then transform it to make a proposal for $$\omega$$ as,


$$\begin{align}
\omega = h(\epsilon \vert \kappa, m) = \frac{1-(1+b(\kappa, m))\epsilon}{1-(1-b(\kappa, m))\epsilon},
\end{align}$$

we'll get to the specific form of $$b$$ later, for now you can just imagine that it's some simple combination of $$\kappa$$ and $$m$$. 

The $$r(\omega, \kappa)$$ referred to in the paper is the so called `envelope` distribution used in rejection sampling. 
For a thorough introduction to rejection sampling I would recommend [my blog post](https://liamcpinchbeck.github.io/posts/2025/01/2025-01-28-rejection-sampling/) (not biased at all...). 


The basics of it is that the envelope is some other distribution that we can sample, and then after some multiplicative factor, we reject samples with a smaller probability than the actual
distribution.
Differences between good and bad envelopes is exemplified in the below two GIFs.
One has a non-informative envelope (top one with a uniform distribution) and one has an infomative envelope (bottom, gaussian envelope). 
You can observe that one wastes more proposals (orange dots) than the other.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-01-28/argus_dist_effective_samples.gif" 
    alt="GIF showing animation of rejection sampling principle with an ARGUS distribution." 
    title="GIF showing animation of rejection sampling principle with an ARGUS distribution." 
    style="width: 75%; height: auto; border-radius: 8px;">
<img 
    src="/files/BlogPostData/2025-01-28/argus_dist_with_better_proposal.gif" 
    alt="GIF showing animation of rejection sampling principle with an ARGUS distribution with a better _proposal_ distribution." 
    title="GIF showing animation of rejection sampling principle with an ARGUS distribution with a better _proposal_ distribution." 
    style="width: 75%; height: auto; border-radius: 8px;">
</div>

What we want, is the envelope to have as similar of a form as the distribution we are trying to sample from. If we look at the implied distribution on $$\omega$$, based on 
injecting the samples/density in $$s(\epsilon)$$ into $$h(\epsilon \vert \kappa, m)$$ we find that we've actually already done this. 
First we need to rearrange $$\epsilon$$ in terms of $$\omega$$, i.e. $$\epsilon = h^{-1}(\omega\vert\kappa, m)$$.

$$\begin{align}
\omega &= \frac{1-(1+b)\epsilon}{1-(1-b)\epsilon} \\
\left(1-(1-b)\epsilon\right)\omega &= 1-(1+b)\epsilon \\
\omega-\omega(1-b)\epsilon &= 1-(1+b)\epsilon \\
\omega - 1 &= (\omega(1-b)-(1+b))\epsilon \\
\epsilon &= \frac{\omega - 1}{\omega(1-b)-(1+b)} \\
\end{align}$$

This allows us to get the jacobian between $$\epsilon$$ and $$\omega$$ in terms of $$\omega$$.

$$\begin{align}
\frac{d\epsilon}{d\omega} &= \frac{-2b}{(1+b - \omega(1-b))^2}. \\
\end{align}$$

Hence,

$$\begin{align}
r(\omega\vert\kappa, m) &= \left\lVert\frac{d\epsilon}{d\omega}\right\rVert s(\epsilon(\omega)) \\
&= \left\lVert\frac{d\epsilon}{d\omega}\right\rVert \epsilon^{\frac{m-3}{2}}\left(1-\epsilon\right)^{\frac{m-3}{2}} \\
&= \frac{2b}{(1+b - \omega(1-b))^2} \left(\frac{\omega - 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}}\left(1 - \frac{\omega - 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}} \\
&= \frac{2b}{(1+b - \omega(1-b))^2} \left(\frac{\omega - 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}}\left(\frac{\omega(1-b)-(1+b) - \omega + 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}} \\
&= \frac{2b}{(1+b - \omega(1-b))^2} \left(\frac{\omega - 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}}\left(\frac{\omega -b\omega - 1 - b - \omega + 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}} \\
&= \frac{2b}{(1+b - \omega(1-b))^2} \left(\frac{\omega - 1}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}}\left(\frac{-b( \omega + 1)}{\omega(1-b)-(1+b)}\right)^{\frac{m-3}{2}} \\
&= \frac{2b^{\frac{m-1}{2}}}{(1+b - \omega(1-b))^2} \left(\frac{(1 - \omega)(1 + \omega)}{(\omega(1-b)-(1+b))^2}\right)^{\frac{m-3}{2}} \\
&= 2b^{\frac{m-1}{2}}  \frac{(1 - \omega^2)^{\frac{m-3}{2}} }{(1 + b - \omega(1-b))^{m-1}}\\
\end{align}$$

The key similar here, and again we want the envelope to be as similar to the distribution we are trying to sample, is that they both share a $$(1 - \omega^2)^{\frac{m-3}{2}}$$ factor.
The polynomial factor actually roughly gets larger with larger $$\omega$$, which is also similar to $$\exp(\kappa \omega)$$, but crucially will always be larger than $$\exp(\kappa \omega)$$ 
within the range of $$\omega$$. i.e. We've found a pretty good envelope.

The actual sampling then comes down to rejection sampling using $$s(\epsilon)$$, $$r(\omega\vert\kappa, m)$$ (dependence on $$\kappa$$ comes through $$b$$ by the way) and $$g(\omega\vert\kappa, m)$$. 
This process is described in Algorithm 2 from [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) with some algebraic trickery that I'm not going to go into as to me at least, that's all it is.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/DavidsonAlgorithm2.png" 
    style="width: 66%; height: auto; border-radius: 8px;">
    <figcaption> Algorithm 2 from Davidson et al. 2018 </figcaption>
</div>


I've coded this up in the below python function.


```python
from scipy.stats import beta, uniform, uniform_direction

def sample_vMF_mixture(k, m, num=10):
    b = (-2*k + np.sqrt(4*k**2 + (m-1)**2))/(m-1)
    a = ((m-1) + 2*k + np.sqrt(4*k**2 + (m-1)**2))/4
    d = 4*a*b/(1+b) - (m-1)*np.log(m-1)


    samples = []

    condition = True
    while condition:
        Y = beta((m-1)/2, (m-1)/2).rvs()
        u = uniform(0, 1).rvs()
        W = (1 - (1 + b) * Y)/(1 - (1 - b) * Y)

        T = 2*a*b/(1 - (1-b)*Y)

        if (m-1)*np.log(T) - T + d > np.log(u):
            samples.append(W)

            if len(samples)==num:
                condition = False
    return np.array(samples)
```




## Putting it all together

We can then put this into action with the function we defined way above.

```python
custom_vMF_samples = sample_vMF(np.array([0., 0., 1.0]), 5.0, num=5000)
```
<iframe 
    src="/files/BlogPostData/2025-constant-curvature-vaes/SVAE_custom_vMF_3d_scatter_with_sphere.html" 
    width="89%" 
    height="500px"
    style="border:none;"
></iframe>

Voila! Terrifique! Magnifique! (Not sure if that's how I'm meant to spell that!)


<br>

## Taking Derivatives

So we made a sampling regime similar to the standard reparameterisation trick, but the key thing in the original was that the derivatives of the loss worked out. 
The next question we need to ask ourselves is whether that is the case? 

It's not immediately obvious because what we did isn't an exact 1-to-1 to the original, because of the rejection sampling step (which isn't in the original).

What [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) do is use a result from ["Reparameterization Gradients through Acceptance-Rejection Sampling Algorithms" - Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683).
In this paper they show that for many rejection sampling setups you can move gradients of the parameters being optimised, into the expectations within the loss.

If we represent our loss as some expectation over the KL divergence between some exact posterior probability distribution $$p(z\vert x) = p(x, z)/p(x)$$ and our (variational) approximation $$q(z\vert x ; \phi)$$ that we are optimising with respect to $$\phi$$
(for those familiar with variational inference I'm just quickly deriving the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound)),

$$\begin{align}
KL(q\vert\vert p ; \phi) &= \mathbb{E}_{z\sim q(z\vert x)}\left[ \log \frac{q(z\vert x ; \phi)}{p(z\vert x)}\right] \\
&= \mathbb{E}_{z\sim q(z\vert x)}\left[ \log q(z\vert x ; \phi) - \log p(z\vert x) \right] \\
&= \mathbb{E}_{z\sim q(z\vert x)}\left[ \log q(z\vert x ; \phi) - \log p(z, x) + \log p(x) \right] \;\;\;\;\;\;\ \text{(Bayes' theorem)}\\
&= \mathbb{E}_{z\sim q(z\vert x)}\left[ \log q(z\vert x ; \phi)\right] \\
&\;\;\;\;-  \mathbb{E}_{z\sim q(z\vert x ; \phi)}\left[\log p(z, x)\right] \\
&\;\;\;\;+ 1 \cdot \log p(x). \;\;\;\;\;\;\ \text{(p(x) doesn't involve z)}\\
\end{align}$$

If we are constructing a loss to optimise, then that last term doesn't matter as it doesn't involve $$\phi$$ so we will drop it. 
Also in statistics the first term has a more formal definition as the [_entropy_](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of $$q(z\vert x ; \phi)$$, denoted $$\mathbb{H}[q(z\vert x ;\phi)]$$.
We will then follow the notation of [Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683) and denote the remaining term as the following (dropping the dependence on $$x$$ from now on, presuming it to be a constant effectively),

$$\begin{align}
\mathbb{E}_{z\sim q(z; \phi)}\left[\log p(z, x)\right] = \mathbb{E}_{z\sim q(z; \phi)}\left[f(z)\right].
\end{align}$$

We will then write the loss, or simply the function that we are trying to minise, as,

$$\begin{align}
L(\phi) = \mathbb{E}_{z\sim q(z ; \phi)}\left[f(z)\right] + \mathbb{H}[q(z ;\phi)].
\end{align}$$

What [Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683) then show is that if you have a target distribution that you are trying to sample, 
in this case $$q(z;\phi)$$, a proposal distribution $$r(z;\phi)$$ and constant $$M_\phi$$ such that $$q(z;\phi)\leq M_\phi r(z;\phi)$$. 
We perform rejection sampling following the algorithm below that I'm stealing from [Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683) because I cannot figure out how to do the formatting
for algorithms in GitHub markdown (just replace the $$\theta$$'s with $$\phi$$'s).


<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/NaessethAlgorithm1.png" 
    style="width: 50%; height: auto; border-radius: 8px;">
    <figcaption> Algorithm 1: "Reparameterized Rejection Sampling" from  Naesseth et al. (2016)</figcaption>
</div>

When the sampling is set up in this manner we can represent the probability of accepting a given sample as the folliwing (following very closely to equation 4 in [Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683)),

$$\begin{align}
\pi(\epsilon;\phi) = \int \pi(\epsilon, u ;\phi) du
\end{align}$$

$$\pi(\epsilon, u ;\phi)$$ is the probability of accepting a given $$\epsilon$$ for a specific $$u$$.
We can split this into the base probability of sampling a given $$\epsilon \sim s(\epsilon)$$ and [_indicator function_](https://en.wikipedia.org/wiki/Indicator_function) (denoted $$\mathbb{1}$$) for whether we accept this given $$\epsilon$$, 
along a normalisation constant which turns out to be $$M_\phi$$.

$$\begin{align}
\pi(\epsilon;\phi) &= \int \pi(\epsilon, u ;\phi) du \\
 &= \int M_\phi s(\epsilon) \mathbb{1} \left[0 < u < \frac{q(h(\epsilon, \phi) ; \phi)}{M_\phi r(h(\epsilon, \phi) ; \phi)} \right] du \\
 &=M_\phi s(\epsilon)  \int \mathbb{1} \left[0 < u < \frac{q(h(\epsilon, \phi) ; \phi)}{M_\phi r(h(\epsilon, \phi) ; \phi)} \right] du \\
\end{align}$$

Where $$h(\epsilon, \phi)$$ still represents the transform between $$\epsilon$$ and the parameter of interest $$z$$ which for our specific use-case is $$\omega$$. The integration then follows that,

$$\begin{align}
\pi(\epsilon;\phi) &=M_\phi s(\epsilon)  \int \mathbb{1} \left[0 < u < \frac{q(h(\epsilon, \phi) \vert \phi)}{M_\phi r(h(\epsilon, \phi) ; \phi)} \right] du \\
&= s(\epsilon) \frac{q(h(\epsilon, \phi) ; \phi)}{r(h(\epsilon, \phi) ; \phi)},\\
\end{align}$$

which if doesn't make sense I've attempted to represent the indicator function graphically below. The function outputs 1 until it hits the threshold so the integral of it with respect to $$u$$ just ends up being the threshold.

<div style="text-align: center;">
<img 
    src="/files/BlogPostData/2025-constant-curvature-vaes/rejection_sampling_integration_figure.png" 
    style="width: 50%; height: auto; border-radius: 8px;">
    <figcaption> Graphical representation of the integration of the indicator function above.</figcaption>
</div>
<br>

And once we have the above form of $$\pi$$ we can re-write our loss as the following,

$$\begin{align}
L(\phi) &=\mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[f(h(\epsilon, \phi))\right] + \mathbb{H}_{\epsilon \sim \pi(\epsilon ; \phi)}[q(h(\epsilon, \phi) ;\phi)].
\end{align}$$


Now that we have the rejection sampling distribution encoded like this it's pretty simple to take derivatives of our loss. 
Like [Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683) I'll focus on the first term, as the derivation for the second is basically exactly the same (and then I don't have to do it myself either!).

$$\begin{align}
\nabla_\phi L(\phi) &= \nabla_\phi \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[f(h(\epsilon, \phi))\right] \\
&= \int s(\epsilon) \nabla_\phi \left(f(h(\epsilon, \phi)) \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} \right) d\epsilon \\
&= \int s(\epsilon) \nabla_\phi \left[f(h(\epsilon, \phi)\right] \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} d\epsilon \\
&\;\;\;\;\;\;\;\;\;\; + \int s(\epsilon) f(h(\epsilon, \phi))\nabla_\phi \left( \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} \right) d\epsilon \\
&=  \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[\nabla_\phi f(h(\epsilon, \phi))\right]\\
&\;\;\;\;\;\;\;\;\;\; + \int s(\epsilon) f(h(\epsilon, \phi))\left( \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} \right) \nabla_\phi \log \left( \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} \right) d\epsilon \\
&=  \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[\nabla_\phi f(h(\epsilon, \phi))\right]\\
&\;\;\;\;\;\;\;\;\;\; + \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[\nabla_\phi \log \left( \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} \right) \right]  \\
\end{align}$$


Along with the above, [Naesseth et al. (2016)](https://arxiv.org/abs/1610.05683) also note that if $$h(\epsilon, \phi)$$ is invertible 
(which in our case it obviously is because we swapped the outputs and inputs above) then you can also show that the final term simplifies further into (remembering that $$r = s \circ h^{-1}$$),

$$\begin{align}
&\nabla_\phi \log \left( \frac{q(h(\epsilon, \phi);\phi)}{r(h(\epsilon, \phi);\phi)} \right) \\
&= \nabla_\phi \log q(h(\epsilon, \phi);\phi) -  \nabla_\phi \log  r(h(\epsilon, \phi);\phi) + \nabla_\phi \log \left\lvert \frac{dh}{d\epsilon}(\epsilon, \phi) \right\rvert \\
&= \nabla_\phi \log q(h(\epsilon, \phi);\phi) -  \nabla_\phi \log  s(h^{-1}(h(\epsilon, \phi);\phi)) + \nabla_\phi \log \left\lvert \frac{dh}{d\epsilon}(\epsilon, \phi) \right\rvert \\
&= \nabla_\phi \log q(h(\epsilon, \phi);\phi) -  \nabla_\phi \log  s(\epsilon) + \nabla_\phi \log \left\lvert \frac{dh}{d\epsilon}(\epsilon, \phi) \right\rvert \\
&= \nabla_\phi \log q(h(\epsilon, \phi);\phi) -  0 + \nabla_\phi \log \left\lvert \frac{dh}{d\epsilon}(\epsilon, \phi) \right\rvert \\
&= \nabla_\phi \log q(h(\epsilon, \phi);\phi) + \nabla_\phi \log \left\lvert \frac{dh}{d\epsilon}(\epsilon, \phi) \right\rvert. \\
\end{align}$$

So the derivative of our loss becomes,


$$\begin{align}
\nabla_\phi L(\phi) &=  \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[\nabla_\phi f(h(\epsilon, \phi))\right]\\
&\;\;\;\;\;\;\; + \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[\nabla_\phi \log q(h(\epsilon, \phi);\phi)\right]  \\
&\;\;\;\;\;\;\; + \mathbb{E}_{\epsilon \sim \pi(\epsilon ; \phi)}\left[\nabla_\phi \log \left\lvert \frac{dh}{d\epsilon}(\epsilon, \phi) \right\rvert\right]. \\
\end{align}$$

So in essence this is great because we went from somehow having to take a derivative of an algorithm (rejection sampling) into some nice monte carlo estimate.

## Some final things involving the loss

Similar to how in standard VAEs we enforce a regularisation on the distribution formed in the latent space by including a non-informative prior on the result, 
for spherical VAEs we enforce the non-informative uniform distribution on the sphere. Beyond the changes in geometry, this is fundamentally different to standard VAEs
that enforce a normal distribution which still yields some information about the 'centres' and 'spread' of the distributions, the regularisation/prior here is completely
uniform. This is fine essentially because the space we are constructing our latent distribution in is already finite.

With this [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) calculate the regularisation term of the loss, which is interpreted as the KL divergence between the 
'posterior distribution' which is the vMF distribution we've been discussing and sampling above described by $$ \mathcal{C}_m(\kappa) \exp(\kappa \vec{\mu}^T \vec{z})$$ 
and the prior (our uniform distribution on the unit sphere). 

The normalisation constant $$ \mathcal{C}_m(\kappa)$$ for the von-Mises Fisher distribution is given by $$\mathcal{C}_m(\kappa) = \frac{\kappa^{m/2-1}}{(2\pi)^{m/2}\mathcal{I}_{m/2-1}(\kappa)}$$. 

The surface area of a sphere in $$m$$ dimensions is given by
$$A(S^{m-1}) = \frac{2\pi^{m/2}}{\Gamma(m/2)}$$, hence the uniform probability distribution is $$p(\vec{z}) = \left(A(S^{m-1})\right)^{-1}$$. 

Hence the KL divergence is given by,

$$\begin{align}
&KL\left[q(\vec{z}\vert \vec{\mu}, \kappa) \lVert p(\vec{z}) \right] \\
&= \int_{S^{m-1}} q(\vec{z}| \vec{\mu}, \kappa) \log \frac{q(\vec{z}| \vec{\mu}, \kappa) }{p(\vec{z})} d\vec{z} \\
&= \int_{S^{m-1}} q(\vec{z}| \vec{\mu}, \kappa) \left( \log q(\vec{z}| \vec{\mu}, \kappa) - \log p(\vec{z}) \right) d\vec{z} \\
&= \int_{S^{m-1}} q(\vec{z}| \vec{\mu}, \kappa) \left( \log \underbrace{\left( \mathcal{C}_m(\kappa) \exp(\kappa \vec{\mu}^T \vec{z}) \right)}_{\text{von-Mises Fisher distribution}} - \log \underbrace{\left(\frac{2\pi^{m/2}}{\Gamma(m/2)} \right)}_{U(S^{m-1})}  \right) d\vec{z} \\
&= \log \frac{\kappa^{m/2-1}}{(2\pi)^{m/2}\mathcal{I}_{m/2-1}(\kappa)} + \int_{S^{m-1}} q(\vec{z}| \vec{\mu}, \kappa) \left( \kappa \vec{\mu}^T \vec{z}\right) d\vec{z} \\
&\;\;\;\;\;\; - \log 2 - \frac{m}{2} \log \pi + \log \Gamma(m/2)\\
&= \log \frac{\kappa^{m/2-1}}{(2\pi)^{m/2}\mathcal{I}_{m/2-1}(\kappa)} + \kappa \vec{\mu}^T \mathbb{E}_{\vec{z}\sim q(\vec{z}|\vec{\mu}, \kappa)}\left[\vec{z}\right] \\
&\;\;\;\;\;\;- \log 2 - \frac{m}{2} \log \pi + \log \Gamma(m/2).\\
\end{align}$$

To continue this derivation we're going to use the fact that,

$$\begin{align}
\mathbb{E}_{\vec{z}\sim q(\vec{z}\vert\vec{\mu}, \kappa)}\left[\vec{z}\right]=\vec{\mu}\frac{\mathcal{I}_{m/2}(\kappa)}{\mathcal{I}_{m/2-1}(\kappa)},
\end{align}$$

which didn't make immediate sense but if you imagine the non-zero concentration parameter examples above, the mean coordinate is along $$\vec{\mu}$$ but has to have a smaller magnitude. 
The $$\frac{\mathcal{I}_{m/2}(\kappa)}{\mathcal{I}_{m/2-1}(\kappa)}$$ is then simply the degree that the vector is contracted. Continuing our analytical construction of the loss (we'll see why this is important in a sec),


$$\begin{align}
&KL\left[q(\vec{z}\vert \vec{\mu}, \kappa) \lVert p(\vec{z}) \right] \\
&= \underbrace{\left(\frac{m}{2}-1\right) \log \kappa - \frac{m}{2} \log\left(2\pi\right) -  \log \mathcal{I}_{m/2-1}(\kappa)}_{\text{norm. const. from vMF}}\\
&\;\;\;\;\;\; + \underbrace{\kappa \frac{\mathcal{I}_{m/2}(\kappa)}{\mathcal{I}_{m/2-1}(\kappa)}}_{\text{from vMF expectation}} -  \underbrace{\log 2 - \frac{m}{2} \log \pi + \log \Gamma(m/2)}_{\text{from prior}}.
\end{align}$$

And this is our final loss (for the latent space/stuff involving the sphere explicitly[^mu], you would also need the reconstruction loss). 
The issue is then that we have [modified bessel functions](https://en.wikipedia.org/wiki/Bessel_function) that non-trivially depend on one of the parameters of interest $$\kappa$$, and automatic differentiation can't handle them.
This means we have to come up with some expression ourselves.

[^mu]: Notice that the loss _only_ depends on $$\kappa$$, not on $$\mu$$. Meaning that $$\mu$$ is only updated by the reconstruction loss. This makes a little sense as the value of $$\mu$$ doesn't really mean anything regularisation-wise.


I'll leave the derivation of this gradient to you (which after the above we know that we can perform nicely). The only information that you might need is that $$\nabla_\kappa \mathcal{I}_\nu(\kappa) = \frac{1}{2} \left(\mathcal{I}_{\nu-1}(\kappa) + \mathcal{I}_{\nu+1}(\kappa) \right)$$. 
It then comes out (using roughly the same formatting as [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) as they derived in Equation 6 of their paper) that the derivatives of the above loss can be calculated as,

$$\begin{align}
&\nabla_\kappa KL(vMF(\vec{\mu}, \kappa) \lVert U(S^{m-1})) \\
&= \frac{k}{2}\left(\frac{\mathcal{I}_{m/2+1}(\kappa)}{\mathcal{I}_{m/2-1}(\kappa)}- \frac{\mathcal{I}_{m/2}(\kappa) \left(\mathcal{I}_{m/2-2}(\kappa) + \mathcal{I}_{m/2}(\kappa)\right)}{\left(\mathcal{I}_{m/2-1}(\kappa)\right)^2} + 1 \right).
\end{align}$$



## Coding it all up

I was thinking of coding this all up myself but in the end figured it would almost be a carbon copy of [the PyTorch version of the code produced by Davidson et al. (2018)](https://github.com/nicola-decao/s-vae-pytorch) anyways. 
Additionally, they were kind enough to _also_ make a [Tensorflow version of their code as well](https://github.com/nicola-decao/s-vae-tf/tree/master).
So, have a look in either of those if you're interested. 
In the PyTorch version the calculations for the gradients/bessel functions are specifically calculated [in this file](https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/ops/ive.py). 
Not quite sure where the equivalent is in the case of the Tensorflow version.

<br>

# Hyperbolic VAE






<br>

# Image Classification and Generation for MNIST and CelebA with constant curavture VAEs



<br>

# Molecular Property Prediction for QM9 dataset with constant curavture VAEs




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