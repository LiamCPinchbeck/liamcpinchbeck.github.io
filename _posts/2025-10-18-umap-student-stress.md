---
title: 'Investigating student stress indicators using UMAP (Kaggle Dataset)'
date: 2025-10-18
permalink: /posts/2025/10/2025-10-18-Student-Stress-UMAP/
tags:
  - Introductory
  - Dimensional Reduction
  - Machine Learning
header-includes:
   - \usepackage{amsmath}
---

In this post, I’m going to investigate the underlying relationships between various physical and mental health indicators and student stress levels. In the process I will give an introduction to the _Uniform Manifold Approximation and Projection_ or ___UMAP___ dimensional reduction technique.


## Resources

As usual, here are some of the resources I’m using as references for this post. Feel free to explore them directly if you want more information or if my explanations don’t quite click for you.

- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426) - Leland McInnes, John Healy, James Melville
- [UMAP Uniform Manifold Approximation and Projection for Dimension Reduction \| SciPy 2018 \|](https://www.youtube.com/watch?v=nq6iPZVUxZU&t=979s) - Leland McInnes
- [How UMAP Works - UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)
- [Uniform Manifold Approximation and Projection (UMAP) and its Variants: Tutorial and Survey](https://arxiv.org/abs/2109.02508) - Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction (Documentation)](https://umap-learn.readthedocs.io/en/latest/)
- [UMAP Dimension Reduction, Main Ideas!!!](https://youtu.be/eN0wFzBA4Sc?si=OOijBHLZVVOlH9pL) - [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
- [UMAP: Mathematical Details (clearly explained!!!)](https://youtu.be/jth4kEvJ3P8) - [StatQuest with Josh Starmer](https://www.youtube.com/@statquest)
- [Uniform Manifold Approximation and Projection (UMAP) \| Dimensionality Reduction Techniques (5/5)](https://youtu.be/iPV7mLaFWyE?si=4k7SQDNw-O9O3C1Z) - [DeepFindr](https://www.youtube.com/@DeepFindr)
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/) - Andy Coenen, Adam Pearce \| Google PAIR
- [](https://giotto-ai.github.io/gtda-docs/0.5.1/notebooks/persistent_homology_graphs.html#id8)

## Table of Contents

- [Motivation](#motivationtraditional-autoencoders)
- [Core Idea](#core-idea)

---


# Motivation

In machine learning contexts it is often the case that we have to deal with very high dimensional data. One of the most familiar cases is that of image data, which even for low resolutions, like that of the MNIST dataset, goes into hundreds of dimensions, and in more common photos hundreds of thousands if not more pixels/dimensions. 

To make these problems more tractable, dimensional reduction techniques are employed, such as Variational Autoencoders and PCA among a plethora of others. However, they all share the common idea that although the data is high dimensional, there exists a low-dimensional representation or components that would aptly explain most of the variation within. For example in the MNIST dataset, due to the nature of the dataset, one might expect that most of the data could be explained by a single variable which encodes the information for "1", "2", "3", etc[^FLIG]. 

[^FLIG]: I essentially showed this in my Flow Matching for image generation post in the [conditional flow section](https://liamcpinchbeck.github.io/posts/2025/09/2025-09-28-FM-ImageGen/#:~:text=Generating%20MNIST%2Dlike%20images%20(conditional%20flow%20matching)) where I generated MNIST-like images by simply giving the relevant numbers.

One particularly interesting approach to me is the [_Uniform Manifold Approximation and Projection_ or ___UMAP___ dimensional reduction technique](https://arxiv.org/abs/1802.03426)[^NFLOWS]. It can encode both something between local and global features of the data, is non-linear is very cheap[^cost] and as stated in many of it's initial publications has a robust theoretical foundation which has enabled many evolutions including parameteric UMAP (_P-UMAP_) that learns a parametric transform from the complicated data space into the lower dimensional space that can very naturally be included in larger machine learning architectures[^SBI].

[^cost]: UMAP is ~$$\mathcal{O}(n^{1.14})$$ compared to t-SNE which is $$\mathcal{O}(n^2)$$ which can be very constraining in very high dimensional spaces among other aspects

[^NFLOWS]: In part because it shares a similar principle to flow-based density estimation approaches that wish to learn a projection from a simpler latent space into the data space.

[^SBI]: One of the main reasons I was initially interested in the approach was as a technique to learn a low dimensional summary statistic for use in [SBI](https://liamcpinchbeck.github.io/posts/2025/08/2025-08-11-SBI-w-NPE-NLE/) approaches.


# Core Ideas

UMAP is composed of two steps: the creation of a fuzzy fiducial simplicial map (will get to those words in a sec) and optimising for the coordinates in the lower dimensional space to be similar to this graph. 

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-10-StudentStressUMAP/general_steps.png" 
      alt="https://umap-learn.readthedocs.io/en/latest/parametric_umap.html" 
      title="https://umap-learn.readthedocs.io/en/latest/parametric_umap.html" 
      style="width: 80%; height: auto; border-radius: 8px;">
</div>



## Constructing the graphical representation of the data

For me the hardest part of the algorithm to understand was the construction of the initial graph. 


#### Constructing a simplicial complex between neighbours within radius

#### Constructing a simplicial complex between k-nearest neighbours

#### Constructing a new metric between k-nearest neighbours

#### Constructing a fuzzy simplicial complex between k-nearest neighbours



## Projecting the graph into the lower dimensional space








# Investigating the Student Stress Dataset

Before getting into the examples, the dataset is available here [mdsultanulislamovi/student-stress-monitoring-datasets](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets).


## 2D UMAP Projection

<div style="text-align: center;">
  <img 
      src="/files/BlogPostData/2025-10-StudentStressUMAP/2dUMAP_Projections/StressLevel2DUMAP_Projection.png" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 49%; height: auto; border-radius: 8px;">
</div>

<div style="text-align: center;">

  <img 
      src="/files/BlogPostData/2025-10-StudentStressUMAP/2dUMAP_Projections/AllIndicators_StressLevel2DUMAP_Projections.png" 
      alt="Nothing to see here." 
      title="Nothing to see here." 
      style="width: 99%; height: auto; border-radius: 8px;">
</div>


## 3D UMAP Projection

(Below plots are all interactive, and annoyingly I couldn't pick the initial camera position)


<iframe 
    src="/files/BlogPostData/2025-10-StudentStressUMAP/3DUMAP_Projections/stress_level_3D_UMAP.html" 
    width="80%" 
    height="400px"
    title="Embedded Content"
    style="border:none;"
    ></iframe>



<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">

    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/3DUMAP_Projections/blood_pressure_3D_UMAP.html" 
        width="49%" 
        height="400px" 
        style="border:none; margin-bottom: 20px;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>

    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/3DUMAP_Projections/sleep_quality_3D_UMAP.html" 
        width="49%" 
        height="400px" 
        style="border:none; margin-bottom: 20px;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>

    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/3DUMAP_Projections/extracurricular_activities_3D_UMAP.html" 
        width="49%" 
        height="400px" 
        style="border:none; margin-bottom: 20px;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>

    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/3DUMAP_Projections/mental_health_history_3D_UMAP.html" 
        width="49%" 
        height="400px" 
        style="border:none; margin-bottom: 20px;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>

</div>

# Conclusion
