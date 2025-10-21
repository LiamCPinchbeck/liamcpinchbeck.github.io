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
- [The mathematics of UMAP](https://adelejackson.com/files/Maths_of_UMAP.pdf) - [Adele Jackson](https://adelejackson.com/)
- [Wikipedia Homotopy Type Page](https://en.wikipedia.org/wiki/Homotopy)
- [Wikipedia page on the Čech complex](https://en.wikipedia.org/wiki/%C4%8Cech_complex)
- The series of [Melvin Leok](https://www.youtube.com/@melvinleok) videos/lectures/tutorials
    - [Homotopy type](https://www.youtube.com/watch?v=leG2KnK5PKo&list=PLHZhjPByiV3JLOtqsO_Bowj4DZIg5VP72&index=28&pp=iAQB)
    - [Nerve Theorem](https://www.youtube.com/watch?v=zDRjz8tVI1Y&list=PLHZhjPByiV3JLOtqsO_Bowj4DZIg5VP72&index=29)
    - [Čech complex](https://www.youtube.com/watch?v=zDRjz8tVI1Y&list=PLHZhjPByiV3JLOtqsO_Bowj4DZIg5VP72&index=30)
    - [Smallest Enclosing Ball](https://www.youtube.com/watch?v=zDRjz8tVI1Y&list=PLHZhjPByiV3JLOtqsO_Bowj4DZIg5VP72&index=31)
    - [Vietoris-Rips complex](https://www.youtube.com/watch?v=w1E-daFPS00&list=PLHZhjPByiV3JLOtqsO_Bowj4DZIg5VP72&index=32)
- [](https://giotto-ai.github.io/gtda-docs/0.5.1/notebooks/persistent_homology_graphs.html#id8)

## Table of Contents

- [Motivation](#motivationtraditional-autoencoders)
- [Core Idea](#core-idea)
- [Constructing the graphical representation of the data](#constructing-the-graphical-representation-of-the-data)
- [Optimising the graph in the lower dimensional space](#optimising-the-graph-in-the-lower-dimensional-space)
- [Student Stress Example](#investigating-the-student-stress-dataset)
- [Conclusion](#conclusion)

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

For me the hardest part of the algorithm to understand was the construction of the initial graph. And from the references you can see it's theoretical underpinnings are way more complex/rigorous than I may be able to properly convey. I'll try and break down the fundamental steps of the final algorithm from an intuitive perspective and then afterwards try and go through some of underpinning math.


#### Constructing a complex between neighbours within radius

The main idea behind UMAP is that we want to construct a graph in the higher dimensional space and reproduce a similar graph (specifically an approximate [Homotopy](https://en.wikipedia.org/wiki/Homotopy)) in the lower dimensional space of our choosing. In the case of high dimensions and large data though (pretty common in ML context like thousands of images) building connections between every single point and recording the vector distance between them would be very computationally difficult. 

So, we might want to limit the number of points we consider, by setting a parameter such as a max distance after which we don't care about any other points anymore. This is demonstrated in the interactive plots below.

<iframe 
    src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/sine_with_radii_and_lines.html" 
    width="99%" 
    height="600px" 
    style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
    sandbox="allow-scripts allow-pointer-lock allow-same-origin"
    allow="fullscreen"
></iframe>

<iframe 
    src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/swiss_roll_with_radii_and_lines.html" 
    width="99%" 
    height="800px" 
    style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
    sandbox="allow-scripts allow-pointer-lock allow-same-origin"
    allow="fullscreen"
></iframe>

You can see that both of these datasets' information/shape can be encoded on a simpler lower dimensional shape, in fancy math speak a lower dimensional [manifold](https://en.wikipedia.org/wiki/Topological_manifold). If unfamiliar with the concept this just refers to a shape or volume particularly in higher dimensions. In the sine wave case if the data didn't have noise it would be perfectly distributed according to the sine wave, and our manifold would be a straight line. So most of our 2D information could be equivalently represented in a 1-dimensional manner.

This however, has a few issues. One of the main ones is that we inherently assume that the points are distributed _uniformly_ about the manifold. e.g. With the concept of a fixed distance function, this is not inducive to common real world datasets. You can see this issue yourself by fixing the radii for the above figures and seeing that many points can be connected for a given radius, but often we introduce clumps when there really shouldn't be (e.g. sine at ~0.5 and swirl at ~2). And if we make the radius too large then too many points would be connected and we'd lose the sub-structures of our data (you can see this by just maxing out the radii for either plot). 

So either we have to really fine tune the value of this parameter or go about it in a different way.


#### Constructing a complex between k-nearest neighbours

So what we do is play with the concept of distance a little bit. We first instead ask what are the $$K$$ nearest neighbours to every point. You can see this demonstrated in the below figure.

<iframe 
    src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/swiss_roll_with_neighbours_single_point.html" 
    width="89%" 
    height="600px" 
    style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
    sandbox="allow-scripts allow-pointer-lock allow-same-origin"
    allow="fullscreen"
></iframe>

So we have a similar concept to a radius, where increasing it leads to more points being connected, but now under-dense regions can still be well connected because now we're just asking what points are closest. Leading to less clumps and less sensitivity to random noise. 

We can now construct a graph between the first $$K$$ points closest to any given point. You can play around with this in the below plot.

<iframe 
    src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/swiss_roll_with_neighbours.html" 
    width="89%" 
    height="600px" 
    style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
    sandbox="allow-scripts allow-pointer-lock allow-same-origin"
    allow="fullscreen"
></iframe>


#### Constructing a new metric between k-nearest neighbours

This is great, we've constructed a graph that is less sensitive to noise perturbations. But now also less sensitive to actual distance values. All of the $$K$$ nearest points are no different to any other contained point so we may still lose some important small scale structure. So, what we do is weight the connections of the $$K$$ nearest neighbours based on the distances between them and the reference point (that we used for the $$K$$-nearest neighbours). 

The specific formula for calculating what we can think of as modified distance is as follows,

$$\begin{align}
D(x_j \vert x_{\textrm{ref}}) = \begin{cases}
			-\frac{dist(x_j, x_{\textrm{ref}}) - \rho^{\textrm{min}}_\textrm{ref}}{\sigma_\textrm{ref}} , & \text{if $x_j$ is one of the K nearest neighbours of $x_{\textrm{ref}}$}\\
            \infty, & \text{otherwise}
		 \end{cases}
\end{align}$$

where $$\rho^{\textrm{min}}_\textrm{ref}$$ is the closest distance of the neighbours to $$x_{\textrm{ref}}$$. The weight of a given conditional connection is simply given by,

$$\begin{align}
p(x_j \vert x_{\textrm{ref}}) = \exp\left( - D(x_j \vert x_{\textrm{ref}})\right) .
\end{align}$$


The weights of the connections are represented like a conditional probability, which they aren't, but are close enough that it makes thinking of the situation easier. So 🤷. 

You can see what these conditional weights look like for different $$K$$ values below. The larger circle is the distance to the furtherest neighbour and the inner circle $$\rho^\textrm{min}_\textrm{ref}$$.


<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/swiss_roll_with_k_neighbour_metric_single_point.html" 
        width="89%" 
        height="600px" 
        style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>
</div>

You may have noticed that I didn't say what $$\sigma_\textrm{ref}$$ was. Strictly, it is a normalisation factor such that the following is satisfied,

$$\begin{align}
\log_2(K) &=\sum_j^K \exp\left(D(x_j \vert x_{\textrm{ref}}) \right) \\
&= \sum_j^K \exp\left(-\frac{dist(x_j, x_{\textrm{ref}}) - \rho^{\textrm{min}}_\textrm{ref}}{\sigma_\textrm{ref}}\right).
\end{align}$$

Where $$K$$ is still representing the choice of $$K$$ nearest neighbours. Specifically why the normalisation is the way it is eludes me. Intuitively, I think of it as purely a regularisation factor that accounts for the effective increase in radii as $$K$$ increases.

#### Constructing a fuzzy simplicial complex between k-nearest neighbours

The above just encodes the directional weights of the graph, but we actually an undirected graph where the points only have at most a single connection between them. For a given connection between $$x_i$$ and $$x_j$$, then the weight of the edge connecting them is,

$$\begin{align}
p(x_i, x_j) = p(x_i|x_j) + p(x_j|x_i) - p(x_i|x_j) \cdot p(x_j|x_i).
\end{align}$$

There's an anology between the weights and conditional probabilities here but I'll let you find that as I don't like actually treating the weights as probabilities.

This new graph now leads to an efficiently stored representation of the graph because we only store those $$K$$ nearest neighbours, but also still have a notion of distance or that points closer together are more important than those further away that is stupid quick and easy to calculate.

You can see these kind of graphs for $$\sigma=1$$ (I'm lazy and couldn't be bothered coding up the normalisation) below.

<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/sine_wave_weighted_graph.html" 
        width="89%" 
        height="600px" 
        style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>
    <iframe 
        src="/files/BlogPostData/2025-10-StudentStressUMAP/interactive_plots/swiss_roll_weighted_graph.html" 
        width="89%" 
        height="600px" 
        style="border:none; margin-bottom: 0px; margin-left: auto; margin-right: auto;"
        sandbox="allow-scripts allow-pointer-lock allow-same-origin"
        allow="fullscreen"
    ></iframe>
</div>

We then encode this information into an adjacency matrix that contains the weights between the points.  

$$\begin{align}
\begin{bmatrix}
0 & p(x_1, x_2) & p(x_1, x_3) & \cdots & p(x_1, x_{n-1}) & p(x_1, x_n)\\
p(x_2, x_1) & 0 & p(x_2, x_3) & \cdots & p(x_2, x_{n-1}) & p(x_2, x_n)\\
p(x_3, x_1) & p(x_3, x_2) & 0 & \cdots & p(x_3, x_{n-1}) & p(x_3, x_n)\\
\vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
p(x_{n-1}, x_1) & p(x_{n-1}, x_2) & p(x_{n-1}, x_3) & \cdots & 0 & p(x_{n-1}, x_{n})\\
p(x_n, x_1) & p(x_n, x_2) & p(x_{n}, x_3) & \cdots & p(x_{n}, x_{n-1}) & 0\\

\end{bmatrix}
\end{align}$$

The weights are symmetric though and the weights between points and themselves is 0, so we only need the lower triangle of information and the matrix will be pretty sparse as we will only care about the closest $$K$$ points to any given point.


#### A quick aside on the fundamental math going on




## Optimising the graph in the lower dimensional space








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
