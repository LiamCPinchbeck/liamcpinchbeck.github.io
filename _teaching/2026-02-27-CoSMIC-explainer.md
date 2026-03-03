---
title: "Trans-dimensional 2D GMM with VTI: a mathematical walkthrough"
date: 2026-02-27
collection: teaching
type: "Workshop"
location: "Melbourne, Australia"
permalink: /posts/teaching/2026-02-VTI-GMM/
tags:
  - Normalising Flows
  - VTI
  - Trans-dimensional inference
header-includes:
  - \usepackage{amsmath}
---

A deep walkthrough of [`gmm_2d_k100.py`](https://github.com/...), building up the full probabilistic model and VTI framework from scratch, with a mathematical explanation for every class and method. Equations are cross-referenced to Davies et al. (2025)[^vti].

## Table of Contents

- [Part 1: The probability model](#part-1-the-probability-model)
  - [The generative story](#the-generative-story)
  - [The transdimensional support](#the-transdimensional-support)
  - [What we want to compute](#what-we-want-to-compute)
  - [Why this is hard](#why-this-is-hard)
- [Part 2: VTI — dimension saturation and the CoSMIC factorisation](#part-2-vti--dimension-saturation-and-the-cosmic-factorisation)
  - [Dimension saturation and the augmented target](#dimension-saturation-and-the-augmented-target)
  - [The IAF variational density on saturated space](#the-iaf-variational-density-on-saturated-space)
  - [The CoSMIC construction: making the Jacobian block-diagonal](#the-cosmic-construction-making-the-jacobian-block-diagonal)
  - [The VTI loss function](#the-vti-loss-function)
- [Part 3: GMM2DDGP — building the joint](#part-3-gmm2ddgp--building-the-joint)
  - [\_generate\_data and the bounding box buffers](#_generate_data-and-the-bounding-box-buffers)
  - [num\_categories, num\_inputs, num\_context\_features](#num_categories-num_inputs-num_context_features)
  - [mk\_identifiers and mk\_cat\_to\_identifier](#mk_identifiers-and-mk_cat_to_identifier)
  - [mk\_to\_context](#mk_to_context)
  - [mk\_to\_mask: the context-to-mask map C(m)](#mk_to_mask-the-context-to-mask-map-cm)
  - [mk\_to\_component\_mask](#mk_to_component_mask)
  - [\_decode\_theta: the change-of-variables and its Jacobian](#_decode_theta-the-change-of-variables-and-its-jacobian)
  - [log\_prob: the four terms of the log joint](#log_prob-the-four-terms-of-the-log-joint)
- [Part 4: AbstractDGP — the framework contract](#part-4-abstractdgp--the-framework-contract)
  - [reference\_dist\_sample\_and\_log\_prob](#reference_dist_sample_and_log_prob)
  - [reference\_log\_prob: the auxiliary variable prior](#reference_log_prob-the-auxiliary-variable-prior)
  - [mk\_prior\_dist](#mk_prior_dist)
  - [construct\_param\_transform](#construct_param_transform)
- [Part 5: The CoSMIC flow — param\_transform in detail](#part-5-the-cosmic-flow--param_transform-in-detail)
  - [The left-right permutation Pm](#the-left-right-permutation-pm)
  - [The CoSMIC IAF step](#the-cosmic-iaf-step)
  - [The composite transform and its log-det](#the-composite-transform-and-its-log-det)
  - [flow.inverse vs flow.forward](#flowinverse-vs-flowforward)
- [Part 6: The surrogate — learning which models matter](#part-6-the-surrogate--learning-which-models-matter)
  - [The optimal model distribution q\*(m)](#the-optimal-model-distribution-qm)
  - [DiagonalGaussianSurrogate: GP surrogate as mean-field Gaussian](#diagonalgaussiansurrogate-gp-surrogate-as-mean-field-gaussian)
  - [observe: Gaussian conjugate update](#observe-gaussian-conjugate-update)
  - [evolve: variance inflation from Adam steps](#evolve-variance-inflation-from-adam-steps)
  - [utility\_UCB and utility\_Thompson](#utility_ucb-and-utility_thompson)
  - [SoftmaxSurrogateSampler: qu,t in code](#softmaxsurrogatesampler-qut-in-code)
  - [action\_dist, squish\_utility, and the double softmax](#action_dist-squish_utility-and-the-double-softmax)
  - [\_NormalisedSurrogate: scaling the ELBO observations](#_normalisedsurrogate-scaling-the-elbo-observations)
  - [\_clamp\_state: preventing posterior collapse](#_clamp_state-preventing-posterior-collapse)
- [Part 7: VTISurrogateEstimator — the training loop](#part-7-vtisurrogateestimator--the-training-loop)
  - [setup\_optimizer: AdamW and the cosine scheduler](#setup_optimizer-adamw-and-the-cosine-scheduler)
  - [loss\_and\_sample\_and\_log\_prob: assembling the full loss](#loss_and_sample_and_log_prob-assembling-the-full-loss)
  - [step: one iteration of SGD](#step-one-iteration-of-sgd)
  - [optimize: the full loop with callbacks](#optimize-the-full-loop-with-callbacks)
  - [Zero-init: why and when](#zero-init-why-and-when)
- [Part 8: After training — reading the posterior](#part-8-after-training--reading-the-posterior)
  - [The model posterior q(k)](#the-model-posterior-qk)
  - [Sampling the parameter posterior for model k](#sampling-the-parameter-posterior-for-model-k)
  - [Bayesian Model Averaging](#bayesian-model-averaging)
- [Adapting to other problems](#adapting-to-other-problems)
- [Numerical notes](#numerical-notes)

---

# Part 1: The probability model

## The generative story

We have $$N$$ points $$\mathbf{y}_1, \ldots, \mathbf{y}_N \in \mathbb{R}^2$$. We believe they came from a mixture of $$k$$ isotropic Gaussians with known standard deviation $$\sigma$$. The generative process is:

$$k \sim p(k), \quad \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_k \overset{\text{iid}}{\sim} \text{Uniform}([x_{\text{lo}}, x_{\text{hi}}] \times [y_{\text{lo}}, y_{\text{hi}}])$$

$$\mathbf{y}_i \mid k, \boldsymbol{\mu}_{1:k} \sim \frac{1}{k} \sum_{j=1}^k \mathcal{N}(\boldsymbol{\mu}_j, \sigma^2 \mathbf{I}_2)$$

We treat the number of components $$k$$ as unknown — a random variable with prior $$p(k)$$ over $$\{1, 2, \ldots, K_{\max}\}$$. We call each value of $$k$$ a *model*, and write $$m \equiv k$$ throughout to match the paper's notation. The parameter vector for model $$m$$ is $$\boldsymbol{\theta}_m = (\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_m) \in \mathbb{R}^{2m}$$. The log joint is:

$$\log p(\mathbf{y}, \boldsymbol{\theta}_m, m) = \log p(\mathbf{y} \mid \boldsymbol{\theta}_m, m) + \log p(\boldsymbol{\theta}_m \mid m) + \log p(m)$$

This is what `log_prob` returns.

## The transdimensional support

The paper (eq. 1) defines the transdimensional support as the disjoint union:

$$\mathcal{X} = \bigsqcup_{m \in \mathcal{M}} \left(\{m\} \times \Theta_m\right)$$

where $$\mathcal{M} = \{1, \ldots, K_{\max}\}$$ and $$\Theta_m = \mathbb{R}^{2m}$$. The dimension $$d_m = 2m$$ varies with $$m$$. This is what makes the problem transdimensional: there is no single fixed vector space that contains all $$\boldsymbol{\theta}_m$$ for all $$m$$ simultaneously. Standard variational inference, which fixes a single distribution over a single $$\mathbb{R}^d$$, cannot directly be applied here.

## What we want to compute

We want the joint posterior:

$$\pi(m, \boldsymbol{\theta}_m \mid \mathbf{y}) \propto p(\mathbf{y} \mid \boldsymbol{\theta}_m, m) \cdot p(\boldsymbol{\theta}_m \mid m) \cdot p(m)$$

and in particular the marginal model posterior:

$$\pi(m \mid \mathbf{y}) = \int_{\Theta_m} \pi(m, \boldsymbol{\theta}_m \mid \mathbf{y}) \, d\boldsymbol{\theta}_m$$

The target density factorises as (paper: $$\eta(m, \theta_m) = \eta(\theta_m \mid m)\pi(m)$$) with unnormalized conditional $$\eta(\boldsymbol{\theta}_m \mid m) = Z_m \pi(\boldsymbol{\theta}_m \mid m)$$, where $$Z_m = \int_{\Theta_m} \eta(\boldsymbol{\theta}_m \mid m) d\boldsymbol{\theta}_m$$. VTI estimates both $$\eta(\boldsymbol{\theta}_m \mid m)$$ and $$\pi(m)$$ simultaneously.

## Why this is hard

**Different parameter dimensions.** Model $$m=3$$ has $$d_3 = 6$$ parameters; model $$m=10$$ has $$d_{10} = 20$$. A single normalising flow $$T_\phi: \mathbb{R}^d \to \mathbb{R}^d$$ has a fixed dimension $$d$$. Naively, you'd need $$K_{\max}$$ separate flows. VTI avoids this with dimension saturation (Part 2).

**Label switching.** Any permutation of $$\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_m$$ gives identical likelihood. The $$m!$$ equivalent labellings produce a multimodal posterior inside each model; a flow must either represent all modes or break symmetry.

**Non-stationarity.** The optimal model distribution $$q^*(m)$$ depends on the current flow parameters $$\phi$$. As $$\phi$$ changes during training, so does which models are "good". Estimating $$q(m)$$ and the flow simultaneously creates a coupled, non-stationary optimisation (paper section 3.4).

---

# Part 2: VTI — dimension saturation and the CoSMIC factorisation

## Dimension saturation and the augmented target

To handle varying dimensions, VTI adopts the dimension saturation approach of Brooks et al. (2003), cited as ref [5] in the paper. Let $$d_{\max} = \max_m d_m = 2 K_{\max}$$. For model $$m$$ with $$d_m < d_{\max}$$, introduce auxiliary variables $$\mathbf{u}_{\setminus m} \in \mathbb{R}^{d_{\max} - d_m}$$ drawn from the reference distribution $$\nu_{\setminus m}$$ (standard normal in each coordinate). The *saturated* parameter vector $$(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m}) \in \mathbb{R}^{d_{\max}}$$ has the same dimension for all models. The augmented unnormalized target density is (paper eq. 3):

$$\tilde{\eta}(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m} \mid m) = \eta(\boldsymbol{\theta}_m \mid m) \cdot \nu_{\setminus m}(\mathbf{u}_{\setminus m})$$

This is the joint density of "real" parameters and auxiliary variables. The auxiliary part $$\nu_{\setminus m}(\mathbf{u}_{\setminus m})$$ is a product of standard normal densities — in code this is `reference_log_prob`, evaluated only on the inactive slots. The augmentation is exact: marginalising out $$\mathbf{u}_{\setminus m}$$ recovers $$\eta(\boldsymbol{\theta}_m \mid m)$$ exactly. The key insight is that $$\mathbf{u}_{\setminus m}$$ should ideally be distributed as $$\nu_{\setminus m}$$ in the variational approximation, which is what the reference prior enforces.

## The IAF variational density on saturated space

The variational family (paper eq. 4–5) is an IAF on $$\mathbb{R}^{d_{\max}}$$ conditioned on $$m$$:

$$\tilde{q}_\phi(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m} \mid m) := \nu_{d_{\max}}(z) \cdot \vert\det \nabla T_\phi(z \mid m)\vert^{-1}$$

where $$\mathbf{z} \sim \nu_{d_{\max}} = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_{\max}})$$ and $$T_\phi(\mathbf{z} \mid m)$$ is the CoSMIC flow (section 2 and 5 below). The change-of-variables formula gives the density at $$(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m})$$ in terms of the reference density at the pre-image $$\mathbf{z} = T_\phi^{-1}(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m} \mid m)$$. The term $$\vert\det \nabla T_\phi(z \mid m)\vert^{-1}$$ is the inverse Jacobian, which appears because we evaluate the density at $$\boldsymbol{\theta}$$ (the output) while the reference density lives at $$\mathbf{z}$$ (the input).

In code: `log q(θ\vert k) = base_log_prob - log_det`, where `base_log_prob` = $$\log \nu_{d_{\max}}(\mathbf{z})$$ and `log_det` = $$\log\vert\det \nabla T_\phi(\mathbf{z} \mid m)\vert$$.

## The CoSMIC construction: making the Jacobian block-diagonal

The paper proves (Lemma 2.1, Proposition 2.2) that if the flow is constructed via the CoSMIC masking mechanism, the saturated density factorises as (paper eq. 6):

$$\tilde{q}_\phi(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m} \mid m) = q_\phi(\boldsymbol{\theta}_m \mid m) \cdot \nu_{d_{\setminus m}}(\mathbf{u}_{\setminus m})$$

This is the key result. It says that the CoSMIC flow simultaneously approximates $$q_\phi(\boldsymbol{\theta}_m \mid m)$$ for the active dimensions *and* leaves the auxiliary dimensions as $$\nu$$ (standard normal), matching the augmented target exactly on those dimensions. The variational density factorises correctly.

The construction relies on two ingredients. First, the context-to-mask map $$C(m)$$ (paper eq. 7):

$$C(m) := (C_1(m), \ldots, C_{d_{\max}}(m)) \in \{0,1\}^{\vert\rho\vert}$$

For each dimension $$i$$, $$A_i: \mathcal{M} \to \{0,1\}$$ indicates whether coordinate $$i$$ is active in model $$m$$. $$B_i$$ broadcasts this bit to the full parameter block $$\vert\rho_i\vert$$ of the $$i$$-th autoregressive transform. The composition $$C_i = B_i \circ A_i$$ selects which NN outputs to use (active) versus which to replace with the identity point $$\rho^{\text{Id}}$$ (inactive). In code, $$A(m)$$ is `mk_to_mask` and $$C(m)$$ is what gets passed to each coupling layer's `context_to_mask` argument.

Second, for each IAF step with bijection $$\tau_{\rho_i}: \mathbb{R} \to \mathbb{R}$$, the CoSMIC masking (paper eq. 8) linearly interpolates between the learned parameters $$\rho_i$$ and the identity point $$\rho^{\text{Id}}$$:

$$\rho_i^C = (1 - C_i(m)) \cdot \rho^{\text{Id}} + C_i(m) \cdot \rho_i$$

When $$C_i(m) = 0$$ (inactive), $$\rho_i^C = \rho^{\text{Id}}$$ and $$\tau_{\rho^{\text{Id}}}(z^{(i)}) = z^{(i)}$$ — exact identity, zero log-det contribution. When $$C_i(m) = 1$$ (active), $$\rho_i^C = \rho_i$$ and the full learned transform applies.

Lemma 2.1 then shows $$\mathbf{u}_{\setminus m} = \mathbf{z}_{\setminus m}$$ (the inactive outputs equal the inactive inputs — the identity is preserved). Proposition 2.2 shows that the left-right permutation $$P_m$$ (placing active dimensions before inactive ones) makes the Jacobian block-diagonal, so $$\vert\det \nabla T_\phi(\mathbf{z} \mid m)\vert$$ depends only on the active coordinates. This is what makes the density factorisation in eq. 6 exact.

## The VTI loss function

By Corollary 2.3 (paper eq. 9–11), the dimension saturation cancels out of the ELBO:

$$\frac{\nu_{d_{\max}}(\mathbf{z}) \cdot \vert\det \nabla T_\phi(\mathbf{z} \mid m)\vert^{-1}}{\tilde{\eta}(T_\phi(\mathbf{z} \mid m) \mid m)} = \underbrace{\frac{\nu_{d_m}(\mathbf{z}_m) \cdot \vert\det \nabla T_\phi(\mathbf{z} \mid m)\vert^{-1}}{\eta(\boldsymbol{\theta}_m \mid m)}}_{:= h_\phi(\mathbf{z} \mid m)}$$

The auxiliary $$\nu_{\setminus m}(\mathbf{z}_{\setminus m})$$ terms cancel exactly between the numerator (from $$\nu_{d_{\max}}(\mathbf{z}) = \nu_{d_m}(\mathbf{z}_m) \nu_{d_{\setminus m}}(\mathbf{z}_{\setminus m})$$) and the denominator (from $$\tilde{\eta} = \eta \cdot \nu_{\setminus m}$$). This means in practice we only need to evaluate the reference density $$\nu_{d_m}$$ on the *active* dimensions, which is what `reference_log_prob` does (it evaluates $$\nu_{\setminus m}$$ on the *inactive* dimensions to match the augmented target, not the active reference).

The full VTI loss (paper eq. 10) is:

$$L(\psi, \phi) = \mathbb{E}_{m \sim q_\psi}\!\left[\ell(m;\phi) - \log p(m) + \log q_\psi(m)\right]$$

where $$\ell(m; \phi) := \mathbb{E}_{\mathbf{z} \sim \nu_{d_{\max}}}[\log h_\phi(\mathbf{z} \mid m)]$$ (paper eq. 11). Substituting $$h_\phi$$:

$$\ell(m; \phi) = \mathbb{E}_\mathbf{z}\!\left[\log \nu_{d_m}(\mathbf{z}_m) - \log\vert\det \nabla T_\phi(\mathbf{z} \mid m)\vert - \log \eta(\boldsymbol{\theta}_m \mid m)\right]$$

In code: `ell = base_log_prob_active - log_det - log_p`, which is the negative ELBO per sample. The three terms:

- $$\log \nu_{d_m}(\mathbf{z}_m)$$: reference log-density, `base_log_prob` restricted to active dims
- $$\log\vert\det \nabla T_\phi\vert$$: log-det of the flow Jacobian, `params_tf_log_prob`
- $$\log \eta(\boldsymbol{\theta}_m \mid m)$$: the unnormalised log-joint returned by `dgp.log_prob`

---

# Part 3: GMM2DDGP — building the joint

`GMM2DDGP(AbstractDGP)` specifies the entire probability model: the data, the parameterisation, and the log-joint $$\log \eta(\boldsymbol{\theta}_m \mid m)$$. Everything in this part is your code when you have a new problem.

**Constructor parameters:**

| Argument | Type | Description |
|---|---|---|
| `seed` | `int` | RNG seed for data generation |
| `num_data` | `int` | $$N$$ — number of data points |
| `max_components` | `int` | $$K_{\max}$$ — size of model space |
| `true_means` | `list[(float, float)]` | True component means (only for data generation) |
| `sigma` | `float` | Known noise std $$\sigma$$ |
| `complexity_penalty` | `float` | BIC prefactor $$\lambda$$ |
| `device`, `dtype` | — | Passed to `AbstractDGP` |

**Registered buffers** (move with `.to(device)`, excluded from gradient graph):

| Buffer | Shape | Role in the calculation |
|---|---|---|
| `y_data` | `[N, 2]` | The observed data $$\mathbf{y}$$; enters the likelihood |
| `x_lo`, `x_hi` | scalar | Bounds of the uniform prior on $$\mu_x$$; define the sigmoid range |
| `y_lo`, `y_hi` | scalar | Bounds of the uniform prior on $$\mu_y$$ |

## `_generate_data` and the bounding box buffers

```python
def _generate_data(self, seed, num_data, true_means)
```

Samples $$N$$ data points from the true GMM and registers the bounding box. The bounding box is the axis-aligned rectangle containing all data points, inflated by 20% on each side:

$$x_{\text{lo}} = \min_j y_{j,x} - 0.2(x_{\text{hi}}' - x_{\text{lo}}'), \quad x_{\text{hi}} = \max_j y_{j,x} + 0.2(x_{\text{hi}}' - x_{\text{lo}}')$$

These buffers define the support of the uniform prior $$p(\boldsymbol{\mu}_i \mid m) = \text{Uniform}([x_{\text{lo}}, x_{\text{hi}}] \times [y_{\text{lo}}, y_{\text{hi}}])$$. They also define the range of the sigmoid transform applied in `_decode_theta`. All four scalars are `register_buffer` so they move to GPU with `.to(device)` and are not treated as learnable parameters.

## `num_categories`, `num_inputs`, `num_context_features`

These three methods provide the shape information the framework needs to size the flow, surrogate, and coupling MLPs. They must be consistent with each other.

`num_categories() → int` returns $$K_{\max}$$. This is $$\vert\mathcal{M}\vert$$, the cardinality of the model space. The surrogate maintains one Gaussian per category; the sampler has $$K_{\max}$$ logits; `mk_identifiers` returns a $$K_{\max} \times K_{\max}$$ matrix.

`num_inputs() → int` returns $$2K_{\max} = d_{\max}$$. This is the dimension of the saturated parameter vector $$(\boldsymbol{\theta}_m, \mathbf{u}_{\setminus m}) \in \mathbb{R}^{d_{\max}}$$. It determines the input/output size of the flow $$T_\phi: \mathbb{R}^{d_{\max}} \to \mathbb{R}^{d_{\max}}$$, the reference distribution $$\nu_{d_{\max}} = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_{\max}})$$, and the batch dimension of $$\mathbf{z}$$ and $$\boldsymbol{\theta}$$ tensors. For model $$m$$, only the first $$2m$$ dimensions are "real" parameters; the remaining $$2(K_{\max} - m)$$ are auxiliary $$\mathbf{u}_{\setminus m}$$.

`num_context_features() → int` returns $$2K_{\max}$$. This is the width of the context vector $$\xi$$ fed to the autoregressive network $$\text{NN}_\phi(\boldsymbol{\theta}_{\setminus i}; \xi)$$ at each coupling layer. It must match the output width of `mk_to_context`. The doubled one-hot $$\xi = [\mathbf{e}_m, \mathbf{e}_m]$$ gives each MLP 2K features to learn from, which empirically helps the MLPs condition reliably on $$m$$.

## `mk_identifiers` and `mk_cat_to_identifier`

`mk_identifiers() → [K, K]` returns the $$K_{\max} \times K_{\max}$$ identity matrix. Each row $$\mathbf{e}_m$$ is the one-hot identifier for model $$m$$. VTI uses this to iterate over the full model space, e.g. when extracting posterior model probabilities after training.

`mk_cat_to_identifier([N] int) → [N, K] one-hot` converts integer indices to one-hot vectors. The one-hot is the canonical form of the model identifier throughout VTI. Integer indices go into `Categorical.sample`, but everywhere the flow, mask, or context are computed, the identifier must be one-hot so that `mk_to_context` and `mk_to_mask` can compute a batched operation over the $$K_{\max}$$ slots.

## `mk_to_context`

```python
def mk_to_context(mk_samples) → [N, 2K]
```

Builds the context vector $$\xi$$ for the autoregressive network. In our implementation:

$$\xi(m) = [\mathbf{e}_m; \mathbf{e}_m] \in \{0,1\}^{2K_{\max}}$$

the one-hot repeated twice. This is the $$\xi$$ in $$\text{NN}_\phi(\boldsymbol{\theta}_{\setminus i}; \xi)$$. The MLP uses it to learn the coupling parameters $$\rho_i(m)$$ separately for each model; the doubled context gives the network extra bandwidth to condition on $$m$$ reliably for large $$K_{\max}$$.

The method is polymorphic over three input types because VTI calls it at different stages:

- `scalar` (int): called once in the training loop during setup — `unsqueeze → one-hot`
- `[N] int`: the main training call — `F.one_hot` directly
- `[N, K] one-hot`: pass-through when the identifier is already computed

## `mk_to_mask`: the context-to-mask map $$C(m)$$

```python
def mk_to_mask(mk) → [N, 2K]
```

This implements the global context-to-mask map $$C(m) \in \{0,1\}^{\vert\rho\vert}$$ from paper eq. 7. It determines, for each dimension $$i \in \{1, \ldots, d_{\max}\}$$, whether the CoSMIC masking formula (eq. 8) uses the learned transform $$\rho_i$$ (active, $$C_i(m) = 1$$) or the identity point $$\rho^{\text{Id}}$$ (inactive, $$C_i(m) = 0$$).

For model $$m = k$$ ($$k$$ components), the active dimensions are $$\{1, \ldots, 2k\}$$ — the $$k$$ $$x$$-coordinates and $$k$$ $$y$$-coordinates. In our flattened layout the first $$K$$ slots hold $$\mu_x$$-coordinates and the next $$K$$ slots hold $$\mu_y$$-coordinates, so:

```python
# Model k=3, K=5 → [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
slot_mask = (arange(K) <= k[:,None])   # first k slots of each block active
return slot_mask.repeat(1, 2)          # [N, 2K]
```

The mask has two roles:

1. **At each CoSMIC coupling layer**: passed as `context_to_mask` to determine $$\rho_i^C$$ in eq. 8, making the transform identity for inactive dimensions.
2. **In `reference_log_prob`**: the complement $$1 - C(m)$$ identifies inactive dimensions; their log-density under $$\nu$$ is computed and added to `log_prob` as the auxiliary variable term.

## `mk_to_component_mask`

```python
def mk_to_component_mask(mk) → [N, K]
```

A GMM-specific convenience: returns a binary mask over the $$K$$ component *slots* (not dimensions) — `[1, 1, 1, 0, 0]` for $$k=3$$, $$K=5$$. This is the single-block version of `mk_to_mask`, used inside `log_prob` to:

- Sum the parameter prior only over active components
- Sum the Jacobian correction only over active components

It is not part of the AbstractDGP interface and is not called by the framework.

## `_decode_theta`: the change-of-variables and its Jacobian

```python
def _decode_theta(theta: [N, 2K]) → (mu_x: [N, K], mu_y: [N, K])
```

The flow operates in unconstrained space $$\boldsymbol{\theta} \in \mathbb{R}^{2K}$$, but the component means $$\boldsymbol{\mu}$$ should lie inside the data bounding box. `_decode_theta` applies a sigmoid change-of-variables:

$$\mu_{x,i} = x_{\text{lo}} + (x_{\text{hi}} - x_{\text{lo}}) \cdot \sigma(\theta_i), \quad \mu_{y,i} = y_{\text{lo}} + (y_{\text{hi}} - y_{\text{lo}}) \cdot \sigma(\theta_{K+i})$$

where $$\sigma(x) = 1/(1 + e^{-x})$$. Writing $$w_x = x_{\text{hi}} - x_{\text{lo}}$$, this defines a bijection $$\boldsymbol{\theta} \mapsto \boldsymbol{\mu}$$ with diagonal Jacobian. Since the transform is applied coordinate-by-coordinate, the Jacobian matrix is diagonal and the log-det factors as a sum:

$$\log\vert J_i^x\vert = \log\!\left(w_x \cdot \sigma(\theta_i)(1 - \sigma(\theta_i))\right)$$

$$\log\vert J_i\vert := \log\vert J_i^x\vert + \log\vert J_i^y\vert$$

This is stashed in `self._last_log_jac` (shape `[N, K]`) for immediate use in `log_prob`. The `.clamp(min=1e-30)` before `log` prevents `log(0)` when $$\sigma \to 0$$ or $$\sigma \to 1$$ at the sigmoid boundary.

**Why we bother with the Jacobian:** `log_prob` is the unnormalized log-joint in $$\boldsymbol{\theta}$$-space (the space the flow operates in). The uniform prior $$p(\boldsymbol{\mu} \mid m)$$ is a density in $$\boldsymbol{\mu}$$-space. When changing variables from $$\boldsymbol{\mu}$$ to $$\boldsymbol{\theta}$$, the density transforms as:

$$p(\boldsymbol{\theta} \mid m) = p(\boldsymbol{\mu}(\boldsymbol{\theta}) \mid m) \cdot \left\vert\frac{d\boldsymbol{\mu}}{d\boldsymbol{\theta}}\right\vert = \frac{1}{w_x w_y} \cdot \prod_{i=1}^m \vert J_i\vert$$

The Jacobian $$\vert J_i\vert$$ is the correction that makes the prior in $$\boldsymbol{\theta}$$-space consistent with the original uniform prior in $$\boldsymbol{\mu}$$-space. Without it, `log_prob` would be computing the wrong density, and the ELBO would be biased.

## `log_prob`: the four terms of the log joint

```python
def log_prob(mk, theta) → [N]
```

Returns $$\log \eta(\boldsymbol{\theta}_m \mid m) = \log p(\mathbf{y} \mid \boldsymbol{\theta}_m, m) + \log p(\boldsymbol{\theta}_m \mid m) + \log p(m) + \log \nu_{\setminus m}(\boldsymbol{\theta}_{\setminus m})$$. This is the numerator of $$h_\phi(\mathbf{z} \mid m)$$ (paper eq. 9), evaluated at the flow output $$\boldsymbol{\theta} = T_\phi(\mathbf{z} \mid m)$$.

**Term 1. Data log-likelihood** — $$\log p(\mathbf{y} \mid \boldsymbol{\theta}_m, m)$$

$$\log p(\mathbf{y} \mid \boldsymbol{\theta}_m, m) = \sum_{j=1}^{N} \log \left(\frac{1}{m} \sum_{i=1}^{m} \mathcal{N}(\mathbf{y}_j; \boldsymbol{\mu}_i, \sigma^2 \mathbf{I})\right)$$

Evaluated as `logsumexp` over components minus `log(m)`. This is $$O(N \cdot m)$$ and grows linearly with $$N$$, which is why raw ELBO values are large and the surrogate must be normalised. The outer sum over $$j$$ is accumulated in a float64 scalar to prevent precision loss — see [numerical notes](#numerical-notes).

**Term 2. Parameter prior** — $$\log p(\boldsymbol{\theta}_m \mid m)$$

The prior $$p(\boldsymbol{\mu}_i \mid m) = \text{Uniform}(\mathcal{B})$$ where $$\mathcal{B} = [x_{\text{lo}}, x_{\text{hi}}] \times [y_{\text{lo}}, y_{\text{hi}}]$$. In $$\boldsymbol{\mu}$$-space this is $$-\log(w_x w_y)$$ per active component. After the change of variables:

$$\log p(\boldsymbol{\theta}_m \mid m) = \sum_{i=1}^{m} \left[-\log(w_x w_y) + \log\vert J_i\vert\right]$$

where the sum runs only over active components (enforced by multiplying with `cmask` before summing, so inactive components contribute 0). The $$\log\vert J_i\vert $$ are `self._last_log_jac` from `_decode_theta`.

**Term 3. Model prior** — $$\log p(m)$$

We use a BIC-style complexity penalty (Schwarz criterion):

$$\log p(m) = -\lambda \cdot \tfrac{1}{2}\log N \cdot (m - 1), \qquad \lambda = \texttt{complexity\_penalty}$$

Each additional component beyond the first costs $$\lambda \cdot \tfrac{1}{2}\log N$$ nats. With $$N = 4000$$ and $$\lambda = 2$$: each extra component costs $$2 \cdot \tfrac{1}{2}\log 4000 \approx 8.3$$ nats. Without this term the posterior concentrates on $$m = K_{\max}$$ because additional components always reduce the likelihood in-sample. The BIC prefactor $$\tfrac{1}{2}\log N$$ comes from the asymptotic penalty for adding a free parameter in a model with $$N$$ observations.

**Term 4. Reference log-probability (auxiliary prior)** — $$\log \nu_{\setminus m}(\boldsymbol{\theta}_{\setminus m})$$

```python
reference_lp = self.reference_log_prob(mk, theta)
```

This term corresponds to $$\log \nu_{\setminus m}(\mathbf{u}_{\setminus m})$$ from the augmented target (paper eq. 3). It evaluates $$\log \mathcal{N}(\boldsymbol{\theta}_{\text{inactive}}; \mathbf{0}, \mathbf{I})$$ for the *inactive* parameter slots — those where `mk_to_mask = 0`. By Lemma 2.1, the CoSMIC flow maps $$\mathbf{z}_{\setminus m} \mapsto \mathbf{u}_{\setminus m} = \mathbf{z}_{\setminus m}$$ (identity), so ideally $$\boldsymbol{\theta}_{\text{inactive}} = \mathbf{z}_{\text{inactive}} \sim \nu$$. Including this term in `log_prob` ensures that the ELBO penalises the flow for moving inactive dimensions away from the reference distribution. Without it, inactive dimensions have no likelihood signal and drift freely, wasting flow capacity.

The method is provided by `AbstractDGP` and simply evaluates:

$$\log \nu_{\setminus m}(\boldsymbol{\theta}_{\text{inactive}}) = -\tfrac{1}{2}(d_{\max} - d_m)\log(2\pi) - \tfrac{1}{2}\|\boldsymbol{\theta}_{\text{inactive}}\|^2$$

using `1 - mk_to_mask(mk)` to select the inactive dimensions.

---

# Part 4: AbstractDGP — the framework contract

`AbstractDGP(nn.Module)` is the abstract base class for all VTI probability models. It defines the interface that `VTISurrogateEstimator` and the flow builder require, and provides default implementations for several helper methods.

## `reference_dist_sample_and_log_prob`

```python
def reference_dist_sample_and_log_prob(batch_size) → (z: [B, D], log_p: [B])
```

Samples $$\mathbf{z}^{(1)}, \ldots, \mathbf{z}^{(B)} \overset{\text{iid}}{\sim} \nu_{d_{\max}} = \mathcal{N}(\mathbf{0}, \mathbf{I}_{d_{\max}})$$ and returns both the samples and their log-densities:

$$\log \nu_{d_{\max}}(\mathbf{z}) = -\tfrac{d_{\max}}{2}\log(2\pi) - \tfrac{1}{2}\|\mathbf{z}\|^2$$

This is $$\log \nu_{d_{\max}}(\mathbf{z})$$ from paper eq. 5 — the reference density at the pre-image of $$\boldsymbol{\theta}$$ under the flow. It enters the ELBO as `base_log_prob` in the expression `log q(θ\vert k) = base_log_prob - log_det`. The dimension here is `num_inputs()` = $$2K_{\max}$$.

## `reference_log_prob`: the auxiliary variable prior

```python
def reference_log_prob(mk, theta) → [N]
```

Evaluates $$\log \nu_{\setminus m}(\mathbf{u}_{\setminus m})$$ from paper eq. 3 — the reference density restricted to *inactive* dimensions. Uses `1 - mk_to_mask(mk)` to identify which of the $$2K_{\max}$$ slots are inactive for each model in the batch, then:

$$\log \nu_{\setminus m}(\boldsymbol{\theta}_{\text{inactive}}) = \sum_{i: C_i(m) = 0} \log \mathcal{N}(\theta_i; 0, 1)$$

This is called from `log_prob` as the fourth term of the log-joint. As explained in Part 2, its inclusion ensures the augmented target $$\tilde{\eta}$$ cancels correctly in the ELBO, so we are effectively training toward the true posterior $$\pi(\boldsymbol{\theta}_m \mid m)$$ rather than the saturated version.

## `mk_prior_dist`

```python
def mk_prior_dist() → Categorical
```

Returns $$p(m) = \text{Categorical}(1/K_{\max}, \ldots, 1/K_{\max})$$ — a uniform distribution over models. This is the prior $$p(m)$$ from paper eq. 10. It appears in two places in the training loop:

1. `log_prior_mk = prior_mk_dist.log_prob(mk)` — the $$\log p(m)$$ term in the VTI loss
2. `mk_prior_log_prob` is negated and added to the loss to form `loss_hat2 = -log_prior_mk + log_q_mk`, which is the KL $$\text{KL}(q_\psi \| p)$$ between the sampler and the prior

## `construct_param_transform`

```python
def construct_param_transform(flow_type: str) → CompositeTransform
```

Builds the CoSMIC flow $$T_\phi$$ by calling `param_transform_factory.construct_param_transform` with:

- `num_inputs = self.num_inputs()` = $$d_{\max}$$
- `num_context_inputs = self.num_context_features()`
- `context_to_mask = self.mk_to_mask` — the context-to-mask map $$C(m)$$
- `context_transform = self.mk_to_context` — the context embedding $$\xi(m)$$

The factory wraps these functions as callables that each CoSMIC layer invokes at runtime. This is how the flow learns model-conditional posteriors: at each forward/inverse pass, the coupling layers call `mk_to_mask(m)` to get the current mask and `mk_to_context(m)` to get the context for the NN.

---

# Part 5: The CoSMIC flow — `param_transform` in detail

The normalising flow $$T_\phi = T_{\phi,L} \circ \cdots \circ T_{\phi,1}$$ is a composition of $$L$$ CoSMIC IAF steps, bookended by left-right permutations (Figure 1(a) in the paper).

## The left-right permutation $$P_m$$

Proposition 2.2 requires a permutation $$P_m$$ that places active dimensions before inactive ones. In code this is `StrictLeftPermutation(num_inputs, context_to_mask=mk_to_mask)`. The permutation depends on $$m$$ and is applied at the start:

$$\mathbf{z}' = P_m \mathbf{z} = [\mathbf{z}_m ; \mathbf{z}_{\setminus m}]$$

After this permutation, the active dimensions occupy the first $$d_m$$ positions. The paper proves (Prop. 2.2) that with this permutation, the full Jacobian $$\nabla T_\phi(\mathbf{z} \mid m)$$ is block-triangular, so its determinant factors as:

$$\det \nabla T_\phi(\mathbf{z} \mid m) = \det \nabla T_\phi^{(m)}(\mathbf{z}_m) \cdot \det I_{d_{\max}-d_m} = \det \nabla T_\phi^{(m)}(\mathbf{z}_m)$$

The inactive block contributes $$\det I = 1$$ to the log-det (i.e. 0 log contribution), matching Lemma 2.1. `InverseTransform(StrictLeftPermutation)` undoes this permutation at the end of the composition, so the overall transform $$T_\phi = P_m^{-1} \circ (T_{\phi,L} \circ \cdots \circ T_{\phi,1}) \circ P_m$$ maps $$\mathbb{R}^{d_{\max}} \to \mathbb{R}^{d_{\max}}$$ with the original dimension ordering.

Between each IAF step, `PartialReversePermutation` reverses only the *active* part of the vector (the first $$d_m$$ positions), while leaving inactive positions unchanged. This plays the role of the `ReversePermutation` used in standard MAF — it ensures each coupling layer conditions on a different set of preceding dimensions — but it respects the active/inactive split.

## The CoSMIC IAF step

Each step $$T_{\phi,\ell}$$ is a `CoSMICMaskedAffineAutoregressiveTransform` wrapped in `InverseTransform(...)`. For an affine transform, the univariate bijection is:

$$\theta^{(i)} = \tau_{\rho_i^C}(z^{(i)}) = \rho_i^{C,(0)} + \rho_i^{C,(1)} \cdot z^{(i)}$$

where the parameters are computed by the autoregressive network:

$$\rho_i = \text{NN}_\phi(\theta^{(1)}, \ldots, \theta^{(i-1)}; \xi(m)), \qquad \rho^{\text{Id}} = (0, 1)$$

$$\rho_i^C = (1 - C_i(m)) \cdot \rho^{\text{Id}} + C_i(m) \cdot \rho_i = \begin{cases} \rho_i & \text{if } C_i(m) = 1 \text{ (active)} \\ (0, 1) & \text{if } C_i(m) = 0 \text{ (inactive)} \end{cases}$$

For an active dimension, $$\theta^{(i)} = \rho_i^{(0)} + \rho_i^{(1)} z^{(i)}$$ with log-det $$\log\vert \rho_i^{(1)}\vert $$. For an inactive dimension, $$\theta^{(i)} = 0 + 1 \cdot z^{(i)} = z^{(i)}$$ with log-det $$\log\vert 1\vert  = 0$$. The log-det of the full composition is thus the sum over active dimensions only:

$$\log\vert \det \nabla T_\phi(\mathbf{z} \mid m)\vert  = \sum_{\ell=1}^{L} \sum_{i: C_i(m)=1} \log\vert \rho_{\ell,i}^{(1)}\vert $$

This quantity is what `params_tf_log_prob` holds in the training loop.

For `affine55`, the architecture is 5 + 5 = 10 MAAT layers, each with 2 ResNet blocks. The hidden feature size defaults to `num_pmaf_hidden_features`, which scales with `num_inputs`. This gives roughly 1500 terms in the above log-det sum for $$K_{\max} = 100$$ — why float64 is needed.

## The composite transform and its log-det

The full `param_transform` is a `CompositeTransform` wrapping the sequence:

```
StrictLeftPermutation(Pm)
  [PartialReversePermutation + InverseTransform(CoSMICMaskedAffineAutoregressiveTransform)] × 5
  [PartialReversePermutation + InverseTransform(CoSMICMaskedAffineAutoregressiveTransform)] × 5  ← leaky_relu
InverseTransform(StrictLeftPermutation)
```

The permutations have trivial Jacobians (orthogonal matrices, $$\vert \det P\vert  = 1$$, log-det $$= 0$$). Only the affine coupling layers contribute to `params_tf_log_prob`.

## `flow.inverse` vs `flow.forward`

VTI uses an IAF (Inverse Autoregressive Flow) where generation is the fast direction:

`flow.inverse(z, context=ctx) → (theta, log_det)`: maps $$\mathbf{z} \to \boldsymbol{\theta}$$, i.e. reference → parameter space. Each coordinate is computed sequentially from the autoregressive network — $$O(d_{\max})$$ total. Returns `log_det` = $$\log\vert \det \nabla T_\phi(\mathbf{z} \mid m)\vert $$. This is used during training.

`flow.forward(theta, context=ctx) → (z, log_det)`: maps $$\boldsymbol{\theta} \to \mathbf{z}$$. For an IAF, each $$z^{(i)}$$ depends on $$\theta^{(i)}$$ and the autoregressive network output computed from $$\theta^{(1)}, \ldots, \theta^{(i-1)}$$, so evaluation requires $$d_{\max}$$ sequential NN calls — $$O(d_{\max}^2)$$ total. Not used during VTI training.

The naming (`.inverse` for generation, `.forward` for density evaluation) follows the normalising flows convention where "forward" means density evaluation direction. Here $$z = T_\phi(\theta)$$ is the "forward" map (normalizing flow perspective), and $$\theta = T_\phi^{-1}(z)$$ = `.inverse(z)` is generation.

---

# Part 6: The surrogate — learning which models matter

## The optimal model distribution $$q^*(m)$$

The VTI loss (paper eq. 12) can be rewritten as a max-entropy objective over $$q_\psi$$:

$$\max_{q_\psi} \, \mathbb{E}_{m \sim q_\psi}[-\ell(m;\phi) + \log p(m)] + H[q_\psi]$$

This is a free energy maximisation. If we optimise over the full space $$\mathcal{P}(\mathcal{M})$$, the optimal distribution has a closed-form expression (paper eq. 13):

$$q^*_{\ell,\phi}(m) = \frac{p(m)\exp(-\ell(m;\phi))}{\sum_{m'} p(m')\exp(-\ell(m';\phi))}$$

This is a Boltzmann/softmax distribution where each model's weight is proportional to its prior probability times $$\exp(-\ell(m;\phi))$$. Since $$\ell(m;\phi) = \mathbb{E}_\mathbf{z}[\log h_\phi(\mathbf{z} \mid m)]$$ is the expected log-ratio (lower is better for the target), models with smaller $$\ell$$ (better ELBO) get higher weight. With a uniform prior $$p(m) \propto 1$$, this simplifies to $$q^*(m) \propto \exp(-\ell(m;\phi))$$.

The problem is that computing $$q^*_{\ell,\phi}$$ exactly requires evaluating $$\ell(m;\phi)$$ for all $$K_{\max}$$ models at every gradient step — that's $$K_{\max}$$ flow forward passes per step. The surrogate approximates this at $$O(B)$$ cost.

## `DiagonalGaussianSurrogate`: GP surrogate as mean-field Gaussian

```python
# vti.surrogates
class DiagonalGaussianSurrogate(GaussianSurrogate)
```

The paper (section 3.1) proposes a Gaussian Process (GP) surrogate over $$-\ell(m;\phi)$$ with UCB acquisition. The diagonal Gaussian surrogate is the mean-field (diagonal covariance) approximation to this GP — making the cost $$O(B)$$ per step instead of $$O(B^3 t^3)$$.

The surrogate maintains, for each model $$m \in \mathcal{M}$$, an independent Gaussian posterior:

$$-\ell(m; \phi) \mid \text{observations} \sim \mathcal{N}(\mu_m, v_m)$$

Here $$\mu_m$$ is our current best estimate of $$-\ell(m; \phi_t)$$ (the negative ELBO for model $$m$$) and $$v_m$$ is the uncertainty about that estimate. All $$K_{\max}$$ Gaussians are independent (mean-field). All parameters are registered as buffers.

```python
DiagonalGaussianSurrogate(
    num_categories = K_max,
    prior_mean = 0.0,               # initial μ_m for all m
    prior_diag_variance = 1e4,      # initial v_m for all m — high = explore
    obs_variance = 1e1,             # σ²_ε — noise on ELBO observations
    f_coupling = 1e2,               # Adam-step variance inflation coefficient
)
```

## `observe`: Gaussian conjugate update

```python
def observe(idx_tensor: [B], x: [B])
```

Performs an exact Bayesian posterior update for each visited model. When model $$m$$ is visited with observed ELBO value $$x$$ (which equals `-loss_hat1` = $$\log p - \log q_\theta$$, an unbiased estimate of $$-\ell(m;\phi)$$), the Gaussian conjugate update gives:

$$v_m^{\text{new}} = \left(\frac{1}{v_m} + \frac{1}{\sigma^2_\varepsilon}\right)^{-1}, \qquad \mu_m^{\text{new}} = v_m^{\text{new}} \left(\frac{\mu_m}{v_m} + \frac{x}{\sigma^2_\varepsilon}\right)$$

This is the standard Gaussian-Gaussian conjugate update: prior $$\mathcal{N}(\mu_m, v_m)$$, likelihood $$\mathcal{N}(x; \mu_m, \sigma^2_\varepsilon)$$, posterior $$\mathcal{N}(\mu_m^{\text{new}}, v_m^{\text{new}})$$. Multiple observations within a batch are processed sequentially. The observation noise $$\sigma^2_\varepsilon$$ controls how much each observation updates the posterior — larger means more scepticism of individual ELBO estimates.

If `max_entropy_gain > 0`, the minimum allowed $$\sigma^2_\varepsilon$$ per observation is $$v_m / (\exp(2 \cdot \texttt{max\_entropy\_gain}) - 1)$$, which limits the per-observation entropy decrease to at most `max_entropy_gain` nats. This prevents a single very-accurate ELBO estimate from locking in $$\mu_m$$ too early.

## `evolve`: variance inflation from Adam steps

```python
def evolve(cat_samples, ell, optimizer, loss)
```

After the flow takes a gradient step, the ELBO estimates $$\ell(m;\phi)$$ for all models change (because $$\phi$$ has changed). The surrogate's posterior means $$\mu_m$$ are now stale. `evolve` inflates all variances to reflect this staleness. Three independent mechanisms (any can be disabled):

**Adam coupling** (`f_coupling > 0`, the primary mechanism). Extracts the current Adam step vector $$\hat{\mathbf{s}}_t = \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \varepsilon)$$ and computes:

$$q = f_{\text{coupling}} \cdot \frac{1}{n_\phi} \|\hat{\mathbf{s}}_t\|^2$$

then adds $$q$$ to all $$v_m$$. The Adam step $$\hat{\mathbf{s}}_t$$ is an estimate of how much the flow parameters moved in this iteration. Large flow movement ↔ large change in $$\ell(m;\phi)$$ ↔ old observations are less informative ↔ increase uncertainty. The coupling constant $$f_{\text{coupling}} = 10^2$$ controls the sensitivity.

**Observation noise adaptation** (`obs_beta < 1`, disabled by default here). Updates $$\sigma^2_\varepsilon$$ from the empirical residuals between surrogate means and observed ELBOs:

$$\sigma^2_\varepsilon \leftarrow \beta \sigma^2_\varepsilon + (1-\beta) \cdot \frac{\vert \mathcal{K}_{\text{visited}}\vert }{K} \cdot \text{Var}(\mu_m - \ell_{\text{obs}})^{-1}$$

**Prior diffusion** (`diffuse_prior < 1`, not used here). Inflates all variances by $$v_m \leftarrow v_m / \texttt{diffuse\_prior}^{B/K}$$ at a rate proportional to visitation frequency.

## `utility_UCB` and `utility_Thompson`

```python
def utility_UCB() → [K]:     return self.mean() + 2 * self.sd()
def utility_Thompson() → [K]: return self.mean() + randn(K) * 2 * self.sd()
```

These implement the UCB acquisition function from paper eq. 14:

$$u_t(m) = \mu_t(m) + \beta \sigma_t(m), \quad \beta = 2$$

The UCB selects models that either have high estimated $$-\ell(m;\phi)$$ (exploitation: models with good ELBOs) or high $$\sigma_m$$ (exploration: models not yet well-characterised). The paper proves (Corollary 3.2) that using UCB sampling from the surrogate gives:

$$D_{KL}(q_{u,t} \| q^*_{\ell,\phi}) \in \mathcal{O}_P(t^{-1/2})$$

i.e. the surrogate-based model distribution converges to the optimal model distribution at rate $$t^{-1/2}$$ as long as the flow parameters follow a convergent sequence.

`utility_Thompson` is the Thompson sampling alternative — sample from the posterior over $$-\ell(m;\phi)$$ instead of taking the mean + 2 std. It has similar theoretical guarantees and can sometimes explore more efficiently, but is not the default.

## `SoftmaxSurrogateSampler`: $$q_{u,t}$$ in code

```python
# vti.model_samplers
class SoftmaxSurrogateSampler(AbstractModelSampler)
```

Wraps `DiagonalGaussianSurrogate` and implements the surrogate model distribution $$q_{u,t}(m)$$ from paper eq. 14:

$$q_{u,t}(m) = \frac{p(m)\exp(u_t(m))}{\sum_{m'} p(m')\exp(u_t(m'))} \stackrel{p(m)=1/K}{=} \text{softmax}(u_t)_m$$

With a uniform prior, $$q_{u,t}(m) = \text{softmax}(\mathbf{u}_t)_m$$ where $$\mathbf{u}_t = \boldsymbol{\mu}_t + 2\boldsymbol{\sigma}_t$$.

`action_logits() → [K]` returns `surrogate.utility_UCB()` = $$\mathbf{u}_t$$.

`logits() → [K]` returns `surrogate.mean()` = $$\boldsymbol{\mu}_t$$ — the greedy distribution without exploration bonus, used for posterior extraction after training.

## `action_dist`, `squish_utility`, and the double softmax

```python
def action_dist():
    if squish_utility:
        return Categorical(logits=softmax(surrogate.utility_UCB()))
    return Categorical(logits=surrogate.utility_UCB())
```

`Categorical(logits=x)` internally computes $$\text{softmax}(x)$$ before sampling. With `squish_utility=True`, softmax is applied first to $$\mathbf{u}_t$$, then again inside `Categorical`:

$$q^{\text{train}}(m) \propto \exp(\text{softmax}(\mathbf{u}_t)_m)$$

This double-softmax compresses the dynamic range of $$\mathbf{u}_t$$. Early in training, $$\mathbf{u}_t$$ has high variance — a model visited once with a lucky ELBO might have UCB 500, while an unvisited model has UCB $$\approx 2\sqrt{10^4} = 200$$ (from the prior variance). Without squishing, $$\text{softmax}(500, 200, \ldots) \approx (1, 0, 0, \ldots)$$ — the sampler collapses to one model. After squishing, $$\text{softmax}(500, 200, \ldots) \approx (0.99, 0.01, \ldots)$$ which after the second softmax gives a distribution with meaningful spread. This corresponds to controlling the exploration–exploitation trade-off by bounding information gain, as discussed in paper section 3.4.

`action_sample_and_log_prob(batch_size) → ([B], [B])` samples $$m^{(b)} \sim q^{\text{train}}$$ and returns both samples and $$\log q^{\text{train}}(m^{(b)})$$. The log-probabilities enter `loss_hat2 = -log_prior_mk + log_q_mk`, which is the KL $$\text{KL}(q^{\text{train}} \| p)$$ appearing in the VTI loss (paper eq. 10).

`log_prob(mk_catsamples) → [N]` evaluates $$\log q^{\text{train}}(m)$$ for given indices under the current `action_dist()`. After training this is called as:

```python
mk_probs = sampler.log_prob(torch.arange(K)).exp()
```

to extract the approximate model posterior.

## `_NormalisedSurrogate`: scaling the ELBO observations

```python
class _NormalisedSurrogate(DiagonalGaussianSurrogate)
```

The ELBO values $$-\ell(m;\phi)$$ are $$O(N \cdot \log K)$$. With $$N=4000$$ and $$K=100$$: each ELBO observation is $$\approx -18{,}000$$ early in training. The surrogate prior has variance $$v_m = 10^4$$ and observation noise $$\sigma^2_\varepsilon = 10$$. A single update shifts $$\mu_m$$ by:

$$\mu_m^{\text{new}} \approx \frac{-18000/10}{1/10^4 + 1/10} \approx -1800$$

in one step, which is orders of magnitude outside the prior range. Subsequent conjugate updates become numerically ill-conditioned and the surrogate essentially gives up learning.

`_NormalisedSurrogate` divides all ELBO values by `scale = NUM_DATA` before passing them to the parent class:

```python
def observe(self, idx_tensor, x):
    super().observe(idx_tensor, x / self._scale)   # x/N ≈ O(log K) ≈ O(4.6)
    self._clamp_state()

def evolve(self, cat_samples, ell, optimizer, loss):
    super().evolve(cat_samples, ell / self._scale, optimizer, loss / self._scale)
    self._clamp_state()
```

After normalisation, ELBO observations are $$O(\log K) \approx 4.6$$, well within the prior range of $$\pm 2\sqrt{10^4} = \pm 200$$. The surrogate posterior means $$\mu_m$$ then represent $$-\ell(m;\phi)/N$$ (the per-data-point ELBO). This does not affect the model distribution $$q_{u,t}(m)$$ because the softmax is scale-invariant: $$\text{softmax}(\mathbf{u}/N) \neq \text{softmax}(\mathbf{u})$$ but the surrogate hyperparameters (`prior_diag_variance`, `obs_variance`, `f_coupling`) are already tuned for the normalised scale.

## `_clamp_state`: preventing posterior collapse

```python
def _clamp_state(self)
```

Called after every `observe` and `evolve`. Performs two operations:

1. `_prior_diag_variance_diag.clamp_(min=1e-10, max=1e6)`: prevents $$v_m$$ from reaching zero (which would give infinite precision — the next conjugate update divides by $$v_m$$ and overflows) or from exceeding $$10^6$$ (unnecessary exploration).

2. Reset NaN/inf entries in `_prior_diag_variance_diag` and `_prior_mean` to their initial prior values. NaNs can arise if `f_coupling` inflation adds an inf, or if the Adam step contains non-finite values from degenerate flow samples.

Without `_clamp_state`, a single degenerate sample can cause the surrogate to collapse and never recover.

---

# Part 7: VTISurrogateEstimator — the training loop

```python
# vti.infer
class VTISurrogateEstimator(nn.Module)
```

Owns the CoSMIC flow $$T_\phi$$, the optimiser, and the training loop. Ties together the DGP, the surrogate, and the flow into a single `nn.Module` so `.to(device)` and checkpointing work uniformly.

## `setup_optimizer`: AdamW and the cosine scheduler

```python
def setup_optimizer()
```

Creates:

**AdamW** on `self.param_transform.parameters()` (flow weights $$\phi$$) with `lr = dgp.flow_lr = 1e-3`. AdamW is Adam with decoupled weight decay, which stabilises training when the flow weights grow large.

**ChainedScheduler**:
- `CosineAnnealingWarmRestarts(T_0=100, eta_min=1e-7)`: lr oscillates as $$\eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T_0))$$ with period $$T_0 = 100$$ steps. The warm restarts allow the optimiser to escape shallow local minima by periodically resetting the lr to its maximum.
- `ExponentialLR(gamma=1-1e-3)`: multiplies the lr by $$(1-10^{-3})$$ every step. Over 5000 steps this decays the overall scale by $$(1-10^{-3})^{5000} \approx e^{-5} \approx 0.007$$.

Must be called before `zero_init` because the optimiser must be built before its internal moment buffers exist for the zero-init to be applied correctly.

## `loss_and_sample_and_log_prob`: assembling the full loss

```python
def loss_and_sample_and_log_prob(batch_size, i) → (loss, ell, mk_samples)
```

Assembles the VTI loss (paper eq. 10) in one function. The computation:

```python
z, log_p_ref   = dgp.reference_dist_sample_and_log_prob(batch_size)
mk, log_q_mk   = model_sampler.action_sample_and_log_prob(batch_size)
log_prior_mk   = prior_mk_dist.log_prob(mk)
theta, log_det = flow.inverse(z, context=mk_to_context(mk))
log_q_theta    = log_p_ref - log_det          # log q(θ|k) via change-of-variables
log_p          = dgp.log_prob(mk, theta)      # log η(θ_m | m)
loss_hat1      = -log_p + log_q_theta         # = -ELBO(θ, m) = ℓ(m;ϕ) per sample
loss_hat2      = -log_prior_mk + log_q_mk     # = KL(q_train || p) per sample
loss           = (loss_hat1 + loss_hat2).nanmean()
ell            = -loss_hat1.detach()          # ELBO = -ℓ(m;ϕ), passed to surrogate
```

`loss_hat1` is $$-\log \eta(\boldsymbol{\theta}_m \mid m) + \log q_\phi(\boldsymbol{\theta}_m \mid m) = \ell(m;\phi)$$ per sample — this is the Monte Carlo estimate of $$\ell(m;\phi)$$ for the sampled $$m$$ and $$\boldsymbol{\theta} = T_\phi(\mathbf{z} \mid m)$$.

`loss_hat2` is $$-\log p(m) + \log q_\psi(m)$$ per sample — the per-sample contribution to $$\text{KL}(q_\psi \| p)$$.

The full loss $$L = \text{mean}(\ell + \text{KL})$$ is an unbiased Monte Carlo estimate of the VTI objective (paper eq. 10).

`nanmean` silently drops NaN batch elements. NaNs arise when the flow maps $$\mathbf{z}$$ to a degenerate $$\boldsymbol{\theta}$$ (e.g. identical component means), making the log-likelihood $$-\infty$$. Dropping these is safer than averaging them in.

## `step`: one iteration of SGD

```python
def step(batch_size, iteration) → loss
```

Full sequence for one training step:

```
1.  zero_grad()
2.  loss, ell, mk = loss_and_sample_and_log_prob(batch_size, iteration)
3.  loss.backward()                                # ∇_ϕ L̂
4.  clip_grad_norm_(flow.params, max_norm=20.0)   # prevent exploding gradients
5.  flow_optimizer.step()                          # ϕ ← ϕ - η ∇_ϕ L̂
6.  model_sampler.observe(mk, ell)                # Gaussian conjugate update
7.  model_sampler.evolve(mk, ell, optimizer, loss) # variance inflation
8.  flow_scheduler.step()                          # lr schedule
```

Step 4 clips the gradient norm at 20.0. This is the only explicit gradient regularisation. The value 20.0 is generous — it only fires when the gradient explodes, not during normal training.

Steps 6 and 7 update the surrogate using $$\ell^{(b)} = -\text{loss\_hat1}^{(b)}$$ as the ELBO observation for each visited model $$m^{(b)}$$. The surrogate update uses the detached ELBO values (`.detach()`) — gradients do not flow through the surrogate. The surrogate is updated by the Bayesian conjugate rule, not by gradient descent.

## `optimize`: the full loop with callbacks

```python
def optimize(batch_size, num_iterations, callbacks=()) → loss
```

Calls `step(batch_size, i)` for `i = 0, ..., num_iterations-1`. `callbacks` is a list of objects with `.on_start()`, `.on_step(i, loss)`, `.on_end(i, loss)` — useful for early stopping, checkpoint saving, or custom logging without modifying the training loop. Callbacks are not used in the default `main()` setup.

## Zero-init: why and when

Before training, zero-initialise the final `Linear` layer of every coupling network sub-module:

```python
for name, module in problem.named_modules():
    children = list(module.children())
    if children:
        last = children[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
```

For an affine coupling transform, the final linear layer outputs the shift $$\rho^{(0)}$$ and log-scale $$\log \rho^{(1)}$$. Zero-init makes:
- $$\rho^{(0)} = 0$$, $$\log \rho^{(1)} = 0$$ → scale = $$e^0 = 1$$, shift = 0 → identity transform
- $$\log\vert \det J_{\text{flow}}\vert  = \sum_i \log \rho^{(1)}_i = 0$$ at initialisation

Without zero-init, the 1500 log-scale outputs of `affine55` at $$K_{\max} = 100$$ sum to $$O(\pm 500)$$ at random init. This makes `log_q_theta = base_log_prob - log_det ≈ ±500 - (-150) ≈ ±650`, the initial loss is $$O(600)$$ instead of $$O(0)$$, and Adam's second-moment estimate $$\hat{v}_0$$ is corrupted by this huge initial gradient — a corruption that persists because Adam accumulates $$\hat{v}_t = \beta_2 \hat{v}_{t-1} + (1-\beta_2) g_t^2$$ with $$\beta_2 = 0.999$$, meaning it takes ~1000 steps for the bad initial estimate to decay.

Applied after `setup_optimizer()` because the AdamW optimiser must be created before its moment buffers are initialised (the zero-init modifies module weights, not optimiser state).

---

# Part 8: After training — reading the posterior

## The model posterior $$q(k)$$

After training, the surrogate means $$\mu_m$$ are converged estimates of $$-\ell(m;\phi^*)/N$$. The model posterior is recovered via:

```python
mk_cat   = torch.arange(K_max, dtype=torch.long)
mk_probs = sampler.log_prob(mk_cat).exp()
```

`sampler.log_prob` evaluates `action_dist().log_prob` = $$\log \text{softmax}(\text{softmax}(\boldsymbol{\mu}_t + 2\boldsymbol{\sigma}_t))$$. For a well-converged surrogate (small $$\sigma_m$$ for all $$m$$), $$\boldsymbol{\mu}_t + 2\boldsymbol{\sigma}_t \approx \boldsymbol{\mu}_t$$ and `mk_probs[m]` $$\approx \text{softmax}(\text{softmax}(\boldsymbol{\mu}))_m$$. This approximates $$q^*(m)$$ from paper eq. 13.

## Sampling the parameter posterior for model $$k$$

```python
z, _     = dgp.reference_dist_sample_and_log_prob(NUM_FLOW_SAMPLES)
ctx      = dgp.mk_to_context(tensor([k_idx])).expand(S, -1)
theta, _ = problem.param_transform.inverse(z, context=ctx)
mu_x, mu_y = dgp.decode_params(theta, k_idx + 1)
```

Draws $$S$$ samples from $$q_\phi(\boldsymbol{\theta}_m \mid m = k)$$ by:

1. Sampling $$\mathbf{z}^{(s)} \overset{\text{iid}}{\sim} \nu_{d_{\max}}$$
2. Passing through the flow: $$\boldsymbol{\theta}^{(s)} = T_\phi(\mathbf{z}^{(s)} \mid k)$$ via `flow.inverse`
3. Decoding via `_decode_theta`: $$(\boldsymbol{\mu}_x^{(s)}, \boldsymbol{\mu}_y^{(s)})$$ for the first $$k$$ active components

Each $$(\boldsymbol{\mu}_x^{(s)}, \boldsymbol{\mu}_y^{(s)})$$ is a sample from the approximate posterior over component locations given $$k$$ components. The context is fixed to `k_idx` for all $$S$$ samples, so we're sampling from the conditional $$q_\phi(\boldsymbol{\theta} \mid m=k)$$ — not averaging over $$m$$.

## Bayesian Model Averaging

The BMA predictive density (paper section 6):

$$p_{\text{BMA}}(\mathbf{y}^*) = \sum_{m=1}^{K_{\max}} q(m) \cdot \mathbb{E}_{\boldsymbol{\theta} \sim q_\phi(\boldsymbol{\theta}_m \mid m)}\!\left[p(\mathbf{y}^* \mid \boldsymbol{\theta}_m, m)\right]$$

```python
bma_density = zeros(G, G)
for k_idx in range(MAX_COMPONENTS):
    w_k = mk_probs[k_idx].item()
    if w_k < 1e-4: continue
    bma_density += w_k * compute_density_2d(flow_samples_for_k(k_idx), ...)
```

`compute_density_2d` approximates the inner expectation via Monte Carlo:

$$\mathbb{E}_{\boldsymbol{\theta} \sim q}\!\left[p(\mathbf{y}^* \mid \boldsymbol{\theta}, m)\right] \approx \frac{1}{S} \sum_{s=1}^{S} \frac{1}{m} \sum_{i=1}^{m} \mathcal{N}(\mathbf{y}^*; \boldsymbol{\mu}_i^{(s)}, \sigma^2 \mathbf{I})$$

The outer sum in BMA then marginalises over model uncertainty. This is strictly better calibrated than using the MAP model when several values of $$k$$ have comparable posterior probability.

---

# Adapting to other problems

## Different parameter dimensions per model

`num_inputs()` always returns $$d_{\max}$$ — the dimension of the *largest* model. Smaller models use the same flow with inactive slots masked. Only `mk_to_mask` changes. For model $$m$$ with $$p_m$$ parameters:

```python
def num_inputs(self): return self.p_max

def mk_to_mask(self, mk):
    k = mk.argmax(dim=-1)
    p_k = self.p_of_k[k]   # precomputed: p_of_k[i] = dim of model i
    arange = torch.arange(self.p_max, device=self.device)
    return (arange < p_k.unsqueeze(-1)).float()
```

For variable selection where the mask *is* the model identifier:

```python
def mk_to_mask(self, mk):
    return mk.float()   # mk is [N, p] binary — directly the active predictor mask
```

## Model index ≠ number of components

The category index is just a label for $$m \in \mathcal{M}$$. For a regression problem with 10 models (linear 1–5 predictors, categories 0–4; quadratic 1–5 predictors, categories 5–9):

```python
def num_categories(self): return 10
def num_inputs(self): return 10   # 5 linear + 5 quadratic = 10 max parameters

def mk_to_mask(self, mk):
    k = mk.argmax(dim=-1)
    is_quad = (k >= 5).float()
    n_pred  = (k % 5) + 1
    linear_mask = (arange(5) < n_pred.unsqueeze(-1)).float()
    quad_mask   = linear_mask * is_quad.unsqueeze(-1)
    return cat([linear_mask, quad_mask], dim=-1)   # [N, 10]
```

## Model 1 has more parameters than model 2

No special handling — the mask for each category index is independent. Category 0 can activate more slots than category 3:

```python
def mk_to_mask(self, mk):
    k = mk.argmax(dim=-1)
    extra = torch.where(k == 0, torch.tensor(5), k)   # category 0 gets 5 extra dims
    shared_mask = torch.ones(mk.shape[0], self.p_shared, device=self.device)
    extra_mask  = (arange(self.max_extra) < extra.unsqueeze(-1)).float()
    return cat([shared_mask, extra_mask], dim=-1)
```

## Non-one-hot context

`mk_to_context` can return any fixed-width vector; `num_context_features()` must match. For a context that also encodes the fraction of active parameters (useful when models have a natural ordering):

```python
def num_context_features(self): return self.max_components + 1

def mk_to_context(self, mk):
    k = mk.argmax(dim=-1).float()
    one_hot = F.one_hot(k.long(), self.max_components).float()
    k_norm  = (k / (self.max_components - 1)).unsqueeze(-1)
    return cat([one_hot, k_norm], dim=-1)   # [N, K+1]
```

A practical rule: context width ≈ `num_inputs`. If context is too narrow relative to `num_inputs`, the coupling MLPs can't distinguish models reliably and posteriors for nearby $$m$$ values bleed into each other.

## Choosing `flow_type`

| Scenario | Recommendation |
|---|---|
| Small model space ($$K \leq 20$$), unimodal per-model posteriors | `"diagnorm"` — diagonal Gaussian, fast |
| Standard trans-dim ($$K \leq 50$$, $$D \leq 100$$) | `"affine33"` or `"affine55"` |
| Large model space or complex posteriors | `"affine55"` with float64 |
| Multimodal per-model posteriors (e.g. label switching) | `"spline13"` or `"spline33"` |
| Heavy-tailed per-model posteriors | `"sas3"` |

For very large `num_inputs` ($$> 200$$), prefer fewer layers (`"affine33"`) to control the log-det sum length before considering other changes.

---

# Numerical notes

## Why float64?

The log-det $$\log\vert \det \nabla T_\phi\vert $$ is the sum of $$L \times d_{\max}$$ scalar NN outputs (one per active dimension per layer). For `affine55` with $$K_{\max} = 100$$: $$L = 10$$ layers, $$d_{\max} = 200$$, giving ~2000 terms in the sum. In float32 (machine epsilon $$\varepsilon \approx 10^{-7}$$), the accumulated rounding error in a sum of $$n$$ terms is $$O(n \varepsilon \vert x\vert )$$. For $$n = 2000$$, $$\vert x\vert  \approx 0.1$$ per term: error $$\approx 2000 \times 10^{-7} \times 0.1 = 2 \times 10^{-5}$$ per forward pass. After 100 Adam steps this accumulates in the moment estimates. When gradients of individual terms are $$O(10^{-3})$$, the rounding error is comparable to the signal. Float64 ($$\varepsilon \approx 10^{-15}$$) has 8 orders of magnitude more headroom. Float32 is safe for $$K_{\max} \leq 20$$ and is about 2× faster on GPU.

## Chunked data likelihood

The log-likelihood requires storing `[B, N, K, 2]` intermediate tensors, which at $$B=256$$, $$N=4000$$, $$K=100$$ would be $$256 \times 4000 \times 100 \times 2 \times 8 \approx 1.6$$ GB per forward pass. Chunking over the data axis (chunk size = `_data_chunk = max(8, 12000 // K_max)`): at $$K=100$$, chunk = 120, so the intermediate tensor is $$256 \times 120 \times 100 \times 2 \times 8 \approx 50$$ MB per pass.

The float64 accumulator for the data likelihood sum is a separate issue. The per-chunk contributions are $$O(\text{chunk} \times \log K) \approx O(550)$$; summing 34 chunks totals $$O(18{,}000)$$. Float32 absolute precision at this scale is $$18{,}000 \times 10^{-7} = 0.002$$. Gradients of the likelihood with respect to $$\boldsymbol{\mu}$$ are $$O(1)$$, so float32 accumulation introduces $$\sim 0.2\%$$ relative gradient error near convergence — enough to mislead Adam. The accumulator is cast back to the working dtype after each step.

## The normalisation trap

An earlier version divided `log_prob` by `n_obs`. The VTI loss is:

$$\text{loss} = \underbrace{-\log \eta(\boldsymbol{\theta}_m \mid m)}_{\texttt{-log\_p}} + \underbrace{\log \nu_{d_{\max}}(\mathbf{z}) - \log\vert \det \nabla T_\phi(\mathbf{z} \mid m)\vert }_{\texttt{log\_q\_theta}}$$

`log_q_theta` = `params_log_prob` is $$O(d_{\max}) = O(200)$$, independent of $$N$$. If `log_p` is divided by $$N$$, the data term becomes $$O(\log K) / N \approx O(0.001)$$ while the log-det term remains $$O(200)$$. The gradient of the loss with respect to $$\phi$$ becomes dominated by $$-\nabla_\phi \log\vert \det \nabla T_\phi\vert $$, which is $$-\partial/\partial\phi \sum_i \log \rho_i^{(1)}$$. Maximising this maximises the log-determinant — volume expansion of the flow — which diverges. The `_NormalisedSurrogate` with `scale = NUM_DATA` handles the $$O(N)$$ scale of the ELBO on the *surrogate side only*, where gradients don't flow. The flow's loss must remain unnormalised.

---

# Footnotes

[^vti]: Davies, L., Mackinlay, D., Oliveira, R., & Sisson, S. A. (2025). Amortized Variational Transdimensional Inference. *arXiv:2506.04749*.