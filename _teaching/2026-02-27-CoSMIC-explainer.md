---
title: "Trans-dimensional 2D GMM with VTI: code reference"
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

Code reference for [`gmm_2d_k100.py`](https://github.com/...). Covers every class, method, and design decision in execution order.

## Table of Contents

- [Module layout](#module-layout)
- [GMM2DDGP](#gmm2ddgp)
  - [\_\_init\_\_](#__init__)
  - [\_generate\_data](#_generate_data)
  - [AbstractDGP interface](#abstractdgp-interface)
  - [\_decode\_theta](#_decode_theta)
  - [log\_prob](#log_prob)
- [CoSMIC/VTI framework classes](#cosmicvti-framework-classes)
  - [AbstractDGP](#abstractdgp)
  - [VTISurrogateEstimator](#vtisurrogateestimator)
  - [The CoSMIC flow](#the-cosmic-flow)
  - [DiagonalGaussianSurrogate](#diagonalgaussiansurrogate)
  - [SoftmaxSurrogateSampler](#softmaxsurrogatesampler)
- [\_NormalisedSurrogate](#_normalisedsurrogate)
- [main()](#main)
  - [Configuration](#configuration)
  - [Assembly](#assembly)
  - [Zero-init](#zero-init)
  - [Training](#training)
  - [Posterior extraction](#posterior-extraction)
  - [Visualisation](#visualisation)
- [Adapting to other problems](#adapting-to-other-problems)
- [Numerical notes](#numerical-notes)

---

# Module layout

```
GMM2DDGP                AbstractDGP subclass — the full probability model
_NormalisedSurrogate    DiagonalGaussianSurrogate subclass — scales ELBO by 1/N
corner_plot             utility — N×N posterior corner plots
make_corner_dict_2d     utility — builds {param_name: samples} dict for corner_plot
compute_density_2d      utility — MC-averaged GMM density on a grid
main()                  config, assembly, training, plotting
```

---

# GMM2DDGP

```python
class GMM2DDGP(AbstractDGP)
```

The full probability model. Owns the data, the parameterisation, and the log-joint.

**Parameters to constructor:**

| Argument | Type | Description |
|---|---|---|
| `seed` | `int` | RNG seed for data generation |
| `num_data` | `int` | Number of data points $$N$$ |
| `max_components` | `int` | $$K_{\max}$$ — size of the model space |
| `true_means` | `list[(float, float)]` | True component means (used only to generate data) |
| `sigma` | `float` | Known noise std (shared across all components) |
| `complexity_penalty` | `float` | BIC prefactor; penalty per extra component = `complexity_penalty × 0.5 × log(N)` |
| `device`, `dtype` | — | Passed to `AbstractDGP` |

**Buffers registered on `self`:**

| Buffer | Shape | Description |
|---|---|---|
| `y_data` | `[N, 2]` | The dataset |
| `x_lo`, `x_hi` | scalar | Data x-range + 20% margin — defines the prior bbox |
| `y_lo`, `y_hi` | scalar | Data y-range + 20% margin |

**Other attributes:**

| Attribute | Description |
|---|---|
| `self.sigma` | Noise std as a tensor |
| `self._data_chunk` | `max(8, 12000 // K_max)` — chunk size for likelihood loop |
| `self.flow_lr` | `1e-3` — read by `VTISurrogateEstimator` to set AdamW lr |
| `self.dgp_mk_identifier` | One-hot of true $$k$$, used for diagnostics |


## `__init__`

```python
def __init__(self, seed, num_data, max_components, true_means, sigma,
             complexity_penalty, device=None, dtype=None, **kwargs)
```

Calls `_generate_data`, sets `_data_chunk` and `flow_lr`. `flow_lr` must be set here because `VTISurrogateEstimator` reads `self.dgp.flow_lr` during its own `__init__`; if absent it falls back to an internal default.


## `_generate_data`

```python
def _generate_data(self, seed, num_data, true_means)
```

Generates data by sampling uniformly from the true components, then adding Gaussian noise. Computes the axis-aligned bounding box of the data (plus 20% margin on each side) and registers it as `x_lo`, `x_hi`, `y_lo`, `y_hi`. These bounds define both the prior support for component means and the range of the sigmoid parameterisation.

All tensors are registered with `register_buffer` so they move with `.to(device)` and are excluded from the gradient graph.


## AbstractDGP interface

VTI calls these methods on every DGP. All must be overridden.

---

### `num_categories() → int`

Returns `K_max`. Determines the size of the surrogate (one ELBO estimate per model) and the number of softmax logits in the sampler.

---

### `num_inputs() → int`

Returns `2 * K_max`. Dimensionality of the parameter vector $$\boldsymbol{\theta}$$. This is the input and output size of the normalising flow.

---

### `num_context_features() → int`

Returns `2 * K_max`. Size of the context vector fed to the flow's coupling network MLPs. The context vector is a doubled one-hot `[one_hot(k), one_hot(k)]` — see `mk_to_context`.

---

### `mk_identifiers() → [K, K]`

Returns the identity matrix. Each row is the one-hot identifier for model $$k$$. Used by VTI to enumerate all models.

---

### `mk_cat_to_identifier(cat_samples) → [N, K]`

```
Input:  [N]  integer indices in {0, ..., K-1}
Output: [N, K]  one-hot float tensor
```

Converts integer model indices to one-hot representations. The one-hot is the canonical internal identifier format throughout VTI.

---

### `mk_to_context(mk_samples) → [N, C]`

```
Input:  scalar / [N] int / [N, K] one-hot  — any of these
Output: [N, C]  float  where C = num_context_features()
```

Builds the context vector for the flow. In our implementation doubles the one-hot: `cat([one_hot, one_hot], dim=-1)`. Handles three input forms because VTI calls this in different ways at different points in the training loop:
- scalar → unsqueeze then one-hot
- `[N]` int → one-hot directly  
- `[N, K]` one-hot → pass through

---

### `mk_to_mask(mk) → [N, D]`

```
Input:  [N] int or [N, K] one-hot
Output: [N, D]  binary float  where D = num_inputs()
```

Returns a parameter-slot mask. For model $$k$$, slots corresponding to active parameters are 1, inactive slots are 0. Shape matches the full parameter vector `num_inputs()`. This mask is passed to the CoSMIC flow's coupling layers at runtime to determine which dimensions to transform and which to leave as identity. It is also used by `reference_log_prob` to identify inactive dimensions for the reference prior.

```python
# Example: K=5, k=3  →  [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
k = mk.argmax(dim=-1)                  # [N]
slot_mask = (arange(K) <= k[:,None])   # [N, K]
return slot_mask.repeat(1, 2)          # [N, 2K]
```

---

### `mk_to_component_mask(mk) → [N, K]`

```
Input:  [N] int or [N, K] one-hot
Output: [N, K]  binary float
```

Like `mk_to_mask` but only over the $$K$$ component slots (not doubled). Used in `log_prob` to mask the mixture weights and the Jacobian sum. Not required by the framework — this is a convenience helper specific to the GMM.

```python
# Example: K=5, k=3  →  [1, 1, 1, 0, 0]
return (arange(K) <= k[:,None])
```

---


## `_decode_theta`

```python
def _decode_theta(self, theta)
```

```
Input:  theta  [N, 2K]  unconstrained
Output: mu_x   [N, K]   in [x_lo, x_hi]
        mu_y   [N, K]   in [y_lo, y_hi]
Side effect: sets self._last_log_jac  [N, K]
```

Applies the sigmoid change-of-variables to map unconstrained flow outputs to component means:

$$\mu_{x,i} = x_{\text{lo}} + (x_{\text{hi}} - x_{\text{lo}}) \cdot \sigma(\theta_i)$$

The Jacobian is diagonal (each $$\mu_{x,i}$$ depends only on $$\theta_i$$):

$$\log|J_i| = \log\big(w_x \cdot \sigma(\theta_i)(1-\sigma(\theta_i))\big)$$

where $$w_x = x_{\text{hi}} - x_{\text{lo}}$$. The full per-component log-Jacobian $$\log\vert J_i^x\vert + \log\vert J_i^y\vert$$ is stashed in `self._last_log_jac` for immediate use by `log_prob`.

The `.clamp(min=1e-30)` before `log` prevents `log(0)` when the sigmoid saturates near 0 or 1 at the boundary of the box.

---

### `decode_params(theta, k) → (mu_x, mu_y)`

```python
def decode_params(self, theta, k)
```

Public wrapper around `_decode_theta`. Returns only the first $$k$$ active components. Used during posterior sampling.

---


## `log_prob`

```python
def log_prob(self, mk, theta) -> [N]
```

```
Input:  mk     [N] int or [N, K] one-hot  — which model
        theta  [N, 2K]  — parameter sample from the flow
Output: [N]  float — log p(y, θ, k) for each batch element
```

Returns the un-normalised log joint. VTI computes the ELBO as `log_prob(mk, theta) − params_log_prob`, where `params_log_prob = log p_ref(z) − log|det J_flow|` is the flow's own entropy term. **Do not normalise by $$N$$** — see [numerical notes](#the-normalisation-trap).

The log joint has four terms:

**1. Data log-likelihood**

$$\log p(\mathbf{y} \mid \boldsymbol{\theta}, k) = \sum_{j=1}^{N} \log \frac{1}{k} \sum_{i=1}^{k} \mathcal{N}(\mathbf{y}_j; \boldsymbol{\mu}_i, \sigma^2 \mathbf{I})$$

Computed via `logsumexp` over components. Accumulated in a float64 scalar to avoid precision loss over the $$N$$-term sum — see [chunked likelihood](#chunked-data-likelihood).

**2. Parameter prior**

Uniform over the bounding box, transformed to $$\boldsymbol{\theta}$$-space:

$$\log p(\boldsymbol{\theta} \mid k) = \sum_{i=1}^{k} \big[ -\log(w_x w_y) + \log|J_i| \big]$$

where $$\log\vert J_i \vert$$ is `self._last_log_jac` from `_decode_theta`. Only active components ($$\texttt{cmask} = 1$$) contribute.

**3. Model prior**

BIC-style complexity penalty:

$$\log p(k) = -\lambda \cdot \tfrac{1}{2} \log N \cdot (k - 1), \qquad \lambda = \texttt{complexity\_penalty}$$

Baseline is $$k=1$$ (penalty = 0). Each additional component costs $$\lambda \cdot \tfrac{1}{2} \log N$$ nats.

**4. Reference log-probability**

```python
reference_lp = self.reference_log_prob(mk, theta)
```

Provided by `AbstractDGP`. Evaluates $$\log \mathcal{N}(\boldsymbol{\theta}_{\text{inactive}}; \mathbf{0}, \mathbf{I})$$ — the log-density under the standard normal reference for the **inactive** dimensions (components $$i > k$$). Anchors inactive slots near zero and prevents the flow from wasting capacity on them.

---

### `get_sfe_lr() → float`

Returns `1e-3`. Learning rate for the Score Function Estimator if VTI falls back to SFE mode (not used in normal operation with the surrogate estimator, but required by the interface).

---

### `printVTIResults(mk_probs)`

Prints the posterior model probabilities $$q(k)$$ for all $$k$$, with a marker at the true $$k$$.

---


# CoSMIC/VTI framework classes

These are the framework classes you don't write but need to understand to use correctly.

---

## AbstractDGP

```python
# vti.dgp  (abstract base inherited by GMM2DDGP)
class AbstractDGP(nn.Module)
```

Base class for all VTI data-generating processes. Provides default implementations that you typically do not override.

**Methods you must override** (the full interface requirement):

| Method | Signature | Purpose |
|---|---|---|
| `num_categories` | `() → int` | Number of models |
| `num_inputs` | `() → int` | Dimension of $$\boldsymbol{\theta}$$ |
| `num_context_features` | `() → int` | Dimension of context to flow MLPs |
| `mk_identifiers` | `() → [K, K]` | Identity matrix |
| `mk_cat_to_identifier` | `([N] int) → [N, K]` | Index to one-hot |
| `mk_to_context` | `([N] or [N,K]) → [N, C]` | Model index to context vector |
| `mk_to_mask` | `([N] or [N,K]) → [N, D]` | Active parameter slot mask |
| `log_prob` | `(mk, theta) → [N]` | Log joint |

**Methods provided by `AbstractDGP`:**

`reference_dist_sample_and_log_prob(batch_size) → (z, log_p)`
: Samples $$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_D)$$ where $$D$$ = `num_inputs()`. Returns both the sample and $$\log p_{\text{ref}}(\mathbf{z})$$. This is the starting distribution for all flow sampling.

`reference_log_prob(mk, theta) → [N]`
: Uses `mk_to_mask` to identify inactive dimensions, then evaluates $$\log \mathcal{N}(\boldsymbol{\theta}_{\text{inactive}}; \mathbf{0}, \mathbf{I})$$. Call this from your `log_prob` as the fourth term of the log joint. If you don't include it, the flow has no incentive to keep inactive dimensions near zero, and they will drift freely.

`mk_prior_dist() → Categorical`
: Returns `Categorical(logits=zeros(K))`, i.e. a uniform distribution over the $$K$$ models. Used by `VTISurrogateEstimator` to compute `mk_prior_log_prob` in `loss_hat2`.

`construct_param_transform(flow_type) → CompositeTransform`
: Builds the CoSMIC flow using `self.mk_to_mask` as the `context_to_mask` function and `self.mk_to_context` as the `context_transform`. These are passed as callable arguments into the transform factory so the flow can call them at runtime during both training and inference.

---

## VTISurrogateEstimator

```python
# vti.infer
class VTISurrogateEstimator(nn.Module)
```

Owns the flow and runs the training loop.

```python
VTISurrogateEstimator(
    dgp,           # AbstractDGP subclass
    model_sampler, # SoftmaxSurrogateSampler
    flow_type,     # str — see flow type table
    output_dir,    # Path for checkpoints
    device, dtype,
)
```

On `__init__` reads `dgp.flow_lr`, builds the flow via `dgp.construct_param_transform(flow_type)`, and registers everything as submodules so `.to(device)` moves the whole stack.

---

### `setup_optimizer()`

Must be called before training. Creates:

- **AdamW** on `self.param_transform.parameters()` with `lr = self.flow_lr`
- **Chained scheduler**: `CosineAnnealingWarmRestarts(T_0=100, eta_min=1e-7)` + `ExponentialLR(gamma=1-1e-3)`

The cosine restarts with period 100 let the lr oscillate to escape local optima; the exponential decay gradually reduces it over the full run. The cosine period of 100 is hardcoded — if you run for 500 iterations total you get 5 full cycles; for 5000 iterations you get 50.

---

### `step(batch_size, iteration) → loss`

One training iteration. Full sequence:

```
1.  z, log_p_ref   = dgp.reference_dist_sample_and_log_prob(batch_size)  [B, D]
2.  mk, log_q_mk   = sampler.action_sample_and_log_prob(batch_size)       [B]
3.  log_prior_mk   = prior_mk_dist.log_prob(mk)          # log(1/K)        [B]
4.  ctx            = dgp.mk_to_context(mk)                                 [B, C]
5.  theta, log_det = flow.inverse(z, context=ctx)         # z → θ          [B, D]
6.  log_q_theta    = log_p_ref - log_det                  # flow entropy    [B]
7.  log_p          = dgp.log_prob(mk, theta)              # your log_prob   [B]
8.  loss_hat1      = -log_p + log_q_theta                 # neg ELBO θ-part
9.  loss_hat2      = -log_prior_mk + log_q_mk             # neg ELBO k-part
10. loss           = (loss_hat1 + loss_hat2).nanmean()
11. loss.backward()
12. clip_grad_norm_(flow.parameters(), max_norm=20.0)
13. flow_optimizer.step()
14. sampler.observe(mk, ell=-loss_hat1.detach())          # feed ELBO to surrogate
15. sampler.evolve(mk, ell, flow_optimizer, loss)         # inflate surrogate variance
16. flow_scheduler.step()
```

`loss_hat2` is the KL between the sampler's $$q(k)$$ and the uniform prior $$p(k) = 1/K$$. It penalises the sampler for concentrating too heavily on a few models.

`ell` on line 14 is $$\log p - \log q_\theta = \text{ELBO}(\boldsymbol{\theta}, k)$$ per batch element. This is the signal the surrogate uses to learn ELBO$$(k)$$.

`nanmean` silently drops NaN batch elements — important early in training or when the flow samples a degenerate $$\boldsymbol{\theta}$$.

The gradient clip at `max_norm=20.0` is the only gradient clipping in the framework.

---

### `optimize(batch_size, num_iterations, callbacks=()) → loss`

Calls `step` in a loop. `callbacks` is a list of objects with `.on_start()`, `.on_step(i, loss)`, `.on_end(i, loss)`. Useful for logging or early stopping without modifying the loop.

---

### `save_training_checkpoint` / `load_training_checkpoint`

Serialises flow parameters, optimiser state, and surrogate state to `output_dir/checkpoint.pt`. The `state_dict` override ensures all three are included.

---

## The CoSMIC flow

The normalising flow is a **CoSMIC** — **Co**nditionally **S**tatic **M**asked **I**nverse **C**oupling — flow. The defining feature is that the mask applied in each autoregressive coupling layer is computed at runtime from the context vector, not fixed at construction. This allows a single flow with fixed architecture to represent posteriors over parameter spaces of different dimensionalities: the mask zeros out inactive dimensions so they pass through as identity, and the flow learns model-specific posteriors for the active ones.

The flow is built by `AbstractDGP.construct_param_transform(flow_type)` which calls `vti.dgp.param_transform_factory.construct_param_transform` with:
- `context_to_mask = self.mk_to_mask` — called at each layer to get the current binary mask
- `context_transform = self.mk_to_context` — the MLP uses this to embed the context

**Available `flow_type` strings:**

| String | Architecture | Approximate parameter count (D=150) |
|---|---|---|
| `"diagnorm"` | Single `CoSMICDiagonalAffineTransform` | ~5K |
| `"affine{N}{M}"` | $$N$$ pre + $$M$$ post MAAT layers, 2 ResNet blocks each | `"affine55"` ≈ 2M |
| `"spline{N}{M}"` | Affine pre + $$N$$ PRQS spline + $$M$$ affine post | More capacity, slower |
| `"sas{N}"` | DiagAffine + $$N$$ MAAT + SinhArcSinh tail | Heavy-tailed posteriors |

For `"affine55"` the full architecture is:

```
StrictLeftPermutation                         # shifts active dims to front
  5 × [PartialReversePermutation
       + InverseTransform(CoSMICMaskedAffineAutoregressiveTransform)]
  5 × [PartialReversePermutation
       + InverseTransform(CoSMICMaskedAffineAutoregressiveTransform)]   ← leaky_relu activation
InverseTransform(StrictLeftPermutation)       # undoes the permutation
```

The `StrictLeftPermutation` at the start permutes the parameter vector so active dimensions (mask=1) come first. The `PartialReversePermutation` between layers reverses only the active part of the vector, which improves coupling by changing which dimensions condition which. Together these replace the simple `ReversePermutation` used in standard MAF, and they're context-dependent (they read `mk_to_mask` at runtime).

Each `CoSMICMaskedAffineAutoregressiveTransform` is a MAAT layer with 2 ResNet blocks. Hidden size defaults to `num_inputs * 2 + num_context_inputs` in `affine55`.

**Key interface:**

`problem.param_transform.inverse(z, context=ctx) → (theta, log_det)`
: Generation direction. $$\mathbf{z} \to \boldsymbol{\theta}$$, returns `log_det = log|det J_{z→θ}|`. This is the fast $$O(D)$$ path used during training (IAF).

`problem.param_transform.forward(theta, context=ctx) → (z, log_det)`
: Density evaluation direction. $$\boldsymbol{\theta} \to \mathbf{z}$$, returns `log_det = log|det J_{θ→z}|`. This is $$O(D^2)$$ for autoregressive transforms and is not used during VTI training.

The two log-dets are negatives of each other: $$\log\vert\det J_{z→\theta}\vert = -\log\vert\det J_{\theta→z}\vert$$.

---

## DiagonalGaussianSurrogate

```python
# vti.surrogates
class DiagonalGaussianSurrogate(GaussianSurrogate)
```

Maintains an independent Gaussian posterior $$\text{ELBO}(k) \sim \mathcal{N}(\mu_k, v_k)$$ for each of the $$K$$ models. Updates are exact Gaussian conjugate updates.

```python
DiagonalGaussianSurrogate(
    num_categories,          # K
    prior_mean=0.0,          # initial μ_k for all k
    prior_diag_variance,     # initial v_k for all k
    obs_variance,            # σ²_obs — noise variance on ELBO observations
    f_coupling,              # Adam step² inflation coefficient
    obs_beta=0.99,           # EMA for obs_variance adaptation
    diffuse_prior=1.0,       # <1 inflates variance by 1/diffuse_prior per step
    max_entropy_gain=0.0,    # >0 caps per-obs entropy gain
    device, dtype,
)
```

**Internal state (all `register_buffer`, so they move with `.to(device)`):**

| Buffer | Shape | Description |
|---|---|---|
| `_prior_mean` | `[K]` | Posterior means $$\mu_k$$ |
| `_prior_diag_variance_diag` | `[K]` | Posterior variances $$v_k$$ |
| `_obs_variance` | scalar | Current observation noise $$\sigma^2_{\text{obs}}$$ |
| `f_coupling` | scalar | Adam-coupling inflation coefficient |
| `_obs_beta` | scalar | EMA coefficient for obs_variance |

---

### `observe(idx_tensor, x)`

```
Input:  idx_tensor  [B]  — which model categories were visited
        x           [B]  — observed ELBO values
```

Gaussian conjugate update for each visited model (processed sequentially within the batch):

$$v_k^{\text{new}} = \left(\frac{1}{v_k} + \frac{1}{\sigma^2_{\text{obs}}}\right)^{-1}, \qquad \mu_k^{\text{new}} = v_k^{\text{new}} \left(\frac{\mu_k}{v_k} + \frac{x}{\sigma^2_{\text{obs}}}\right)$$

If `max_entropy_gain > 0`, the minimum allowable `obs_variance` per observation is $$v_k / (\exp(2 \cdot \texttt{max\_entropy\_gain}) - 1)$$, capping how much information a single observation can add.

---

### `evolve(cat_samples, ell, optimizer, loss)`

Inflates the posterior variances to reflect that the flow has changed since ELBO estimates were made. Three independent mechanisms:

**1. Observation noise adaptation** (`obs_beta < 1.0`, disabled in our setup since default `obs_beta=0.99`)

$$\sigma^2_{\text{obs}} \leftarrow \beta\,\sigma^2_{\text{obs}} + (1-\beta)\,\frac{|\mathcal{K}_{\text{visited}}|}{K} \cdot \text{Var}(\mu_k - \text{ell})^{-1}$$

Adapts the observation noise from empirical residuals, weighted by fraction of models visited.

**2. Prior diffusion** (`diffuse_prior < 1.0`, not used in our setup)

$$v_k \leftarrow v_k / \texttt{diffuse\_prior}^{B/K}$$

Inflates all variances uniformly at a rate proportional to visitation frequency.

**3. Adam coupling** (`f_coupling > 0`, the main mechanism)

Extracts the current Adam step vector $$\hat{m}_t / \sqrt{\hat{v}_t + \varepsilon}$$ and computes:

$$q = f_{\text{coupling}} \cdot \text{mean}\!\left[\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}\right)^2\right]$$

then adds $$q$$ to all $$v_k$$. Large Adam steps ↔ large flow change ↔ previous ELBO estimates are stale ↔ inflate uncertainty and re-explore.

---

### `mean() → [K]`, `sd() → [K]`, `diag_variance() → [K]`

Current posterior mean, standard deviation, and variance for all $$K$$ models.

---

### `utility_UCB() → [K]`

```python
return self.mean() + 2 * self.sd()
```

UCB logits: two standard deviations above the mean. This is what `SoftmaxSurrogateSampler.action_logits()` returns.

---

### `utility_Thompson() → [K]`

Thompson sampling alternative: `mean + randn(K) * 2 * sd`. Available as a method but not used by default.

---

### `LowRankGaussianSurrogate`

A variant where the covariance is $$\text{diag}(\sigma^2) + \mathbf{L}\mathbf{L}^\top$$. Models correlations between ELBO values for nearby models in the model space (e.g. $$k=4$$ and $$k=5$$ may have correlated ELBOs). Available as `vti.surrogates.LowRankGaussianSurrogate`. Same interface as `DiagonalGaussianSurrogate` with additional `prior_dev` and `lr_variance_scale` arguments.

---

## SoftmaxSurrogateSampler

```python
# vti.model_samplers
class SoftmaxSurrogateSampler(AbstractModelSampler)
```

Wraps a `GaussianSurrogate` and exposes it as a categorical distribution $$q(k)$$ for sampling.

```python
SoftmaxSurrogateSampler(
    surrogate,           # GaussianSurrogate instance
    squish_utility=True, # apply softmax to UCB before passing to Categorical
    check_nans=True,
    device, dtype,
)
```

---

### `action_logits() → [K]`

Returns `surrogate.utility_UCB()` — the raw UCB values for all $$K$$ models.

---

### `logits() → [K]`

Returns `surrogate.mean()` — the posterior means, without the exploration bonus. This is the "greedy" distribution, used by `log_prob` for posterior extraction.

---

### `action_dist() → Categorical`

```python
if squish_utility:
    return Categorical(logits=softmax(surrogate.utility_UCB()))
else:
    return Categorical(logits=surrogate.utility_UCB())
```

With `squish_utility=True`, softmax is applied to the UCB values before passing them as logits to `Categorical`. Since `Categorical(logits=x)` applies softmax internally, the effective probabilities are $$q(k) \propto \exp(\text{softmax}(\text{UCB}_k))$$, compressing the logit spread. This prevents a single dominant model from being sampled with probability ≈1 early in training when UCB estimates are noisy.

---

### `action_sample_and_log_prob(batch_size) → (mk_catsamples [B], mk_log_probs [B])`

Samples from `action_dist()` and returns sample indices plus their log-probabilities. The log-probs go into `loss_hat2`.

---

### `log_prob(mk_catsamples) → [N]`

$$\log q(k)$$ for given indices under the current `action_dist()`. Called after training to extract the posterior:

```python
mk_probs = sampler.log_prob(torch.arange(K)).exp()
```

Note this uses the UCB-based `action_dist`, not the greedy `logits`. The resulting $$q(k)$$ includes the exploration bias.

---

### `observe(mk_catsamples, loss_hat, iteration)` / `evolve(...)`

Both delegate directly to `surrogate.observe` and `surrogate.evolve`. The `iteration` argument is accepted but unused.

---

### `BinaryStringSSSampler`

A subclass for model spaces indexed by binary strings rather than integers. Returns `[N, B]` binary tensors where $$B = \lceil \log_2 K \rceil$$ instead of `[N]` integer indices. Useful when the model identifier naturally has a binary structure (e.g. variable selection where each bit indicates predictor inclusion). Internally the surrogate still uses integer indices; the conversion is handled in `observe` and `evolve`.

---


# `_NormalisedSurrogate`

```python
class _NormalisedSurrogate(DiagonalGaussianSurrogate)
```

Wraps `DiagonalGaussianSurrogate` with a division by `scale` on all incoming ELBO observations, keeping the surrogate's inputs $$O(1)$$ regardless of dataset size.

**Constructor:**

```python
_NormalisedSurrogate(scale=float(NUM_DATA), num_categories=K_max,
                     prior_mean=0.0, prior_diag_variance=1e4,
                     obs_variance=1e1, f_coupling=1e2,
                     device=..., dtype=...)
```

`scale = NUM_DATA`: the ELBO equals `log_prob − params_log_prob`, where `log_prob` is $$O(N \log K)$$. Dividing by $$N$$ keeps surrogate observations $$O(1)$$–$$O(10)$$ and prevents the Bayesian update from becoming ill-conditioned.

**Overridden methods:**

```python
def observe(self, idx_tensor, x):
    super().observe(idx_tensor, x / self._scale)
    self._clamp_state()

def evolve(self, cat_samples, ell, optimizer, loss):
    super().evolve(cat_samples, ell / self._scale, optimizer, loss / self._scale)
    self._clamp_state()
```

Both divide their ELBO argument by `scale` before the parent call, then call `_clamp_state`.

---

### `_clamp_state()`

Called after every `observe` and `evolve`. Clamps `_prior_diag_variance_diag` to $$[10^{-10}, 10^6]$$ and resets any NaN/inf entries in the variance or mean buffers to the prior values. Prevents the Gaussian posterior from collapsing to zero variance (infinite precision → divergent next update) or overflowing float32.

---


# `main()`

## Configuration

```python
SEED               = 0        # training RNG seed
DGP_SEED           = 42       # data generation seed
NUM_DATA           = 4000     # N
num_comps          = 50       # true k
MAX_COMPONENTS     = 100      # K_max = 2 × k_true
SIGMA              = 0.5      # known noise std
COMPLEXITY_PENALTY = 2.0      # BIC prefactor
FLOW_TYPE          = "affine55"
NUM_ITERATIONS     = 5000
BATCH_SIZE         = 256
NUM_FLOW_SAMPLES   = 500      # posterior samples for plotting
DTYPE              = "float64"  # required for K > 40; see numerical notes

# Surrogate hyperparameters
SURROGATE_PRIOR_MEAN          = 0.0
SURROGATE_PRIOR_DIAG_VARIANCE = 1e4
SURROGATE_OBS_VARIANCE        = 1e1
SURROGATE_F_COUPLING          = 1e2
```

True means are 50 uniform random points in $$[0, 10]^2$$, generated from `DGP_SEED`. `MAX_COMPONENTS = 2 × k_true` gives the flow room to overfit upward.

## Assembly

```python
surrogate = _NormalisedSurrogate(scale=NUM_DATA, ...)
sampler   = SoftmaxSurrogateSampler(surrogate, squish_utility=True)
problem   = VTISurrogateEstimator(dgp, sampler, flow_type=FLOW_TYPE, ...)
problem.setup_optimizer()
```

`setup_optimizer()` must be called before the zero-init step below.


## Zero-init

```python
for name, module in problem.named_modules():
    children = list(module.children())
    if children:
        last = children[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
```

Zero-initialises the **last `Linear` layer** of every coupling network sub-module. This makes every affine coupling transform start as the identity map (scale=1, shift=0, $$\log\vert\det J\vert=0$$). 

Without this, `affine55` at 100+ dimensions has $$\log\vert\det J_{\text{flow}}\vert = O(\pm 500)$$ at random init, producing large `params_log_prob` and corrupting Adam's moment estimates at step 0. Applied after `setup_optimizer()` because the optimiser must be built before parameters are modified.


## Training

```python
loss = problem.optimize(batch_size=BATCH_SIZE, num_iterations=NUM_ITERATIONS)
```

Calls `problem.step(batch_size, i)` in a loop, as described in [VTISurrogateEstimator.step](#stepbatch_size-iteration--loss).


## Posterior extraction

```python
mk_cat   = torch.arange(K_max, dtype=torch.long)
mk_probs = sampler.log_prob(mk_cat).exp()
```

`sampler.log_prob(k)` evaluates the action distribution (softmax of softmax of UCB) for all $$K$$ models simultaneously.


## Visualisation

**`flow_samples_for_k(k_idx)`**

```python
z, _   = dgp.reference_dist_sample_and_log_prob(NUM_FLOW_SAMPLES)
ctx    = dgp.mk_to_context(tensor([k_idx])).expand(N, -1)
theta  = problem.param_transform.inverse(z, context=ctx)
mu_x, mu_y = dgp.decode_params(theta, k_idx + 1)
```

Samples posterior mean configurations for model $$k$$ by running the flow generation direction.

---

**`compute_density_2d(mu_x, mu_y, sigma, grid_x, grid_y)`**

```
Input:  mu_x, mu_y  [S, k]  — S posterior samples of the k component means
Output: [G, G]  density averaged over samples
```

$$\hat{p}(\mathbf{y}^*) = \frac{1}{S} \sum_{s=1}^{S} \frac{1}{k} \sum_{i=1}^{k} \mathcal{N}(\mathbf{y}^*; \boldsymbol{\mu}_i^{(s)}, \sigma^2 \mathbf{I})$$

Chunked over samples (100 at a time) to avoid OOM.

---

**BMA density**

$$p_{\text{BMA}}(\mathbf{y}^*) = \sum_{k} q(k) \cdot \hat{p}(\mathbf{y}^* \mid k)$$

Models with $$q(k) < 10^{-4}$$ skipped for speed.

---


# Adapting to other problems

---

## Different parameter dimensions per model

`num_inputs()` always returns the dimension of the *largest* model. Smaller models use the same flow dimension but have their inactive slots masked. `mk_to_mask` is the key: it returns a `[N, D]` binary tensor telling the flow which dimensions are active for each sample.

For a problem where model $$k$$ has $$p_k$$ parameters and $$p_1 < p_2 < \cdots < p_K$$:

```python
def num_inputs(self): return self.p_max

def mk_to_mask(self, mk):
    k = mk.argmax(dim=-1)        # [N]
    p_k = self.p_of_k[k]        # [N]  precomputed lookup: p_k[i] = dim of model i
    arange = torch.arange(self.p_max, device=self.device)
    return (arange < p_k.unsqueeze(-1)).float()   # [N, p_max]
```

The mask doesn't need to be contiguous. Variable selection with an arbitrary subset of $$p$$ predictors active:

```python
def mk_to_mask(self, mk):
    # mk is [N, p] binary — directly the active predictor mask
    return mk.float()
```

---

## Model index ≠ number of components

The integer category index is just a label. `log_prob` and `mk_to_mask` are free to interpret it arbitrarily.

**Example: two structurally different model families**

Regression with 10 models: linear (categories 0–4, 1–5 predictors) and quadratic (categories 5–9, 1–5 predictors):

```python
def num_categories(self): return 10
def num_inputs(self): return 10   # 5 linear params + 5 quadratic params

def mk_to_mask(self, mk):
    k = mk.argmax(dim=-1)            # [N]
    is_quad = (k >= 5).float()
    n_pred  = (k % 5) + 1            # 1-5 predictors
    arange  = torch.arange(5, device=self.device)
    linear_mask = (arange < n_pred.unsqueeze(-1)).float()  # [N, 5]
    quad_mask   = linear_mask * is_quad.unsqueeze(-1)       # [N, 5]
    return torch.cat([linear_mask, quad_mask], dim=-1)      # [N, 10]

def log_prob(self, mk, theta):
    k = mk.argmax(dim=-1)
    is_quad = k >= 5
    n_pred  = (k % 5) + 1
    linear_params = theta[:, :5]
    quad_params   = theta[:, 5:]
    # compute likelihood using n_pred linear params,
    # plus n_pred quadratic params if is_quad ...
```

---

## Model 1 has more parameters than model 2

No special handling needed — the mask for each model index is fully independent. A lower category index can have a larger mask than a higher one.

**Example: base model with shared params plus model-specific extras**

Model 0 has $$p$$ shared + 5 extra parameters. Models 1–4 have $$p$$ shared + $$k$$ extra:

```python
def mk_to_mask(self, mk):
    k = mk.argmax(dim=-1)            # [N]
    p = self.p_shared
    extra = torch.where(k == 0, torch.tensor(5), k)
    shared_mask = torch.ones(mk.shape[0], p, device=self.device)
    extra_mask  = (torch.arange(self.max_extra, device=self.device)
                   < extra.unsqueeze(-1)).float()
    return torch.cat([shared_mask, extra_mask], dim=-1)
```

---

## Non-one-hot context

The context can be any function of the model index that's useful to the coupling MLPs. `num_context_features()` must return the actual width of whatever `mk_to_context` returns. For example, an embedding that also encodes the number of active parameters:

```python
def num_context_features(self): return self.max_components + 1

def mk_to_context(self, mk):
    k = mk.argmax(dim=-1).float()
    one_hot = F.one_hot(k.long(), self.max_components).float()
    k_norm  = (k / (self.max_components - 1)).unsqueeze(-1)   # [N, 1]
    return torch.cat([one_hot, k_norm], dim=-1)                # [N, K+1]
```

A useful rule of thumb: context width ≈ flow input width. If context is too narrow relative to `num_inputs`, the coupling MLPs can't distinguish models well and the flow will conflate posteriors across models.

---

## Choosing `flow_type`

| Scenario | Recommendation |
|---|---|
| Small model space ($$K \leq 20$$), unimodal posteriors | `"diagnorm"` — fast, interpretable |
| Standard trans-dim ($$K \leq 50$$, $$D \leq 100$$) | `"affine33"` or `"affine55"` |
| Large model space or complex posteriors | `"affine55"` with float64 |
| Multimodal per-model posteriors | `"spline13"` or `"spline33"` |
| Heavy-tailed per-model posteriors | `"sas3"` |

The hidden feature size in `affine{N}{M}` layers scales with `num_inputs`. For very large `num_inputs` ($$> 200$$), consider reducing the number of layers (e.g. `"affine33"`) before increasing `flow_lr` or reducing `BATCH_SIZE`.

---


# Numerical notes

## Why float64?

Required for $$K > 40$$ (flow dimension $$> 80$$). `affine55` stacks 10+ Masked Affine Autoregressive Transform (MAAT) layers, each contributing a sum of NN scalar outputs to the total $$\log\vert\det J_{\text{flow}}\vert$$. At 150 dims × 10 layers that's 1500 terms in the log-det gradient chain. In float32 ($$\sim 10^{-7}$$ relative error), rounding accumulates enough over $$\sim 100$$ Adam steps to corrupt the gradient direction. Float64 ($$\sim 10^{-15}$$) has 8 orders of magnitude more headroom. Float32 is fine and ~2× faster for $$K \leq 20$$.

## Chunked data likelihood

The tensor `[batch, chunk, K, 2]` in float64 uses `batch × chunk × K × 2 × 8` bytes. At $$K=75$$, batch=256: `307,200 × chunk` bytes. Capping at 50 MB gives `chunk ≤ 162`, hence `max(8, 12000 // K_max) = 160`. Without chunking the full $$D=4000$$ would allocate ~1.2 GB per forward pass.

The float64 *accumulator* for `data_log_lik` is a separate fix: summing $$D$$ values each $$O(\log K)$$ in float32 gives a total $$O(16{,}000)$$, where float32 absolute precision is $$16{,}000 \times 10^{-7} \approx 0.002$$. Gradients of magnitude $$O(1)$$ sit close to this noise floor near convergence. The accumulator is the only float64 object; all chunk tensors stay in the working dtype.

## The normalisation trap

An earlier version divided `log_prob` by `n_obs`. This is wrong. VTI computes:

$$\text{loss} = -\log p(\mathbf{y}, \boldsymbol{\theta}, k) + \underbrace{\log p_{\text{ref}}(\mathbf{z}) - \log|\det J_{\text{flow}}|}_{\text{params\_log\_prob}}$$

`params_log_prob` is $$O(2K) \approx O(150)$$ regardless of $$N$$. Dividing `log_prob` by $$N=1000$$ makes the data term $$O(0.04)$$, leaving the gradient dominated by $$-\partial \log\vert\det J_{\text{flow}}\vert/\partial \text{flow}$$. The flow learns to maximise its log-determinant (volume expansion) rather than fit data, and diverges. The `_NormalisedSurrogate` with `scale=N` handles the large ELBO values on the surrogate side where they belong.

---

# Footnotes

[^vti]: Davies, L., Mackinlay, D., Oliveira, R., & Sisson, S. A. (2025). Variational Trans-dimensional Inference. *arXiv:2501.12280*.