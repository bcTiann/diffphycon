# An Introduction to Flow Matching and Diffusion Models

**Authors:** Peter Holderrieth and Ezra Erives
**Course:** MIT 6.S184: Generative AI With Stochastic Differential Equations, 2026
**Website:** https://diffusion.csail.mit.edu/

---

## Table of Contents

1. Introduction
2. Flow and Diffusion Models
3. Flow Matching
4. Score Functions and Score Matching
5. Guidance: How To Condition on a Prompt
6. Building Large-Scale Image or Video Generators
7. Discrete Diffusion Models
8. References
- Appendix A: Probability Theory Reminder
- Appendix B: Proof of the Fokker-Planck equation
- Appendix C: Existence and Uniqueness of CTMCs
- Appendix D: Additional Perspectives on VAEs
- Appendix E: A Guide to the Diffusion Model Literature

---

# 1 Introduction

> *Creating noise from data is easy; creating data from noise is generative modeling.* — Song et al.

## 1.1 Overview

The goal of this class is to teach two of the most widely used generative AI algorithms: **denoising diffusion models** and **flow matching**. These models are the backbone of the best image, audio, and video generation models (e.g., Nano Banana, FLUX, VEO-3), and have most recently become state-of-the-art in scientific applications such as protein structures (e.g., AlphaFold3 is a diffusion model).

All of these generative models generate objects by iteratively converting noise into data. This evolution from noise to data is facilitated by the simulation of **ordinary or stochastic differential equations (ODEs/SDEs)**. Flow matching and denoising diffusion models are a family of techniques that allow us to construct, train, and simulate such ODEs/SDEs at large scale with deep neural networks.

## 1.2 Course Structure

- **Section 1, Generative Modeling as Sampling:** Translate the problem of generation into the more precise problem of sampling from a probability distribution.
- **Section 2, Flow and Diffusion Models:** Introduction to ODEs/SDEs and how to use them to construct generative models.
- **Section 3, Flow Matching:** A simple and scalable algorithm at the core of all afore-mentioned large-scale generative models.
- **Section 4, Score Matching:** Score functions and how they can be learnt via score matching. Training algorithm for diffusion models.
- **Section 5, Guidance:** How to condition samples on a prompt via classifier-free guidance.
- **Section 6, Latent Spaces & Architectures:** How to build large-scale image and video generators.
- **Section 7 (Optional), Discrete Diffusion Models:** Building language models using diffusion principles.

## 1.3 Generative Modeling As Sampling

**Data modalities:**
1. **Image:** $z \in \mathbb{R}^{H \times W \times 3}$
2. **Video:** $z \in \mathbb{R}^{T \times H \times W \times 3}$
3. **Molecular structure:** $z = (z_1, \ldots, z_N) \in \mathbb{R}^{3 \times N}$

**Key Idea 1 (Objects as Vectors):** We identify the objects being generated as vectors $z \in \mathbb{R}^d$.

**Key Idea 2 (Generation as Sampling):** Generating an object $z$ is modeled as sampling from the data distribution $z \sim p_{\text{data}}$.

**Key Idea 3 (Dataset):** A dataset consists of a finite number of samples $z_1, \ldots, z_N \sim p_{\text{data}}$.

**Key Idea 4 (Guided Generation):** Guided generation involves sampling from $z \sim p_{\text{data}}(\cdot|y)$, where $y$ is a conditioning variable.

---

# 2 Flow and Diffusion Models

## 2.1 Flow Models

A solution to an ODE is defined by a **trajectory** $X : [0, 1] \to \mathbb{R}^d$, $t \mapsto X_t$.

Every ODE is defined by a **vector field** $u : \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$, $(x, t) \mapsto u_t(x)$.

**ODE:**
$$\frac{d}{dt} X_t = u_t(X_t), \quad X_0 = x_0$$

The **flow** $\psi_t$ is the solution map:
$$\frac{d}{dt} \psi_t(x_0) = u_t(\psi_t(x_0)), \quad \psi_0(x_0) = x_0$$

**Theorem 3 (Flow existence and uniqueness):** If $u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is continuously differentiable with a bounded derivative, then the ODE has a unique solution given by a flow $\psi_t$, which is a diffeomorphism for all $t$.

**Example 4 (Linear Vector Fields):** For $u_t(x) = -\theta x$ with $\theta > 0$:
$$\psi_t(x_0) = \exp(-\theta t) x_0$$

### Simulating an ODE

**Euler method** with step size $h = 1/n$:
$$X_{t+h} = X_t + h u_t(X_t)$$

**Heun's method:**
$$X'_{t+h} = X_t + h u_t(X_t)$$
$$X_{t+h} = X_t + \frac{h}{2}(u_t(X_t) + u_{t+h}(X'_{t+h}))$$

### Flow Models (Definition)

A flow model is described by:
$$X_0 \sim p_{\text{init}}, \quad \frac{d}{dt} X_t = u_t^\theta(X_t)$$

Goal: $X_1 \sim p_{\text{data}}$, i.e., $\psi_1^\theta(X_0) \sim p_{\text{data}}$.

**Algorithm 1: Sampling from a Flow Model (Euler method)**
```
Require: Neural network vector field u^θ_t, number of steps n
1: Set t = 0
2: Set step size h = 1/n
3: Draw X_0 ~ p_init
4: for i = 1, ..., n do
5:   X_{t+h} = X_t + h * u^θ_t(X_t)
6:   t ← t + h
7: end for
8: return X_1
```

## 2.2 Diffusion Models

A **stochastic process** $(X_t)_{0 \le t \le 1}$ is a family of random variables, one per time $t$.

### Brownian Motion

A **Brownian motion** $W = (W_t)_{0 \le t \le 1}$ is a stochastic process such that:
- $W_0 = 0$
- Trajectories $t \mapsto W_t$ are continuous
- **Normal increments:** $W_t - W_s \sim \mathcal{N}(0, (t-s) I_d)$ for $0 \le s < t$
- **Independent increments:** For $0 \le t_0 < t_1 < \cdots < t_n = 1$, the increments $W_{t_1} - W_{t_0}, \ldots, W_{t_n} - W_{t_{n-1}}$ are independent

Approximate simulation with step size $h > 0$:
$$W_{t+h} = W_t + \sqrt{h} \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)$$

### From ODEs to SDEs

The infinitesimal update form of an SDE:
$$X_{t+h} = X_t + \underbrace{h u_t(X_t)}_{\text{deterministic}} + \underbrace{\sigma_t (W_{t+h} - W_t)}_{\text{stochastic}} + \underbrace{h R_t(h)}_{\text{error}}$$

where $\sigma_t \ge 0$ is the **diffusion coefficient**. Symbolic notation:
$$dX_t = u_t(X_t) dt + \sigma_t dW_t, \quad X_0 = x_0$$

**Theorem 5 (SDE Solution Existence and Uniqueness):** If $u$ is continuously differentiable with bounded derivative and $\sigma_t$ is continuous, then the SDE has a unique solution.

Every ODE is also an SDE (with $\sigma_t = 0$).

**Example 6 (Ornstein-Uhlenbeck Process):** With $\sigma_t = \sigma$ and $u_t(x) = -\theta x$:
$$dX_t = -\theta X_t dt + \sigma dW_t$$
Converges to $\mathcal{N}(0, \sigma^2/(2\theta))$ as $t \to \infty$.

### Simulating an SDE: Euler-Maruyama method

$$X_{t+h} = X_t + h u_t(X_t) + \sqrt{h} \sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)$$

### Diffusion Models (Definition)

$$X_0 \sim p_{\text{init}}, \quad dX_t = u_t^\theta(X_t) dt + \sigma_t dW_t$$

**Algorithm 2: Sampling from a Diffusion Model (Euler-Maruyama)**
```
Require: NN u^θ_t, number of steps n, diffusion coefficient σ_t
1: Set t = 0; step size h = 1/n
2: Draw X_0 ~ p_init
3: for i = 1, ..., n do
4:   Draw ε ~ N(0, I_d)
5:   X_{t+h} = X_t + h * u^θ_t(X_t) + σ_t * √h * ε
6:   t ← t + h
7: end for
8: return X_1
```

A diffusion model with $\sigma_t = 0$ is a flow model.

---

# 3 Flow Matching

In Section 2, we constructed flow/diffusion models as generative models parameterized by neural network vector fields $u_t^\theta$. **Flow matching** is the algorithm for training $u_t^\theta$ — simple, scalable, and state-of-the-art.

We restrict to flow models:
$$X_0 \sim p_{\text{init}}, \quad dX_t = u_t^\theta(X_t) dt$$

Goal: $X_1 \sim p_{\text{data}}$.

## 3.1 Conditional and Marginal Probability Path

Let $\delta_z$ denote the Dirac delta. A **conditional (interpolating) probability path** is a set of distributions $p_t(x|z)$ over $\mathbb{R}^d$ such that:
$$p_0(\cdot|z) = p_{\text{init}}, \quad p_1(\cdot|z) = \delta_z \quad \text{for all } z \in \mathbb{R}^d$$

The induced **marginal probability path**:
$$z \sim p_{\text{data}}, \ x \sim p_t(\cdot|z) \Rightarrow x \sim p_t$$
$$p_t(x) = \int p_t(x|z) p_{\text{data}}(z) \, dz$$

We can sample from $p_t$ but cannot compute the density $p_t(x)$ (intractable integral). The marginal path interpolates between noise and data:
$$p_0 = p_{\text{init}}, \quad p_1 = p_{\text{data}}$$

**Example 8 (Gaussian Conditional Probability Path):** Let $\alpha_t, \beta_t$ be **noise schedulers**: continuously differentiable, monotonic, with $\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$. Define:
$$p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)$$

Sampling from the marginal:
$$z \sim p_{\text{data}}, \ \epsilon \sim \mathcal{N}(0, I_d) \Rightarrow x = \alpha_t z + \beta_t \epsilon \sim p_t$$

## 3.2 Conditional and Marginal Vector Fields

For every data point $z$, let $u_t^{\text{target}}(\cdot|z)$ denote a **conditional vector field** such that:
$$X_0 \sim p_{\text{init}}, \ \frac{d}{dt} X_t = u_t^{\text{target}}(X_t|z) \Rightarrow X_t \sim p_t(\cdot|z)$$

**Theorem 9 (Marginalization trick):** The **marginal vector field**:
$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz$$

follows the marginal probability path:
$$X_0 \sim p_{\text{init}}, \ \frac{d}{dt} X_t = u_t^{\text{target}}(X_t) \Rightarrow X_t \sim p_t$$

In particular $X_1 \sim p_{\text{data}}$.

**Example 10 (Target ODE for Gaussian probability paths):** With $\dot\alpha_t = \partial_t \alpha_t$, $\dot\beta_t = \partial_t \beta_t$, the conditional Gaussian vector field is:
$$u_t^{\text{target}}(x|z) = \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t} \alpha_t\right) z + \frac{\dot\beta_t}{\beta_t} x$$

*Proof sketch:* Define the conditional flow $\psi_t^{\text{target}}(x|z) = \alpha_t z + \beta_t x$. Then if $X_0 \sim \mathcal{N}(0, I_d)$:
$$X_t = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot|z)$$

Taking the time derivative and using the flow ODE definition gives the result.

**Intuition for the marginal vector field:** Using Bayes' rule, $\frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)}$ is the posterior over data points $z$ given noisy data $x$. The marginal vector field is the *average* of conditional velocities, weighted by this posterior.

### The Continuity Equation

Define the divergence:
$$\text{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t^i(x)$$

**Theorem 11 (Continuity Equation):** For a flow model with vector field $u_t^{\text{target}}$ and $X_0 \sim p_{\text{init}} = p_0$, then $X_t \sim p_t$ for all $0 \le t \le 1$ if and only if:
$$\partial_t p_t(x) = -\text{div}(p_t u_t^{\text{target}})(x)$$

*Proof of Theorem 9 (Marginalization trick):* By the continuity equation, we show:
$$\partial_t p_t(x) = \partial_t \int p_t(x|z) p_{\text{data}}(z) dz = \int \partial_t p_t(x|z) p_{\text{data}}(z) dz$$
$$= \int -\text{div}(p_t(\cdot|z) u_t^{\text{target}}(\cdot|z))(x) p_{\text{data}}(z) dz$$
$$= -\text{div}\left(p_t(x) \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz\right)$$
$$= -\text{div}(p_t u_t^{\text{target}})(x) \qquad \square$$

## 3.3 Learning the Marginal Vector Field

**Flow matching loss:**
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}\left[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2\right]$$

Cannot compute directly since $u_t^{\text{target}}$ is intractable.

**Conditional flow matching loss:**
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}\left[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\right]$$

**Theorem 12:** $\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C$ where $C$ is independent of $\theta$. Hence $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$.

*Proof sketch:* Expand $\|a-b\|^2 = \|a\|^2 - 2a^T b + \|b\|^2$. The key cross term:
$$\mathbb{E}_{t,x \sim p_t}[u_t^\theta(x)^T u_t^{\text{target}}(x)] = \int_0^1 \int p_t(x) u_t^\theta(x)^T \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz \, dx \, dt$$
$$= \mathbb{E}_{t, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}[u_t^\theta(x)^T u_t^{\text{target}}(x|z)]$$

The marginal vector field becomes the conditional vector field inside the expectation. $\square$

**Algorithm 3: Flow Matching Training (Gaussian CondOT path $p_t(x|z) = \mathcal{N}(tz, (1-t)^2)$)**
```
Require: Dataset z ~ p_data, neural network u^θ_t
1: for each mini-batch do
2:   Sample data example z
3:   Sample t ~ Unif[0,1]
4:   Sample noise ε ~ N(0, I_d)
5:   x = t*z + (1-t)*ε       (general case: x ~ p_t(·|z))
6:   L(θ) = ||u^θ_t(x) - (z - ε)||²     (general case: ||u^θ_t(x) - u_t^target(x|z)||²)
7:   θ ← grad_update(L(θ))
8: end for
```

Key features:
- **Simulation-free:** Never simulate ODE during training
- Simple regression objective
- Used in Stable Diffusion 3, Meta's Movie Gen Video, and many other state-of-the-art models

**Example 13 (Flow Matching for Gaussian paths):** Sample $\epsilon \sim \mathcal{N}(0, I_d) \Rightarrow x_t = \alpha_t z + \beta_t \epsilon$. The loss becomes:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\|u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot\alpha_t z + \dot\beta_t \epsilon)\|^2\right]$$

For $\alpha_t = t, \beta_t = 1-t$ (Gaussian CondOT path): $\dot\alpha_t = 1, \dot\beta_t = -1$:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\|u_t^\theta(tz + (1-t)\epsilon) - (z - \epsilon)\|^2\right]$$

### Summary 14: Flow Matching

Choose a conditional probability path $p_t(x|z)$ with $p_0(\cdot|z) = p_{\text{init}}$, $p_1(\cdot|z) = \delta_z$. Find conditional vector field $u_t^{\text{target}}(x|z)$ satisfying:
$$X_0 \sim p_{\text{init}} \Rightarrow X_t = \psi_t^{\text{target}}(X_0|z) \sim p_t(\cdot|z)$$

The marginal vector field:
$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz$$

follows the marginal probability path. Train via:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, z, x}\left[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\right]$$

**Gaussian path summary:**
$$p_t(x|z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$$
$$u_t^{\text{flow}}(x|z) = \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right) z + \frac{\dot\beta_t}{\beta_t} x$$
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}\left[\|u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot\alpha_t z + \dot\beta_t \epsilon)\|^2\right]$$

---

# 4 Score Functions and Score Matching

## 4.1 Conditional and Marginal Score Functions

For an arbitrary distribution $q(x)$, the **score function** is $\nabla \log q(x)$ — the direction of steepest ascent of log-likelihood.

- **Conditional score:** $\nabla \log p_t(x|z)$
- **Marginal score:** $\nabla \log p_t(x)$

Marginal-conditional relation:
$$\nabla \log p_t(x) = \int \nabla \log p_t(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz$$

*Derivation:*
$$\nabla \log p_t(x) = \frac{\nabla p_t(x)}{p_t(x)} = \frac{\int \nabla p_t(x|z) p_{\text{data}}(z) dz}{p_t(x)} = \int \nabla \log p_t(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz$$

**Example 15 (Gaussian Score):** For $p_t(x|z) = \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d)$:
$$\nabla \log p_t(x|z) = -\frac{x - \alpha_t z}{\beta_t^2}$$

**Proposition 1 (Conversion Formula for Gaussian paths):**
$$u_t^{\text{target}}(x|z) = a_t \nabla \log p_t(x|z) + b_t x$$
$$u_t^{\text{target}}(x) = a_t \nabla \log p_t(x) + b_t x$$
where $a_t = \beta_t^2 \frac{\dot\alpha_t}{\alpha_t} - \dot\beta_t \beta_t$ and $b_t = \frac{\dot\alpha_t}{\alpha_t}$.

*Proof:*
$$u_t^{\text{target}}(x|z) = \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right) z + \frac{\dot\beta_t}{\beta_t} x = \left(\beta_t^2 \frac{\dot\alpha_t}{\alpha_t} - \dot\beta_t \beta_t\right)\frac{\alpha_t z - x}{\beta_t^2} + \frac{\dot\alpha_t}{\alpha_t} x$$

The conditional vector field can be recovered from the conditional score, and vice versa. Once you've learned $u_t^{\text{target}}$, you've also learned $\nabla \log p_t$.

**Remark 16 (Reparameterization / Denoiser):** Define the **denoiser**:
$$D_t(x|z) = z, \quad D_t(x) = \int z \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} dz = \frac{1}{\dot\alpha_t \beta_t - \alpha_t \dot\beta_t}(\beta_t u_t^{\text{target}}(x_t) - \dot\beta_t x_t)$$

The denoiser is the expected value of clean data $z$ given noisy data $x$. Models that learn this are called **denoising diffusion models**.

## 4.2 Sampling with SDEs

**Theorem 17 (SDE Extension Trick):** For any diffusion coefficient $\sigma_t \ge 0$:
$$X_0 \sim p_{\text{init}}, \quad dX_t = \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\right] dt + \sigma_t dW_t$$
$$\Rightarrow X_t \sim p_t \quad (0 \le t \le 1)$$

In particular $X_1 \sim p_{\text{data}}$. The stochastic dynamics are related to **Langevin dynamics**.

In theory, any $\sigma_t \ge 0$ works. In practice, training error and simulation error mean there's an empirically optimal $\sigma_t$.

**Example 18 (Gaussian SDE Extension):** Using Proposition 1:
$$dX_t = \left[\left(a_t + \frac{\sigma_t^2}{2}\right) \nabla \log p_t(X_t) + b_t X_t\right] dt + \sigma_t dW_t$$

### Fokker-Planck Equation

Laplacian: $\Delta w_t(x) = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2} w_t(x) = \text{div}(\nabla w_t)(x)$

**Theorem 19 (Fokker-Planck Equation):** For SDE $dX_t = u_t(X_t) dt + \sigma_t dW_t$, then $X_t \sim p_t$ for all $0 \le t \le 1$ iff:
$$\partial_t p_t(x) = -\text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$

When $\sigma_t = 0$, recovers the continuity equation.

*Proof of Theorem 17:* Direct calculation shows the SDE satisfies the Fokker-Planck equation:
$$\partial_t p_t(x) = -\text{div}(p_t u_t^{\text{target}})(x) - \text{div}\left(p_t \frac{\sigma_t^2}{2} \nabla \log p_t\right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$
$$= -\text{div}\left(p_t \left[u_t^{\text{target}} + \frac{\sigma_t^2}{2} \nabla \log p_t\right]\right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \qquad \square$$

**Remark 20 (Langevin Dynamics):** When $p_t = p$ (constant), set $u_t^{\text{target}} = 0$:
$$dX_t = \frac{\sigma_t^2}{2} \nabla \log p(X_t) dt + \sigma_t dW_t$$
$p$ is the stationary distribution: $X_0 \sim p \Rightarrow X_t \sim p$. Under mild conditions, the dynamics converge to $p$ even from $X_0 \sim p' \ne p$. Basis for MCMC methods.

## 4.3 Score Matching

To learn the marginal score, use a **score network** $s_t^\theta : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$:

**Score matching loss:**
$$\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{t, z, x}\left[\|s_t^\theta(x) - \nabla \log p_t(x)\|^2\right]$$

**Conditional score matching loss (denoising score matching):**
$$\mathcal{L}_{\text{CSM}}(\theta) = \mathbb{E}_{t, z, x}\left[\|s_t^\theta(x) - \nabla \log p_t(x|z)\|^2\right]$$

**Theorem 22:** $\mathcal{L}_{\text{SM}}(\theta) = \mathcal{L}_{\text{CSM}}(\theta) + C$ where $C$ is independent of $\theta$.

*Proof:* Identical to proof of Theorem 12, replacing $u_t^{\text{target}}$ with $\nabla \log p_t$.

**Example 23 (Denoising Diffusion / DDPM):** For Gaussian paths:
$$\mathcal{L}_{\text{CSM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\left\|s_t^\theta(\alpha_t z + \beta_t \epsilon) + \frac{\epsilon}{\beta_t}\right\|^2\right] = \mathbb{E}\left[\frac{1}{\beta_t^2}\|\beta_t s_t^\theta(\alpha_t z + \beta_t \epsilon) + \epsilon\|^2\right]$$

Reparameterize to a **noise predictor** $\epsilon_t^\theta$: define $-\beta_t s_t^\theta(x) = \epsilon_t^\theta(x)$. Then (DDPM loss):
$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\|\epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon\|^2\right]$$

The network learns to predict the noise that corrupted the data.

**Algorithm 4: Score Matching Training (Gaussian path)**
```
Require: Dataset z ~ p_data, score network s^θ_t or noise predictor ε^θ_t
1: for each mini-batch do
2:   Sample z from dataset
3:   Sample t ~ Unif[0,1]
4:   Sample ε ~ N(0, I_d)
5:   x_t = α_t * z + β_t * ε
6:   L(θ) = ||s^θ_t(x_t) + ε/β_t||²
       (Alternative: L(θ) = ||ε^θ_t(x_t) - ε||²)
7:   θ ← grad_update(L(θ))
8: end for
```

### Summary 24

Score-based SDE sampling:
$$X_0 \sim p_{\text{init}}, \quad dX_t = \left[u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t)\right] dt + \sigma_t dW_t \Rightarrow X_t \sim p_t$$

For Gaussian paths, train via denoising score matching, then sample using:
$$dX_t = \left[\left(a_t + \frac{\sigma_t^2}{2}\right) s_t^\theta(X_t) + b_t X_t\right] dt + \sigma_t dW_t$$

---

# 5 Guidance: How To Condition on a Prompt

We want to sample from $p_{\text{data}}(z|y)$ — the **guided data distribution** conditioned on $y$ (e.g., a text prompt).

**Terminology note:** "Conditional" refers to conditioning on $z$ (probability path/vector field); "guided" refers to conditioning on $y$ (text prompt).

## 5.1 Vanilla Guidance

A **guided diffusion model** consists of:
- Neural network: $u^\theta : \mathbb{R}^d \times \mathcal{Y} \times [0,1] \to \mathbb{R}^d$, $(x, y, t) \mapsto u_t^\theta(x|y)$
- Diffusion coefficient: $\sigma_t$

Sampling:
$$X_0 \sim p_{\text{init}}, \quad dX_t = u_t^\theta(X_t|y) dt + \sigma_t dW_t$$

**Guided conditional flow matching objective:**
$$\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta) = \mathbb{E}_{(z,y) \sim p_{\text{data}}, t, x \sim p_t(\cdot|z)}\|u_t^\theta(x|y) - u_t^{\text{target}}(x|z)\|^2$$

Key difference from unguided: sample $(z, y) \sim p_{\text{data}}$ jointly rather than just $z$.

## 5.2 Classifier-Free Guidance

Vanilla guidance often fails to adhere strongly enough to the prompt. **Classifier-free guidance (CFG)** is the main fix used in state-of-the-art diffusion models.

### Classifier Guidance (Background)

Using Bayes' rule for Gaussian paths:
$$\nabla \log p_t(x|y) = \nabla \log p_t(x) + \nabla \log p_t(y|x)$$

Therefore:
$$u_t^{\text{target}}(x|y) = u_t^{\text{target}}(x) + a_t \nabla \log p_t(y|x)$$

**Classifier guidance** scales up the prompt-dependent term:
$$\tilde u_t(x|y) = u_t^{\text{target}}(x) + w a_t \nabla \log p_t(y|x)$$
where $w > 1$ is the **guidance scale**. Requires training a classifier $\log p_t(y|x)$ on noisy data.

### Classifier-Free Guidance

By rearranging:
$$\tilde u_t(x|y) = u_t^{\text{target}}(x) + w a_t (\nabla \log p_t(x|y) - \nabla \log p_t(x))$$
$$= (1-w) u_t^{\text{target}}(x) + w \, u_t^{\text{target}}(x|y)$$

This expresses the guided vector field as a linear combination of unguided and guided vector fields. **Both can be trained as one model** by augmenting the label set with $\emptyset$ (no conditioning):
$$u_t^{\text{target}}(x) = u_t^{\text{target}}(x|\emptyset)$$

During training, randomly drop the label $y \to \emptyset$ with probability $\eta$.

**Remark 26:** $\tilde u_t(x|y) = (1-w) u_t^{\text{target}}(x|\emptyset) + w u_t^{\text{target}}(x|y)$ works for any probability path, not just Gaussian. When $w = 1$, $\tilde u_t = u_t^{\text{target}}(x|y)$.

**Algorithm 5: CFG Training (Gaussian path)**
```
Require: Paired dataset (z, y) ~ p_data, neural network u^θ_t
1: for each mini-batch do
2:   Sample (z, y) from dataset
3:   Sample t ~ Unif[0,1]
4:   Sample ε ~ N(0, I_d)
5:   x = α_t * z + β_t * ε
6:   With probability η: y ← ∅
7:   L(θ) = ||u^θ_t(x|y) - (α̇_t * z + β̇_t * ε)||²
8:   θ ← grad_update(L(θ))
9: end for
```

### Summary 27: CFG for Flow Models

$$\tilde u_t(x|y) = (1-w) u_t^{\text{target}}(x|\emptyset) + w \, u_t^{\text{target}}(x|y)$$

**Training:**
- Sample $(z, y) \sim p_{\text{data}}$
- Sample $t \sim \text{Unif}[0,1]$, $x \sim p_t(\cdot|z)$
- With probability $\eta$, replace $y \leftarrow \emptyset$
- Regress $\|u_t^\theta(x|y) - u_t^{\text{target}}(x|z)\|^2$

**Inference:** Simulate ODE $dX_t = \tilde u_t^\theta(X_t|y) dt$ from $X_0 \sim p_{\text{init}}$.

Note: For $w > 1$, $X_1$ is NOT distributed exactly like $p_{\text{data}}(\cdot|y)$, but empirically gives better prompt adherence. CFG is a **heuristic** justified by excellent empirical results. Almost all AI-generated images/videos use $w \ge 4$.

**Remark 28:** Extension to diffusion models: replace $u_t^\theta(x|y)$ with $\tilde u_t^\theta(x|y)$ and sample via SDE (Section 4).

---

# 6 Building Large-Scale Image or Video Generators

This section covers Stable Diffusion 3, Meta Movie Gen Video, etc.

## 6.1 Neural Network Architectures

The model $u_t^\theta(x|y)$ has three inputs:
- $x \in \mathbb{R}^d$: vector input
- $y \in \mathcal{Y}$: conditioning variable
- $t \in [0,1]$: time

For low-dim data, an MLP suffices. For images/videos, use specialized architectures.

### 6.1.1 Embedding the Conditioning Variables

**Time embedding (Fourier features):**
$$\text{TimeEmb}(t) = \sqrt{\frac{2}{d}}\left[\cos(2\pi w_1 t), \ldots, \cos(2\pi w_{d/2} t), \sin(2\pi w_1 t), \ldots, \sin(2\pi w_{d/2} t)\right]^T$$
where $w_i = w_{\min}(w_{\max}/w_{\min})^{(i-1)/(d/2-1)}$. Note $\|\text{TimeEmb}(t)\| = 1$.

**Class label embedding:** For $y_{\text{raw}} \in \{0, \ldots, N\}$, learn a separate embedding vector for each value.

**Text embedding:** Use frozen, pre-trained models like:
- **CLIP** (Contrastive Language-Image Pre-training): shared image-text embedding space
- Pre-trained transformers (e.g., T5, UL2, ByT5) for sequence-level embeddings

Common shape: $\text{PromptEmbed}(y_{\text{raw}}) \in \mathbb{R}^{S \times k}$.

### 6.1.2 Diffusion Transformers (DiT)

An image $x \in \mathbb{R}^{C \times H \times W}$. Notation: $d$ = hidden dim, $L$ = transformer layers, $h$ = heads per layer.

**Patchification:**
$$\text{Patchify}(x) \in \mathbb{R}^{N \times C'}, \quad C' = C P^2, \quad N = (H/P)(W/P)$$

**Patch embedding:** $\text{PatchEmb}(x) = \text{Patchify}(x) W \in \mathbb{R}^{N \times d}$ with $W \in \mathbb{R}^{C' \times d}$.

**Inputs to DiT:**
$$\tilde t = \text{TimeEmb}(t) \in \mathbb{R}^d$$
$$\tilde y = \text{PromptEmb}(y) \in \mathbb{R}^{S \times d}$$
$$\tilde x_0 = \text{PatchEmb}(x) \in \mathbb{R}^{N \times d}$$

**Iterative updates:**
$$\tilde x_{i+1} = \text{DiTBlock}(\tilde x_i, \tilde t, \tilde y) \in \mathbb{R}^{N \times d}, \quad i = 0, \ldots, L-1$$

**Output:** $u = \text{Depatchify}(\tilde x_L \tilde W) \in \mathbb{R}^{C \times H \times W}$.

**Remark 29 (DiT Block):**

Scaled dot product attention:
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right) V$$

Multi-head attention with $h$ heads, $d_h = d/h$:
$$\text{head}_h(x, z) = \text{Attn}(x W_Q^{(h)}, z W_K^{(h)}, z W_V^{(h)})$$
$$\text{MultiHead}(x, z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

Self-attention: $z = x$. Cross-attention to prompt: $z = y$.

**Adaptive Layer Normalization (AdaLN):** Let $g: \mathbb{R}^d \to \mathbb{R}^{2d}$ be an MLP. Set $(\gamma, \beta) = g(\tilde t)$. Then:
$$\text{AdaNorm}_{\tilde t}(x) = (1 + \gamma) \odot \text{Norm}(x) + \beta$$

**Combined DiT block:**
$$x \leftarrow x + g_{\text{self}}(\tilde t) \odot \text{MultiHead}(\text{AdaNorm}_{\tilde t}(x), \text{AdaNorm}_{\tilde t}(x))$$
$$x \leftarrow x + g_{\text{cross}}(\tilde t) \cdot \text{MultiHead}(\text{AdaNorm}_{\tilde t}(x), y)$$
$$x \leftarrow x + g_{\text{MLP}}(\tilde t) \cdot \text{MLP}(\text{AdaNorm}_{\tilde t}(x))$$

Class-conditioned DiTs typically drop cross-attention and use time+class AdaNorm.

### 6.1.3 U-Net

The U-Net architecture is an alternative — a convolutional network whose input and output both have image shape. Used in early diffusion model literature (DDPM, etc.).

Components:
- Encoders $E_i$: reduce spatial dims, increase channels
- Midcoder $M$: latent processing block
- Decoders $D_i$: restore spatial dims, decrease channels
- Residual connections between encoders and decoders

Example flow:
$$x_t^{\text{input}} \in \mathbb{R}^{3 \times 256 \times 256} \to x_t^{\text{latent}} = E(\cdot) \in \mathbb{R}^{512 \times 32 \times 32}$$
$$\to M(x_t^{\text{latent}}) \in \mathbb{R}^{512 \times 32 \times 32} \to D(\cdot) = x_t^{\text{output}} \in \mathbb{R}^{3 \times 256 \times 256}$$

Modern U-Nets often include attention layers in encoders/decoders.

## 6.2 Working in Latent Space: (Variational) Autoencoders

A $1024 \times 1024$ RGB image has $d \approx 3 \times 10^6$ — too large to model directly. **Solution: compress into latent space.**

### 6.2.1 Standard Autoencoders

- **Encoder:** $\mu_\phi : \mathbb{R}^d \to \mathbb{R}^k$
- **Decoder:** $\mu_\theta : \mathbb{R}^k \to \mathbb{R}^d$ (with $k \ll d$)

**Reconstruction loss:**
$$\mathcal{L}_{\text{Recon}}(\phi, \theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\|\mu_\theta(\mu_\phi(x)) - x\|^2\right]$$

**Problem:** No control over the latent distribution $p_{\text{latent}}(z) = $ distribution of $z = \mu_\phi(x)$ for $x \sim p_{\text{data}}$. May be hard to learn a generative model in latent space.

### 6.2.2 Variational Autoencoders (VAE)

Relax determinism: stochastic encoder/decoder.

**Standard choice (Gaussian):**
$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$
$$p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma_\theta^2(z) I_d)$$

**Reconstruction loss:**
$$\mathcal{L}_{\text{VAE-Recon}}(\phi, \theta) = -\mathbb{E}_{x \sim p_{\text{data}}, z \sim q_\phi(\cdot|x)}[\log p_\theta(x|z)]$$

For Gaussian decoder:
$$\mathcal{L}_{\text{VAE-Recon}}(\phi, \theta) = \mathbb{E}\left[\frac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2 + \frac{d}{2}\log \sigma_\theta^2(z)\right] + \text{const}$$

**Prior loss:** Introduce prior $p_{\text{prior}}(z) = \mathcal{N}(0, I_k)$. Regularize encoder via:
$$\mathcal{L}_{\text{VAE-Prior}}(\phi) = \mathbb{E}_{x \sim p_{\text{data}}}[D_{\text{KL}}(q_\phi(\cdot|x) \| p_{\text{prior}})]$$

**Total VAE loss:**
$$\mathcal{L}_{\text{VAE}}(\phi, \theta) = \mathcal{L}_{\text{VAE-Recon}}(\phi, \theta) + \beta \mathcal{L}_{\text{VAE-Prior}}(\phi)$$

**Remark 30 (KL divergence):** $D_{\text{KL}}(q \| p) = \int q(x) \log(q(x)/p(x)) dx = \mathbb{E}_q[\log(q/p)]$. Properties: $\ge 0$ always, $= 0$ iff $q = p$.

**Example 31 (KL between Gaussians):** For $q = \mathcal{N}(\mu_q, \text{diag}(\sigma_q^2))$, $p = \mathcal{N}(\mu_p, \text{diag}(\sigma_p^2))$:
$$D_{\text{KL}}(q \| p) = \frac{1}{2}\left(K\left(\frac{\sigma_q^2}{\sigma_p^2}\right) + \frac{\|\mu_q - \mu_p\|^2}{\sigma_p^2}\right), \quad K(\alpha) = \sum_i \alpha_i - \log \alpha_i - 1$$

With Gaussian encoder and standard normal prior:
$$\mathcal{L}_{\text{VAE-Prior}}(\phi) = \mathbb{E}\left[\frac{1}{2} K(\sigma_\phi^2(x)) + \frac{1}{2}\|\mu_\phi(x)\|^2\right]$$

**Total loss** (intuitive form):
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}\left[\underbrace{\frac{1}{2\sigma_\theta^2(z)}\|x - \mu_\theta(z)\|^2}_{\text{recon error}} + \underbrace{\frac{d}{2}\log \sigma_\theta^2(z)}_{\text{decoder confidence}} + \underbrace{\frac{\beta}{2} K(\sigma_\phi^2(x))}_{\text{latent var}=1} + \underbrace{\frac{\beta}{2}\|\mu_\phi(x)\|^2}_{\text{latent mean}=0}\right]$$

**Reparameterization trick:** Sample $z \sim q_\phi(\cdot|x)$ as $z = \mu_\phi(x) + \sigma_\phi(x) \epsilon$ with $\epsilon \sim \mathcal{N}(0, I_k)$. Now $\epsilon$ doesn't depend on $\phi$, so gradients flow.

**Algorithm 6: β-VAE Training** (Gaussian decoder with fixed variance)
```
Require: Dataset x ~ p_data, encoder (μ_φ, log σ²_φ), decoder μ_θ, latent dim k, β ≥ 0, σ² > 0
1: for each mini-batch {x_i} do
2:   Encode: μ_i ← μ_φ(x_i), log σ²_i ← log σ²_φ(x_i)
3:   Sample ε_i ~ N(0, I_k)
4:   Reparameterize: z_i ← μ_i + σ_i ⊙ ε_i
5:   Decode mean: x̂_i ← μ_θ(z_i)
6:   L_recon ← (1/B) Σ (1/(2σ̃²)) ||x_i - x̂_i||²
7:   L_KL ← (1/B) Σ (1/2) Σ_j (μ²_{i,j} + σ²_{i,j} - log σ²_{i,j} - 1)
8:   L ← L_recon + β * L_KL
9:   (φ, θ) ← grad_update(L)
10: end for
```

**Practical remarks:**
1. **β choice:** Large β can cause posterior collapse. Modern autoencoders use very small β.
2. **Decoder variance:** Often fix $\sigma_\theta^2 = $ constant for stability.
3. **Reconstruction loss:** Pixel MSE often too smooth; add perceptual losses (features from pre-trained nets).
4. **Adversarial losses:** Combine with GAN discriminator (VAE-GAN) for sharper outputs.

**Remark 32 (Working in Latent Space):** At training, sample $z \sim q_\phi(z|x)$ with $x \sim p_{\text{data}}$. At inference, sample $z$ from latent generative model, decode via $x = \mu_{\text{mean}}(z)$ (mean, not random sample). Nearly all state-of-the-art image/video generators use this **latent diffusion** paradigm.

## 6.3 Case Study: Stable Diffusion 3 and Meta Movie Gen

### 6.3.1 Stable Diffusion 3

- Uses **conditional flow matching** objective (Algorithm 4 in the paper, equivalent to our Algorithm 3)
- **Classifier-free guidance** training with label dropping
- Operates in **latent space** of a pre-trained autoencoder
- **Three text embeddings:** CLIP (coarse) + T5-XXL encoder outputs (granular, sequential)
- **Multi-modal DiT (MM-DiT):** Extended DiT attending to both image patches AND text embeddings
- Largest model: **8 billion parameters**
- Sampling: 50 Euler steps with CFG weight 2.0–5.0

### 6.3.2 Meta Movie Gen Video

Data: $x \in \mathbb{R}^{T \times C \times H \times W}$ (videos with $T$ frames).

- Uses **conditional flow matching** with straight line schedulers $\alpha_t = t$, $\beta_t = 1 - t$
- Operates in latent space of pretrained **temporal autoencoder (TAE)** mapping $\mathbb{R}^{T' \times 3 \times H' \times W'} \to \mathbb{R}^{T \times C \times H \times W}$ with $T'/T = H'/H = W'/W = 8$
- **Temporal tiling:** Chop video into pieces, encode separately, stitch latents
- DiT-like backbone with self-attention on image patches + cross-attention to language embeddings
- **Three text embeddings:** UL2 (granular reasoning) + ByT5 (character-level) + MetaCLIP (image-text shared space)
- Largest model: **30 billion parameters**

---

# 7 Discrete Diffusion Models: Building Language Models with Diffusion

Not all data is naturally in Euclidean space — text and DNA are sequences of discrete tokens. The principles of flow matching extend to discrete state spaces via **Continuous-Time Markov Chains (CTMCs)** instead of SDEs.

## 7.1 Continuous-Time Markov chain (CTMC) models

**State space:** $S = \mathcal{V}^d$ where $\mathcal{V} = \{v_1, \ldots, v_V\}$ is a vocabulary, $d$ is sequence length.

**Markov property:**
$$p(X_{t+h} | X_t, X_{t_1}, \ldots, X_{t_k}) = p(X_{t+h} | X_t)$$

A CTMC is fully determined by **transition probabilities** $p_{t+h|t}(X_{t+h}|X_t)$ and initial distribution $X_0 \sim p_0$.

**Rate matrix:** Function $Q : S \times S \times [0,1] \to \mathbb{R}$, $Q_t(y|x)$ describing rate of switching $x \to y$:
1. **Outgoing rates positive:** $Q_t(y|x) \ge 0$ for $x \ne y$
2. **Consistency:** $Q_t(x|x) = -\sum_{y \ne x} Q_t(y|x)$

The CTMC follows the rate matrix if:
$$\frac{d}{dh} p_{t+h|t}(X_{t+h} = y | X_t = x)\bigg|_{h=0} = Q_t(y|x)$$

**Theorem 33 (CTMC existence and uniqueness):** For any bounded, time-continuous rate matrix $Q_t$, there is a unique Markov chain satisfying the rate equation.

**Example 34 (Two-state CTMC):** $S = \{a, b\}$ with rate $\lambda > 0$:
$$Q = \begin{pmatrix} -\lambda & \lambda \\ \lambda & -\lambda \end{pmatrix}$$
Transition probabilities:
$$P(h) = \frac{1}{2}\begin{pmatrix} 1 + e^{-2\lambda h} & 1 - e^{-2\lambda h} \\ 1 - e^{-2\lambda h} & 1 + e^{-2\lambda h} \end{pmatrix}$$
As $h \to \infty$, converges to uniform on $\{a, b\}$.

### Simulation of CTMC

Using the approximation:
$$p_{t+h|t}(X_{t+h} = y | X_t = x) \approx \mathbb{1}_{y=x} + h Q_t(y|x) =: \tilde p_{t+h|t}(y|x)$$

Sample: $X_{t+h} \sim \tilde p_{t+h|t}(\cdot | x)$ as a categorical distribution.

### CTMC Model & Factorization

A CTMC model: neural net $Q_t^\theta(y|x)$ returning a column of the rate matrix.

**Problem:** $|S| = V^d$ grows exponentially. Solution: **factorized CTMC** — only allow jumps between neighbors (states differing in at most one position):
$$Q_t^\theta(y|x) = 0 \text{ whenever } y_i \ne x_i \text{ for more than one position } i$$

Output shape: $d \times V$ (linear, not exponential).

Per-position constraints:
$$Q_t^\theta(v, i|x) \ge 0 \text{ if } v \ne x_i, \quad Q_t(x_i, i|x) = -\sum_{v \ne x_i} Q_t^\theta(v, i|x)$$

**Algorithm 7: Sampling from a Factorized CTMC**
```
Require: Rate network Q^θ_t (factorized), p_init, n steps
1: t ← 0, h ← 1/n
2: Draw X_0 = (X^(1)_0, ..., X^(d)_0) ~ p_init
3: for i = 1, ..., n do
4:   Compute rates {q_j(v)}_{j, v} ← Q^θ_t(· | X_t)
5:   for j = 1, ..., d (in parallel) do
6:     x ← X^(j)_t
7:     Define transition probabilities:
        p̃_{j,t}(v|x) = h*q_j(v) if v ≠ x
        p̃_{j,t}(v|x) = 1 - h*Σ_{v'≠x} q_j(v') if v = x
8:     Sample X^(j)_{t+h} ~ Categorical(p̃_{j,t}(·|x))
9:   end for
10:  t ← t + h
11: end for
12: return X_1
```

## 7.2 Training CTMC models

Same recipe as flow matching: (1) conditional probability path, (2) conditional rate matrix, (3) marginal rate matrix learnable simulation-free.

### 7.2.1 Conditional and Marginal Probability Path

A **discrete conditional probability path:** $p_t(x|z)$ with $p_0(\cdot|z) = p_{\text{init}}$, $p_1(\cdot|z) = \delta_z$.

**Marginal:**
$$p_t(x) = \sum_{z \in S} p_t(x|z) p_{\text{data}}(z)$$

Satisfies $p_0 = p_{\text{init}}$, $p_1 = p_{\text{data}}$.

**Example 35 (Factorized mixture path):** With $p_{\text{init}}(x) = \prod_j p_{\text{init}}^{(j)}(x_j)$ and scheduler $0 \le \kappa_t \le 1$, $\kappa_0 = 0$, $\kappa_1 = 1$:
$$p_t(x|z) = \prod_{j=1}^d \left[(1 - \kappa_t) p_{\text{init}}^{(j)}(x_j) + \kappa_t \delta_{z_j}(x_j)\right]$$

Equivalently, sample $x \sim p_t(\cdot|z)$ via:
$$m_j \sim \text{Bernoulli}(\kappa_t), \quad \xi_j \sim p_{\text{init}}^{(j)}$$
$$x_j = m_j z_j + (1 - m_j) \xi_j$$

The factorized mixture path "destroys" the $j$-th token independently with probability $1 - \kappa_t$. Unlike Gaussian paths, this doesn't transport mass — it just fades distributions in/out.

### 7.2.2 Conditional and Marginal Rate Matrix

A **conditional rate matrix** $Q_t^z(y|x)$ satisfies:
$$X_0 \sim p_{\text{init}}, \ X_t \text{ CTMC of } Q_t^z \Rightarrow X_t \sim p_t(\cdot|z)$$

**Theorem 36 (Discrete marginalization trick):** The **marginal rate matrix**:
$$Q_t(y|x) = \sum_{z \in S} Q_t^z(y|x) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} = \sum_z Q_t^z(y|x) p_{1|t}(z|x)$$

where $p_{1|t}(z|x) = p_t(x|z) p_{\text{data}}(z) / p_t(x)$. The marginal CTMC follows the marginal probability path.

**Proposition 2 (Kolmogorov Forward Equation):** $X_t \sim p_t$ for all $t$ iff:
$$\frac{d}{dt} p_t(x) = \sum_{y \in S} Q_t(x|y) p_t(y)$$

**Example 37 (Conditional rate for factorized mixture path):**
$$Q_t^z(v_i, j | x_j) = \frac{\dot\kappa_t}{1 - \kappa_t}(\delta_{z_j}(v_i) - \delta_{x_j}(v_i))$$

In matrix form:
$$Q_t^z(v_i, j | x_j) = \frac{\dot\kappa_t}{1-\kappa_t} \cdot \begin{cases} 0 & \text{if } x_j = z_j \\ 1 & \text{if } v_i = z_j, x_j \ne z_j \\ 0 & \text{if } v_i \ne z_j, x_j \ne z_j \\ -1 & \text{if } v_i = x_j, x_j \ne z_j \end{cases}$$

Only allows jumps directly to $z_j$.

### 7.2.3 Learning the Marginal Rate Matrix

**Theorem 38 (Marginalization for factorized mixture path):**
$$Q_t(v_i, j | x) = \frac{\dot\kappa_t}{1 - \kappa_t}(p_{1|t}(z_j = v_i | x) - \delta_{x_j}(v_i))$$

**Key insight:** The marginal rate matrix is a reparameterization of $p_{1|t}(z_j = v_i | x)$ — just a classifier per token position!

Define **denoising probabilities network:**
$$p_{1|t}^\theta : x \mapsto (p_{1|t}^\theta(z_j = v_i | x))_{j=1,\ldots,d, v_i \in \mathcal{V}}$$

Output shape: $d \times V$. Use softmax per position. Standard sequence model (e.g., transformer) works.

**Discrete Flow Matching loss** (cross-entropy per position):
$$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{z, t, x \sim p_t(\cdot|z)}\left[\sum_{j=1}^d -\log p_{1|t}^\theta(z_j | x)\right]$$

**Discrete diffusion ≈ classification training**, just like continuous flow matching ≈ regression.

**Algorithm 8: Training Factorized CTMC Model (Discrete Diffusion)**
```
Require: Dataset z ~ p_data, z = (z_1,...,z_d), p_init, schedule κ_t, posterior net f_θ
1: for each iteration do
2:   Sample z ~ p_data
3:   Sample t ~ Unif[0,1], κ ← κ_t
4:   For j = 1,...,d in parallel:
5:     m_j ~ Bernoulli(κ)
6:     ξ_j ~ p_init^(j)
7:     x_j ← m_j * z_j + (1 - m_j) * ξ_j
8:   x ← (x_1, ..., x_d)
9:   Predict logits: l_j(·) ← f_θ(x, t)_j
10:  p^θ_{1|t}(v|x)_j = Softmax(l_j)(v)
11:  L_DFM(θ) ← Σ_j [-log p^θ_{1|t}(z_j | x)_j]
12:  θ ← Opt.step(∇_θ L_DFM(θ))
13: end for
```

**Example 39 (Masked Diffusion Language Models / MDLMs):** Augment vocabulary with `[mask]` token: $\mathcal{V} = \{v_1, \ldots, v_V, \text{[mask]}\}$. Initial state: all-masked, $p_{\text{init}} = \delta_{\text{[mask]}^d}$.

Example trajectory:
```
t = 0:    [MASK] [MASK] [MASK] [MASK]  [MASK] [MASK] [MASK]
t = 0.25: [MASK] [MASK] [MASK] on     [MASK] [MASK] [MASK]
t = 0.75: [MASK] cat    [MASK] on     the   mat    [MASK]
t = 1:    The   cat    sat    on     the   mat    .
```

Current SOTA discrete diffusion models (e.g., LLaDA 2.0) use this recipe with transformers trained on web-scale data.

**Remark 40 (Generator Matching):** The principles of flow matching extend to general Markov processes via **Generator Matching**. A generator generalizes both vector fields and rate matrices. Allows models for: smooth manifolds (geometric data), mixed state spaces (joint text-image), jump processes, etc.

---

# 8 References (Key References)

Note: Full reference list in original document. Key references:

- **[1]** Albergo, Boffi, Vanden-Eijnden (2023). Stochastic interpolants.
- **[2]** Anderson (1982). Reverse-time diffusion equation models.
- **[11]** Dhariwal, Nichol (2021). Diffusion Models Beat GANs on Image Synthesis (classifier guidance).
- **[14]** Esser et al. (2024). Scaling Rectified Flow Transformers (Stable Diffusion 3).
- **[16]** Gat et al. (2024). Discrete flow matching.
- **[17]** Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models (DDPM).
- **[18]** Ho, Salimans (2022). Classifier-Free Diffusion Guidance.
- **[19]** Holderrieth et al. (2024). Generator matching.
- **[23]** Karras et al. (2022). Elucidating the design space of diffusion-based generative models.
- **[25]** Lipman et al. (2022). Flow matching for generative modeling.
- **[26]** Lipman et al. (2024). Flow Matching Guide and Code.
- **[27]** Liu, Gong, Liu (2022). Flow straight and fast / rectified flow.
- **[30]** Peebles, Xie (2023). Scalable Diffusion Models with Transformers (DiT).
- **[33]** Polyak et al. (2024). Movie Gen.
- **[34]** Radford et al. (2021). CLIP.
- **[36]** Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models.
- **[41]** Sohl-Dickstein et al. (2015). Deep unsupervised learning using nonequilibrium thermodynamics.
- **[42]** Song, Ermon (2019). Generative modeling by estimating gradients of the data distribution.
- **[43-45]** Song et al. Score-Based Generative Modeling through SDEs.
- **[49]** Vaswani et al. (2017). Attention Is All You Need.

---

# Appendix A: Probability Theory Reminder

## A.1 Random Vectors

Data $x = (x^1, \ldots, x^d) \in \mathbb{R}^d$ with inner product $\langle x, y \rangle = \sum_i x^i y^i$, norm $\|x\| = \sqrt{\langle x, x\rangle}$.

RV $X \in \mathbb{R}^d$ with continuous PDF $p_X : \mathbb{R}^d \to \mathbb{R}_{\ge 0}$:
$$\mathbb{P}(X \in A) = \int_A p_X(x) dx, \quad \int p_X(x) dx = 1$$

**Isotropic Gaussian:**
$$\mathcal{N}(x; \mu, \sigma^2 I) = (2\pi \sigma^2)^{-d/2} \exp\left(-\frac{\|x - \mu\|_2^2}{2\sigma^2}\right)$$

**Expectation:** $\mathbb{E}[X] = \int x \, p_X(x) dx$

**Law of unconscious statistician:** $\mathbb{E}[f(X)] = \int f(x) p_X(x) dx$

## A.2 Conditional Densities and Expectations

Joint PDF $p_{X,Y}(x, y)$ with marginals $p_X(x) = \int p_{X,Y}(x, y) dy$, $p_Y(y) = \int p_{X,Y}(x, y) dx$.

**Conditional PDF:**
$$p_{X|Y}(x|y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}$$

**Bayes' rule:**
$$p_{Y|X}(y|x) = \frac{p_{X|Y}(x|y) p_Y(y)}{p_X(x)}$$

**Conditional expectation function:**
$$\mathbb{E}[X | Y = y] = \int x \, p_{X|Y}(x|y) dx$$

**Tower property:** $\mathbb{E}[\mathbb{E}[X|Y]] = \mathbb{E}[X]$

---

# Appendix B: Proof of the Fokker-Planck Equation

**Theorem 41 (Fokker-Planck):** For SDE $dX_t = u_t(X_t) dt + \sigma_t dW_t$ with $X_0 \sim p_{\text{init}}$:
$$X_t \sim p_t \iff \partial_t p_t(x) = -\text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)$$

**Proof sketch (necessity):** Use test functions $f$ (smooth, compactly supported). For trajectory $X_{t+h} \approx X_t + h u_t(X_t) + \sigma_t(W_{t+h} - W_t)$, expand $f(X_{t+h})$ as a Taylor series:

$$f(X_{t+h}) - f(X_t) = \nabla f(X_t)^T (h u_t(X_t) + \sigma_t(W_{t+h} - W_t)) + \frac{1}{2}(\cdots)^T \nabla^2 f(X_t)(\cdots)$$

Taking conditional expectation $\mathbb{E}[\cdot | X_t]$ uses:
- $\mathbb{E}[W_{t+h} - W_t | X_t] = 0$
- $W_{t+h} - W_t | X_t \sim \mathcal{N}(0, h I_d)$
- $\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I_d)}[\epsilon^T A \epsilon] = \text{trace}(A)$

Yields:
$$\partial_t \mathbb{E}[f(X_t)] = \mathbb{E}\left[\nabla f(X_t)^T u_t(X_t) + \frac{1}{2} \sigma_t^2 \Delta f(X_t)\right]$$

Using integration by parts (twice):
$$\int f(x) \partial_t p_t(x) dx = \int f(x)\left[-\text{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2}\Delta p_t(x)\right] dx$$

Since this holds for all test functions $f$, the Fokker-Planck equation follows.

**Sufficiency** follows from uniqueness of parabolic PDE solutions.

---

# Appendix C: Existence and Uniqueness of CTMCs

**Proof of Theorem 33:**

**Uniqueness:** The rate matrix equation implies the Kolmogorov forward equation:
$$\frac{d}{dt'} p_{t'|t}(X_{t'} = y | X_t = x) = \sum_z Q_{t'}(y|z) p_{t'|t}(X_{t'} = z | X_t = x)$$

This is a linear ODE in $t'$ with fixed initial condition $p_{t|t}(y|x) = \delta_y(x)$. Linear ODEs have unique solutions, so transition probabilities are unique.

**Existence:** Any linear ODE has a solution, so $p_{t'|t}$ exists. Must verify three properties:
1. **Normalization:** $\sum_y p_{t'|t}(y|x) = 1$
2. **Non-negativity:** $p_{t'|t}(y|x) \ge 0$
3. **Chain rule consistency:** $\sum_z p_{t_2|t_1}(y|z) p_{t_1|t_0}(z|x) = p_{t_2|t_0}(y|x)$

Each verified using the rate matrix conditions and ODE uniqueness. (Full proof in original.)

---

# Appendix D: Additional Perspectives on VAEs

Define joint distributions:
$$q_\phi(x, z) = p_{\text{data}}(x) q_\phi(z|x) \quad \text{(encoder joint)}$$
$$p_\theta(x, z) = p_\theta(x|z) p_{\text{prior}}(z) \quad \text{(decoder joint)}$$

**KL between joints:**
$$D_{\text{KL}}(q_\phi(x, z) \| p_\theta(x, z)) = \mathbb{E}[\log p_{\text{data}}(x)] + \mathbb{E}[D_{\text{KL}}(q_\phi(z|x) \| p_{\text{prior}}(z))] - \mathbb{E}[\log p_\theta(x|z)]$$

So the VAE loss equals (up to a constant):
$$\mathcal{L}_{\text{VAE}} = D_{\text{KL}}(q_\phi(x, z) \| p_\theta(x, z)) + \text{const}$$

**Proposition 3 (KL chain rule):**
$$D_{\text{KL}}(q(z, x) \| p(z, x)) = D_{\text{KL}}(q(x) \| p(x)) + \mathbb{E}_{x \sim q}[D_{\text{KL}}(q(z|x) \| p(z|x))]$$

**Data-processing inequality:**
$$D_{\text{KL}}(q(x) \| p(x)) \le D_{\text{KL}}(q(z, x) \| p(z, x))$$

VAEs minimize an upper bound on $D_{\text{KL}}(p_{\text{data}}(x) \| p_\theta(x))$.

**ELBO (Evidence Lower Bound):**
$$\text{ELBO}(x; \phi, \theta) = \mathbb{E}_{z \sim q_\phi(z|x)}\left[\log \frac{p_\theta(x|z) p_{\text{prior}}(z)}{q_\phi(z|x)}\right] \le \log p_\theta(x)$$

VAE training $\equiv$ maximize $\mathbb{E}_{x \sim p_{\text{data}}}[\text{ELBO}(x; \phi, \theta)]$ (up to constant).

**Amortization gap:** $D_{\text{KL}}(q_\phi(x, z) \| p_\theta(x, z)) - D_{\text{KL}}(q_\phi(z) \| p_{\text{prior}}(z))$. Zero iff $q_\phi(z|x) = p_\theta(z|x)$ (encoder = true posterior).

**Why not stop at VAE?** Decoder learns to reconstruct from $q_\phi(z)$, not $p_{\text{prior}}(z)$ — switching at inference time goes out-of-distribution. Flow/diffusion models are more capable, so we farm generative complexity to latent generative model.

**Reconstruction vs Generation tradeoff:**
- **Low rFID** (good reconstruction) ↔ low info loss ↔ harder latent distribution to learn ↔ higher gFID
- **High rFID** ↔ high info loss ↔ easier latent distribution ↔ lower gFID

The optimal "division of labor" is at the knee of the rate-distortion Pareto frontier: high compression (low rate) without high distortion.

---

# Appendix E: A Guide to the Diffusion Model Literature

## Discrete time vs Continuous time

- **First DDPM papers** (Sohl-Dickstein 2015, DDPM 2020): used discrete-time Markov chains
- **Disadvantage:** Forces time discretization choice before training; loss requires ELBO approximation
- **Song et al. 2021:** showed discrete formulation approximates continuous SDE; ELBO becomes tight in continuous limit
- Both use same loss; not fundamentally different

## "Forward process" vs probability paths

Original diffusion models constructed a **forward process** SDE:
$$\bar X_0 = z, \quad d\bar X_t = u_t^{\text{forw}}(\bar X_t) dt + \sigma_t^{\text{forw}} d\bar W_t$$

Restricted to affine $u_t^{\text{forw}}(x) = a_t x$ for tractability:
$$\bar X_t | \bar X_0 = z \sim \mathcal{N}(\alpha_t z, \beta_t^2 I)$$
$$\alpha_t = \exp\left(\int_0^t a_r dr\right), \quad \beta_t^2 = \alpha_t^2 \int_0^t (\sigma_r^{\text{forw}})^2 / \alpha_r^2 dr$$

Just specific Gaussian probability paths. **Probability paths** (Lipman et al. 2022, flow matching) generalize this:
- Forward processes converge only as $t \to \infty$ (never reach $p_{\text{init}}$ in finite time)
- Probability paths reach $p_{\text{init}}$ at $t = 0$ exactly
- Simpler and more general

## Time-reversals vs Fokker-Planck

Original diffusion: training target constructed via **time-reversal** (Anderson 1982):
$$dX_t = [-u_t(X_t) + \sigma_t^2 \nabla \log p_t(X_t)] dt + \sigma_t dW_t$$

This is a specific instance of our training target (Proposition 1). For generation we only use $X_1$, so "true" time-reversal is not necessary — and often suboptimal (probability flow ODE is better).

## Flow Matching and Stochastic Interpolants

- **Flow matching** (Lipman 2022): pure flows, no SDE construction needed. Trained scalably.
- **Stochastic interpolants** (Albergo et al. 2023): includes flow + SDE extension via Langevin dynamics. Uses interpolant function $I(t, x, z)$.
- **Advantages over DDPM:** simplicity, generality (arbitrary $p_{\text{init}} \to p_{\text{data}}$, not just Gaussian initialization)

### Summary 45: Alternative Formulations

Common variations in the literature:
1. **Discrete-time** Markov chain approximations
2. **Inverted time convention** ($t = 0$ ↔ $p_{\text{data}}$ instead of $p_{\text{init}}$)
3. **Forward process** as way to construct Gaussian probability paths
4. **Time-reversal** training target (specific case of marginalization trick)
