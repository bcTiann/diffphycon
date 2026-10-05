# Score-to-Velocity Adaptation of Prior Reweighting

Algebraic score-to-velocity conversion for an experimental FM adaptation of DiffPhyCon's prior-reweighted sampling (paper §3.2, Eq. 8–9). This note records the proposed conversion, not a proof that the resulting sampler produces the intended endpoint-reweighted distribution.

## Setup

Joint variables $(u, w)$ with condition $c$. Base model: joint $p(u, w \mid c)$ with marginal prior $p(w \mid c)$.

Goal: sample from the reweighted distribution

$$
\tilde p_\gamma(u, w \mid c) \propto p(w \mid c)^{\gamma - 1} \cdot p(u, w \mid c).
$$

Generative path: Gaussian CondOT,

$$
x_\tau = \alpha_\tau z + \beta_\tau \varepsilon, \qquad \alpha_\tau = \tau, \quad \beta_\tau = 1 - \tau.
$$

For any Gaussian conditional path, the score–velocity identity (Proposition 1) gives

$$
u_\tau(x \mid c) = a_\tau \nabla \log p_\tau(x \mid c) + b_\tau x,
$$

where $a_\tau, b_\tau$ are path-dependent scalars.

## Score-to-velocity conversion

**Step 1.** Take log and gradient of $\tilde p_\gamma$:

$$
\nabla \log \tilde p_\gamma = \nabla \log p(u, w \mid c) + (\gamma - 1) \nabla \log p(w \mid c).
$$

**Step 2 (proposed correction).** Use the score–velocity identity to motivate an additive velocity correction:

$$
\tilde u_\tau^{\text{rw}} = u_\tau^{\text{target}} + a_\tau (\gamma - 1) \nabla_{(u, w)} \log p(w \mid c).
$$

**Step 3.** Since $p(w \mid c)$ does not depend on $u$, its gradient in joint space has zero $u$-block:

$$
\nabla_{(u, w)} \log p(w \mid c) = \bigl[ 0, \nabla_w \log p(w \mid c) \bigr].
$$

**Step 4.** Apply Proposition 1 to the prior $p_\tau(w \mid c)$ (using the same path as the joint):

$$
\nabla_w \log p_\tau(w \mid c) = \frac{1}{a_\tau} \bigl[ u_\tau^{\text{prior}}(w \mid c) - b_\tau w \bigr].
$$

Substituting into Step 2, the $a_\tau$ factors cancel exactly:

$$
\boxed{ \tilde u_\tau^{\text{rw}} = u_\tau^{\text{target}} + (\gamma - 1) \bigl[ u_\tau^{\text{prior}} - b_\tau [0, w] \bigr] }
$$

This expression is the proposed prior correction obtained by converting the score term to velocity form. It is used as a sampling heuristic; this calculation does not establish exact sampling of the endpoint-reweighted distribution.

**Step 5 (empirical schedule).** Strong reweighting near $\tau \to 0$ (high noise) destabilises sampling. In practice, multiply $(\gamma - 1)$ by a time schedule $\tilde\eta(\tau)$:

$$
\boxed{ \tilde u_\tau^{\text{rw}} = u_\tau^{\text{target}} + (\gamma - 1) \tilde\eta(\tau) \bigl[ u_\tau^{\text{prior}} - b_\tau [0, w] \bigr] }
$$

with $\tilde\eta(0) \approx 0$, $\tilde\eta(1) \approx 1$, and monotonically increasing in $\tau$.
