"""
Post-processor for flow/lab_four.ipynb.

⚠️ NEW WORKFLOW (2026-05-23) — do NOT run `jupytext --to notebook` before this.
   That command would OVERWRITE the .ipynb (losing user's filled-in code).
   This script now reads the EXISTING .ipynb in-place and only modifies
   markdown cells (never touches code cells).

What it does, idempotently:
- Top docstring → proper intro markdown (if not already)
- Part separators (============ Part N ============) → clean H1 markdown
- Question separators (---- Question N.M ----) → clean H2 markdown
- Inject extra explainer markdown cells (TWO_FORMS, TRAINER, DATASET_INIT,
  W_SCHEDULER) before specific code cells, UPDATING in place if already
  present (won't duplicate on repeated runs).

User's filled-in code is fully preserved.

Run:    python flow/_build_lab_four_nb.py
"""
import os
import re
import nbformat

NB_PATH = os.path.join(os.path.dirname(__file__), "lab_four.ipynb")


PART_INTROS = {
    "Part 0": """# Part 0: Setup + Base Classes

These are the same abstractions you implemented/used in `lab_three.ipynb`. Copied here so this lab is self-contained.

**No fill-ins in Part 0** — just read through to remember the interfaces:

- `Sampleable` / `LabeledSampleable` — distributions you can sample from
- `ConditionalProbabilityPath` — defines $p_t(x \\mid z)$
- `LinearAlpha` / `LinearBeta` — the schedule $\\alpha_\\tau = \\tau,\\; \\beta_\\tau = 1 - \\tau$
- `GaussianConditionalProbabilityPath` — Gaussian path with this schedule
- `VectorFieldNet` — abstract velocity-field network
- `Trainer` — generic FM trainer (you'll subclass this)""",

    "Part 1": """# Part 1: Get a Feel for Burgers Data

Like lab_three Part 1 (where you visualized MNIST), we first look at the Burgers dataset before building any models.

A Burgers trajectory is a **2-channel "image"**:
- channel 0 = $u(t, x)$, the velocity field of the fluid
- channel 1 = $w(t, x)$, the external control force we applied
- time axis = 11 actual steps (padded to 16), space axis = 128 points

Conditioning $c = (u_0, u_T^*)$ = the initial state and target terminal state. We control $w(t, x)$ to drive the system from $u_0$ to $u_T^*$.

**No fill-ins in Part 1** — just helper functions for you to play with.""",

    "Part 2": """# Part 2: Train the Joint FM Model

**Goal**: learn the velocity field $u_t^\\theta(x \\mid c)$ of the joint distribution $p(u, w \\mid u_0, u_T^*)$ over Burgers trajectories.

**Three classes to fill in**:
- **Q2.1** — `BurgersDataset.sample` (sampling from $p_{\\text{data}}$)
- **Q2.2** — `BurgersFlowTrainer.get_train_loss` (CFM loss + inpainting trick)
- **Q2.3** — `BurgersVectorField.forward` (Unet2D wrapped + $c$ injection)

**How Burgers differs from MNIST (lab_three Part 2)**:
- Conditioning $c$ is a **continuous vector** $(u_0, u_T^*)$, not a class index. So no `null label` and no CFG dropout ($\\eta = 0$).
- We inject $c$ via **inpainting overwrite** (force `x[:, 0, 0, :] = u_0`, `x[:, 0, T_IDX, :] = u_T*`) instead of cross-attention or class embedding. This is the DiffPhyCon trick — see `notes_diffphycon_flow_bridge.md §4.3`.
- We train with an additional "inpainting trick" loss term: the target velocity at the boundary rows is **forced to 0**, teaching the model "if you see a clean boundary, don't change it". See `notes_diffphycon_flow_bridge.md §4.4`.""",

    "Part 3": """# Part 3: Inpainting + ODE Sampling

**Goal**: sample $z$ from the trained velocity field.

In lab_three Part 3 you built a DiT transformer; here we don't need that — `Unet2D` is already capable. The novelty for Burgers is **how** we sample, specifically the inpainting overwrite that injects $(u_0, u_T^*)$.

**Three things to fill in**:
- **Q3.1** — `inpaint_overwrite` (helper)
- **Q3.2** — `BurgersEulerSampler.sample`
- **Q3.3** — sanity check (provided, no fill-in)""",

    "Part 4": """# Part 4: Prior Model + γ-Reweighting

This Part has **no parallel in lab_three** — it is the DiffPhyCon novelty.

We train a **second** model `net_prior` that learns the marginal $p(w \\mid c)$ (i.e., the controls alone, no fluid dynamics). At sampling time we combine joint + prior to bias the velocity field toward ($\\gamma > 1$) or away from ($\\gamma < 1$) the prior, per the DiffPhyCon Eq. 9 reweighting.

You derived the FM-side formula yourself — see `notes_fm_prior_reweighting.md`:

$$\\tilde{u}_\\tau(x \\mid c) = u_{\\text{joint}}(x \\mid c) + (\\gamma - 1) \\cdot \\tilde{\\eta}(\\tau) \\cdot \\big[\\,u_{\\text{prior}}(x \\mid c) - b_\\tau \\cdot [0, w]\\,\\big]$$

**Four things to fill in**:
- **Q4.1** — `BurgersPriorDataset` (u-channel zeroed in data)
- **Q4.2** — `BurgersPriorTrainer` (target velocity u-channel = 0)
- **Q4.3** — `w_scheduler_fm` (DDPM sigmoid_flip → FM time)
- **Q4.4** — `ReweightedVectorField` (the actual formula above)""",

    "Part 5": """# Part 5: γ Sweep + Baseline Comparison

This part has **no fill-ins** — once Parts 2-4 are working, this just plugs them together and runs the sweep that reproduces `notes_baseline_summary.md §3.1`.

We sweep $\\gamma \\in \\{0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.5\\}$, compute $J$ and Energy, and compare directly against the existing DDPM baseline numbers.""",

    "Sanity Checks": """# Sanity Checks

Run these in order after filling in each Part:

1. `sanity_check_part1()` — visualize Burgers data (no fill-in needed; already runs)
2. `sanity_check_2_4()` — after Q2.1-Q2.3: joint FM trains and samples (loss drops, output not noise)
3. `sanity_check_3_3()` — after Q3.1-Q3.2: sample with γ=1 and check J
4. `sanity_check_4_5()` — after Q4.1-Q4.4: γ=1 reweighted == joint output, γ≠1 differs""",

    "Main entry point": """# Main entry point

The `if __name__ == "__main__"` block below just runs `sanity_check_part1()` and prints what to do next.

You won't typically run this cell in the notebook — just call the individual `sanity_check_*` functions as you finish each Part.""",
}


# Map each Question section header to a richer markdown intro
QUESTION_INTROS = {
    "Question 2.1": """## Question 2.1 — `BurgersDataset.sample`

Build the data sampler. For each batch, return:
- `z` of shape `(b, 2, 16, 128)` — the full clean trajectory
- `c` of shape `(b, 2, 128)` — the boundary condition $(u_0, u_T^*)$, extracted from `z`'s u-channel rows 0 and 10

**2 fill-in Steps**.""",

    "Question 2.2": """## Question 2.2 — `BurgersFlowTrainer.get_train_loss`

This is **the heart of Flow Matching** — the conditional flow matching loss.

For each batch:
- sample $z \\sim p_{\\text{data}}$ and time $\\tau \\sim U[0, 1]$
- form noisy $x_\\tau = \\alpha_\\tau z + \\beta_\\tau \\epsilon$
- compute target velocity $u^{\\text{target}}(x_\\tau \\mid z) = \\dot\\alpha_\\tau z + \\dot\\beta_\\tau \\epsilon$
- regress: $\\|u^\\theta_\\tau(x_\\tau \\mid c) - u^{\\text{target}}(x_\\tau \\mid z)\\|^2$

Plus the **inpainting trick** at Step 5: force target velocity to 0 at the boundary rows.

**6 fill-in Steps**. Compare to `lab_three.ipynb` Q2.2.""",

    "Question 2.3": """## Question 2.3 — `BurgersVectorField.forward`

Wraps `Unet2D` as a `VectorFieldNet`. The trick: **no embedding for $c$** — every forward call overwrites the boundary rows of $x$ with clean $c$ values before passing to the Unet.

This way train/inference are consistent: at both training time (Q2.2) and sampling time (Q3.2), the network sees clean boundaries at the same rows.

**3 fill-in Steps**.""",

    "Question 3.1": """## Question 3.1 — `inpaint_overwrite`

A tiny helper called **before every Euler step** during sampling. Forces:
- `x[:, 0, 0, :]    = u_0`
- `x[:, 0, T_idx, :] = u_T*`

This is the DiffPhyCon way of injecting hard conditioning — see `notes_diffphycon_flow_bridge.md §4.3`. Different from RePaint (no re-noising); just clean overwrite.

**2 fill-in Steps**.""",

    "Question 3.2": """## Question 3.2 — `BurgersEulerSampler.sample`

Euler ODE integration from $\\tau = 0$ (noise) to $\\tau = 1$ (clean data), with `inpaint_overwrite` before each step.

The flow ODE is $\\dfrac{dx}{d\\tau} = v_\\tau^\\theta(x \\mid c)$, integrated with step size $d\\tau = 1 / n_{\\text{steps}}$.

**5 fill-in Steps**.""",

    "Question 4.1": """## Question 4.1 — `BurgersPriorDataset.sample`

Same as `BurgersDataset.sample`, but the u-channel of $z$ is **zeroed out** before returning. The prior model only learns $p(w \\mid c)$ — it shouldn't see fluid dynamics.

The DDPM equivalent is `diffusion_1d_burgers.py:400-402`.

**1 fill-in Step**.""",

    "Question 4.2": """## Question 4.2 — `BurgersPriorTrainer` extras

Inherits from `BurgersFlowTrainer`, with **two additional zeroing steps**:
- Force the entire u-channel of `u_target` to 0 (not just rows 0 and T_IDX)
- Force the u-channel of `u_pred` to 0 (safety; the network shouldn't predict u dynamics)

This cleanly embeds $u_{\\text{prior}}$ into the joint $(u, w)$ space with u-block = 0 — see `notes_fm_prior_reweighting.md §2.4`.

**2 fill-in Steps**.

---

### Design note — why not just set `out_dim=1` for the prior net?

Reasonable question: if we're going to zero the u-channel of `u_pred` anyway, why not build a smaller network with `out_dim=1` (only outputs the w-channel)?

**Two reasons we keep `out_dim=2`**:

1. **The "wasted" parameters are negligible** — measured exactly:
   - The only layer that differs is `final_conv = nn.Conv2d(dim, out_dim, 1)`:
     - `out_dim=2`: weight `(2, 64, 1, 1)` = 128 params + bias 2 = **130 params**
     - `out_dim=1`: weight `(1, 64, 1, 1)` = 64 params + bias 1 = **65 params**
     - Difference: **65 params**
   - Total Unet2D parameters: **35,707,906**
   - Waste ratio: $65 / 35{,}707{,}906 \\approx 1.8 \\times 10^{-6} = 0.0002\\%$
   - Like 65 pixels out of a 1024×1024 image.

2. **The γ-reweighting formula stays clean** (see `notes_fm_prior_reweighting.md §3`):

   $$\\tilde{u}_\\tau(x \\mid c) \\;=\\; u_{\\text{joint}}(x \\mid c) \\;+\\; (\\gamma - 1)\\,\\tilde\\eta(\\tau)\\,\\big[\\,u_{\\text{prior}}(x \\mid c) - b_\\tau\\,[0, w]\\,\\big]$$

   This formula assumes $u_{\\text{joint}}$ and $u_{\\text{prior}}$ have the **same shape** $(b, 2, 16, 128)$ so they add directly. If $u_{\\text{prior}}$ were $(b, 1, 16, 128)$, every call to `ReweightedVectorField.forward` would need `torch.cat([zeros, v_prior], dim=1)` glue — adding code complexity for a 0.0002% saving.

The pattern of **"output same shape, mask the irrelevant channel"** is also exactly what the DDPM baseline does (`diffusion_1d_burgers.py:402`), so this keeps our FM comparison apples-to-apples.""",

    "Question 4.3": """## Question 4.3 — `w_scheduler_fm`

FM-time equivalent of DDPM's `sigmoid_schedule_flip`. The schedule is **small at τ=0** (noise end, weak reweighting) and **large at τ=1** (clean end, strong reweighting).

Map $\\tau \\to t_{\\text{ddpm}} = \\text{round}((1 - \\tau) \\cdot 999)$, then call existing `sigmoid_schedule_flip(t_ddpm)`.

**1 fill-in Step**.""",

    "Question 4.4": """## Question 4.4 — `ReweightedVectorField.forward`

This is the **core of γ-reweighting in FM** — the formula you derived in `notes_fm_prior_reweighting.md §3`:

$$\\boxed{\\;\\tilde{u}_\\tau(x \\mid c) = u_{\\text{joint}}(x \\mid c) + (\\gamma - 1) \\cdot \\tilde{\\eta}(\\tau) \\cdot \\big[\\,u_{\\text{prior}}(x \\mid c) - b_\\tau \\cdot [0, w]\\,\\big]\\;}$$

where:
- $b_\\tau = \\dot\\alpha_\\tau / \\alpha_\\tau = 1/\\tau$ (for the CondOT path $\\alpha_\\tau = \\tau$)
- $\\tilde{\\eta}(\\tau)$ from Q4.3
- $[0, w]$ = the $x$ vector with u-channel zeroed (only $w$ survives)

When $\\gamma = 1$ the correction vanishes — sanity-checked in Q4.5.

**6 fill-in Steps**.""",
}


TOP_INTRO_MD = """# Flow Matching for 1D Burgers' Control — `lab_four`

Companion to `lab_three.ipynb` (MIT 6.S184 lab 3: CFG-FM on MNIST).

This lab walks you through implementing Flow Matching for the **1D Burgers control problem** (DiffPhyCon paper Experiment 1), using your existing diffusion baselines from `trained_models/burgers/` as the comparison target.

## How to use this lab

1. Read the markdown cell at the top of each Part — it frames why this piece is needed and what's different from MNIST.
2. Find each `# Question N.M` cell and **fill in** the methods marked `raise NotImplementedError("Fill me in!")`. Each fill-in has a `# Step N:` comment above it that says exactly what to compute.
3. After each Part, **run the corresponding sanity check cell**.
4. When all sanity checks pass, run `part5_gamma_sweep` to reproduce the baseline comparison.

## Background reading (cross-referenced inline)

- `flow_matching_diffusion.md` — MIT FM theory (Prop 1 + Example 13)
- `notes_diffphycon_flow_bridge.md` — DDPM ↔ FM translation, §4 inpainting
- `notes_fm_prior_reweighting.md` — γ-reweighting math + code skeleton
- `notes_baseline_summary.md` — target numbers to beat
- `diffusion/diffusion_1d_burgers.py` — the DDPM reference impl this mirrors

## Data shape conventions

Throughout this file:

- `x` or `z` has shape `(b, 2, 16, 128)` — Burgers trajectory:
  - channel 0 = $u(t, x)$, the state field
  - channel 1 = $w(t, x)$, the control field
  - **time axis = 16**:
    - rows **0..10** = real physical time steps ($N_t = 11$, $t = 0, 1, \\ldots, 10$)
    - rows **11..15** = **zero-padding** so the UNet's `dim_mults=(1,2,4,8)` can downsample 3 times ($16 \\to 8 \\to 4 \\to 2$). No physical meaning.
    - w-channel row 10 is also 0 because $w$ drives transitions $t \\to t{+}1$ and there's no transition out of $t = T$.
  - space axis = 128 ($N_x = 128$)
- `c` has shape `(b, 2, 128)` — boundary conditions:
  - `c[:, 0, :]` = $u_0$ (initial state = `x[:, 0, 0, :]`)
  - `c[:, 1, :]` = $u_T^*$ (target terminal = `x[:, 0, 10, :]`) — **row 10, not 15!** Row 10 is the last real time step; rows 11..15 are padding.
- `t` (FM time) has shape `(b, 1, 1, 1)` or scalar — $\\tau \\in [0, 1]$
"""


TWO_FORMS_MD = r"""## Two equivalent forms of the target velocity $u^{\text{target}}(x_t \mid z)$

The next cell — `GaussianConditionalProbabilityPath` — implements the **conditional vector field** $u^{\text{target}}(x_t \mid z)$. There are **two mathematically equivalent ways** to write it. **They give identical numerical values** — the next cell implements both as `target_velocity_formA` and `target_velocity_formB`, with a dispatcher to switch between them. Read this to understand why.

### Setup

The conditional path is

$$
x_t \;=\; \alpha_t\, z \;+\; \beta_t\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

The conditional vector field is the time derivative of this flow.

### Form A — express in $\varepsilon$ (the direct form)

Differentiate the path directly:

$$
u^{\text{target}}(x_t \mid z) \;=\; \frac{d}{dt}\bigl[\alpha_t z + \beta_t \varepsilon\bigr] \;=\; \dot\alpha_t\, z + \dot\beta_t\, \varepsilon
$$

$$
\boxed{\;u^{\text{target}}(x_t \mid z) \;=\; \dot\alpha_t\, z \;+\; \dot\beta_t\, \varepsilon\;} \qquad \text{(Form A — ε form)}
$$

For our CondOT path ($\alpha_t = t,\;\beta_t = 1-t,\;\dot\alpha = 1,\;\dot\beta = -1$):

$$
u^{\text{target}} \;=\; z - \varepsilon
$$

**Signature**: `(x_t, z, t, eps)` — needs $\varepsilon$ (we already have it from sampling).

### Form B — express in $x_t$ alone (`lab_three` style)

Eliminate $\varepsilon$ by solving the path equation: $\varepsilon = (x_t - \alpha_t z)/\beta_t$. Substitute:

$$
\dot\alpha_t z + \dot\beta_t \cdot \frac{x_t - \alpha_t z}{\beta_t}
\;=\; \left(\dot\alpha_t - \frac{\dot\beta_t \alpha_t}{\beta_t}\right) z + \frac{\dot\beta_t}{\beta_t}\, x_t
$$

$$
\boxed{\;u^{\text{target}}(x_t \mid z) \;=\; \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t}\alpha_t\right) z \;+\; \frac{\dot\beta_t}{\beta_t}\, x_t\;} \qquad \text{(Form B — $x_t$ form)}
$$

For our CondOT path:

$$
u^{\text{target}} \;=\; \frac{z - x_t}{1 - t}
$$

**Signature**: `(x_t, z, t)` — no $\varepsilon$ needed.

### They're identical (by construction)

Form B was derived FROM Form A by algebraic substitution. So they output the same number for the same `(z, x_t, t, ε)` consistent with $x_t = \alpha_t z + \beta_t \varepsilon$.

Run `sanity_check_two_forms()` (defined a few cells below) to verify: it samples $z$, $\varepsilon$, $x_t$ and asserts $\|u_A - u_B\|_\infty < 10^{-4}$.

### Trade-offs

| | Form A (ε form) | Form B ($x_t$ form, `lab_three`) |
|:---|:---|:---|
| **Signature** | `(x_t, z, t, eps)` | `(x_t, z, t)` |
| **Computation** | one multiply per term | one division by $\beta_t$ |
| **Numerical issue** | None | $\beta_t \to 0$ at $t \to 1$ ⇒ divide by zero |
| **Needs ε?** | Yes (have it from sampler) | No (uses $x_t$ instead) |
| **Consistency with `lab_three`** | ❌ different signature | ✅ same signature |
| **Pedagogical value** | Direct from path definition | Forces you to use the change-of-variable trick |

### Why does any of this matter?

**At inference** (sampling new data), we never compute $u^{\text{target}}$ — we call the **trained neural network** $u^\theta_t(x_t)$, which only takes $x_t$. So neither form's "advantage" actually applies to inference.

**At training**, both forms are valid CFM loss targets. Theorem 12 says the regression MSE has the same expectation either way; the per-batch values differ pointwise but in a way that averages out.

**Bottom line**: this is a notational choice. **Default is Form A** (ε form, direct derivative). If you want `lab_three` style exactly, change the dispatcher body to call `_formB`."""


def fix_cell(cell):
    """Returns (new_cell or None to drop, replacement_md or None)."""
    src = cell.source.strip()

    # Detect the top docstring (cell 0)
    if cell.cell_type == "code" and src.startswith('"""') and "lab_four.py" in src:
        return nbformat.v4.new_markdown_cell(TOP_INTRO_MD)

    # Detect Part / section markdown cells
    if cell.cell_type == "markdown":
        for key, body in PART_INTROS.items():
            if key in src:
                return nbformat.v4.new_markdown_cell(body)
        for key, body in QUESTION_INTROS.items():
            if key in src:
                return nbformat.v4.new_markdown_cell(body)

    return cell


TRAINER_EXPLAINER_MD = r"""## How the `Trainer.train` loop works

The next cell defines the generic FM trainer. Most of it is obvious — `loss.backward()`, `opt.step()`, etc. — but the **printing logic** can look cryptic if you haven't seen this pattern. Let's walk through it.

### The two key lines

```python
self.loss_history.append(loss.item())
if step % print_every == 0:
    avg = float(np.mean(self.loss_history[-print_every:]))
    print(f"  step {step:6d}  loss={loss.item():.5f}  avg{print_every}={avg:.5f}")
```

### What `print_every` does

It's a **stride** — "print a progress line every $N$ steps". If `print_every = 100`, we print at step 0, 100, 200, 300, ... and stay quiet in between. This keeps the log readable when training for thousands of steps (you don't want a line per step).

### What `step % print_every == 0` means

`%` is the **modulo** operator (余数). `step % 100` is the remainder when `step` is divided by 100:

| `step` | `step % 100` | `step % 100 == 0`? |
|:---:|:---:|:---:|
| 0   | 0  | ✅ |
| 1   | 1  | ❌ |
| 50  | 50 | ❌ |
| 99  | 99 | ❌ |
| 100 | 0  | ✅ |
| 199 | 99 | ❌ |
| 200 | 0  | ✅ |

So `if step % print_every == 0:` is the idiomatic Python way to say "is `step` a multiple of `print_every`?". This fires exactly when we want a progress line.

### Why we print a rolling average, not the single-step loss

A single batch's loss is **noisy** — it depends on which random samples ended up in that batch. Looking at one number doesn't tell you much about whether training is actually working.

So we additionally compute:

```python
avg = float(np.mean(self.loss_history[-print_every:]))
```

`self.loss_history[-print_every:]` is **slice notation** that grabs the **last `print_every` entries** of the list (Python lets negative indices count from the end). Then `np.mean(...)` averages them.

So the output looks like:

```
step      0  loss=1.04832  avg100=1.04832
step    100  loss=0.83217  avg100=0.91408
step    200  loss=0.62113  avg100=0.74829
step    300  loss=0.51002  avg100=0.58217
```

The `avg100` column is what you actually watch — it's much smoother and shows the real trend. The single-step `loss` is just a sanity check that the latest batch isn't blowing up.

### How to tune `print_every`

- **Too small** (e.g. `print_every=1`): log floods, hard to spot trends
- **Too large** (e.g. `print_every=10000`): can't tell if training is broken until 10k steps in
- **Good default**: roughly $1/10$ to $1/50$ of `num_steps`. For `num_steps=500` (our smoke check), `print_every=50` is fine."""


def _insert_or_update_md(cells, target_match_fn, md_content):
    """Idempotent insertion. Use the first line of md_content as a fingerprint.

    If a markdown cell starting with that fingerprint already exists anywhere
    in the notebook → replace its source with md_content (keeps text in sync).
    Otherwise → insert a new markdown cell right before the next cell that
    matches target_match_fn.
    """
    fingerprint = md_content.split("\n", 1)[0].strip()

    # Pass 1: look for existing copy by fingerprint
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith(fingerprint):
            cells[i] = nbformat.v4.new_markdown_cell(md_content)
            return cells

    # Pass 2: not found → insert before target
    out = []
    inserted = False
    for c in cells:
        if not inserted and target_match_fn(c):
            out.append(nbformat.v4.new_markdown_cell(md_content))
            inserted = True
        out.append(c)
    return out


def insert_two_forms_md(cells):
    """Inject the 'Two equivalent forms...' markdown cell before
    GaussianConditionalProbabilityPath (idempotent)."""
    return _insert_or_update_md(
        cells,
        lambda c: c.cell_type == "code" and "class GaussianConditionalProbabilityPath" in c.source,
        TWO_FORMS_MD,
    )


def insert_trainer_md(cells):
    """Idempotent inject of TRAINER_EXPLAINER_MD before class Trainer."""
    return _insert_or_update_md(
        cells,
        lambda c: c.cell_type == "code" and "class Trainer(ABC)" in c.source,
        TRAINER_EXPLAINER_MD,
    )


DATASET_INIT_MD = r"""### Breaking down `BurgersDataset.__init__`

Before you fill in `.sample()` (Q2.1), let's walk through what the constructor does. It looks like a lot but it's just **wrap a PyTorch Dataset + pre-load everything into memory for speed**.

```python
def __init__(self, dataset: Burgers1D, device: str = "cpu"):
    self.ds = dataset                                                      # 1
    self.device = device                                                   # 2
    all_z = torch.stack([self.ds[i] for i in range(len(self.ds))], dim=0)  # 3
    self.all_z = all_z.to(device)                                          # 4
    self.N = self.all_z.shape[0]                                           # 5
```

### Line 1 — `self.ds = dataset`

`dataset` is a `Burgers1D` instance — a PyTorch **Dataset** class. The thing to remember:

- `dataset[i]` returns one sample, a tensor of shape `(2, 16, 128)`
- `len(dataset)` returns the total number of samples (160 for our train set)

Mental model: a list-like object of `(2, 16, 128)` tensors that **lazily** reads from the HDF5 file on disk each time you access it.

### Line 2 — `self.device = device`

Stores a string like `"cpu"`, `"mps"`, or `"cuda"`. Used in Line 4.

### Line 3 — the list comprehension + `torch.stack`

This is two things glued together. Let's split:

**Part A — list comprehension**:
```python
[self.ds[i] for i in range(len(self.ds))]
```
is equivalent to:
```python
samples = []
for i in range(len(self.ds)):     # i = 0, 1, ..., 159
    samples.append(self.ds[i])    # each ds[i] has shape (2, 16, 128)
```
Result: a Python `list` of 160 tensors.

**Part B — `torch.stack`**:
```python
torch.stack(samples, dim=0)
```
takes the list of 160 tensors (each shape `(2, 16, 128)`) and stacks them along a **new dimension 0**:

| Input | Output |
|:---|:---|
| 160 tensors of shape `(2, 16, 128)` | 1 tensor of shape `(160, 2, 16, 128)` |

The first dim is now the "batch index" — `all_z[5]` gives you sample number 5.

> Quick distinction: `torch.stack` adds a new dim. `torch.cat` concatenates along an existing dim. If you tried `torch.cat(samples, dim=0)` here you'd get shape `(320, 16, 128)` — channels and batches glued together, which is **not** what we want.

### Line 4 — `.to(device)`

Moves the entire pre-loaded tensor to GPU/MPS if requested. On CPU this is a no-op.

### Line 5 — `self.N = self.all_z.shape[0]`

Stores 160 (the number of samples) so `.sample()` can pick valid indices in `[0, N)`.

### Why pre-stack instead of lazy-loading?

| Approach | Pros | Cons |
|:---|:---|:---|
| **Lazy** (call `self.ds[i]` each time in `sample`) | Low memory | Slow — each call reads HDF5 + applies transforms |
| **Pre-stack** (this lab) | Fast — `sample` becomes pure tensor indexing | Holds all data in RAM |

For Burgers (~5 MB total) pre-stacking is a clear win. For ImageNet-scale data you'd stay lazy and rely on PyTorch's `DataLoader` workers."""


def insert_dataset_init_md(cells):
    """Idempotent inject of DATASET_INIT_MD before class BurgersDataset."""
    return _insert_or_update_md(
        cells,
        lambda c: c.cell_type == "code" and "class BurgersDataset(LabeledSampleable)" in c.source,
        DATASET_INIT_MD,
    )


W_SCHEDULER_EXPLAINER_MD = r"""### What is `w_scheduler_fm` and where does it fit?

`w_scheduler_fm(τ)` returns a single number $\tilde\eta(\tau) \in [0, 1]$ — the **time-dependent weight** in the γ-reweighting formula. It's not γ itself; it's a multiplier on $(\gamma - 1)$.

> ⚠️ **Naming convention heads-up** (read this if you've also looked at the Jellyfish experiment in this repo).
>
> Two papers in the DiffPhyCon family decompose the per-step reweighting strength differently:
>
> | Role | Jellyfish paper (L.1) | Burgers paper / this lab |
> |:---|:---|:---|
> | **Scalar knob** (you pick) | $\xi$ (`coeff_ratio_w`) | $\gamma$ (`prior_beta`); we use $(\gamma - 1)$ |
> | **Time function** | $\beta_{K-k}$ (hardcoded) | $\tilde\eta(\tau)$ (a schedule we can swap) |
> | **Combined per-step weight** | $\xi \cdot \beta_{K-k}$ | $(\gamma - 1) \cdot \tilde\eta(\tau)$ |
> | **What "γ" refers to** | the time-varying sequence $\gamma_k = 1 - \xi \beta_{K-k}$ | the scalar $\gamma$ |
>
> **Mathematically identical** — both formulas come out as `score_joint + (scalar × time-function) × score_prior`. The Jellyfish convention bundles the time variation INTO γ; the Burgers convention keeps γ scalar and exposes the schedule separately.
>
> **In this lab (lab_four)**: `gamma` is a Python scalar (you'll pass `gamma=0.3` or `gamma=2.5` to `ReweightedVectorField`). The time variation lives entirely in `w_scheduler_fm(τ)`. So when you read "γ is constant" in this lab, it's true *in this convention* — but the **effective** reweighting strength $(\gamma - 1) \cdot \tilde\eta(\tau)$ is still time-varying, exactly like Jellyfish's $\gamma_k$.

### Where it lives in the formula (`notes_fm_prior_reweighting.md §3 Step 5`)

$$
\boxed{\;\tilde u_\tau(x \mid c) \;=\; u_{\text{joint}}(x \mid c) \;+\; (\gamma - 1)\,\underbrace{\tilde\eta(\tau)}_{\text{this scheduler}}\,\big[\,u_{\text{prior}}(x \mid c) \;-\; b_\tau\,[0, w]\,\big]}
$$

So **γ alone is constant** (scalar knob you pick), and **η̃(τ) modulates how strongly γ acts at each time** in the ODE integration. Together they form the effective time-varying reweighting strength $(\gamma - 1)\cdot \tilde\eta(\tau)$.

### What it looks like

The function is borrowed from DDPM's `sigmoid_schedule_flip` (in `diffusion/diffusion_1d_burgers.py:110-111`) and remapped from DDPM time $t \in [0, 999]$ to FM time $\tau \in [0, 1]$ via:

$$
t_{\text{ddpm}} \;=\; \text{round}\big((1 - \tau) \cdot 999\big)
$$

so that $\tau = 0$ (noise end in FM) corresponds to $t = 999$ (noise end in DDPM), and $\tau = 1$ (clean end in FM) corresponds to $t = 0$ (clean end in DDPM).

Here's the actual curve (left = FM-time view, right = the original DDPM-time curve mirrored):

![η̃(τ) schedule plot](lab_four_eta_schedule.png)

Numerical values at a few representative τ:

| $\tau$ | $\tilde\eta(\tau)$ |
|:---:|:---:|
| 0.00 | 0.0003 |
| 0.30 | 0.0015 |
| 0.50 | 0.0033 |
| 0.70 | 0.0058 |
| 0.90 | 0.0127 |
| 0.99 | 0.0934 |
| 1.00 | 0.9990 |

So η̃ is **essentially zero** for most of integration (τ ∈ [0, 0.9]) and **suddenly ramps up** to ~1 right at the end (τ → 1).

### Why this shape — the "only reweight at the clean end" intuition

When integrating the ODE from $\tau = 0$ (pure noise) to $\tau = 1$ (clean data), the meaningful structure only emerges late. Reweighting the velocity field is essentially "biasing toward $p(w \mid c)$" — but **at the noise end, you don't yet know what your sample looks like**, so biasing toward the prior amplifies meaningless noise and destabilizes the trajectory.

By keeping η̃ ≈ 0 for the noisy portion and only "opening the gate" near the clean end, the reweighting fires when it can actually help: when $x_\tau$ is already a meaningful trajectory that just needs to be nudged closer to typical $w$.

**Empirical evidence** (`notes_baseline_summary.md §4.2`): without this scheduler (i.e., η̃ ≡ 1 across all τ), γ=0.3 gives $J = 0.0607$ — a disaster. With this scheduler, $J = 0.0083$ — basically as good as γ=1. **The scheduler is what makes off-baseline γ usable at all.**

### How to read the next cell

The fill-in is a single-line translation: take `τ`, convert to a DDPM step index via the formula above, look up `sigmoid_schedule_flip(t_ddpm)`. The dispatcher returns a torch tensor of the same shape as `τ` (scalar in, scalar out; 1-D in, 1-D out)."""


def insert_w_scheduler_md(cells):
    """Idempotent inject of W_SCHEDULER_EXPLAINER_MD before def w_scheduler_fm."""
    return _insert_or_update_md(
        cells,
        lambda c: c.cell_type == "code" and "def w_scheduler_fm" in c.source,
        W_SCHEDULER_EXPLAINER_MD,
    )


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    new_cells = [fix_cell(c) for c in nb.cells]
    new_cells = insert_two_forms_md(new_cells)
    new_cells = insert_trainer_md(new_cells)
    new_cells = insert_dataset_init_md(new_cells)
    new_cells = insert_w_scheduler_md(new_cells)
    nb.cells = [c for c in new_cells if c.source.strip()]
    nbformat.write(nb, NB_PATH)
    print(f"Polished {NB_PATH}: {len(nb.cells)} cells")
    md_count = sum(1 for c in nb.cells if c.cell_type == "markdown")
    code_count = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"  markdown cells: {md_count}")
    print(f"  code cells:     {code_count}")


if __name__ == "__main__":
    main()
