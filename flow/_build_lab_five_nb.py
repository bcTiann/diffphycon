"""
Post-processor for flow/lab_five.ipynb.

⚠️ NEW WORKFLOW — do NOT run `jupytext --to notebook` before this.
   That command would OVERWRITE the .ipynb (losing user's filled-in code).
   This script reads the EXISTING .ipynb in-place and only modifies
   markdown cells (never touches code cells).

What it does, idempotently:
- Top docstring → proper intro markdown (if not already converted)
- Part separators (============ Part N ============) → clean H1 markdown
- Question separators (---- Question N.M ----) → clean H2 markdown
- Inject 1 jellyfish-specific layout explainer (5D shape gotcha)

User's filled-in code is fully preserved.

Run:    python flow/_build_lab_five_nb.py
"""
import os
import nbformat

NB_PATH = os.path.join(os.path.dirname(__file__), "lab_five.ipynb")


TOP_INTRO_MD = r"""# Flow Matching for 2D Jellyfish Control — `lab_five`

Companion to `lab_four.ipynb` (Burgers 1D, DiffPhyCon Experiment 1).

This lab is the **toy / local** version of paper Experiment 3 (2D jellyfish control):

- Smaller `Unet3D` ($\dim = 32$, $\dim\_mults = (1,2)$, ~1.9M params vs paper's 22M)
- Trains on the 200 test_data samples (split 160 train / 40 eval), instead of the full 30k-sample training set that lives on Modal
- **Skips** prior-reweighting + LilyPad eval (those are `lab_five_modal.py` future work)
- Designed to run end-to-end on M4 Pro MPS in one overnight

**Why toy**: verify FM scaffolding works on 3D (T, H, W) data layout and learn how jellyfish data + Unet3D differ from Burgers. Paper-faithful Table 28 comparison needs Modal A100.

## How to use this lab

1. Read the markdown cell at the top of each Part.
2. Find each `# Question N.M` cell and **fill in** the methods marked `raise NotImplementedError(...)`. Each fill-in has a `# Step N:` comment that says exactly what to compute.
3. After each Part, **run the sanity check cell**.
4. When all sanity checks pass, run `train_jellyfish_for_part5(num_steps=15000)` (overnight on M4 Pro), then use Part 5 helpers to visualize.

## ⚠️ Data layout — `(b, T, C, H, W)` not `(b, C, T, H, W)`

`Unet3D_with_Conv3D.forward` internally does `x.permute(0, 2, 1, 3, 4)`, so it expects input in `(b, T, C, H, W)`. **The channel dim is dim 2, not dim 1.**

This means:
- `z` has shape `(b, 20, 4, 64, 64)` — 4 channels (state vx/vy/pressure + theta-broadcast)
- `c` has shape `(b, 20, 3, 64, 64)` — 3 channels (bd mask + 2 offsets), pre-padded 62→64
- **`torch.cat([x, c], dim=2)`** when injecting boundary — NOT `dim=1` (which would concat over time)
- `t` (FM time) shape `(b, 1, 1, 1, 1)` for broadcasting

This is the only **systematic difference** from `lab_four`. Most fill-ins are just `lab_four` with shape `(b, 1, 1, 1, 1)` instead of `(b, 1, 1, 1)`.

## Background reading

- `flow/lab_four.py` — the Burgers FM lab this mirrors (abstractions imported from there)
- `LILYPAD_DEEP_DIVE.md` — jellyfish eval pipeline (used in `lab_five_modal.py` future work)
- `notes_diffphycon_flow_bridge.md` — DDPM ↔ FM translation
- `train/train_2d_jellyfish.py` — paper DDPM trainer (we mirror the model arch)
"""


PART_INTROS = {
    "Part 0": """# Part 0: Setup + Re-imports

We **reuse all the FM scaffolding from lab_four**:

- `Sampleable` / `LabeledSampleable` — distribution ABCs
- `ConditionalProbabilityPath` / `GaussianConditionalProbabilityPath` — Gaussian path with $\\alpha_\\tau = \\tau, \\beta_\\tau = 1 - \\tau$
- `LinearAlpha` / `LinearBeta` — schedules
- `VectorFieldNet` — ABC for velocity-field networks
- `Trainer` — generic FM training loop (tqdm + loss_history)
- `EMA` / `finetune_with_ema` — optional EMA helpers

Then we import jellyfish-specific repo modules:

- `Unet3D_with_Conv3D` from `model/video_diffusion_pytorch/video_diffusion_pytorch_conv3d.py` — the 3D conv backbone (same architecture as paper DDPM, just smaller).

**No fill-ins in Part 0.**""",

    "Part 1": """# Part 1: Get a Feel for Jellyfish Data

Like `lab_four` Part 1, we visualize the data before building any models.

Each jellyfish simulation has **40 frames** of:

- **state** $(40, 3, 64, 64)$ — channels are $(v_x, v_y, p)$ (velocity x/y + pressure)
- **boundary mask + offsets** $(40, 3, 62, 62)$ — describes where the jellyfish wings are at each frame (mask + horizontal/vertical offset)
- **θ** $(40,)$ — the wing rotation angle (the control variable)

We work with the **first 20 frames** of each sim → 20-frame windows.

**No fill-ins in Part 1.**""",

    "Part 2": """# Part 2: Train Joint FM (Q5.1 - Q5.4)

**Goal**: learn the velocity field $u_\\tau^\\theta(z \\mid c)$ of the joint distribution $p(\\text{state}, \\theta \\mid \\text{boundary})$ over jellyfish trajectories.

**Three classes to fill in**:

- **Q5.1** — `JellyfishDataset.sample` (3 Steps) — extract 20-frame windows, pack into $(z, c)$
- **Q5.2** — `JellyfishFlowTrainer.get_train_loss` (5 Steps) — CFM loss, same as `lab_four` Q2.2 but 5D
- **Q5.3** — `JellyfishVectorField.forward` (3 Steps) — wraps `Unet3D`, concat $c$ along **dim=2**

**Key differences vs `lab_four`**:

| | Burgers (`lab_four`) | Jellyfish (this lab) |
|:---|:---|:---|
| $z$ shape | $(b, 2, 16, 128)$ | $(b, 20, 4, 64, 64)$ |
| $c$ injection | inpaint_overwrite (replace rows) | concat on channel dim=2 |
| $c$ scope | 2 slices (initial + terminal $u$) | full 20-frame boundary trajectory |
| FM time $t$ shape | $(b, 1, 1, 1)$ | $(b, 1, 1, 1, 1)$ |
| Inpainting trick on `u_target` | yes (Step 5 of Q2.2) | **no** — $c$ never noised, lives outside $x$ |

After filling, run `sanity_check_2_4()` — loss should drop to ~0.5 in 200 steps.""",

    "Part 3": """# Part 3: ODE Sampling (Q5.5 + Q5.6)

**Goal**: sample $z$ from the trained velocity field.

Unlike Burgers, we **don't need `inpaint_overwrite`** — the boundary $c$ lives in a separate tensor (not inside $x$), so it never gets noised. Sampling is pure Euler ODE.

**Q5.5** — `JellyfishEulerSampler.sample` (3 Steps) — Euler integration $\\frac{dx}{d\\tau} = v_\\tau^\\theta(x \\mid c)$ from $\\tau = 0$ (noise) to $\\tau = 1$ (clean).

After filling, run `sanity_check_3_3()` — trains 300 steps and visualizes one sample. The FM L2 should be < random L2.""",

    "Part 4": """# Part 4 — **SKIPPED in toy v1**

Prior model + `ReweightedVectorField` + $\\gamma$ sweep + LilyPad evaluation.

**Why skipped here**:

1. LilyPad eval requires Processing IDE + ~30 min per 50-sample sweep — not iteration-friendly with a tiny toy model
2. Without LilyPad we **can't compute** $\\bar{v}$ / $R(\\theta)$ / $J$ — no quantitative way to validate $\\gamma$ sweep results
3. Toy net (1.9M params, 15k step training) may not produce realistic enough $\\theta$ trajectories for $\\gamma$ sweep to matter

**Future**: `lab_five_modal.py` adds:

- `JellyfishPriorDataset` (zero state channels)
- `JellyfishPriorTrainer`
- `ReweightedVectorField` for 3D
- $\\gamma$ sweep via existing `jellyfish_modal.py`
- LilyPad eval via existing `lilypad_prepare.py` + `lilypad_parse.py`""",

    "Part 5": """# Part 5: Visual Evaluation

Once you've trained a real net (via `train_jellyfish_for_part5(num_steps=15000)`), use these helpers to look at samples:

- `visualize_sample_vs_gt(net, ds_eval, idx)` — side-by-side gt vs predicted state + $\\theta$
- `compute_l2_per_channel(net, ds_eval)` — mean L2 per channel + comparison to random baseline

We **don't** compare to paper Table 28 here (no LilyPad in toy mode). For that, see `lab_five_modal.py` future work.""",

    "Sanity Checks": """# Sanity Checks

Run these in order after filling in each Part:

1. `sanity_check_part1()` — visualize jellyfish data (no fill-in needed)
2. `sanity_check_2_4()` — after Q5.1-Q5.3: trainer runs and loss drops
3. `sanity_check_3_3()` — after Q5.5: sample from trained net, FM L2 < random L2""",

    "Main entry": """# Main entry

The `if __name__ == "__main__"` block prints quickstart instructions. You won't typically run this cell in the notebook — just call the individual `sanity_check_*` functions as you finish each Part.""",
}


QUESTION_INTROS = {
    "Question 5.1": """## Question 5.1 — `JellyfishDataset.sample`

Build the data sampler. For each batch, return:

- `z` of shape $(b, 20, 4, 64, 64)$ — joint = state (3 channels) + theta-broadcast (1 channel)
- `c` of shape $(b, 20, 3, 64, 64)$ — boundary mask+offsets, **padded from 62×62 → 64×64**

⚠️ **Channel dim is dim=2** (not dim=1!) — see top-of-file layout note.

**3 fill-in Steps**.""",

    "Question 5.2": """## Question 5.2 — `JellyfishFlowTrainer.get_train_loss`

**Structurally identical to `lab_four` Q2.2.** Only difference: shapes are 5D `(b, 20, 4, 64, 64)` instead of 4D `(b, 2, 16, 128)`.

So when sampling FM time, use `(b, 1, 1, 1, 1)` (5 ones) instead of `(b, 1, 1, 1)` (4 ones).

**No inpainting trick** (`lab_four` Step 5) needed — the boundary $c$ lives outside $x$ and never gets noised, so target velocity rows don't need zeroing.

**5 fill-in Steps** (vs 6 in `lab_four` Q2.2 — one less because no inpaint trick).""",

    "Question 5.3": """## Question 5.3 — `JellyfishVectorField.forward`

Wraps `Unet3D_with_Conv3D` as a `VectorFieldNet`. The trick: **concat $c$ onto $x$ along the channel dim** before passing to the Unet.

⚠️ **`torch.cat([x, c], dim=2)`** — NOT `dim=1`! Dim 1 is time (20 frames), dim 2 is channel. Concat on the wrong dim gives a runtime error from Unet3D's first conv layer.

After concat the shape is $(b, 20, 7, 64, 64)$ — 4 z-channels + 3 c-channels. Unet3D's `out_dim=4` returns only the velocity for $z$.

**3 fill-in Steps**.""",

    "Question 5.5": """## Question 5.5 — `JellyfishEulerSampler.sample`

Euler ODE integration from $\\tau = 0$ (noise) to $\\tau = 1$ (clean data). **No `inpaint_overwrite`** — boundary $c$ lives outside $x$.

**3 fill-in Steps**.""",
}


PACKAGING_MD = r"""## What `z` and `c` are — packaging the 3 raw tensors into 2 FM inputs

You've seen the 3 raw tensors per sim:

- `state` $(40, 3, 64, 64)$ — water flow (vx, vy, pressure)
- `bd_mask_offset` $(40, 3, 62, 62)$ — where the wings are
- `theta` $(40,)$ — wing angle scalar per frame

But FM wants just **two** things:

| FM concept | Role | What it contains | Shape |
|:---|:---|:---|:---|
| **z** | the thing we GENERATE (output) | state + θ-broadcast | $(b, 20, 4, 64, 64)$ |
| **c** | the thing we're CONDITIONED on (input) | bd_mask + offsets | $(b, 20, 3, 64, 64)$ |

In words: **"given the wing position trajectory (c), generate the water flow and the angle sequence (z)"**.

### Why `state` AND `θ` both go into `z`?

Physically `state` is a *result* (water gets pushed by the wings) and `θ` is a *cause* (control variable). But FM doesn't care about causality — it learns the **joint distribution** $p(\text{state}, \theta \mid \text{boundary})$.

Think of it like GPT generating an A/B dialogue — it doesn't know who "speaks first", it just learns the joint over the whole conversation. Same here: state and θ are generated together as one $z$.

(The full Modal version of this lab will add a **second** model that learns the marginal $p(\theta \mid \text{boundary})$ — the "prior" — then γ-reweights joint and prior at sampling time to bias θ toward "fast-swimming" patterns. That's Part 4 we're skipping in toy mode.)

### Why does θ get "broadcast" to 64×64? (the key trick)

`state` is a (64, 64) image per channel. `θ[t]` is a single scalar. They can't be `torch.cat`-ed because shapes don't match.

**The fix**: pretend θ is also a (64, 64) image — just fill the entire image with the scalar value.

```python
theta[t] = 0.871           # one scalar

# After broadcast:
theta_bcast[t] = a (64, 64) image where every pixel = 0.871
# Looks like:
#   0.871  0.871  0.871  ...  0.871
#   0.871  0.871  0.871  ...  0.871
#   ...
#   0.871  0.871  0.871  ...  0.871
```

Visually it's a flat color, but the shape now matches `state`'s channels. We `cat` them as channel 4:

```
state         (40, 3, 64, 64)   ← 3 channels: vx, vy, p
theta_bcast   (40, 1, 64, 64)   ← 1 channel: pixel value = θ
z = cat       (40, 4, 64, 64)   ← 4 channels: vx, vy, p, θ
```

### Why this works for the network

Unet3D sees a "θ channel" that is a flat color per frame. **A flat color carries no spatial information (every pixel identical) but carries temporal info (color changes frame-to-frame)** — exactly what we want, because θ is a per-frame scalar.

The temporal attention layers inside Unet3D pick up "this frame has θ = 0.871, next frame θ = 0.793". This is the standard trick paper DDPM also uses (`train_2d_jellyfish.py` has 7-channel input = state(3) + bd(3) + θ-broadcast(1)).

### Why does `c` not need broadcast?

Because `bd_mask_offset` is already a per-pixel image — each pixel has its own mask/offset value (real spatial information). We only `F.pad` it from 62×62 → 64×64 to match `state`.

### TL;DR

| What | Why this shape |
|:---|:---|
| **z** = state + θ-broadcast → $(b, 20, 4, 64, 64)$ | θ is a scalar; broadcast it to a flat (64, 64) image so it can stack with state |
| **c** = boundary, padded → $(b, 20, 3, 64, 64)$ | already an image; just pad 62→64 |
| FM learns | velocity field $v_\tau(z \mid c)$ — i.e. how to denoise (state, θ) given (boundary) |

The Q5.1 fill-in code in the next cell is literally just this packaging — once you grok this section, the Q5.1 hints will read like obvious one-liners.
"""


LAYOUT_GOTCHA_MD = r"""## ⚠️ The 5D layout gotcha (read before Q5.1!)

`Unet3D_with_Conv3D.forward` does this on its first line:

```python
def forward(self, x, time, cond=None, ...):
    x = x.permute(0, 2, 1, 3, 4)   # swaps dim 1 and dim 2
```

So **it expects input in `(b, T, C, H, W)`**, not the "natural" `(b, C, T, H, W)`.

If you pass `(b, C, T, H, W)` by Burgers-instinct, you'll see an error like:

```
RuntimeError: Given groups=1, weight of size [32, 7, 7, 7, 7],
expected input[1, 20, 7, 64, 64] to have 7 channels, but got 20 channels instead
```

(it treats your time dim as channels and gets the wrong count)

**Throughout this lab**:
- `z` is `(b, T=20, C=4, H=64, W=64)` — channel dim is **dim=2**
- `c` is `(b, T=20, C=3, H=64, W=64)` — channel dim is **dim=2**
- Concat them: `torch.cat([z, c], dim=2)` → `(b, 20, 7, 64, 64)`
- Theta broadcast: `theta[:, :, None, None, None].expand(-1, T, 1, H, W)` → `(b, 20, 1, 64, 64)`

The cached data in `JellyfishDataset` is already in `(N, 40, 3, 64, 64)` per-sim layout = `(N, T, C, H, W)`, so you don't need any `.permute()` calls in Q5.1.

This matches `dataset/data_2d.py::Jellyfish` (the paper's dataset class) and `train/train_2d_jellyfish.py` (the paper's trainer)."""


# Markers used to detect Part / Question separators in source comments
PART_MARKERS = {
    "Part 0": "Part 0: Setup + Re-imports",
    "Part 1": "Part 1: Get a Feel for Jellyfish Data",
    "Part 2": "Part 2: Train Joint FM",
    "Part 3": "Part 3: ODE Sampling",
    "Part 4": "Part 4 [SKIPPED",
    "Part 5": "Part 5: Visual Evaluation",
    "Sanity Checks": "Run sanity checks in order",
    "Main entry": "Main entry",
}

QUESTION_MARKERS = {
    "Question 5.1": "Question 5.1",
    "Question 5.2": "Question 5.2",
    "Question 5.3": "Question 5.3",
    "Question 5.5": "Question 5.5",
}


def fix_cell(cell):
    """Convert separator comments / docstrings into proper markdown cells.

    Idempotent: if cell is already a markdown cell starting with the expected
    intro, we replace its source (keeps in sync with edits to this script).
    """
    src = cell.source.strip()

    # Top docstring — only present on the very first cell as a code-string
    if cell.cell_type == "code" and src.startswith('"""') and "lab_five.py" in src:
        return nbformat.v4.new_markdown_cell(TOP_INTRO_MD)

    # Question markers in markdown separator cells (e.g. "---- Question 5.1 ----")
    if cell.cell_type == "markdown":
        for key, marker in QUESTION_MARKERS.items():
            if marker in src and "---" in src:
                return nbformat.v4.new_markdown_cell(QUESTION_INTROS[key])

    # Part markers — check both code (raw === separator) and markdown
    for key, marker in PART_MARKERS.items():
        if marker in src:
            return nbformat.v4.new_markdown_cell(PART_INTROS[key])

    # Drop noisy separator-only markdown cells (the bare === or --- with nothing else)
    if cell.cell_type == "markdown":
        stripped = src.replace("=", "").replace("-", "").replace("#", "").strip()
        if stripped == "":
            return None    # signal to drop

        # Generic "# === ... === / # Title / # === ... ===" leftover separators
        # → convert to a small H3 markdown using the title line (strip surrounding ---).
        lines = [ln.strip().lstrip("#").strip() for ln in src.split("\n") if ln.strip()]
        title_lines = [ln for ln in lines if ln and not all(c in "=-" for c in ln)]
        if (src.count("=") + src.count("-") > 20) and len(title_lines) == 1:
            clean_title = title_lines[0].strip("=- ").strip()
            return nbformat.v4.new_markdown_cell(f"### {clean_title}")

    return cell


def _insert_or_update_md(cells, target_match_fn, md_content):
    """Insert md_content as a markdown cell before the first cell matching
    target_match_fn. Idempotent — if a markdown cell already starts with the
    same fingerprint (first line), update its source in place instead.
    """
    fingerprint = md_content.split("\n", 1)[0].strip()
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith(fingerprint):
            cells[i] = nbformat.v4.new_markdown_cell(md_content)
            return cells

    out = []
    inserted = False
    for c in cells:
        if not inserted and target_match_fn(c):
            out.append(nbformat.v4.new_markdown_cell(md_content))
            inserted = True
        out.append(c)
    return out


def insert_layout_gotcha(cells):
    """Insert the 5D layout gotcha cell before Q5.1."""
    def match(c):
        return c.cell_type == "markdown" and c.source.lstrip().startswith("## Question 5.1")
    return _insert_or_update_md(cells, match, LAYOUT_GOTCHA_MD)


def insert_packaging_explainer(cells):
    """Insert the 'What z and c are' explainer before the layout gotcha (or before Q5.1 if no gotcha)."""
    def match(c):
        return c.cell_type == "markdown" and c.source.lstrip().startswith("## ⚠️ The 5D layout gotcha")
    return _insert_or_update_md(cells, match, PACKAGING_MD)


def main():
    if not os.path.isfile(NB_PATH):
        raise FileNotFoundError(
            f"{NB_PATH} not found. First run:\n"
            f"  jupytext --to notebook flow/lab_five.py"
        )

    nb = nbformat.read(NB_PATH, as_version=4)
    cells = [fix_cell(c) for c in nb.cells]
    cells = [c for c in cells if c is not None]

    cells = insert_layout_gotcha(cells)
    cells = insert_packaging_explainer(cells)

    nb.cells = cells
    nbformat.write(nb, NB_PATH)
    print(f"✅ updated {NB_PATH}")
    print(f"   {len(nb.cells)} cells total")
    md_count = sum(1 for c in nb.cells if c.cell_type == "markdown")
    code_count = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"   {md_count} markdown / {code_count} code")


if __name__ == "__main__":
    main()
