# DiffPhyCon — Reproduction with Flow Matching

Reproduction of [DiffPhyCon (Wei et al, NeurIPS 2024)](https://github.com/AI4Science-WestlakeU/diffphycon), using **Flow Matching (CondOT)** as the generative backbone instead of the paper's DDPM. Flow Matching implementation follows methods taught in **[MIT 6.S184 — Generative AI with Stochastic Differential Equations and Flow Matching](https://diffusion.csail.mit.edu/)** (link placeholder — update if course site moved).

This is a fork of [AI4Science-WestlakeU/diffphycon](https://github.com/AI4Science-WestlakeU/diffphycon). Scope: **1D Burgers FOPC task only** (paper's Task 1). 2D Jellyfish and 2D Smoke not tested.

- 📄 Full report: [REPORT_fm_burgers_fopc.md](REPORT_fm_burgers_fopc.md)
- 📚 Original paper README: [README_ORIGINAL.md](README_ORIGINAL.md)

---

## Results

| Method | n_steps | J_actual | Sampling time/sample (A100) |
|:---|---:|---:|---:|
| Paper DDPM | 1000 | 0.001231 | 8.0 s |
| Paper DDIM | 100 | 0.000717 | 0.06 s |
| FM (CondOT) | 8 | 0.000174 | 0.005 s |

On 1D Burgers FOPC with the paper's data + evaluation pipeline (500 held-out samples, paper-faithful `burgers_metric`), FM at 8 NFE achieves 4.1× lower J than paper DDIM at 100 NFE, with 12× faster sampling.

---

## Findings

1. FM as DDPM replacement on Burgers FOPC: J 4×, speed 12×, 86% per-sample win rate against paper DDPM 1000-step
2. Paper L.1 (γ-reweighting has near-zero effect on FOPC): quantitatively confirmed — 36/36 (γ, n_steps) cells within ±5% of γ=1.0 baseline, using paper-faithful jellyfish β schedule
3. FM J(n_steps) has a U-shape: n=8 sweet spot, n=1000 degrades 1.74×. Root cause: velocity Lipschitz `L(τ) = 1/(1-τ) → ∞`
4. Dense-jump fix (arxiv [2509.13574](https://arxiv.org/abs/2509.13574)) flattens the U-shape on this task: `--dense_jump_tau 0.875` keeps J(n=1000) at 1.06× baseline n=8 sweet spot (vs 1.75× without it). First validation of dense-jump beyond robotic policies

Math derivations, ablations, pitfalls in [REPORT_fm_burgers_fopc.md](REPORT_fm_burgers_fopc.md).

---

## Files added in this fork

```
flow/burgers_fm_train.py             FM training CLI (dim=128, EMA, cosine LR, paper Table 5 config)
flow/burgers_fm_eval_v2.py           FM eval (γ-sweep, β schedule choice, dense-jump, RK4)
scripts/sweep_500_fresh.sh           Main paper baseline + FM sweep
scripts/sweep_jellyfish_schedule.sh  γ-sweep with paper-faithful jellyfish β
scripts/sweep_ushape_diag.sh         U-shape diagnostic (4 hypothesis tests, 60 cells)
scripts/sweep_dense_jump*.sh         Dense-jump validation (AutoDL + local versions)
scripts/sweep_baseline_local.sh      Local control for fair DJ comparison
scripts/sweep_raw_weights.sh         EMA vs raw weights ablation
scripts/analyze_*.py                 Analysis + plot scripts
scripts/verify_skip_first.py         Data-leak-free test set verification (5 tests)
REPORT_fm_burgers_fopc.md            Full report (~6000 words)
pull_*.sh                            AutoDL ↔ local sync scripts
environment.yml                      Conda environment spec
```

Original paper code under `inference/`, `diffusion/`, `model/` is unchanged. The only modification outside `flow/` and `scripts/` is in `dataset/apps/generate_burgers.py` — added a `--skip_first N` flag for leak-free test set generation (validated by `scripts/verify_skip_first.py`).

---

## Reproduce

```bash
# Setup
conda env create -f environment.yml
conda activate diffphycon

# Train FM joint model (~12 hours on A100)
python flow/burgers_fm_train.py --variant vanilla --model joint \
    --dim 128 --dim_mults 1 2 4 --num_steps 170000 \
    --dataset free_u_f_paper_fopc --batch_size 16 --lr 1e-4

# Main sweep (paper baseline + FM, ~25 min on A100)
bash scripts/sweep_500_fresh.sh

# γ-sweep with paper-faithful jellyfish β
SKIP_PAPER=1 GAMMAS="0.5 0.7 0.8 0.9 1.0 1.1 1.2" bash scripts/sweep_jellyfish_schedule.sh

# U-shape diagnostic
bash scripts/sweep_ushape_diag.sh

# Dense-jump validation
bash scripts/sweep_dense_jump_parallel.sh    # AutoDL
bash scripts/sweep_dense_jump_local.sh       # local Mac/MPS

# Analysis (locally)
python scripts/analyze_500fresh.py
python scripts/analyze_jellyfish_vs_sigmoid.py
python scripts/analyze_ushape_diag.py
```

---

## Limitations

- 1D Burgers FOPC only. Paper's Task 2 (Jellyfish) and Task 3 (Smoke) not tested
- Single seed for FM training (170k step model used on AutoDL, 180k locally)
- `--dense_jump_tau 0.875` chosen empirically (matches n=8 max τ); no principled procedure to pick τ_jump per task

---

## Citation

```
@misc{bctiann2026fmdiffphycon,
  author = {bcTiann},
  title  = {DiffPhyCon (Wei et al, NeurIPS 2024): reproduction with Flow Matching},
  year   = {2026},
  url    = {https://github.com/bcTiann/diffphycon}
}

@inproceedings{wei2024diffphycon,
  title     = {DiffPhyCon: A Generative Approach to Control Complex Physical Systems},
  author    = {Wei, Long and Hu, Peiyan and Feng, Ruiqi and Feng, Haodong and Du, Yixuan and Zhang, Tao and Wang, Rui and Wang, Yue and Ma, Zhi-Ming and Wu, Tailin},
  booktitle = {NeurIPS},
  year      = {2024}
}

@misc{densejump2025,
  title         = {Dense-Jump Flow Matching with Non-Uniform Time Scheduling for Robotic Policies: Mitigating Multi-Step Inference Degradation},
  year          = {2025},
  eprint        = {2509.13574},
  archivePrefix = {arXiv}
}
```

---

## License

Same as upstream (see [LICENSE](LICENSE) if present in original repo).
