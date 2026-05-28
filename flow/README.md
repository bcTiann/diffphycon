# flow/ — Flow Matching for DiffPhyCon

教学风格的 Flow Matching 实现,对照 MIT 6.S184 lab_three.ipynb 的 fill-in-the-blank 模式。

## Files

| 文件 | 说明 |
|:---|:---|
| **`lab_four.ipynb`** | **主 lab — 用 Jupyter 打开,按 cell 填 Q2.1 → Q4.4** |
| `lab_four.py` | 同样内容的 .py 版本(jupytext 源文件),git diff 友好 |
| `_build_lab_four_nb.py` | 后处理器,把 .py 转出的 .ipynb 美化 markdown cells |
| `lab_five.{py,ipynb}` | Jellyfish 2D 控制(paper Experiment 3)— 后续 |

### Explore notebooks(完善 Experiment 1,不打补丁,共享 helper)

| 文件 | 说明 |
|:---|:---|
| `lab_four_explore.py` | **共享 helper** — `load_fm` / `infer` / `sweep` / `LiveLossTrainer` / `JGradEulerSampler` / `plot_trajectory_grid` / `savefig`。自动从 `lab_four.ipynb` 抽 user 填好的代码生成 `lab_four_solved.py`(只留定义,丢顶层执行)。 |
| `lab_four_ood.{py,ipynb}` | **Q3 OOD test** — 4 种 OOD u_0(3-peak/jagged/same-sign/step)喂给现有 FM,看泛化崩多少 |
| `lab_four_jgrad.{py,ipynb}` | **Q1 J-gradient** — 给 FM 加 J-grad 引导,sweep wfs/wu。结论:wfs 是「省力↔精准」trade-off 旋钮,不免费降 J;wu 在 inpaint 下是 no-op |
| `lab_four_popc_lib.py` | **Q2 POPC 类库** — `BurgersPOPC{Dataset,FlowTrainer,VectorField,EulerSampler,Prior*}` + observed_mask + inpaint_overwrite_popc |
| `lab_four_popc.{py,ipynb}` | **Q2 POPC notebook** — 训 POPC joint+prior+EMA,γ sweep 对比 DDPM POPC baseline §3.2 |

**workflow**:`.py`(percent 格式)是 source of truth,改完跑 `jupytext --to notebook flow/lab_four_xxx.py` 重生成 `.ipynb`。结果图自动存 `flow/results/<workstream>/<timestamp>_*.png`。

## 怎么用

**推荐**:Jupyter 打开 `lab_four.ipynb`,按 cell 顺序读 markdown → 填 code cell 的 `raise NotImplementedError` → run sanity check cell。

```bash
cd /Users/baochen/diffphycon
jupyter notebook flow/lab_four.ipynb
```

Sanity check 顺序:
1. `sanity_check_part1()` — 数据可视化(已能跑)
2. 填 Q2.1-Q2.3 → `sanity_check_2_4()` — joint FM 训练
3. 填 Q3.1-Q3.2 → `sanity_check_3_3()` — γ=1 采样
4. 填 Q4.1-Q4.4 → `sanity_check_4_5()` — γ=1 退化 + γ≠1 区别
5. `part5_gamma_sweep(net_joint, net_prior)` — 复现 baseline 7 个 γ

## ⚠️ .ipynb 是 source of truth — 不要再跑 jupytext regen

`lab_four.ipynb` 里有你**填好的 code**。如果跑 `jupytext --to notebook lab_four.py` 就会用 `.py`(里面是 `raise NotImplementedError` placeholder)覆盖整个 `.ipynb`,**你填的代码全没**。

### 正确流程

1. **改 markdown 解释 / inject 新讲解**:改 `_build_lab_four_nb.py`,然后跑:
   ```bash
   python flow/_build_lab_four_nb.py
   ```
   这个脚本**原地修改 .ipynb,只动 markdown cell,不碰 code cell**(idempotent)。

2. **填代码**:直接在 Jupyter / IDE 里编辑 `lab_four.ipynb`,不需要任何同步命令。

3. **如果一定要同步回 .py**(比如要 git diff):
   ```bash
   jupytext --to py:percent flow/lab_four.ipynb -o flow/lab_four.py
   ```
   注意:这一步是 `.ipynb → .py`,**安全方向**(把你填的代码导到 .py,不丢)。

### 千万不要做
```bash
jupytext --to notebook flow/lab_four.py   # ← 反向同步,会覆盖 .ipynb!绝对不要
```

## 参考资料(都在 repo 根目录)

- `flow_matching_diffusion.md` — MIT FM 理论(Prop 1, Example 13)
- `notes_diffphycon_flow_bridge.md` — DDPM ↔ FM 翻译,§4 inpainting trick
- `notes_fm_prior_reweighting.md` — γ-reweighting 完整推导 + 代码骨架
- `notes_baseline_summary.md` — 要复现的 baseline 数字
- `/Users/baochen/Documents/MIT Flow Matching and Diffusion Models/lab_three.ipynb` — 教学风格参照
