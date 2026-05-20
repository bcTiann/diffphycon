# DiffPhyCon 1D Burgers' 控制 — Baseline 数字汇总

写这份的目的:把所有 baseline 实验结果集中起来,方便后续 flow matching 实现做对比。

---

## 1. 实验设置(我们 vs 论文)

| 维度 | 我们(small-scale)| 论文(full-scale)| 比例 |
|------|------------------|------------------|------|
| 训练数据集 | 10k(8k train + 2k test)| 100k | **1 : 10** |
| Joint 模型训练步数 | 25000(FOPC) / 6250(POPC) | 200000 | **~1 : 30** |
| Prior 模型训练步数 | 6250 | 200000 | **~1 : 30** |
| U-Net dim | 64(dim_muls = 1,2,4,8)| 128 (FOPC) / 64 (POPC) | small |
| Batch size | 16~64 | 16 | 同范围 |
| n_test_samples | 8 | 50 | small |

→ **整体 ~ 1/30 论文规模**,适合"概念验证"型对比,不适合冲绝对数字。

---

## 2. 已训模型清单

| 模型 | 文件位置 | 角色 | 训练步数 |
|------|---------|------|---------|
| `FOPC_10k` | `trained_models/burgers/FOPC_10k/cos10000-model-10.pt` | joint $p(u,f)$,FOPC | 25000 |
| `FOPC_w_10k` | `trained_models/burgers_w/FOPC_w_10k/cos10000-model-10.pt` | prior $p(f)$,FOPC | 6250 |
| `POPC_10k` | `trained_models/burgers/POPC_10k/cos10000-model-10.pt` | joint $p(u,f \mid u_{\text{partial}})$,POPC | 6250 |
| `POPC_w_10k` | `trained_models/burgers_w/POPC_w_10k/cos10000-model-10.pt` | prior $p(f \mid u_{\text{partial}})$,POPC | 6250 |

(另外还有论文原版的 paper-grade FOPC checkpoint(170k 步,dim=128),保留在 `trained_models/burgers/FOPC/cos10000-model-170.pt`,但我们不在主线对比里用它。)

---

## 3. Baseline 数字表(核心结果)

### 3.1 FOPC sweep(论文 paper-config:`sigmoid_flip` + γ ∈ {0.3 ... 2.5})

| γ | J_actual | Energy | ΔJ vs γ=1 | ΔE vs γ=1 |
|---|----------|--------|-----------|-----------|
| 0.3 | 0.00830 | 1670.7 | +1.3% | +0.9% |
| 0.5 | 0.00828 | 1666.0 | +1.0% | +0.6% |
| 0.7 | 0.00825 | 1661.8 | +0.6% | +0.3% |
| 0.9 | 0.00821 | 1658.0 | +0.2% | +0.1% |
| **1.0** | **0.00820** | **1656.2** | **0(baseline)** | **0** |
| 1.5 | 0.00811 | 1648.1 | −1.0% | −0.5% |
| **2.5** | **0.00796** | **1634.0** | **−3.0% ⭐** | **−1.3%** |

→ **趋势**:γ 越大,J 和 Energy 双双改善(单调),γ=2.5 是 sweep 内最优。
→ **符合论文 Thm 3.1 F(1) > 0 的预期**。

### 3.2 POPC sweep(论文 paper-config:`sigmoid_flip` + γ ∈ {0.3 ... 2.5})

| γ | J_actual | Energy | ΔJ vs γ=1 | ΔE vs γ=1 |
|---|----------|--------|-----------|-----------|
| 0.3 | 0.02014 | 1424.1 | +0.0% | +1.1% |
| 0.5 | 0.02014 | 1419.2 | +0.0% | +0.7% |
| 0.7 | 0.02014 | 1414.9 | +0.0% | +0.4% |
| 0.9 | 0.02013 | 1411.1 | +0.0% | +0.1% |
| **1.0** | **0.02013** | **1409.2** | **0(baseline)** | **0** |
| 1.5 | 0.02011 | 1401.0 | −0.1% | −0.6% |
| **2.5** | **0.02006** | **1387.3** | **−0.3%** | **−1.6%** |

→ **趋势**:γ ↑ 时 J 几乎不动,但 Energy 持续下降。
→ **没看到论文预期的 "γ < 1 救场"**(可能因为我们规模太小,F(1) 符号未翻转;详见 §5)。

---

## 4. 关键发现(给后续研究)

### 发现 1:γ 是个**对称的 reweighting 旋钮**(纠正之前误解)

代码 `--prior_beta` **等于** 论文 γ,**同方向,同含义**。公式:
$$
p_\gamma(\mathbf{u}, \mathbf{w} | \mathbf{c}) \propto p(\mathbf{w}|\mathbf{c})^\gamma \cdot p(\mathbf{u}|\mathbf{w}, \mathbf{c})
$$
- **γ < 1**:flatten prior,鼓励探索非典型 $\mathbf{w}$
- **γ > 1**:sharpen prior,逼向训练分布常见 $\mathbf{w}$

### 发现 2:`w_scheduler sigmoid_flip` **强烈影响 γ < 1 的行为**

| γ=0.3 时 J | 无 sigmoid_flip(旧实验)| 有 sigmoid_flip(新实验) |
|------------|--------------------------|--------------------------|
| FOPC | **0.0607**(灾难) | **0.0083**(几乎无伤) |

→ `sigmoid_flip` 让 reweighting 只在末段(t→0,x_start 干净时)发力,**避开早期不稳定区** → γ < 1 不再"崩"。**论文 FOPC/POFC 配置必须用它**。

### 发现 3:**γ > 1 真的有效改善 J 和 Energy**(FOPC 复现)

γ = 2.5 vs γ = 1.0:**J 降 3.0%,Energy 降 1.3%**,**两边都改善**(不是 Pareto trade-off)。

→ 验证论文 Thm 3.1 在 FOPC 上的 F(1) > 0 case。

### 发现 4:小规模下,**POPC 没复现 paper 的"γ < 1 改善 J"现象**

我们的 POPC J 在 γ 各值下几乎一样(0.020 浮动 0.3% 范围)。可能原因:
- POPC 模型欠训练(6250 步 vs paper 200k)→ 模型可能根本没学到论文所说的"信息不足导致 typical w 偏离最优"
- 数据集小 10× → p(u, f|c) 多峰性不充分体现
- 架构 dim=64 容量小 → 难以捕捉细微的 F(1) 翻转

→ **小规模下 FOPC 和 POPC 行为没明显分化**。两边都偏向 γ > 1 改善 Energy。

### 发现 5:**γ 旋钮在 POPC 上主要调 Energy,不调 J**

POPC γ 影响 Energy 但不影响 J,说明 POPC J 受**物理瓶颈**约束(因为部分观测无法消除的不确定性),γ 调不动。

---

## 5. 给后续 Flow Matching 研究的 baseline

### 5.1 比较口径

在你 FM 实现里,**和我们这套 baseline 直接比 J 和 Energy** 才公平:
- 同一份 10k 数据
- 同一份评估 protocol(8 test samples)
- 同一架构(dim=64)

### 5.2 目标数字(超越则算成功)

| Setting | 数字目标(γ=1 baseline)|
|---------|------------------------|
| **FOPC** | J ≤ **0.0082**,Energy ≤ **1656** |
| **POPC** | J ≤ **0.0201**,Energy ≤ **1409** |

如果 FM 实现能在同 setting 同 γ 下达到或超过这俩,**baseline 算复现成功**。

### 5.3 加分项

| 加分点 | 怎么判定 |
|--------|---------|
| FM γ-reweighting 也工作 | 跑 γ ∈ {0.3, ..., 2.5} sweep,看 γ↑ 是否也改善 J/E |
| FM 采样比 DDPM 快 | 同精度下 ODE 步数 < 1000 |
| FM 在小规模下不需 `w_scheduler` | sigmoid_flip 是 DDPM 工件,看 FM 是否天然稳定 |
| 发现 F(1) 符号翻转的真正条件 | 跑更大规模 POPC,看 γ < 1 何时开始改善 J |

---

## 6. 实验产物清单

### 6.1 npz 轨迹文件(`outputs/trajectories/`)

每个 γ 值的 8 个 test samples 的预测和模拟轨迹:
```
inference_trajectories_beta{XX}_FOPC_paper.npz   (7 个 γ)
inference_trajectories_beta{XX}_POPC_paper.npz   (7 个 γ)
```
另外旧实验保留作对照:
```
inference_trajectories_gamma{XX}_10k.npz         (FOPC 无 sigmoid_flip)
inference_trajectories_gamma{XX}_POPC_10k.npz    (POPC 无 sigmoid_flip)
inference_trajectories_withUT.npz                (FOPC paper checkpoint × undertrained prior)
inference_trajectories_noUT.npz, _noU0.npz 等    (hard conditioning 消融)
```

### 6.2 可视化图(`outputs/figures/`)

```
gamma_sweep_FOPC_paper_sample0.png      ← 7 行 × 4 列,FOPC 完整 γ 扫描
gamma_sweep_POPC_paper_sample0.png      ← 7 行 × 4 列,POPC 完整 γ 扫描
gamma_sweep_10k_sample0.png             ← 旧 FOPC sweep(5 行,γ ≤ 1)
gamma_sweep_POPC_10k_sample0.png        ← 旧 POPC sweep
inference_viz_*.png                     ← 4 样本 × 4 panel,单个 γ 配置
training_loss_FOPC_10k.png              ← FOPC joint 训练曲线
training_loss_latest.png                ← 最新一次训练曲线
```

### 6.3 相关 notes
- `notes_p_sample_loop.md`:DDPM 采样 loop 代码视角细节
- `notes_inference_deep.md`:inference 全套理论 + 实操(包括 §07 教程)
- `notes_diffphycon_flow_bridge.md`:DDPM ↔ FM 桥接,以及 FOPC/POPC 在两个框架下的等价
- **本文档**:实验数字汇总 + flow matching 研究起点

---

## 7. 主线下一步

1. **设计 FM 实现框架**(velocity prediction + ODE Euler + Inpainting + γ-mix)
2. **训 FM 等价模型**(同 10k 数据,同架构 dim=64,等价 step 数)
3. **复现 4 个 baseline 数字**(FOPC γ=1, FOPC γ=2.5, POPC γ=1, POPC γ=2.5)
4. **写论文 / 报告:DDPM vs FM 在 Burgers 控制上的对比**

→ 见 `notes_diffphycon_flow_bridge.md §9` 路径图。
