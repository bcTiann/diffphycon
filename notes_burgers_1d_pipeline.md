# 1D Burgers FM Pipeline — 端到端总览

写给「重新回来想搞清楚整个 lab_four 怎么从原始数据走到对比图」的我自己。**不重复**其他 notes 的数学细节,只画**全流程地图 + 数据形状流动**。深入推导链接到对应 notes。

---

## §0 这份文档跟其他 notes 的关系

| Notes | 它讲什么 | 这里讲什么 |
|:---|:---|:---|
| `notes_diffphycon.md` | 论文公式 + MIT 视角翻译 | (不重复)|
| `notes_diffphycon_flow_bridge.md` | DDPM ↔ FM 数学桥接 | (不重复)|
| `notes_fm_prior_reweighting.md` | γ 公式 5 步推导 | (不重复)|
| `notes_baseline_summary.md` | DDPM 训完的数字 | (不重复)|
| `notes_inference_deep.md` | DDPM 采样代码视角 | (不重复)|
| **本文** | **数据 → 模型 → 训练 → 推理 → 评估 的全流程** | ✅ |

---

## §1 任务一句话

**控制 Burgers 物理系统**:给定初始状态 $u_0(x)$ 和目标终态 $u_T^*(x)$,找控制力 $w(t, x)$ 让系统满足:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} + w(t, x), \qquad u(0, x) = u_0(x)
$$

最小化:

$$
J \;=\; \|u(T, x) - u_T^*(x)\|^2 \;+\; \lambda \|w\|^2
$$

**FOPC**(Fully Observed Partial Control):全空间能观测 u,但 w **只能作用在前 1/4 和后 1/4 空间区域**,中间一半 ($x \in [\tfrac{1}{4}, \tfrac{3}{4}]$) 不能加 control。

---

## §2 数据(disk → tensor 形状)

### Disk 文件

`data/free_u_f_1e4_front_rear_quarter/`:
- `burgers_train.h5` — **8000 个** 训练 trajectory
- `burgers_test.h5` — **2000 个** 测试 trajectory

每条 trajectory:
- `u` shape `(11, 128)` — 状态场,$N_t = 11$ 个时间步,$N_x = 128$ 个空间点
- `w` shape `(10, 128)` — 控制力,**10 步**($w$ 驱动 $t \to t+1$,所以比 $u$ 少 1 步)

### Burgers1D 类

加载 HDF5 后 stack + pad,**每个 sample**:

$$
x \in \mathbb{R}^{2 \times 16 \times 128}
$$

- **channel 0** = $u$,**channel 1** = $w$(已扩到 11 步,row 10 = 0)
- **time = 16**:前 11 行是物理,**后 5 行(11..15)是 zero-padding**(为了 Unet 4 级 downsample,16 = $2^4$ 能整除)
- **normalize**:除以 `rescaler=10`,模型看到的范围 ~[-1, 1]

### BurgersDataset 类(我们 lab 的)

```python
class BurgersDataset(LabeledSampleable):
    def __init__(self, dataset, device):
        # pre-stack 全 N=8000 trajectory 到一个 tensor 进内存(~164 MB,M4 Pro OK)
        self.all_z = torch.stack([dataset[i] for i in range(len(dataset))]).to(device)

    def sample(self, batch_size):
        idx = torch.randint(0, self.N, (batch_size,))
        z = self.all_z[idx]                        # (b, 2, 16, 128)
        c = torch.stack([z[:, 0, 0, :], z[:, 0, T_IDX, :]], dim=1)   # (b, 2, 128)
        return z, c
```

**返回**:
- `z`(完整 trajectory)= $(b, 2, 16, 128)$
- `c`(boundary condition)= $(b, 2, 128)$,**channel 0 = $u_0$, channel 1 = $u_T^*$**

---

## §3 模型架构

**`BurgersVectorField`**(`flow/lab_four.py`)= 包了 `Unet2D` 的 `VectorFieldNet`(NN ABC)子类:

```python
BurgersVectorField(dim=64, dim_mults=(1, 2, 4, 8))
  ↓ wraps
Unet2D(dim=64, dim_mults=(1,2,4,8), channels=2)
  ↑
  35,707,906 参数 (~36M)
```

输入 / 输出 shape:
- 输入 `x`:$(b, 2, 16, 128)$,在 forward 内部用 `inpaint_overwrite(x, c)` 把 row 0 / row $T$ 覆盖为 clean $u_0$ / $u_T^*$
- 输入 `t`:$(b,)$ 或 $(b, 1, 1, 1)$ — FM 时间 $\tau \in [0, 1]$
- 输入 `c`:$(b, 2, 128)$ — boundary,**只通过 inpainting 注入**,不直接 embed
- 输出 `v`:$(b, 2, 16, 128)$ — velocity field

**两个独立 net 实例**:
- `net_joint`:学 $p(u, w \mid c)$
- `net_prior`:学 $p(w \mid c)$(u-channel 强制 0,见 §4)

---

## §4 训练管线(2 个模型)

### Joint Training

```python
ds = BurgersDataset(load_burgers_train(...))     # 8000 train
path = GaussianConditionalProbabilityPath(LinearAlpha, LinearBeta)  # α=τ, β=1-τ
net_joint = BurgersVectorField(dim=64, ...).to(device)
trainer = BurgersFlowTrainer(net_joint, path, ds, lr=1e-4)
trainer.train(num_steps=25000, batch_size=64)
```

每一步 loss:

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{z \sim p_{\text{data}}, \tau \sim U[0,1], \varepsilon \sim \mathcal{N}(0,I)} \left\| u_\tau^\theta(x_\tau \mid c) - u^{\text{target}}(x_\tau \mid z) \right\|^2
$$

其中 $x_\tau = \alpha_\tau z + \beta_\tau \varepsilon$ 是 noisy interpolation。

**Inpainting trick**:在算 loss 前强制 $u^{\text{target}}$ 的 row 0 / row $T$ = 0(教模型「看到 clean boundary 别动」,详见 `notes_diffphycon_flow_bridge.md §4.4`)。

### Prior Training

跟 Joint 几乎一样,**3 个差异**:

1. **数据集**用 `BurgersPriorDataset`,sample 时 **u-channel 强制清零** → 网络只学 control 的 marginal $p(w \mid c)$,不学 u dynamics
2. **训练 loss** 多 2 行(`BurgersPriorTrainer.get_train_loss`):
   - `u_target[:, 0] = 0`:整个 u-channel target 强制 0
   - `u_pred[:, 0] = 0`:模型输出 u-channel 也强制 0
3. **step 数少**(`6250` vs joint `25000`):prior 简单,收敛快

数学解释:`notes_fm_prior_reweighting.md §2.4` 的「u-block = 0 嵌入」原理 — prior 自然地嵌入到联合 (u, w) 空间。

---

## §5 推理管线

### γ = 1(纯 joint sampling)

```
x ~ N(0, I)                              # shape (b, 2, 16, 128)
for τ in linspace(τ_min, 1-τ_min, 100):  # 100 Euler 步
    x = inpaint_overwrite(x, c)          # 强制 row 0/T = u_0/u_T*
    v = net_joint(x, τ, c)               # (b, 2, 16, 128)
    x = x + v · dτ
x = inpaint_overwrite(x, c)              # 最终再覆盖一次保 boundary 严格
```

输出 `x`:`x[:, 0, :, :]` 是预测的 u-field,`x[:, 1, :, :]` 是预测的 w-field。

### γ ≠ 1(reweighted sampling)

只换 velocity 函数 — `ReweightedVectorField` 内部:

$$
\tilde u_\tau(x \mid c) \;=\; u_\tau^{\text{joint}}(x \mid c) \;+\; (\gamma - 1) \cdot \tilde\eta(\tau) \cdot \big[\,u_\tau^{\text{prior}}(x \mid c) \;-\; b_\tau \cdot [0, w]\,\big]
$$

完整推导:`notes_fm_prior_reweighting.md §3`。其中:
- `b_τ = α̇_τ / α_τ = 1/τ`(CondOT path)
- `η̃(τ)` 是 sigmoid_flip schedule(noise 端 ≈ 0,clean 端 ≈ 1,详见 `notes_fm_prior_reweighting.md §3 Step 5`)
- $[0, w]$ 是 x 的 u-channel 强制 0 的副本

**γ > 1**:sharpen prior → control 拉向训练集常见模式;**γ < 1**:flatten prior → 鼓励探索。

---

## §6 评估

### J(control quality)

```python
# compute_J_and_energy in lab_four.py
w_pred = x_pred[:, 1, :10, :]           # 取预测 w(只取 10 真实步)
w_pred *= rescaler                       # 反归一化到物理单位

# burgers_metric 内部:
#   1. FOPC mask: w[:, :, Nx//4 : 3*Nx//4] = 0  (中间 50% 强制 0)
#   2. PDE forward solve: u_sim = burgers_numeric_solve_free(u_0, w_pred)
#   3. J = mean((u_sim[:, -1, :] - u_T_star) ** 2)
J = mean over batch
```

**FOPC 约束在评估时强制** — 跟 DDPM baseline 同样的算法(`utils.py::burgers_metric` 同款),保证 apples-to-apples。

### Energy

$$
E \;=\; \sum_{t, x} w(t, x)^2
$$

DDPM baseline(`notes_baseline_summary.md §3.1`):
- γ=1.0:$J = 0.0082$,$E = 1656$
- γ=2.5:$J = 0.00796$,$E = 1634$

---

## §7 全流程图(数据 + 形状流动)

```
                ┌──────────────────────────────────────────┐
                │  DATA                                    │
                │  data/free_u_f_1e4_front_rear_quarter/  │
                │  burgers_train.h5 (8000) + test.h5 (2000)│
                └─────────────────┬────────────────────────┘
                                  │
                                  ▼  load_burgers_train / test
                ┌──────────────────────────────────────────┐
                │  Burgers1D                               │
                │  __getitem__(i) → (2, 16, 128) tensor    │
                └─────────────────┬────────────────────────┘
                                  ▼  pre-stack (BurgersDataset.__init__)
                ┌──────────────────────────────────────────┐
                │  BurgersDataset                          │
                │  all_z: (N, 2, 16, 128) in MPS memory    │
                │  .sample(b) → (z, c)                     │
                │      z: (b, 2, 16, 128)  full trajectory │
                │      c: (b, 2, 128)      (u_0, u_T*)     │
                └────────────┬─────────────────────────────┘
                             │
                ┌────────────┴─────────────┐
                │                          │
                ▼                          ▼
    ┌─────────────────────┐    ┌─────────────────────┐
    │ Joint Training      │    │ Prior Training      │
    │ (BurgersDataset)    │    │ (BurgersPriorDataset│
    │  z: u + w full      │    │  z: w only, u=0)    │
    │  loss: CFM +        │    │  loss: CFM +        │
    │   inpaint trick     │    │   u-channel mask    │
    │  25000 steps        │    │  6250 steps         │
    └──────────┬──────────┘    └──────────┬──────────┘
               │                          │
               ▼                          ▼
           net_joint                  net_prior
           Unet2D dim=64              Unet2D dim=64
           36M params                 36M params
               │                          │
               └────────────┬─────────────┘
                            ▼
            ┌────────────────────────────────────┐
            │ Inference (BurgersEulerSampler)    │
            │                                    │
            │ x₀ ~ N(0, I)                       │
            │ for τ ∈ linspace(0, 1, 100):       │
            │   x ← inpaint_overwrite(x, c)      │
            │   if γ == 1:                       │
            │     v = net_joint(x, τ, c)         │
            │   else:                            │
            │     v = ReweightedVectorField(...) │
            │   x ← x + v · dτ                   │
            │ x ← inpaint_overwrite(x, c)        │
            └────────────┬───────────────────────┘
                         ▼
                  x_pred: (b, 2, 16, 128)
                         │
                         ▼
            ┌────────────────────────────────────┐
            │ Evaluation (compute_J_and_energy)  │
            │                                    │
            │ w = x_pred[:, 1, :10, :] * 10      │
            │ FOPC mask: w[middle 50%] = 0       │
            │ u_sim = PDE_solve(u_0, w)          │
            │ J = MSE(u_sim[T], u_T*)            │
            │ E = sum(w²)                        │
            └────────────────────────────────────┘
```

---

## §8 文件位置一览

### 主代码
- `flow/lab_four.py` / `flow/lab_four.ipynb` — 主 lab 文件

### 复用(不动)
- `model/burgers_1d/unet.py::Unet2D` — 网络架构
- `dataset/data_1d.py::Burgers1D` — HDF5 dataset 类
- `dataset/apps/generate_burgers.py::burgers_numeric_solve_free` — PDE solver(Numpy)
- `utils.py::burgers_metric` — 评估指标(已内置 FOPC mask)
- `diffusion/diffusion_1d_burgers.py::sigmoid_schedule_flip` — γ schedule
- `data/free_u_f_1e4_front_rear_quarter/` — 数据(8k+2k)
- `trained_models/burgers/FOPC_10k/cos10000-model-10.pt` — DDPM joint baseline(对比用)
- `trained_models/burgers_w/FOPC_w_10k/cos10000-model-10.pt` — DDPM prior baseline

### 输出
- `flow/checkpoints/fm_joint_*.pt` — 训练好的 FM joint(每个 checkpoint 一份 ~140 MB)
- `flow/checkpoints/fm_prior_*.pt` — 训练好的 FM prior
- `flow/checkpoints/fm_*_losses.png` — loss 曲线
- `flow/lab_four_gamma_sweep.png` — 7 个 γ 的 J/E 双线图
- `flow/lab_four_inf_gamma*.png` — 每个 γ 的 5-panel 详细图
- `flow/lab_four_part3_sample.png` — sanity_check_3_3 的 5-panel 图
- `flow/lab_four_part1_data.png` — Part 1 数据可视化
- `flow/lab_four_part1_noisy.png` — Part 1 noisy 样本
- `flow/lab_four_eta_schedule.png` — η̃(τ) schedule 曲线

---

## §9 跟 DDPM baseline 的关键对照

| 维度 | DDPM baseline | FM (这个 lab) |
|:---|:---|:---|
| Joint 训练命令 | `run_train_FOPC_10k.sh`,25000 步 | `train_joint_for_part5(num_steps=25000)` |
| Prior 训练命令 | `run_train_FOPC_w_10k.sh`,6250 步 | `train_prior_for_part5(num_steps=6250)` |
| Net architecture | `Unet2D(dim=64, dim_mults=(1,2,4,8))` | **同**(直接复用) |
| Dataset | `free_u_f_1e4_front_rear_quarter`(8000 train)| **同** |
| Batch size | 64 | **同** |
| Learning rate | 1e-4(`train_lr`)| **同** |
| Optimizer | Adam | **同** |
| **Probability path** | DDPM sigmoid β schedule(1000 步)| **FM Gaussian CondOT**(α=τ, β=1-τ)|
| **Training loss** | $\|\varepsilon_\theta - \varepsilon\|^2$(噪声预测)| $\|u_\theta - u^{\text{target}}\|^2$(velocity 预测)|
| **Sampling** | DDIM/DDPM 1000 步反向 | Euler ODE 100 步前向 |
| **EMA** | ✅ 用了(`ema_decay=0.995`)| ❌ 没用(可能略噪)|
| **γ-reweighting** | 同公式不同 parameterization | 同公式,见 `notes_fm_prior_reweighting.md` |

数学等价性:`notes_diffphycon_flow_bridge.md §2-3` 解释 DDPM noise 预测 ↔ FM velocity 预测的转换。

---

## §10 典型使用 recipe(一键运行)

```python
# Stage 0: 加载已训模型(或先训)
import torch
from flow.lab_four import *

device = "mps"

def load_ckpt(path):
    net = BurgersVectorField(dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt['state_dict']); net.eval()
    return net

net_joint = load_ckpt("flow/checkpoints/fm_joint_25k.pt")
net_prior = load_ckpt("flow/checkpoints/fm_prior_6250.pt")

# Stage 1: γ sweep(对比 baseline)
part5_gamma_sweep(net_joint, net_prior, n_samples=8, n_steps=100, split="train")

# Stage 2: 关键过拟合检查 — test 集
part5_gamma_sweep(net_joint, net_prior, n_samples=8, n_steps=100, split="test")

# Stage 3: 详细可视化(单个 γ)
inference_and_plot(net_joint, net_prior, gamma=2.5, n_samples=3,
                   save_path="flow/lab_four_inf_gamma2.5.png")

# Stage 4(可选): 加载早期 checkpoint 对比
net_joint_15k = load_ckpt("flow/checkpoints/fm_joint_step15000.pt")
inference_and_plot(net_joint_15k, net_prior, gamma=2.5,
                   save_path="flow/lab_four_inf_step15k.png")

# Stage 5(可选): loss 曲线
plot_loss_history("flow/checkpoints/fm_joint_25k.pt",
                  save_path="flow/lab_four_joint_loss.png")
```
