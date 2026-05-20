# DiffPhyCon 学习笔记 — 1D Burgers 实验

> 用 **MIT《Flow Matching and Diffusion Models》** 课程笔记的视角,理解 **DiffPhyCon (NeurIPS 2024)** 论文和代码。
>
> Repo: <https://github.com/AI4Science-WestlakeU/diffphycon>
> Paper: <https://openreview.net/forum?id=MbZuh8L0Xg>
> 项目本地路径:`/Users/baochen/diffphycon/`

---

## 0. 我学过的参考框架(MIT 课程)

为了方便后续翻译,这里列出 MIT 课程的核心约定:

- **概率路径** $p_t(x \mid z)$,$t \in [0, 1]$,**$t = 0$ 是噪声,$t = 1$ 是数据**
- **Gaussian path**:$p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I)$
  - $\alpha_0 = 0, \alpha_1 = 1$ (数据缩放)
  - $\beta_0 = 1, \beta_1 = 0$ (噪声缩放)
- **Flow matching loss / Score matching loss**:简化为 denoising 形式
- **Noise predictor (DDPM 形式)**:
$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\|\epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon\|^2\right]$$
- **Score ↔ noise 关系**:$s_t^\theta(x) = -\epsilon_t^\theta(x) / \beta_t$
- **Classifier guidance**:$\tilde u_t(x \mid y) = u_t^{\text{target}}(x) + w \cdot a_t \nabla \log p_t(y \mid x)$
- **CFG**:$\tilde u_t(x \mid y) = (1 - w)\, u_t^{\text{target}}(x \mid \emptyset) + w \cdot u_t^{\text{target}}(x \mid y)$

---

## 1. 控制问题设定(论文 §2.1, Eq 1)

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{J}(\mathbf{u}, \mathbf{w}) \quad \text{s.t.} \quad \mathcal{C}(\mathbf{u}, \mathbf{w}) = 0$$

**符号含义**:

| 符号 | 含义 | 1D Burgers 任务中 |
|---|---|---|
| $\mathbf{u}(t, \mathbf{x})$ | 系统状态轨迹(state) | 1D 流体速度场,$t \in [0, T]$,$x \in [0, 1]$ |
| $\mathbf{w}(t, \mathbf{x})$ | 外部控制力(control) | 施加在管子上的力 |
| $\mathcal{J}$ | 控制目标(要最小化的代价) | 例:$\int \|\mathbf{u} - \mathbf{u}^*\|^2 + \int \|\mathbf{w}\|^2$ |
| $\mathcal{C}$ | 物理约束 | Burgers 方程 |
| $\mathbf{c}$ | 条件信息 | 初值 $\mathbf{u}_0$ 和目标终值 $\mathbf{u}_T$ |

**通俗讲**:给你一根 1D 管子,初始流体形状是 $\mathbf{u}_0$,我想让它在 $T$ 时刻变成 $\mathbf{u}_T$。问应该怎么对它施力 $\mathbf{w}(t, x)$?

---

## 2. DiffPhyCon 的核心思想(论文 §3.1, Eq 3)

把控制问题转成 **带能量正则的优化**

$$\mathbf{u}^*, \mathbf{w}^* = \arg\min_{\mathbf{u}, \mathbf{w}} \left[\underbrace{E_\theta(\mathbf{u}, \mathbf{w}, \mathbf{c})}_{\text{学到的能量,代表"物理上不合理"程度}} + \lambda \cdot \underbrace{\mathcal{J}(\mathbf{u}, \mathbf{w})}_{\text{控制目标}}\right]$$

- $E_\theta$:从扩散模型学到的能量函数。EBM 视角:$p_\theta(\mathbf{u}, \mathbf{w} \mid \mathbf{c}) \propto \exp(-E_\theta)$
- 第一项保证生成的 $(\mathbf{u}, \mathbf{w})$ 满足物理(Burgers 方程)
- 第二项推动它满足控制目标

### MIT 视角翻译

这就是 **guided sampling**。设 $y = $ "我希望 $\mathcal{J}$ 小"。等价于从
$$p(\mathbf{u}, \mathbf{w} \mid y, \mathbf{c}) \propto p(\mathbf{u}, \mathbf{w} \mid \mathbf{c}) \cdot \exp(-\lambda \mathcal{J}(\mathbf{u}, \mathbf{w}))$$
采样。

取 $\log$,$\nabla \log p(\mathbf{u}, \mathbf{w} \mid y, \mathbf{c}) = \nabla \log p(\mathbf{u}, \mathbf{w} \mid \mathbf{c}) - \lambda \nabla \mathcal{J}$,正好是 **classifier guidance**(MD §5.2),"classifier" 就是 $-\mathcal{J}$。

---

## 3. 时间约定的差异 ⚠️ 重要

论文(和大多数 DDPM 代码)用 **离散反向时间** $k$,跟 MIT 的连续正向时间 $t$ **方向相反**。

| | MIT 笔记 | 论文 / 代码 |
|---|---|---|
| 时间索引 | $t \in [0, 1]$ 连续 | $k \in \{K, K-1, \dots, 1, 0\}$ 离散,通常 $K = 1000$ |
| **纯噪声** | $t = 0$ | $k = K$,$\mathbf{z}_K \sim \mathcal{N}(0, I)$ |
| **干净数据** | $t = 1$ | $k = 0$,$\mathbf{z}_0$ |
| 前向(加噪) | $t: 1 \to 0$ | $k: 0 \to K$ |
| 反向(去噪 / 采样) | $t: 0 \to 1$ | $k: K \to 0$ |

近似映射:$t \approx (K - k) / K$。

每次看代码里的 `t`(实际是 $k$),记住:**`t` 大 = 噪声多,`t` 小 = 干净**。

---

## 4. 完整符号 ↔ 代码 对照表

| 概念 | MIT 笔记 | 论文 | 代码(变量 / 函数) |
|---|---|---|---|
| 干净数据 | $z \sim p_{\text{data}}$ | $\mathbf{z}_0 = [\mathbf{u}, \mathbf{w}]$ | `x_start` |
| 加噪样本 | $x_t = \alpha_t z + \beta_t \epsilon$ | $\mathbf{z}_k = \sqrt{\bar\alpha_k}\,\mathbf{z}_0 + \sqrt{1 - \bar\alpha_k}\,\boldsymbol{\epsilon}$ | `q_sample(x_start, t, noise)` |
| 数据缩放 | $\alpha_t$ | $\sqrt{\bar\alpha_k}$ | `sqrt_alphas_cumprod[t]` |
| 噪声缩放 | $\beta_t$ | $\sqrt{1 - \bar\alpha_k}$ | `sqrt_one_minus_alphas_cumprod[t]` |
| 网络(噪声预测) | $\epsilon_t^\theta(x)$ | $\boldsymbol{\epsilon}_\theta(\mathbf{z}_k, \mathbf{c}, k)$ | `self.model(x, t)` 返回 `pred_noise` |
| 等价 score 视角 | $s_t^\theta = -\epsilon_t^\theta / \beta_t$ | $\nabla \log p_k(\mathbf{z}_k) = -\boldsymbol{\epsilon}_\theta / \sqrt{1 - \bar\alpha_k}$ | (推导关系,代码用 epsilon 形式) |
| 干净估计(denoiser) | $D_t(x) = \mathbb{E}[z \mid x]$ | $\hat{\mathbf{z}}_0 = (\mathbf{z}_k - \sqrt{1 - \bar\alpha_k}\,\boldsymbol{\epsilon}_\theta) / \sqrt{\bar\alpha_k}$ | `predict_start_from_noise(x_t, t, noise)` 第 363 行 |
| 训练 loss | $\mathbb{E}\|\epsilon_t^\theta(x_t) - \epsilon\|^2$ | Eq 4: $\mathbb{E}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$ | `p_losses` `diffusion/diffusion_1d_burgers.py:721` |
| 单步去噪 | (SDE Euler-Maruyama) | $\mathbf{z}_{k-1}$ 更新公式 | `p_sample(x, t)` 第 464 行 |
| 完整采样循环 | SDE 模拟 from $t=0$ to $t=1$ | for $k = K \to 1$ | `p_sample_loop(shape)` 第 525 行 |
| Guidance gradient | $\nabla \log p(y \mid x)$ | $-\nabla_{\mathbf{z}} \mathcal{J}(\hat{\mathbf{z}}_k)$ | `nablaJ` 参数 in `p_sample_loop` |
| 条件 | $y$ (prompt) | $\mathbf{c}$ (例:$\mathbf{u}_0, \mathbf{u}_T$) | `u_init`, `u_final` 参数 |

---

## 5. 训练(论文 Eq 4,代码 `p_losses`)

### 论文公式

$$\mathcal{L} = \mathbb{E}_{k \sim U(1, K),\,(\mathbf{z}, \mathbf{c}) \sim p,\,\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)} \left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\sqrt{\bar\alpha_k}\mathbf{z} + \sqrt{1 - \bar\alpha_k}\boldsymbol{\epsilon},\ \mathbf{c},\ k\right)\right\|^2\right]$$

### 对照 MIT Example 23 (DDPM)

$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t, z, \epsilon}\left[\|\epsilon_t^\theta(\alpha_t z + \beta_t \epsilon) - \epsilon\|^2\right]$$

**完全一致**,只是把连续 $t$ 换成离散 $k$,把 $\alpha_t \to \sqrt{\bar\alpha_k}$、$\beta_t \to \sqrt{1 - \bar\alpha_k}$。

### 这个项目的特别之处

- **$\mathbf{z} = [\mathbf{u}, \mathbf{w}]$ 是状态轨迹和控制信号拼起来的张量,一起送进 UNet**
- 张量形状:`(batch, 2, T_padded, N_x)` = `(B, 2, 16, 128)`
  - 通道 0 = $\mathbf{u}$
  - 通道 1 = $\mathbf{w}$
  - 时间维(11)和(10)各 pad 到 16,方便 2D 卷积
- **UNet 把它当作一张 2D 图来处理**(2 通道,16×128"像素")

---

## 6. 推理 / 采样 + Guidance(论文 Eq 7,代码 `p_sample_loop`)

### 论文公式(单步)

$$\mathbf{z}_{k-1} = \mathbf{z}_k - \eta \left(\boldsymbol{\epsilon}_\theta(\mathbf{z}_k, \mathbf{c}, k) + \lambda \nabla_{\mathbf{z}} \mathcal{J}(\hat{\mathbf{z}}_k)\right) + \xi,\quad \xi \sim \mathcal{N}(0, \sigma_k^2 I)$$

其中 $\hat{\mathbf{z}}_k$ 是从 $\mathbf{z}_k$ 估出的干净数据(论文 Eq 6):
$$\hat{\mathbf{z}}_k = \frac{\mathbf{z}_k - \sqrt{1 - \bar\alpha_k}\,\boldsymbol{\epsilon}_\theta(\mathbf{z}_k, \mathbf{c}, k)}{\sqrt{\bar\alpha_k}}$$

为什么要用 $\hat{\mathbf{z}}_k$ 而不是 $\mathbf{z}_k$ 算 $\mathcal{J}$:$\mathbf{z}_k$ 含噪声,直接算 $\mathcal{J}(\mathbf{z}_k)$ 会让梯度被噪声主导。

### MIT 视角翻译

这是 **classifier guidance**:
- 第一项 $-\eta \boldsymbol{\epsilon}_\theta$:无条件去噪(对应 score $\nabla \log p(\mathbf{z}_k)$)
- 第二项 $-\eta \lambda \nabla \mathcal{J}$:对应 $+\eta \lambda \cdot \nabla \log p(y \mid \mathbf{z}_k)$,因为 $\log p(y \mid \mathbf{z}_k) \propto -\mathcal{J}(\hat{\mathbf{z}}_k)$
- 第三项 $\xi$:对应 SDE 里的布朗运动项

合并起来就是从 $p(\mathbf{z} \mid \mathbf{c}, y)$ 的 Langevin 采样。

---

## 7. ⭐ DiffPhyCon 的主要创新:Prior Reweighting(论文 §3.2)

这一节是 DiffPhyCon **跟标准 guided diffusion 的真正区别**,MIT MD 里没有直接对应章节。

### 动机

训练集里的 $\mathbf{w}$ 是**随便生成的**(用 7 个随机高斯叠加),**没有最优性**。如果直接 sample $p(\mathbf{u}, \mathbf{w} \mid \mathbf{c})$,生成的 $\mathbf{w}$ 会"像训练集",但**真正最优的 $\mathbf{w}$ 可能在训练分布的低密度区**。

### 方法

**训练两个独立的扩散模型**:
- $\boldsymbol{\epsilon}_\theta$:学联合 $p(\mathbf{u}, \mathbf{w} \mid \mathbf{c})$
- $\boldsymbol{\epsilon}_\phi$:学边缘先验 $p(\mathbf{w} \mid \mathbf{c})$

引入超参 $\gamma > 0$,定义**重加权分布**(论文 Eq 9):
$$p_\gamma(\mathbf{u}, \mathbf{w} \mid \mathbf{c}) \propto p(\mathbf{w} \mid \mathbf{c})^{\gamma - 1} \cdot p(\mathbf{u}, \mathbf{w} \mid \mathbf{c})$$

- 当 $\gamma = 1$:就是原始联合 $p(\mathbf{u}, \mathbf{w} \mid \mathbf{c})$
- 当 $\boxed{0 < \gamma < 1}$:$p(\mathbf{w} \mid \mathbf{c})^{\gamma - 1}$ **指数为负**,**对训练里常见的 $\mathbf{w}$ 取倒数惩罚**,把 prior **压平**,**鼓励探索低密度区**

### 采样更新(论文 Eq 13, 14)

$$\mathbf{z}_{k-1} = \mathbf{z}_k - \eta\left(\boldsymbol{\epsilon}_\theta(\mathbf{z}_k, \mathbf{c}, k) + \lambda \nabla_{\mathbf{z}} \mathcal{J}(\hat{\mathbf{z}}_k)\right) + \xi_1$$
$$\mathbf{w}_{k-1} \mathrel{-}{=} \eta(\gamma - 1)\,\boldsymbol{\epsilon}_\phi(\mathbf{w}_k, \mathbf{c}, k) + \xi_2$$

当 $\gamma < 1$,$(\gamma - 1) < 0$,**减去了 prior 的 noise prediction**,等于在 score 视角里**减去 prior 的 score**。

### 跟 CFG 的对比

| | MIT CFG | DiffPhyCon Prior Reweighting |
|---|---|---|
| 用了几个模型 | 1 个($y$ 和 $\emptyset$ 共享) | 2 个($\boldsymbol{\epsilon}_\theta$ 和 $\boldsymbol{\epsilon}_\phi$) |
| 公式 | $\tilde u = (1 - w)u(\emptyset) + w\,u(y)$ | $\boldsymbol{\epsilon}_{\text{eff}} = \boldsymbol{\epsilon}_\theta + (\gamma - 1)\boldsymbol{\epsilon}_\phi$ |
| 目的 | 加强 prompt 依从度 | 打破训练分布束缚,探索更优 $\mathbf{w}$ |

数学结构非常相似(都是"主分布减去 baseline 分布"),但语义和目标不一样。

### ⚠️ 关于实际能跑的版本

作者只在 Google Drive 公开了 $\boldsymbol{\epsilon}_\theta$(joint),**没有公开 $\boldsymbol{\epsilon}_\phi$(prior)**。理由(用户原话):"在 1D Burgers 上 DiffPhyCon-lite 和完整版效果一样好,所以可能没放"。

所以**目前只能跑 DiffPhyCon-lite**($\gamma = 1$,只用 joint + guidance,不做 prior reweighting)。

---

## 8. 文件 ↔ 概念 对照

| 概念 | 文件 / 行号 |
|---|---|
| 训练入口 + argparse | `train/train_1d_burgers.py` |
| **DDPM 类** | `diffusion/diffusion_1d_burgers.py`,class `GaussianDiffusion`(第 192 行起) |
| - `q_sample`(前向加噪) | 第 713 行 |
| - `p_losses`(训练 loss) | 第 721 行 |
| - `p_sample`(单步去噪) | 第 464 行 |
| - `p_sample_loop`(完整采样 + guidance) | 第 525 行 |
| - `predict_start_from_noise`($\hat{\mathbf{z}}_0$) | 第 363 行 |
| - `Trainer`(训练循环) | 第 844 行 |
| **UNet**(噪声预测器 $\boldsymbol{\epsilon}_\theta$) | `model/burgers_1d/unet.py`,class `Unet2D`(第 268 行) |
| 推理 + guidance 评估 | `inference/inference_1d_burgers.py` |
| 数据集类(读 .h5) | `dataset/data_1d.py`(`Burgers1D`)+ `dataset/apps/burgers_h5py.py` |
| **$\mathcal{J}$ 的具体定义**(MSE + control energy) | `utils.py:1203` `burgers_metric`、`utils.py:1289` `ddpm_guidance_loss` |
| Burgers 数值求解器(生成训练数据用) | `dataset/apps/generate_burgers.py`,`burgers_numeric_solve_free`(第 207 行) |

---

## 9. 1D Burgers 实验设定细节

### 数据集
- 共 $N = 10^5$ 条轨迹
- 每条:
  - $u_0(x)$:两个 Gaussian 之和(随机参数)
  - $\mathbf{w}(t, x)$:7 个时空 Gaussian 之和(随机)
  - 由 Burgers 数值求解器跑出 $\mathbf{u}(t, x)$
- 时间:11 个时间点(含 $t = 0$),即 $T = 1$,$\Delta t = 0.1$
- 空间:$N_x = 128$ 个空间点
- 物理参数:$\nu = 0.01$(viscosity)

### 三种场景(对应三个独立训练的模型)

| 缩写 | Observation $\mathbf{u}$ | Control $\mathbf{w}$ 可施力区域 |
|---|---|---|
| **FOPC** *(我们要跑的)* | Full(整段可见) | Partial(只在 $x \in [0, 1/4] \cup [3/4, 1]$ 可施力) |
| POFC | Partial(中间 1/2 不可见) | Full(全段可施力) |
| POPC | Partial | Partial |

控制目标:
$$\mathcal{J} = \|\mathbf{u}(T) - \mathbf{u}^*\|^2 + w_f \|\mathbf{w}\|^2$$
(默认 $w_f = 0$,只看终值匹配度)

---

## 10. 学习路径建议

按难度递增:

1. **`utils.py:burgers_metric` + `ddpm_guidance_loss`** — 看 $\mathcal{J}$ 怎么算,**最具体最直观**
2. **`diffusion_1d_burgers.py:p_losses`(line 721)** — 训练 loss 实现,对应论文 Eq 4
3. **`diffusion_1d_burgers.py:p_sample_loop`(line 525)+ `nablaJ`** — 采样 + guidance,对应论文 Eq 7,**最核心**
4. **`model/burgers_1d/unet.py:Unet2D`** — UNet 架构(可选,对应 MD §6.1.3)

---

## 11. 当前进度(2026-05-15)

- ✅ 下载了 1D Burgers FOPC checkpoint:`trained_models/burgers/FOPC/cos10000-model-170.pt`
- ✅ 建好了 conda 环境 `diffphycon`(Python 3.10 + PyTorch 2.4.1 + MPS)
- ⏳ 待做:把代码里的 `.cuda()` 改成 MPS 兼容版本
- ⏳ 待做:用 `generate_burgers.py` 生成一小批 test 数据(几十条)
- ⏳ 待做:跑 `inference_1d_burgers.py` 的 lite 模式,看效果
