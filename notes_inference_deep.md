# DiffPhyCon Inference 深度解析(主文档)

> 这份文件是 inference 学习的**总索引 + 深度内容**。
> 已经写好的两个独立笔记用链接引入,新内容按章节集中在本文件。

---

## 📖 目录(可点击跳转)

| # | 章节 | 状态 | 在哪 |
|---|---|---|---|
| 00 | [概念地图(论文 ↔ 代码 ↔ MIT 框架)](notes_diffphycon.md) | ✅ | 独立文件 |
| 01 | [p_sample_loop 反向采样主循环](notes_p_sample_loop.md) | ✅ | 独立文件 |
| 02 | [model_predictions —— 模型预测 + Prior Reweighting + Guidance](#02-model_predictions) | ✅ | 本文件 ↓ |
| 02.4-bis | [桥梁:$\hat{x}_0 \leftrightarrow E[z\|x_t]$(MIT vs DDPM)](#02-model_predictions) | ✅ | 本文件 ↓ |
| 03 | [J 控制目标函数详解](#03-j-控制目标函数) | ✅ | 本文件 ↓ |
| 04 | [p_sample 单步采样的数学公式 / 后验均值 $\mu_\theta$](#04-p_sample-单步采样的数学公式) | ✅ | 本文件 ↓ |
| 05 | [Metrics —— 怎么读懂 inference 的输出](#05-metrics) | ✅ | 本文件 ↓ |
| 06 | [evaluate 外层批处理 + 整体管道(概览)](#06-evaluate-与整体管道) | ✅ | 本文件 ↓ |
| 07 | [⚙️ 教程:怎么自己跑一次 inference](#07-怎么自己跑一次-inference) | ✅ | 本文件 ↓ |

---

## 🗺️ 整体调用栈回顾

每次跑 inference,调用栈大致是:

```
__main__
  └── evaluate(...)                        ← § 06
        └── diffuse_2dconv(...)            ← § 05/06
              ├── load_2dconv_model(...)
              └── ddpm.sample(...)
                    └── p_sample_loop(...) ← § 01 ✅ 已讲完
                          └── p_sample(x, t, ...)     ← § 04
                                └── p_mean_variance(...)
                                      └── model_predictions(...)  ⭐ § 02 ← 现在讲
                                            ├── U-Net 前向
                                            ├── Prior Reweighting
                                            └── Classifier Guidance (用 nabla_J)
                                                  └── J 函数  ← § 03
```

红色 ⭐ 标的就是这一章要剖析的核心函数。

---

<a id="02-model_predictions"></a>

# 02. `model_predictions` —— 模型预测 + Prior Reweighting + Guidance

> 位置:`diffusion/diffusion_1d_burgers.py:396-450`
>
> 这是整个 DiffPhyCon 论文**核心创新落地的地方**。短短 55 行里塞了 3 件大事:
> 1. 跑 U-Net 拿到对噪声 ε 的预测
> 2. ⭐ Prior Reweighting(论文 Eq 9,核心贡献)
> 3. ⭐ Classifier Guidance(把控制目标 $J$ 的梯度注入到 ε 里)

---

## 02.0 它在哪儿被调用?

回顾 § 01:`p_sample_loop` 每一步去噪都会:

```
p_sample_loop:
  for t = 999..0:
    set_condition(...)               ← 覆盖 u₀, u_T
    img = p_sample(img, t, ...)      ← 调这里
                │
                ▼
          p_mean_variance(...)       ← 算 x_{t-1} 的后验均值
                │
                ▼
          model_predictions(...)     ← 我们现在剖析这里 ⭐
```

**`model_predictions` 干一件事**:输入当前的噪声态 $x_t$ 和时间步 $t$,
输出两个东西:
- $\epsilon_\theta(x_t, t)$:模型预测的噪声(可能被 reweighting 和 guidance 修过)
- $\hat{x}_0$:模型对"干净版本"的估计

然后这俩被传给 `p_mean_variance` 算后验均值,再被 `p_sample` 用来采 $x_{t-1}$。

---

## 02.1 函数签名

```python
def model_predictions(self,
                      x,                     # 当前噪声态 x_t,形状 (B, 2, 16, 128)
                      t,                     # 当前时间步,形状 (B,)
                      x_self_cond=None,      # self-conditioning,不用管(None)
                      residual=None,         # 残差 conditioning,不用管(None)
                      clip_x_start=False,    # 是否把 x_0 估计裁剪到 [-1, 1]
                      rederive_pred_noise=False,  # 裁剪后是否重新算 ε
                      **kwargs               # 传 guidance 相关参数(nablaJ, J_scheduler 等)
                      ):
```

### 关键参数一句话

| 参数 | 含义 | 我们用的值 |
|---|---|---|
| `x` | 当前的噪声态 $x_t$(已经被 `set_condition` 覆盖过 $u_0, u_T$) | shape `(50, 2, 16, 128)` |
| `t` | 当前去噪步(0~999) | 在循环里递减 |
| `clip_x_start` | 是否把 $\hat{x}_0$ 裁到 $[-1, 1]$ | True(`diffuse_2dconv` 传 `clip_denoised=True`) |

---

## 02.2 第一步:跑 U-Net,拿到 ε 预测

代码分成 **3 个分支**,根据 flag 走不同路径:

```python
if self.eval_two_models:
    # 分支 A:同时用 model_uw 和 model_w(完整 DiffPhyCon)
    model_output = self.model_uw(x, t, x_self_cond, residual=residual)
    ...   # 见 § 02.3 (Prior Reweighting)
elif self.is_model_w:
    # 分支 B:只用 model_w(单独训 w 模型时)
    ...
else:
    # 分支 C:只用单个 model(DiffPhyCon-lite,我们跑的就是这个)
    model_output = self.model(x, t, x_self_cond, residual=residual)
```

### 三个分支什么时候用?

| 分支 | flag 设置 | 用途 | 我们跑的吗? |
|---|---|---|---|
| **A** `eval_two_models=True` | `--is_model_w False --eval_two_models True` | DiffPhyCon **完整版**(联合模型 + 先验模型) | ❌ 没下到 model_w checkpoint |
| **B** `is_model_w=True` | `--is_model_w True` | 只跑先验模型(消融实验) | ❌ |
| **C** 默认 | 两个 flag 都 False | DiffPhyCon-**lite**(只用联合模型) | ✅ **就是这个** |

我们跑的是 **分支 C**,所以这一段简单粗暴:
```python
model_output = self.model(x, t, x_self_cond, residual=residual)
```
就是把 $x_t$ 和时间 $t$ 喂给 U-Net,得到一个跟 $x_t$ **形状一样**的输出
$\epsilon_\theta(x_t, t)$,即预测的噪声。

---

## 02.3 ⭐ Prior Reweighting(论文核心创新)

这一段我们这次跑 lite 版用不到,但**必须懂**,因为这是论文的 Eq 9。

### 分支 A 的代码(详细解释每一行)

```python
# 1. 跑联合模型 p(u, w | c)
model_output = self.model_uw(x, t, x_self_cond, residual=residual)

# 2. 准备 model_w 的输入:把 u[1..9] 抹零
#    理由:model_w 只学过 p(w | u_0, u_T),没见过中间的 u
x_w = x.clone()
x_w[..., 0, 1: self.condition_idx, :] = 0

# 3. 跑先验模型 p(w | c)
model_w_output = self.model_w(x_w, t, x_self_cond, residual=residual)

# 4. 把 u 通道的输出抹零(model_w 不预测 u)
model_w_output[..., 0, :, :] = 0

# 5. 步长调度(随时间衰减的权重)
eta = kwargs['w_scheduler'](t[0].item()) if ... else 1

# 6. ⭐⭐⭐ Prior Reweighting 主公式
if self.normalize_beta:
    model_output = (model_output - (1 - self.prior_beta) * model_w_output) / self.prior_beta
else:
    model_output = model_output - (1 - self.prior_beta) * eta * model_w_output
```

### 第 6 步的数学含义

记 $\gamma = $ `prior_beta`,先看 `normalize_beta=False` 的情况:

$$
\tilde\epsilon = \epsilon_{u,w} - (1 - \gamma) \cdot \eta(t) \cdot \epsilon_w
$$

把它翻译到 score:由于 $\epsilon = -\sigma_t \cdot \nabla \log p$,等价于:

$$
\nabla \log \tilde p \;=\; \nabla \log p(u, w | c) \;-\; (1 - \gamma) \cdot \eta(t) \cdot \nabla \log p(w | c)
$$

积分(忽略常数):
$$
\tilde p(u, w \mid c) \;\propto\; \frac{p(u, w \mid c)}{p(w \mid c)^{1-\gamma}}
$$

### 直觉

| $\gamma$ | $(1-\gamma)$ | $p(w)^{1-\gamma}$ | 效果 |
|---|---|---|---|
| $\gamma = 1$ | 0 | $1$ | 不重加权,就是普通采样 $p(u, w)$ |
| $\gamma > 1$(常用) | 负数 | $p(w)^{\text{负}}$ | **贬低常见的 w,鼓励罕见但有用的 w** |
| $\gamma < 1$ | 正数 | $p(w)^{\text{正}}$ | 加强常见的 w(不太用) |

### 直觉再举例

> 设想你在玩"找不同":数据集里大多数样本里的 $w$ 都是平淡无奇的(因为大部分初末状态用平庸的力就能达成)。
> 但你的**控制目标** $u_T$ 是一个**罕见、刁钻的状态**,需要**非典型**的 $w$ 才能达到。
>
> Prior Reweighting 就是告诉模型:**"那些平庸的 $w$ 你别老往那儿采,给点反主流的方案出来。"**

### 跟 Classifier-Free Guidance 的类比

| | CFG(文图生成) | Prior Reweighting(DiffPhyCon) |
|---|---|---|
| 模型 1 | $p(x \| y)$,条件分布 | $p(u, w \| c)$,联合分布 |
| 模型 2 | $p(x)$,无条件分布 | $p(w \| c)$,边缘分布 |
| 组合公式 | $\tilde\epsilon = (1+w)\epsilon_c - w\epsilon_{\text{uncond}}$ | $\tilde\epsilon = \epsilon_{uw} - (1-\gamma)\epsilon_w$ |
| 目的 | 让条件更"强烈" | 让 $w$ 更"非平凡" |

**形式上像,但语义不同**:
- CFG 是"在 prompt 方向上推开无条件分布"
- PR 是"在 $u$ 方向上推开 $w$ 的边缘分布",让模型多考虑 $u$ 跟 $w$ 的耦合

---

## 02.4 第二步:从 ε 反推 $\hat{x}_0$

不管走哪个分支,最后都拿到一个 `model_output` = $\epsilon_\theta(x_t, t)$。
接下来按 `objective` 的设置(我们是 `'pred_noise'`)走:

```python
if self.objective == 'pred_noise':
    if 'pred_noise' in kwargs and kwargs['pred_noise'] is not None:
        pred_noise = kwargs['pred_noise']   # 外面已经算好了(不走这里)
    else:
        pred_noise = model_output           # 我们走这里

    x_start = self.predict_start_from_noise(x, t, pred_noise)
    x_start = maybe_clip(x_start)
```

### `predict_start_from_noise` 公式

代码(`diffusion_1d_burgers.py:363`):
```python
def predict_start_from_noise(self, x_t, t, noise):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )
```

数学(标准 DDPM 公式):
$$
\hat{x}_0 = \frac{1}{\sqrt{\bar\alpha_t}}\, x_t - \frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\, \epsilon_\theta
$$

其中 $\bar\alpha_t = \prod_{s=1}^t \alpha_s = \prod_{s=1}^t (1 - \beta_s)$。

### 为什么要算 $\hat{x}_0$?

**两个理由**:
1. **算 guidance 用**(下一步会算 $\nabla J(\hat{x}_0)$,在干净的 $x_0$ 上算梯度比在噪声态 $x_t$ 上算更有意义)
2. **算后验均值用**(`p_mean_variance` 里用 $\hat{x}_0$ 和 $x_t$ 算 $\mu_\theta$,$\mu_\theta$ 是下一步 $x_{t-1}$ 的均值 —— 详见 [§04](#04-p_sample-单步采样的数学公式))

---

### 🌉 02.4-bis. MIT 框架下,$\hat{x}_0$ 到底是什么?

> 这一节专门回答:**"`predict_start_from_noise` 跟我 MIT 课学的 flow matching 有什么关系?"**

#### 1. 你 MIT 笔记 Sec 3 里的核心一招

MIT 笔记里(Example 8, line 189-210)讲条件概率路径:
$$
z \sim p_{\text{data}},\quad \epsilon \sim \mathcal{N}(0, I) \;\Rightarrow\; x = \alpha_t z + \beta_t \epsilon \sim p_t
$$

其中 $\alpha_0 = \beta_1 = 0$,$\alpha_1 = \beta_0 = 1$($\tau=0$ 噪声,$\tau=1$ 数据 —— MIT 约定)。

条件向量场是:
$$
u_t^{\text{target}}(x \mid z) = \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t} \alpha_t\right) z + \frac{\dot\beta_t}{\beta_t} x \quad \text{(MIT 笔记 line 209)}
$$

> ⚠️ 你之前写的 $u^{\text{target}}(x|z) = \alpha_t z + \beta_t x$ —— **那个其实是 conditional flow $\psi_t^{\text{target}}(x|z) = \alpha_t z + \beta_t x$**(笔记 line 211),不是向量场。
> 不过你直觉对了:**它对 $z$ 是线性的**,这一点很关键。

边缘向量场:
$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \cdot \underbrace{\frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)}}_{=\;p(z|x_t=x)} dz
$$

由于 $u^{\text{target}}(x|z)$ **对 $z$ 线性**,可以把积分提出:
$$
u_t^{\text{target}}(x) = \left(\dot\alpha_t - \frac{\dot\beta_t}{\beta_t} \alpha_t\right) \underbrace{E[z \mid x_t = x]}_{\text{需要估计这个!}} + \frac{\dot\beta_t}{\beta_t} x
$$

**核心观察**:**只要你能估计 $E[z \mid x_t = x]$,你就能算出向量场,就能跑 ODE 采样。**

---

#### 2. DDPM 在干的事:就是估计 $E[z \mid x_t]$!

DDPM 的前向过程:
$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

对照 MIT:$\alpha_t \leftrightarrow \sqrt{\bar\alpha_t}$,$\beta_t \leftrightarrow \sqrt{1-\bar\alpha_t}$,$z \leftrightarrow x_0$。

DDPM 训练 $\epsilon_\theta$ 去预测 $\epsilon$,所以 $\epsilon_\theta(x_t, t) \approx E[\epsilon \mid x_t]$。
反推 $x_0$:
$$
\hat{x}_0 \;=\; \frac{x_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}} \;\approx\; \boxed{E[x_0 \mid x_t]}
$$

**这就是 `predict_start_from_noise` 实际在算的东西**:
> **$\hat{x}_0 \approx E[x_0 \mid x_t]$ —— 给定噪声态 $x_t$ 下,干净数据 $x_0$ 的后验均值估计。**

---

#### 3. 桥梁结论 ✅

| MIT 视角 | DDPM 视角 |
|---|---|
| 想要 $E[z \mid x_t = x]$ | $\hat{x}_0 = $ `predict_start_from_noise(x_t, t, ε_θ)` |
| 用这个估计算 $u_t^{\text{target}}(x)$,然后跑 ODE | 用这个估计算后验均值 $\mu_\theta$,然后采 $x_{t-1}$(下一节 §04 讲)|
| 训练目标:让 $u_\theta \approx u^{\text{target}}$(向量场匹配) | 训练目标:让 $\epsilon_\theta \approx \epsilon$(噪声匹配) |

**两者本质等价**:都是在估计"给定当前噪声态,干净数据最可能是什么"。
DDPM 是用 ε 间接表达,MIT 是用向量场直接表达,**信息量一样**。

> 所以你那段推导是对的,只是 MIT 笔记里 conditional vector field 系数有点复杂,
> 不是简单的 $\alpha_t z + \beta_t x$。但**线性 + 需要 $E[z|x]$** 这个核心结构你抓到了。

### `maybe_clip` 是什么?

```python
maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
```

如果 `clip_x_start=True`(我们传 `clip_denoised=True`),就把 $\hat{x}_0$ 裁到 $[-1, 1]$ 范围。
理由:训练时数据被归一化到 $[-1, 1]$,所以 $x_0$ 的"合法范围"就是这个。
裁剪能防止数值发散。

---

## 02.5 ⭐ 第三步:Classifier Guidance(注入控制目标)

```python
if self.guidance_u0:
    pred_noise = may_proj_guidance(pred_noise,
                                    nablaJ(x_start) * nablaJ_scheduler(t[0].item()))
    x_start = self.predict_start_from_noise(x, t, pred_noise)
    x_start = maybe_clip(x_start)
```

### 一行行拆解

**`nablaJ(x_start)`**
- $\nabla_x J(\hat{x}_0)$ —— 控制目标 $J$ 在干净估计上的梯度
- 形状跟 $x_{\text{start}}$ 一样,**告诉模型"沿哪个方向调整能让 $J$ 变小"**
- $J$ 的具体形式见 § 03

**`nablaJ_scheduler(t[0].item())`**
- 一个**时间相关的权重** $\lambda(t)$
- 我们用 `cosine`(`cosine_beta_J_schedule`):去噪后期权重大,前期权重小
- 直觉:**早期采样还是一团噪声,$J$ 算不准,不要瞎调;后期形状成型了,再用 $J$ 微调**

**`may_proj_guidance(pred_noise, ∇J · λ)`**
- 默认实现是相加:`lambda ep, nabla_J: ep + nabla_J`
- 即:$\tilde\epsilon = \epsilon_\theta + \lambda(t) \cdot \nabla J(\hat{x}_0)$

**最后两行**:用修过的 $\tilde\epsilon$ 重新算一遍 $\hat{x}_0$。

### 这个公式的来源(贝叶斯 + score)

你 MIT 笔记 Sec 5 推过:

$$
\nabla \log p(x \mid y) = \nabla \log p(x) + \nabla \log p(y \mid x)
$$

把 $y$ = "$J$ 很小"这个事件,假设 $p(y \mid x) \propto e^{-J(x)/T}$:
$$
\nabla \log p(y \mid x) = -\frac{1}{T} \nabla J(x)
$$

转换到 ε(用 $\epsilon = -\sigma_t \nabla \log p$):
$$
\tilde\epsilon \;=\; \epsilon_\theta - \sigma_t \cdot \nabla \log p(y \mid x) \;=\; \epsilon_\theta + \frac{\sigma_t}{T} \nabla J(x)
$$

代码里的 $\lambda(t)$ 把 $\sigma_t / T$ 都吸收进去了。

### 关键 trick:对 $\hat{x}_0$ 算梯度,不是对 $x_t$ 算

```python
nablaJ(x_start)       # 在干净估计上算梯度
# 不是:
nablaJ(x)             # 在噪声态上算梯度
```

**为什么?**
- $J$ 是定义在干净轨迹上的(比如 $\|u(T) - u_{\text{target}}\|^2$)
- 直接对噪声态 $x_t$ 算,$J$ 值没意义(它不是干净的 $u$)
- 在 $\hat{x}_0$ 上算 $\nabla J$,**梯度信号才反映真实控制目标**

这是论文里特意提到的设计选择,代码里 `guidance_u0=True`(`guidance` on $\hat{u}_0$ = on $\hat{x}_0$)。

---

## 02.6 返回值

```python
return ModelPrediction(pred_noise, x_start)
```

`ModelPrediction` 是个 `namedtuple`(`diffusion_1d_burgers.py` 顶部定义),
返回两个张量:
- `pred_noise` = $\tilde\epsilon$(可能经 PR + guidance 修过)
- `pred_x_start` = $\hat{x}_0$(也是修过的)

这两个会被外层的 `p_mean_variance` 用来算 $\mu_\theta(x_t, t)$,
然后 `p_sample` 加噪声采出 $x_{t-1}$。

---

## 02.7 总流程伪代码

```
model_predictions(x_t, t):
    # ① 跑 U-Net(可能多个)
    if 双模型:
        ε_uw  = model_uw(x_t, t)
        ε_w   = model_w(x_t_with_u_zeroed, t)
        # ⭐ Prior Reweighting
        ε     = ε_uw - (1 - γ) · η(t) · ε_w
    else (lite):
        ε     = model(x_t, t)

    # ② 反推干净估计
    x_0_hat = (x_t - sqrt(1-α_bar_t) · ε) / sqrt(α_bar_t)
    x_0_hat = clip(x_0_hat, -1, 1)

    # ③ ⭐ Guidance:把控制目标的梯度注入 ε
    if guidance_u0:
        ε       = ε + λ(t) · ∇J(x_0_hat)
        x_0_hat = (x_t - sqrt(1-α_bar_t) · ε) / sqrt(α_bar_t)
        x_0_hat = clip(x_0_hat, -1, 1)

    return (ε, x_0_hat)
```

---

## 02.8 MIT 框架对照

| MIT 视角 | 这里的实现 |
|---|---|
| 学到的 score $\nabla\log p_\tau$ | `model_output` 经过 $\epsilon \to \text{score}$ 转换 |
| 条件 score $\nabla\log p(x \mid y) = \nabla\log p(x) + \nabla\log p(y\mid x)$ | Guidance 那一行 |
| CFG: $\tilde u = (1+w)u_{\text{cond}} - w \, u_{\text{uncond}}$ | Prior Reweighting 公式(类似但语义不同) |
| Score 与 ε 关系:$\epsilon = -\sigma \cdot \text{score}$ | DDPM 直接预测 ε,等价于隐式预测 score |

---

## 02.9 ✅ 检查理解

1. **`model_predictions` 在哪里被调用?它的输入和输出分别是什么?**
   - 在 `p_mean_variance` 里被调用。输入是 $x_t$ 和 $t$,输出是 $(\tilde\epsilon, \hat{x}_0)$。

2. **Prior Reweighting 公式里 `prior_beta = 1` 时会发生什么?**
   - $(1 - \gamma) = 0$,reweighting 项为 0,退化成普通 $p(u, w)$ 采样,没有 PR 效果。

3. **为什么 guidance 要对 $\hat{x}_0$ 算梯度,不是 $x_t$?**
   - $J$ 定义在干净轨迹上,$x_t$ 是噪声态没意义;$\hat{x}_0$ 是模型对干净轨迹的估计,在它上算 $\nabla J$ 才反映真实控制目标。

4. **`nablaJ_scheduler` 为什么用 cosine(后期大、前期小)?**
   - 早期 $\hat{x}_0$ 还是噪声团,$J$ 估计不准;后期形状成型,可信度高,这时再用 $J$ 微调。

5. **我们跑的是哪个分支(A/B/C)?为什么?**
   - 分支 C(默认),因为只下载了 `model_uw` checkpoint,没有 `model_w`,跑的是 DiffPhyCon-lite。

如果这 5 个能口答,你就完全懂 `model_predictions` 了。

---

<a id="03-j-控制目标函数"></a>

# 03. J 控制目标函数详解

> **位置**:
> - `inference/inference_1d_burgers.py:129-165`(`get_loss_fn_2dconv` 构造 $J$)
> - `inference/inference_1d_burgers.py:167-168`(`get_nablaJ_2dconv` 用 autograd 包装成梯度函数)
> - `utils.py:1289-1328`(`ddpm_guidance_loss` 真正的公式实现)
> - `diffusion/diffusion_1d_burgers.py:34-49`(`get_nablaJ` 用 autograd 算梯度)

回顾:在 §02.5 我们看到 guidance 这一步:
```python
pred_noise += nablaJ(x_start) * nablaJ_scheduler(t)
```
这一节回答两个问题:
1. $J$ 到底是什么函数?(数学公式 + 代码实现)
2. $\nabla J$ 怎么算出来?(autograd 一行搞定)

---

## 03.0 ⭐ 你 MIT 视角下,$J$ 是什么角色?

MIT 笔记 Sec 5 讲 guidance,关键公式:
$$
\nabla \log p(x \mid y) \;=\; \nabla \log p(x) \;+\; \nabla \log p(y \mid x)
$$

把 $y$ 当作一个"事件":比如"轨迹满足目标 $u_T = u_{\text{target}}$"。
假设这个事件的似然有玻尔兹曼形式:
$$
p(y \mid x) \;\propto\; \exp(-J(x) / T)
$$

那么:
$$
\nabla \log p(y \mid x) \;=\; -\frac{1}{T} \nabla J(x)
$$

带回 guidance 公式:
$$
\nabla \log p(x \mid y) \;=\; \nabla \log p(x) \;-\; \frac{1}{T} \nabla J(x)
$$

**$J$ 的角色**:**用一个"能量函数"刻画 "条件 $y$ 被满足的程度"**。$J$ 越小,条件越符合;$\nabla J$ 指向"$J$ 增大最快"的方向,所以采样要往 $-\nabla J$ 方向走。

> 这就是为什么 §02.5 说 $J$ 是 $|c$ 的"软约束"部分:
> 它不像 $u_0, u_T$ 那样硬覆盖,而是通过梯度**软推动**采样朝目标方向。

---

## 03.1 代码里 $J$ 的真正公式(`ddpm_guidance_loss`)

`utils.py:1289-1328`:

```python
def ddpm_guidance_loss(u_target, u, f,
                       wu=0, wf=0, wreg=0, wpinn=0,
                       dist_reg=lambda x: 0,
                       pinn_loss_mode='mean',
                       partially_observed=None):
    # 输入形状:
    #   u_target: (B, 11, 128)   测试集真值轨迹
    #   u:        (B, 11, 128)   扩散模型当前估计的 u
    #   f:        (B, 10, 128)   扩散模型当前估计的 w

    # === 第 1 项:端点误差(初末状态拟合)===
    u0_gt = u_target[:, 0, :]      # (B, 128)
    uf_gt = u_target[:, -1, :]     # (B, 128)
    u0 = u[:, 0, :]
    uf = u[:, -1, :]
    loss_u = (u0 - u0_gt).square() + (uf - uf_gt).square()
    if partially_observed == 'front_rear_quarter':
        loss_u[:, nx//4: (nx*3)//4] = 0       # 只在可观测区域算
    loss_u = loss_u.mean()

    # === 第 2 项:能量正则(控制力越小越好) ===
    loss_f = f.square().sum((-1, -2)).mean()

    # === 第 3 项:PINN 残差(物理一致性) ===
    if wpinn != 0:
        loss_pinn = pinn_loss(u, f, ...)
    else:
        loss_pinn = 0

    # === 总 J ===
    return loss_u * wu + loss_f * wf + loss_pinn * wpinn + dist_reg(u) * wreg
```

### 用数学写出来

$$
\boxed{
J(u, w) \;=\; w_u \cdot J_u(u) \;+\; w_f \cdot J_f(w) \;+\; w_{\text{pinn}} \cdot J_{\text{pinn}}(u, w) \;+\; w_{\text{reg}} \cdot J_{\text{reg}}(u)
}
$$

| 项 | 公式 | 物理意义 |
|---|---|---|
| $J_u$ | $\|u(0) - u_{\text{init}}\|^2 + \|u(T) - u_{\text{target}}\|^2$ | 端点拟合误差 |
| $J_f$ | $\sum_{t,x} f(t,x)^2$ | 控制力总能量(越小越省力) |
| $J_{\text{pinn}}$ | Burgers 方程残差 | 物理一致性(本代码未实现) |
| $J_{\text{reg}}$ | $\sum_t \|u(t+1) - u(t)\|^2$ | 时间平滑(`mse_dist_reg`) |

权重 $w_u, w_f, w_{\text{pinn}}, w_{\text{reg}}$ **全部从命令行传**,可调。

---

## 03.2 命令行权重默认值(⚠️ 重要的"暗坑")

argparse 默认(`inference_1d_burgers.py:60-67`):
```
--wfs    default=[0]
--wus    default=[0]
--wreg   default=0
--wpinns default=[0]
```

我们的 FOPC-lite 脚本(`scripts/burgers_inference_full_obs_partial_ctr.sh:33-47`):
```bash
python inference/inference_1d_burgers.py \
    --exp_id FOPC \
    --dataset free_u_f_1e5_front_rear_quarter \
    --is_condition_u0 True \
    --is_condition_uT True \
    --J_scheduler cosine \
    ...
    # ⚠️ 没传 --wus / --wfs / --wpinns / --wreg
```

**所有权重都是默认 0** → **$J \equiv 0$** → **$\nabla J \equiv 0$** → **guidance 这一步什么都不做!**

### 🤯 意外结论

> **跑 lite 的默认配置时,guidance 其实是关闭的。**
> **唯一让 $u_0, u_T$ 被遵守的机制是 §01 讲的 `set_condition` 硬覆盖。**
> **模型完全依赖训练时学到的 $p(u, w \mid u_0, u_T)$ 条件分布,加上硬覆盖,就能生成合理的 $(u, w)$。**

### 想真正用 guidance 怎么办?

加 CLI 参数:
```bash
python inference/inference_1d_burgers.py \
    ...
    --wus 1.0      # 鼓励端点拟合
    --wfs 0.01     # 惩罚大力
    --wreg 0.001   # 鼓励 u 平滑
```

加这些之后 $J$ 才有值,梯度才有信号,采样才会被 $J$ 推动。

---

## 03.3 ⭐ FOPC-lite 默认行为的"双层条件"再梳理

回看你之前的 $|c$ 疑问 —— 在我们这次跑的 lite 默认配置下:

| 条件类别 | 论文里的 $\|c$ | 代码机制 | 是否启用? |
|---|---|---|---|
| **硬约束** $(u_0, u_T)$ | 必须等于真值 | `set_condition` 在 `p_sample_loop` 每步覆盖 | ✅ 启用 |
| **软约束** $J$(端点拟合 / 能量 / 物理) | 软目标越小越好 | `pred_noise += λ(t) · ∇J(x_start)` | ❌ **关闭**(所有权重=0)|

所以默认 lite 跑的就是:

> "**只有硬覆盖,没有梯度引导,纯靠训练好的条件扩散模型 $p(u, w \mid u_0, u_T)$ 自己生成。**"

论文里说 lite 已经"work reasonably well" —— 就是因为模型本身学得够好,
单凭硬覆盖 + 学到的分布就能采到不错的 $(u, w)$。

---

## 03.4 `get_nablaJ`:autograd 怎么把 $J$ 变成 $\nabla J$

`diffusion/diffusion_1d_burgers.py:34-49`:

```python
def get_nablaJ(loss_fn: callable):
    def nablaJ(x: torch.TensorType):
        x.requires_grad_(True)                                # ① 开启梯度追踪
        J = loss_fn(x)                                         # ② 算 J(x),形状 (B,)
        grad = torch.autograd.grad(
            J, x,                                              # ③ 对 x 求导
            grad_outputs=torch.ones_like(J),
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        return grad.detach()                                   # ④ 返回 ∇J,断开计算图
    return nablaJ
```

### 逐行拆解

**① `x.requires_grad_(True)`**
告诉 PyTorch:对 `x` 的所有后续运算都要建立**计算图**,以便后面求导。

**② `J = loss_fn(x)`**
跑一遍 $J$ 的前向计算。`J` 是张量,形状 `(B,)`(每个 batch 元素一个 loss 值)。

**③ `torch.autograd.grad(J, x, grad_outputs=ones)`**
反向传播,得到 $\dfrac{\partial \sum_b J_b}{\partial x}$,形状跟 `x` 完全一样。
`grad_outputs=torch.ones_like(J)` 等同于先把 `J` 求和再求导(这是处理 batch 维度的标准技巧)。

**④ `grad.detach()`**
把梯度从计算图里"剥离",返回纯数值张量。后续不会继续追踪它的梯度。

---

## 03.5 整个 guidance pipeline 串起来

```
inference_1d_burgers.py:
  evaluate(...)
    └── nablaJ = get_nablaJ_2dconv(target_i, wu, wf, ...)   ← 构造 ∇J 函数
                    │
                    └── = get_nablaJ(get_loss_fn_2dconv(...))
                                        │
                                        └── loss_fn = lambda x: ddpm_guidance_loss(...)
    └── diffuse_2dconv(..., nablaJ=nablaJ, ...)
          └── ddpm.sample(..., nablaJ=nablaJ, ...)
                └── p_sample_loop(shape, nablaJ=nablaJ, ...)
                      └── p_sample(...)                       ← 每一步去噪
                            └── model_predictions(...)
                                  └── pred_noise += λ(t) · nablaJ(x_start)
                                                              ↑
                                          (autograd 在这一步实际跑,B×T 次)
```

**注意**:`nablaJ` 函数本身只是个**闭包**,直到 `p_sample` 调它时才会跑前向 + 反向。
1000 步采样里它会被调 1000 次(每个 batch 元素同时算)。

---

## 03.6 一个细节:为什么 J 用的是 `u_target` 而不只是 `u_final`?

代码里 `u_target` 是完整的 `(B, 11, 128)` 真值轨迹,但 $J_u$ 只用到了它的第 0 和第 -1 行:
```python
u0_gt = u_target[:, 0, :]      # 用了
uf_gt = u_target[:, -1, :]     # 用了
# u_target[:, 1:10, :] 没用
```

为什么传整个轨迹?
- 历史原因:作者可能想保留扩展性(将来或许要约束中间时间步)
- 实际上**只用了首尾两行**,中间被忽略

> 这跟 §03.0 直觉一致:控制目标只关心**"开头从哪出发,结尾到哪去"**,中间路径让模型自己选(只要物理一致就行)。

---

## 03.7 MIT 框架对照

| MIT 视角 | DiffPhyCon 代码 |
|---|---|
| 软条件 $y$ | "$u_T = u_{\text{target}}$" 这个事件 |
| $\log p(y\|x) \propto -J(x)/T$ | $J = w_u \|u_T - u_{\text{target}}\|^2 + \ldots$ |
| $\nabla \log p(y\|x) = -\nabla J / T$ | `get_nablaJ(loss_fn)` 通过 autograd |
| Guidance:$\tilde{\text{score}} = \text{score} + s \cdot \nabla \log p(y\|x)$ | `pred_noise += λ(t) · ∇J(x_start)` |
| 温度 $T$ + guidance scale $s$ | 都被吸收进 `nablaJ_scheduler(t)` 即 $\lambda(t)$ |

---

## 03.8 ✅ 检查理解

1. **$J$ 的输入是什么形状?它返回什么形状?**
   - 输入:`x` 形状 `(B, 2, 16, 128)`(batch, 2 通道, 16 时间, 128 空间)。
   - 返回:标量 loss(实际是 `(B,)` 后被求和),用于 autograd 求 $\nabla J$。

2. **跑 lite 默认脚本时,guidance 实际起作用了吗?为什么?**
   - **没起作用**。`--wus, --wfs, --wreg, --wpinns` 都默认 0,$J \equiv 0$,$\nabla J \equiv 0$。
   - 条件全靠 `set_condition` 硬覆盖来实现。

3. **如果只想用能量正则($J_f$),命令行该怎么改?**
   - 加 `--wfs 0.01`(数值要小,别压过条件)。

4. **`get_nablaJ` 里 `x.requires_grad_(True)` 不写会怎样?**
   - `torch.autograd.grad(J, x)` 会报错,因为 `x` 没在计算图里,没法对它求导。

5. **`nablaJ_scheduler(t)` 在 §02.5 里干嘛?跟 $J$ 公式里的权重有什么关系?**
   - `nablaJ_scheduler(t)` 是**时间相关的全局权重** $\lambda(t)$,作用在最后的 $\nabla J$ 上。
   - `wu, wf` 等是 $J$ **内部**的项权重(决定哪一项更重要)。
   - 两层权重独立:$J$ 内部权重决定形状,$\lambda(t)$ 决定时间衰减。

---

<a id="04-p_sample-单步采样的数学公式"></a>

# 04. p_sample 单步采样的数学公式

> 位置:`diffusion_1d_burgers.py:387-470`

---

## 04.0 ⭐ 先用你 MIT 的语言推一遍

> 这一节就是回答:"我用 MIT 框架推出来了 $\hat{x}_0 = E[z|x_t]$,接下来代码到底怎么用它走到 $x_{t-1}$?"

### MIT 视角下,从当前 $X_\tau$ 走到下一步要做什么?

你笔记 Sec 2 里讲的反向 SDE 欧拉一步:
$$
X_{\tau + \Delta\tau} \;=\; X_\tau \;+\; u_\tau^\theta(X_\tau) \cdot \Delta\tau \;+\; \sigma_\tau \sqrt{\Delta\tau} \cdot \xi, \quad \xi \sim \mathcal{N}(0, I)
$$

也就是 3 步:
1. 算 **当前的向量场** $u_\tau^\theta(X_\tau)$
2. 沿着它走一小步 $\Delta\tau$
3. 加一点噪声(SDE 才有,ODE 不加)

### 第 1 步:怎么算 $u_\tau^\theta(X_\tau)$?

笔记 Sec 3 你推过:
$$
u_\tau^{\text{target}}(x) \;=\; \underbrace{\left(\dot\alpha_\tau - \frac{\dot\beta_\tau}{\beta_\tau} \alpha_\tau\right)}_{=:\, a_\tau} \cdot E[z \mid x_\tau = x] \;+\; \underbrace{\frac{\dot\beta_\tau}{\beta_\tau}}_{=:\, b_\tau} \cdot x
$$

记 $a_\tau, b_\tau$ 为两个**只依赖时间的系数**(noise schedule 决定):
$$
\boxed{u_\tau^\theta(x) \;=\; a_\tau \cdot \hat{z}(x) \;+\; b_\tau \cdot x}
$$

其中 $\hat{z}(x) := \text{模型估计的 } E[z|x_\tau = x]$ —— 就是 §02.4-bis 里你已经接受的那个 $\hat{x}_0$。

### 一步代入:把 1 + 2 合起来

把 $u_\tau^\theta$ 代回欧拉一步:
$$
\begin{aligned}
X_{\tau + \Delta\tau}
&= X_\tau + \bigl(a_\tau \hat{z}(X_\tau) + b_\tau X_\tau\bigr) \Delta\tau + \sigma_\tau \sqrt{\Delta\tau}\, \xi \\
&= \underbrace{(1 + b_\tau \Delta\tau)\, X_\tau \;+\; a_\tau \Delta\tau \cdot \hat{z}(X_\tau)}_{\text{均值,记作 } M(X_\tau, \hat{z})} \;+\; \sigma_\tau \sqrt{\Delta\tau}\, \xi
\end{aligned}
$$

**关键观察**:**下一步 $X_{\tau+\Delta\tau}$ 的均值是 $X_\tau$ 和 $\hat{z}(X_\tau)$ 的线性组合**:
$$
M(X_\tau, \hat{z}) \;=\; \underbrace{(1 + b_\tau \Delta\tau)}_{=:\, C_2(\tau)} \cdot X_\tau \;+\; \underbrace{a_\tau \Delta\tau}_{=:\, C_1(\tau)} \cdot \hat{z}(X_\tau)
$$

### 🎯 桥梁来了

这跟代码里的 `q_posterior` 完全是**同一个形式**:

| MIT 推导 | DDPM 代码 |
|---|---|
| $M(X_\tau, \hat{z}) = C_2(\tau) X_\tau + C_1(\tau) \hat{z}$ | `posterior_mean = coef2 * x_t + coef1 * x_start` |
| $\hat{z}$ = $E[z\|X_\tau]$,从 $\hat{x}_0$ 来 | `x_start` 是 `predict_start_from_noise(...)` 的输出 |
| $C_1, C_2$ 由 $\alpha_\tau, \beta_\tau$ 算 | `coef1, coef2` 由 $\bar\alpha_t, \beta_t$ 在 `__init__` 里预先算好 |
| 再加 $\sigma_\tau \sqrt{\Delta\tau} \xi$ | `p_sample` 里加 `(0.5 * log_variance).exp() * noise` |

**结论**:
> **代码里的 `p_mean_variance` + `q_posterior` + `p_sample` 一整套,
> 就是你 MIT 框架里"算 vector field → 欧拉一步 → 加噪声"那三步,
> 只是 DDPM 选择 把"算 vector field"和"走一步"合并成"直接算下一步的均值 $\mu_\theta$"。**

不要被 DDPM 文献里的 "**posterior** $q(x_{t-1} | x_t, x_0)$" 吓到 ——
那是 DDPM 推这个均值的另一条路径(贝叶斯反演加噪过程),
**结论跟你 MIT 推出来的一模一样**。两条路殊途同归。

### 完整对照(MIT 视角的整个 `p_sample`)

```
你想要的:  X_{τ+Δτ} = X_τ + u^θ(X_τ)·Δτ + σ·√Δτ·ξ
                              │
                              ▼  把 u^θ 用 ẑ 写出来
              X_τ + (a_τ ẑ + b_τ X_τ)·Δτ + σ·√Δτ·ξ
                              │
                              ▼  合并 X_τ 项
              (1 + b_τΔτ) X_τ + (a_τΔτ) ẑ  +  σ·√Δτ·ξ
              └──────────┬─────────────────┘  └────┬────┘
                       均值 μ                     噪声
                              │
                              ▼  这就是代码做的
代码做的:    posterior_mean   +  (0.5·log_var).exp() · noise
              └───────┬───────┘       └────────┬────────┘
                  q_posterior              p_sample 最后一行
```

> **如果你完全理解上面这个对照,§04 后面的内容你只是看代码确认实现就行了,
> 不需要再被"后验"这个词卡住**。

---

## 04.1 接下来:DDPM 文献是怎么"反推"出同一个公式的?

> 上面我们用 MIT 框架直接推出了答案。
> DDPM 文献从另一条路推出同一个东西,术语完全不同。下面是 DDPM 的语言,你**对照看**就行。

`p_sample_loop` 里要做的事:从 $x_t$ 推出 $x_{t-1}$。

**DDPM 文献的做法**:定义一个分布 $p_\theta(x_{t-1} \mid x_t)$,然后从这个分布里采。

也就是说:**"给定当前的 $x_t$,下一个去噪态 $x_{t-1}$ 应该长什么样?"**

如果你知道这个分布,采一下就得到了 $x_{t-1}$。
DDPM 的关键结论:**这个分布是个高斯**,均值和方差有解析公式,均值刚好等于 §04.0 推出来的 $M(X_\tau, \hat{z})$。

---

## 04.2 关键定理:这个分布是高斯,且均值方差有解析公式

DDPM 论文证明:
$$
p_\theta(x_{t-1} \mid x_t) \;\approx\; q(x_{t-1} \mid x_t, x_0) \;=\; \mathcal{N}\bigl(\tilde\mu_t(x_t, x_0),\, \tilde\beta_t I\bigr)
$$

其中 $q(x_{t-1} \mid x_t, x_0)$ 叫做**正向过程的后验**(给定 $x_0$ 时,$x_{t-1}$ 在 $x_t$ 下的分布)。

它是个高斯,均值和方差都有**解析公式**:

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t}\, x_0 + \frac{\sqrt{\alpha_t}\, (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t
$$

$$
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t
$$

### 直觉解读

- $\tilde\mu_t$ 是个**线性组合**:$x_0$ 的一部分 + $x_t$ 的一部分
- $t$ 越大(越接近纯噪声),$x_t$ 系数大,$x_0$ 系数小
- $t$ 越小(越接近干净),$x_0$ 系数大
- 等于在说:**"去噪一步就是往 $x_0$ 方向移动一小步,移动幅度由 $\beta$ 表决定"**

---

## 04.3 但 $x_0$ 我们不知道,怎么办?

**关键 trick(也是 DDPM 的核心)**:用模型的估计 $\hat{x}_0$ **代替**真实的 $x_0$:

$$
\mu_\theta(x_t, t) \;:=\; \tilde\mu_t(x_t,\, \hat{x}_0)
$$

把 $\hat{x}_0$(从 § 02 的 `predict_start_from_noise` 算出来)**插进** $\tilde\mu_t$ 的位置 0 里。
得到的 $\mu_\theta$ 就是 $p_\theta(x_{t-1} \mid x_t)$ 的均值,也叫 **"后验均值"** —— 后验 $q$ 的均值,只不过 $x_0$ 是估计的。

---

## 04.4 这就是 `q_posterior` 函数(`diffusion_1d_burgers.py:387`)

```python
def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
        extract(self.posterior_mean_coef1, t, x_t.shape) * x_start   # ← coef1 * x_0_hat
      + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t       # ← coef2 * x_t
    )
    ...
    return posterior_mean, ...
```

对照公式:
- `posterior_mean_coef1` = $\dfrac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1 - \bar\alpha_t}$ → $\hat{x}_0$ 的权重
- `posterior_mean_coef2` = $\dfrac{\sqrt{\alpha_t}\, (1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}$ → $x_t$ 的权重

这两个系数在 `GaussianDiffusion.__init__` 里(line 325-326)预先算好,
作为 PyTorch buffer 存着:
```python
register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
```

---

## 04.5 `p_mean_variance`(`diffusion_1d_burgers.py:452`)

```python
def p_mean_variance(self, x, t, ...):
    preds = self.model_predictions(x, t, ...)       # ← § 02 的输出
    x_start = preds.pred_x_start                     # ← 拿到 x_0_hat
    
    if kwargs['clip_denoised']:
        x_start.clamp_(-1., 1.)                       # 再裁一次防数值发散

    model_mean, posterior_variance, posterior_log_variance = \
        self.q_posterior(x_start=x_start, x_t=x, t=t)  # ← 调 q_posterior 算 μ_θ
    return model_mean, ..., x_start, preds.pred_noise
```

**这函数做 3 件事**:
1. 调 `model_predictions` 拿 $\hat{x}_0$ 和 $\tilde\epsilon$
2. 把 $\hat{x}_0$ 和 $x_t$ 喂给 `q_posterior` 算 $\mu_\theta$
3. 返回 $\mu_\theta$ 和方差

返回值名叫 `model_mean`,就是 $\mu_\theta$。

---

## 04.6 `p_sample`(`diffusion_1d_burgers.py:464`)

```python
def p_sample(self, x, t, ...):
    model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(...)
    noise = torch.randn_like(x) if t > 0 else 0.    # t=0 时不加噪声(到达终点)
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start, pred_noise
```

**最后一步从高斯分布采样**:
$$
x_{t-1} = \mu_\theta + \sigma_t \cdot \xi, \quad \xi \sim \mathcal{N}(0, I), \quad \sigma_t = \sqrt{\tilde\beta_t}
$$

当 $t = 0$ 时,我们已经在干净空间了,不需要再加噪声,所以 `noise = 0`,直接返回 $\mu_\theta$。

---

## 04.7 回答你之前的疑问

> "**算后验均值用(`p_mean_variance` 里用 $\hat{x}_0$ 和 $x_t$ 算 $\mu_\theta$)**" —— 看不懂

现在你应该懂了:

| 名词 | 含义 |
|---|---|
| **后验** | $q(x_{t-1} \mid x_t, x_0)$ —— "给定 $x_0$ 时,$x_{t-1}$ 在 $x_t$ 下的分布" |
| **后验均值** | 这个高斯分布的均值 $\tilde\mu_t(x_t, x_0)$ |
| **$\mu_\theta$** | 把真实 $x_0$ 换成模型估计的 $\hat{x}_0$ 后的 $\tilde\mu_t(x_t, \hat{x}_0)$ |
| **用 $\hat{x}_0$ 和 $x_t$ 算** | 公式里 $\tilde\mu_t = \text{coef1} \cdot \hat{x}_0 + \text{coef2} \cdot x_t$,**就是两者的加权和** |

整体逻辑:
$$
\underbrace{x_t}_{\text{当前}} \xrightarrow{\text{U-Net}} \underbrace{\hat{x}_0}_{\text{干净估计}} \xrightarrow{\text{加权组合}} \underbrace{\mu_\theta}_{\text{下一步均值}} \xrightarrow{+\sigma_t \xi} \underbrace{x_{t-1}}_{\text{下一步}}
$$

---

## 04.8 MIT 框架对照

**MIT 视角下的单步采样**(欧拉离散):
$$
X_{\tau + \Delta\tau} = X_\tau + u_\tau^\theta(X_\tau) \, \Delta\tau + \sigma_\tau \sqrt{\Delta\tau} \cdot \xi
$$

**DDPM 视角**:
$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot \xi
$$

对照:
- DDPM 的 $\mu_\theta - x_t$ ↔ MIT 的 $u_\tau^\theta \cdot \Delta\tau$(向量场积一小步)
- DDPM 的 $\sigma_t$ ↔ MIT 的 $\sigma_\tau \sqrt{\Delta\tau}$
- 时间方向相反($t: T \to 0$ 对应 $\tau: 0 \to 1$)

**结论**:DDPM 的 `p_sample` 就是 MIT 反向 SDE 在离散网格上的欧拉一步。

---

## 04.9 ✅ 检查理解

1. **`q_posterior` 输入两个东西、输出一个东西,分别是什么?**
   - 输入:$\hat{x}_0$($x_{\text{start}}$)和 $x_t$。输出:$\mu_\theta$(`posterior_mean`)。

2. **`p_mean_variance` 跟 `q_posterior` 的区别?**
   - `p_mean_variance` 是**外层**:先调 `model_predictions` 拿 $\hat{x}_0$,再调 `q_posterior` 算 $\mu_\theta$。
   - `q_posterior` 是**内层**:纯粹的数学公式,把两个张量加权求和。

3. **为什么 $t=0$ 时 `noise = 0`?**
   - $t=0$ 已经在干净数据空间,不需要再加噪声,直接返回均值就行。

4. **`posterior_mean_coef1/2` 在哪里被计算?为什么是 buffer?**
   - 在 `GaussianDiffusion.__init__` 里预先算好,作为 buffer 注册到模型里(line 325-326)。
     这样推理时不用每步重算,加速。

---

<a id="05-metrics"></a>

# 05. Metrics —— 怎么读懂 inference 的输出

> **位置**:
> - `inference/inference_1d_burgers.py:261-305`(`diffuse_2dconv` 算 4 个指标)
> - `utils.py:1188-1200`(`mse_deviation`)
> - `utils.py:1203-1280`(`burgers_metric` 算 $J_{\text{actual}}$ + 能量)

跑完 inference 你会看到 4 个数字打印出来。它们各自测什么、怎么读、看什么算"控制效果好",这一节讲清楚。

---

## 05.0 ⭐ 一句话先讲清:**4 个数字回答 3 个独立问题**

| 问题 | 用哪个指标? |
|---|---|
| **A.** 控制目标完成得怎样?(真实物理意义上) | $J_{\text{actual}}$ |
| **B.** 模型生成的 $(u, w)$ 内部自洽吗?(模型相信的 vs 真实物理) | `ddpm_mse`(也叫 `mse_deviation`) |
| **C.** 用了多大力?(成本) | `energy` |

还有一个 $J_{\text{diffused}}$ —— 是 A 的"模型自我感觉",**辅助判断模型是否过度乐观**。

---

## 05.1 数据流回顾(理解 4 个指标的前提)

`diffuse_2dconv` 跑完一批后,你手上有这些张量:

```python
x = ddpm.sample(...) * RESCALER          # 扩散模型生成,(B, 2, 16, 128)

u_pred = x[:, 0, :11, :]                  # 模型生成的 u(11 个时间步)
w_pred = x[:, 1, :10, :]                  # 模型生成的 w(10 个时间步)
u0_pred = u_pred[:, 0, :]                 # 模型生成的初始态

# 拿模型生成的 w_pred + u0_pred 喂给真实数值求解器
u_gt = burgers_numeric_solve_free(
    u0_pred, w_pred, visc=0.01, T=1.0, dt=1e-4, num_t=10
)                                          # 真实物理演化结果,(B, 11, 128)
```

**关键认知**:
- `u_pred` = **模型相信的**(它生成出来的整个轨迹)
- `u_gt` = **真实物理的**(把模型生成的 $w$ 真正代入 PDE 算出来)
- `u_target` = **你想要的**(从测试集取的目标)

3 个 $u$,两两对比就是各种指标。

---

## 05.2 指标 1:`ddpm_mse` (= `mse_deviation`)

**问题**:**模型生成的 $u$ 跟真实物理模拟的 $u$ 差多少?**

```python
ddpm_mse = mse_deviation(u_pred, u_gt)
        = ((u_pred - u_gt) ** 2).mean(dim=(-1, -2))   # (B,)
```

公式:
$$
\text{ddpm\_mse} \;=\; \frac{1}{N_t \cdot N_x}\sum_{t,x}\bigl(u_{\text{pred}}(t,x) - u_{\text{gt}}(t,x)\bigr)^2
$$

### 怎么解读

| 数值 | 含义 |
|---|---|
| **小** | 模型生成的 $(u, w)$ **物理一致** —— 它说"用这个 $w$ 会得到这个 $u$",真的就是这样 |
| **大** | 模型 **"幻觉"** —— 它给的 $u$ 和 $w$ 之间物理上对不上 |

**这是评估扩散模型本身好坏的关键指标**。
跟控制目标无关,只检查"模型懂不懂物理"。

---

## 05.3 指标 2:$J_{\text{actual}}$ (来自 `burgers_metric`)

**问题**:**用模型给的 $w$ 真实模拟,末态 $u(T)$ 跟你的目标差多少?**

```python
J_actual, energy = custom_metric(f_from_x(x))
# custom_metric 内部:
#   u_controlled = solver(u_target, w_pred)
#   J_actual = ((u_controlled[:, -1, :] - u_target[:, -1, :]) ** 2).mean(-1)
```

公式:
$$
J_{\text{actual}} \;=\; \frac{1}{N_x} \sum_x \bigl(u_{\text{controlled}}(T, x) - u_{\text{target}}(T, x)\bigr)^2
$$

其中 $u_{\text{controlled}} = $ 用真实 PDE 求解器跑出来的轨迹(输入: $u_0 = u_{\text{target}}[0]$, $w = w_{\text{pred}}$)。

### 怎么解读

| 数值 | 含义 |
|---|---|
| **小** | ✅ 控制成功 —— 模型找到的 $w$ 真的把状态推到了目标 |
| **大** | ❌ 控制失败 —— $w$ 不够好或者目标根本不可达 |

**这是评估"控制好不好"的金标准**,因为它**完全用物理求解器**验证,不依赖模型自己声称的轨迹。

> 论文里所有的"control performance"对比都用这个。

---

## 05.4 指标 3:$J_{\text{diffused}}$

**问题**:**模型生成的 $u_{\text{pred}}$ 末态跟目标差多少?**

```python
J_diffused, _ = custom_metric(f_from_x(x), diffused_u=u_pred, evaluate_u=True)
#   J_diffused = ((u_pred[:, -1, :] - u_target[:, -1, :]) ** 2).mean(-1)
```

公式:
$$
J_{\text{diffused}} \;=\; \frac{1}{N_x} \sum_x \bigl(u_{\text{pred}}(T, x) - u_{\text{target}}(T, x)\bigr)^2
$$

### 怎么解读

| $J_{\text{diffused}}$ vs $J_{\text{actual}}$ | 含义 |
|---|---|
| $J_{\text{diffused}} \approx J_{\text{actual}}$,都小 | ✅ 模型生成的 $(u, w)$ 物理一致 + 控制成功 |
| $J_{\text{diffused}}$ 小,$J_{\text{actual}}$ 大 | ⚠️ **模型过度乐观** —— 自己以为达到了,实际跑物理发现差很远(典型物理不一致) |
| 两者都大 | ❌ 模型连"自己以为"都没达到 |

> 在 FOPC 设定下,因为 `set_condition` 硬把 $u_{\text{pred}}[T] = u_{\text{target}}[T]$,所以 $J_{\text{diffused}}$ 必定 ≈ 0(只是机械相等)。**真正有信息量的是 $J_{\text{actual}}$ 和 `ddpm_mse`**。

---

## 05.5 指标 4:`energy`

**问题**:**用了多大力?**

```python
energy = f.square().sum((-1, -2))    # (B,)
```

公式:
$$
\text{energy} \;=\; \sum_{t,x} w(t, x)^2
$$

### 怎么解读

| 数值 | 含义 |
|---|---|
| **小** | ✅ 控制省力 |
| **大** | ⚠️ 用蛮力达成,实际工程不好用 |

**单独看 energy 没意义**,要跟 $J_{\text{actual}}$ 一起看:
- $J_{\text{actual}}$ 小、energy 小 → **理想**(精准 + 省力)
- $J_{\text{actual}}$ 小、energy 大 → 蛮力达成
- $J_{\text{actual}}$ 大、energy 小 → 没控住但也没瞎用力
- $J_{\text{actual}}$ 大、energy 大 → 最差

---

## 05.6 4 个指标关系图

```
                  ┌───────────────┐
                  │   u_target    │ ← 你想要的(从测试集)
                  └───────┬───────┘
                          │
              ┌───────────┴──────────────────┐
              │                              │
              ▼                              ▼
        u_pred (扩散生成)             u_gt (真实模拟)
              │   │                              ▲
              │   │                              │
              │   └──► J_diffused                │
              │       (vs u_target[T])           │
              │                                  │
              │       ddpm_mse                   │
              └──────► (vs u_gt) ◄───────────────┘
                          
                       w_pred ──► energy
                          │
                          ▼
                  burgers_solver
                          │
                          ▼
                       u_controlled
                          │
                          └──► J_actual
                              (vs u_target[T])
```

---

## 05.7 看到这样的输出意味着什么(示例)

```
J_actual: 0.05         ← 末态平均偏差 0.05
Energy:   12.3         ← 总能量 12.3
```

| 值的量级 | 解读 |
|---|---|
| $J_{\text{actual}} < 0.1$,$\text{energy} < 20$ | 控制基本成功 |
| $J_{\text{actual}} \sim 0.5$ | 还差不少,但方向对 |
| $J_{\text{actual}} > 1$ | 基本失败,模型生成的 $w$ 跟目标关系不大 |

> ⚠️ 修正 (2026-05-27,读了 PDF 原文):论文 **Table 1**(不是 Table 2,那是 jellyfish dataset outline)报告 1D Burgers 的 $J_{\text{actual}}$。DiffPhyCon **FO-PC = 0.00037**,**PO-PC = 0.00494**,**PO-FC = 0.01103**。0.03-0.08 那个量级是较弱的 baseline(SAC/BC 等),不是 DiffPhyCon。
> metric `J_actual = ∫_Ω|u(T)−u_d|²dx`(对空间积分)≈ 我们 `burgers_metric` 的 `.mean(-1)`(∫dx over [0,1] 128 点 = mean),都是物理单位 → **可直接比**。我们小规模 FM (0.0071) 比论文 0.00037 差 ~19×,纯规模差距。

---

## 05.8 MIT 框架对照

不太适用 —— metric 是工程评估,与训练 / 采样的数学无关。但有一个关键概念:

> **`ddpm_mse` 测的是"模型对物理动力学的隐式学习有多准"**。
> 你 MIT 笔记里讲到 score 模型 $\nabla \log p_t$ 学到了多准,在那个抽象层面对应这里 `ddpm_mse` 反映的"扩散模型对联合 $(u, w)$ 流形是否准确"。

---

## 05.9 ✅ 检查理解

1. **`u_pred` 和 `u_gt` 都是从模型来的吗?**
   - 不完全是。`u_pred` 是扩散模型直接生成的 $u$ 张量;`u_gt` 是把模型生成的 $w$ 喂给**真实数值求解器**算出的 $u$。

2. **如果 $J_{\text{diffused}} = 0.001$ 但 $J_{\text{actual}} = 0.5$,说明什么?**
   - 模型自欺欺人:它生成的 $u_{\text{pred}}$ 看起来达成了目标,但用同一组 $w$ 真实跑 PDE 远远偏离。**物理不一致**严重。

3. **为什么 FOPC 里 $J_{\text{diffused}}$ 几乎必定接近 0?**
   - 因为 `set_condition` 硬把 $u_{\text{pred}}[T]$ 覆盖成了 $u_{\text{target}}[T]$,两者**机械相等**,平方差自然 0。

4. **想综合判断 inference 跑得好不好,主要看哪两个指标?**
   - $J_{\text{actual}}$(控制好不好)和 `ddpm_mse`(物理一致不一致)。

---

<a id="06-evaluate-与整体管道"></a>

# 06. evaluate 外层批处理 + 整体管道(概览)

> **位置**:`inference/inference_1d_burgers.py:327-399`
>
> 这一节是**最后一块拼图**,串起所有前面讲过的东西。短小精悍,看一遍就行。

---

## 06.1 `evaluate` 函数结构(只有 3 层)

```python
def evaluate(model_i, args, wu=0, wf=0, wpinn=0, wf_eval=0, wu_eval=1, conv2d=True):
    n_test_samples = args.n_test_samples       # 默认 50
    batch_size = 50
    rep = n_test_samples // batch_size         # = 1 (默认)

    for i in range(rep):                       # 外层:跑 rep 次(分批,默认就 1 次)
        seed = i
        target_idx = list(range(i * batch_size, (i + 1) * batch_size))
        # ↑ 第 i 批用测试集的 [i*50, i*50+50) 索引作为目标

        _, _, J_actual, energy = diffuse_2dconv(            # ← 一次跑 50 个样本
            args,
            custom_metric=...,                  # ← 评估目标(burgers_metric)
            model_i=model_i,
            seed=seed,
            nablaJ=get_nablaJ_2dconv(...),      # ← 控制目标的梯度(§03)
            J_scheduler=get_scheduler(args.J_scheduler),
            ...
            u_init=get_target(target_idx, ...)[:, 0, :] / RESCALER,    # ← 测试集的 u_0
            u_final=get_target(target_idx, ...)[:, 10, :] / RESCALER,  # ← 测试集的 u_T
        )
        
        l_gts.append(J_actual)
        energies.append(energy)
        
        print('J_actual:', l_gts[0][0].mean())
        print('Energy:',   energies[0].mean())
```

### 关键观察

| 维度 | 值 / 说明 |
|---|---|
| `n_test_samples` | 默认 50。我们小数据集只有 40 条 test,需要改成 ≤ 40 |
| `batch_size` | 硬编码 50(`inference_1d_burgers.py:341`) |
| `rep` | `n_test_samples // batch_size`,所以默认 1 |
| `target_idx` | `[0, 1, ..., 49]`,从测试集前 50 条取 |
| `u_init / u_final` | 直接从测试集真值取(不是用户随便指定的目标)|

---

## 06.2 整体 inference 管道(自顶向下)

```
[命令行 args]
     │
     ▼
__main__ ───► evaluate(...)
                    │
                    ▼
              ┌─────────────────────────────────────────────┐
              │ for batch in test_set[:50]:                  │
              │   target_idx = batch indices                 │
              │                                              │
              │   ┌────── diffuse_2dconv(...) ──────┐       │
              │   │                                  │       │
              │   │  ① load_2dconv_model           │       │
              │   │     - 创建 GaussianDiffusion    │       │
              │   │     - 创建 Unet2D              │       │
              │   │     - load checkpoint         │       │
              │   │                                  │       │
              │   │  ② ddpm.sample(...)            │       │
              │   │     └─ p_sample_loop  (§01)    │       │
              │   │          for t = 999..0:        │       │
              │   │            set_condition  (§01) │       │
              │   │            p_sample       (§04) │       │
              │   │              p_mean_variance    │       │
              │   │                model_predictions│       │
              │   │                  U-Net    (§02) │       │
              │   │                  prior reweight │       │
              │   │                  guidance  (§02)│       │
              │   │                    nablaJ  (§03)│       │
              │   │                                  │       │
              │   │  ③ burgers_numeric_solve_free   │       │
              │   │     (用模型的 w 重跑 PDE)        │       │
              │   │                                  │       │
              │   │  ④ 算 4 个 metrics  (§05)       │       │
              │   └──────────────────────────────────┘       │
              │                                              │
              │   收集 J_actual, energy                      │
              └─────────────────────────────────────────────┘
                              │
                              ▼
                       打印平均值
```

---

## 06.3 跑一次需要多久?(我们的环境预估)

| 阶段 | 操作 | CPU 时间(估)| MPS 时间(估)|
|---|---|---|---|
| ① load checkpoint | 一次性 | < 1 秒 | < 1 秒 |
| ② `p_sample_loop` 1000 步 × 50 batch | U-Net 前向 50000 次 | **几分钟到几十分钟** | 1–5 分钟 |
| ③ PDE 重模拟 | 10000 步数值积分,batch=50 | 5–20 秒 | 5–20 秒 |
| ④ metrics | 张量运算 | < 1 秒 | < 1 秒 |
| **总计** | 一批 50 个样本 | **可能 10+ 分钟** | **2–6 分钟** |

> 我们的小测试集只有 40 个 test 样本,所以会先把 `--n_test_samples` 改成 40,`batch_size` 也改成 40。

---

## 06.4 你需要传的命令行参数(回顾 + 准备跑)

```bash
python inference/inference_1d_burgers.py \
    --exp_id FOPC \                                  # checkpoint 在 trained_models/burgers/FOPC/
    --dataset free_u_f_1e5_front_rear_quarter \      # h5 在 data/{这个名字}/
    --partial_control front_rear_quarter \           # 中间一半不能施力
    --partially_observed None \                      # 全观测
    --train_on_partially_observed None \
    --set_unobserved_to_zero_during_sampling False \
    --is_condition_u0 True \                         # 启用 set_condition u_0
    --is_condition_uT True \                         # 启用 set_condition u_T
    --J_scheduler cosine \                           # λ(t) 用 cosine
    --dim 64 --dim_muls 1 2 4 \                      # U-Net 架构(必须跟训练时一样)
    --checkpoint 170 \                               # 加载 model-170.pt
    --checkpoint_interval 1000 \
    --save_file burgers_results/full_obs_partial_ctr/result_lite.yaml \
    --n_test_samples 40                              # ⚠️ 我们的小测试集只有 40 条
```

跑之前还要改的:
- `--n_test_samples` 默认 50,我们改成 40
- `batch_size = 50` 是**硬编码**(line 341),要改成 40
- 各种 `.cuda()` / `torch.cuda.manual_seed` 改成 MPS / CPU 兼容(详见后续 Mac 改造章节)

---

## 06.6 ✅ 检查理解

1. **`evaluate` 跑一次会调用 `diffuse_2dconv` 多少次?**
   - `n_test_samples / batch_size` 次。默认就 1 次。

2. **`u_init` 和 `u_final` 是用户指定的吗?**
   - 不是。从**测试集真值**里取(测试集本身有完整的 $u$ 轨迹,取 t=0 和 t=10)。这是 evaluate 模式,不是真正的"用户自定义控制"。

3. **整个 inference 的"最贵"步骤是哪一步?**
   - `p_sample_loop` 1000 步,每步跑 U-Net 前向 + autograd 算 $\nabla J$(虽然在 lite 下 $\nabla J = 0$,但仍然要算)。

---

<a id="07-怎么自己跑一次-inference"></a>

# 07. ⚙️ 教程:怎么自己跑一次 inference

> 这一节教你**怎么从零构造命令并跑起来**,不是给你结果。

---

## 07.0 跑 inference 的 4 个阶段

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│ 1. 准备前置物料  │ →  │ 2. 构造命令  │ →  │ 3. 启动 + 观察│ →  │ 4. 读输出  │
└─────────────────┘    └──────────────┘    └──────────────┘    └────────────┘
   - checkpoint           - 4 类参数         - 进度条          - J_actual
   - 数据集               - 复制 / 改写       - 错误诊断        - Energy
   - 修过的代码           - 跟训练匹配
```

---

## 07.1 前置物料检查(必须先确认这 3 样齐全)

跑命令前,**手动**检查:

```bash
# ① checkpoint 在不在?
ls trained_models/burgers/FOPC/cos10000-model-170.pt

# ② 测试数据集在不在?
ls data/free_u_f_1e5_front_rear_quarter/burgers_test.h5

# ③ utils.py 和 inference_1d_burgers.py 的 cuda 改过没?
grep -n "DEVICE" utils.py | head -3
grep -n "DEVICE" inference/inference_1d_burgers.py | head -3
```

3 个都打出文件路径 / 行号 = 准备就绪。

---

## 07.2 命令从哪来?

**起点**:`scripts/burgers_inference_full_obs_partial_ctr.sh`(论文作者写的脚本)

这个文件里有 **两个** 命令:
- 第 1 个(line 4-30):完整版 DiffPhyCon(需要 model_w,我们没下载)
- 第 2 个(line 33-47):**DiffPhyCon-lite** ← 我们用这个

```bash
cat scripts/burgers_inference_full_obs_partial_ctr.sh | sed -n '33,47p'
```

这就是我们的命令模板。

---

## 07.3 命令的 4 类参数(必须看懂)

| 类别 | 例子 | 作用 |
|---|---|---|
| **A. 模型身份** | `--exp_id FOPC`, `--checkpoint 170` | 加载 `trained_models/burgers/FOPC/cos10000-model-170.pt` |
| **B. 数据 + 实验设定** | `--dataset ...`, `--partial_control front_rear_quarter`, `--partially_observed None` | 决定从哪取 u_target,什么实验 |
| **C. Conditioning + Sampling** | `--is_condition_u0 True`, `--J_scheduler cosine` | 决定硬条件 + guidance 行为 |
| **D. U-Net 架构** | `--dim 128`, `--dim_muls 1 2 4` | ⚠️ **必须跟训练时一模一样**,否则 load 失败 |

### ⚠️ D 类是最容易踩坑的地方

不同 checkpoint 用不同架构训出来。**判断方法**:
1. 先看 `scripts/burgers_train_full_obs_partial_ctr.sh` 怎么训的
2. 如果对不上,直接尝试加载报"size mismatch"错误 —— 看错误信息**反推**实际配置

我们这次就踩了这个坑(脚本写 `--dim 64 --dim_muls 1 2 4` 但实际是 `--dim 128 --dim_muls 1 2 4`)。

### 怎么从 size mismatch 反推架构?

错误信息会告诉你:
```
size mismatch for model.final_conv.weight:
    copying a param with shape torch.Size([2, 128, 1, 1]) from checkpoint,
    the shape in current model is torch.Size([2, 64, 1, 1]).
```

`final_conv` 的输入通道 = `dim`(模型 base dim)。
- checkpoint:`128` → 实际训练用了 `dim=128`
- current model:`64` → 你传的 `--dim 64`

→ 把 `--dim 64` 改成 `--dim 128`。

---

## 07.4 实际的命令(我们用过的,完整版)

```bash
python inference/inference_1d_burgers.py \
    --exp_id FOPC \                                  # A. checkpoint 目录
    --checkpoint 170 \                               # A. milestone 编号
    --checkpoint_interval 1000 \                     # A. (load 时也要,不用细究)
    --dataset free_u_f_1e5_front_rear_quarter \      # B. h5 文件夹
    --partial_control front_rear_quarter \           # B. 中间 1/2 不能施力
    --partially_observed None \                      # B. 全观测
    --train_on_partially_observed None \             # B.
    --set_unobserved_to_zero_during_sampling False \ # B.
    --is_condition_u0 True \                         # C. 硬约束 u_0
    --is_condition_uT True \                         # C. 硬约束 u_T
    --J_scheduler cosine \                           # C. λ(t) 用 cosine
    --dim 128 \                                      # D. ⚠️ 跟训练匹配
    --dim_muls 1 2 4 \                               # D. ⚠️ 跟训练匹配
    --save_file burgers_results/full_obs_partial_ctr/result_lite.yaml \
    --n_test_samples 40                              # ⚠️ 我们小测试集大小
```

⚠️ 注意:每行末尾的 `\` 是 shell 续行符。**最后一行不能有 `\`**。

---

## 07.5 启动 + 怎么读进度

激活环境后执行:
```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate diffphycon

python inference/inference_1d_burgers.py \
    ... (上面那一坨参数)
```

### 你会看到 3 个阶段(都有进度条)

```
1. [模型加载]               ← 几秒
   "Load dataset data/free_u_f_1e5_front_rear_quarter/burgers_test.h5" × 3
   "Load dataset data/free_u_f_1e5_front_rear_quarter/burgers_train.h5"

2. [反向采样 1000 步]       ← 主头戏,3-6 分钟
   "sampling loop time step:   0%|...| 0/1000 [00:00<?]"
   慢慢爬到 100%

3. [PDE 重模拟 10000 步]    ← 算 J_actual,6-10 秒
   "  0%|...| 0/10000 [00:00<?]"
   两个进度条(test_indices x 一个) 各跑一遍

4. [打印结果]
   J_actual: 0.0010...
   Energy: 1487...
```

---

## 07.6 常见错误 + 怎么修(我们这次都碰到了)

### 错误 1:`ModuleNotFoundError: No module named 'deepsnap'`
**原因**:utils.py 顶部 import 了非 Burgers 实验才用的库。
**修法**:把 import 改成 try/except 包裹(已做)。

### 错误 2:`FileNotFoundError: data/free_u_f_1e5/burgers_train.h5`
**原因**:`load_burgers_dataset` 里硬编码了路径,跟我们生成的不一样。
**修法**:改成接受参数,默认指向我们有数据的路径(已做)。

### 错误 3:`size mismatch for model.xxx`
**原因**:`--dim` / `--dim_muls` 跟训练时不一致。
**修法**:看错误信息,从 checkpoint shape 反推实际架构(详见 07.3)。

### 错误 4:`AssertionError: n_test_samples % batch_size == 0`
**原因**:代码硬编码 `batch_size = 50`,但 `n_test_samples = 40`(40 不能整除 50)。
**修法**:把 `batch_size = 50` 改成 `batch_size = args.n_test_samples`(已做)。

---

## 07.7 跑一次大概要多久?

| 阶段 | 时长(MPS, M 系列 Mac)|
|---|---|
| 模型加载 | < 5 秒 |
| 反向采样 1000 步 | 3–6 分钟(主要花费)|
| PDE 重模拟 | 6–10 秒 |
| Metric 计算 | < 1 秒 |
| **总计** | **~5 分钟** |

CPU 跑会慢 5-10 倍,大约 25-50 分钟。

---

## 07.8 你的练习题 🎓

跑通一次后,尝试以下 3 个变体:

### 练习 1:换 seed 看方差
默认 seed = 0。改 `evaluate` 函数里的 `seed = i` 为 `seed = i + 100`,重跑,看 J_actual 会不会差很多。

### 练习 2:打开 guidance
原命令加:
```bash
    --wus 1.0 \      # 启用端点拟合 loss
    --wfs 0.01       # 加点能量正则
```
看 J_actual 和 Energy 怎么变化。

### 练习 3:换实验设定(POFC)
找 `scripts/burgers_inference_partial_obs_full_ctr.sh`,把 `--partial_control` 和 `--partially_observed` 的设置改成 POFC 版本看效果。
⚠️ 这个需要对应的 checkpoint,如果没下载会报 FileNotFoundError。

---

## 07.9 ✅ 自检

如果你能口答以下问题,你"会跑 inference"了:

1. checkpoint 文件在哪个文件夹下,叫什么名字格式?
2. `--dim` 和 `--dim_muls` 不匹配会发生什么?怎么从错误信息修?
3. 跑完看到 `J_actual` 数字,怎么判断"控制好不好"?
4. 你的小数据集只有 40 个 test 样本,要怎么改 CLI 参数才能跑?
5. 进度条里"sampling loop time step"和"0/10000"是同一个吗?

