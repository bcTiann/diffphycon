# FM 下的 Prior Reweighting:完整推导 + 代码骨架

## §0 这份文档的目的

把 DiffPhyCon paper 的 **prior reweighting**(γ-reweighting)trick 在 **Flow Matching** 框架下完整推导一遍,产出一个可以直接写进 `flow/burgers_fm.py` 的公式。

**重要更新**:这份文档**取代**之前 `notes_diffphycon_flow_bridge.md §6.4` 那个简化公式。原来的版本

$$
u^{\text{rw}}_\tau = u^{(u,f),\theta}_\tau - (1-\gamma)\,\tilde\eta(\tau)\,u^{(f),\theta}_\tau
$$

**漏掉了一个修正项** `−b_t·[0,w]`(详见 §5),严格说有 bug。这里推出的新公式是正确的:

$$
\boxed{\;\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{joint}}(u,w|c) \;+\; (\gamma-1)\,\tilde\eta(\tau)\,\Big[\,u_\tau^{\text{prior}}(w|c) \;-\; b_\tau\,[\,0,\,w\,]\,\Big]\;}
$$

下面把这个公式怎么来的、每一步意思、怎么写代码,讲清楚。

---

## §1 背景:为什么要 prior reweighting

(熟悉的可跳过,详见 `notes_diffphycon_flow_bridge.md §6`。)

### §1.1 控制问题

Burgers 控制:给初始 $u_0$ 和目标 $u_T^*$,找控制力 $w(t,x)$ 让物理系统从 $u_0$ 走到 $u_T^*$,同时最小化代价 $J = \|u(T) - u_T^*\|^2 + \lambda\|w\|^2$。

DiffPhyCon 思路:学一个生成模型 $p(u, w | c)$(联合分布,$c = (u_0, u_T^*)$),采样时再用 guidance / inpainting / **prior reweighting** 推向低 $J$。

### §1.2 Prior reweighting 是什么(论文 Eq 9)

定义一个**重加权后的目标分布**:

$$
\tilde p_\gamma(u, w \mid c) \;\propto\; p(w \mid c)^{\gamma-1}\, p(u, w \mid c)
$$

直觉:

| $\gamma$ | 系数 $(\gamma - 1)$ | $p(w\|c)^{\gamma-1}$ 行为 | 含义 |
|:---:|:---:|:---|:---|
| $\gamma < 1$ | 负 | 把常见 $w$ 的概率拉低 | **flatten** prior(鼓励探索非典型 $w$)|
| $\gamma = 1$ | 0 | $p^0 = 1$,不变 | 普通 joint sampling |
| $\gamma > 1$ | 正 | 把常见 $w$ 的概率放大 | **sharpen** prior(逼回训练分布常见模式)|

物理动机:

- **FOPC**(完全可观测 + 部分控制):joint $p(u,w|c)$ 已经很集中,但常见 $w$ 不一定是最优。**γ > 1** sharpen 一下能把 $w$ 拉回训练分布常见(更"自然")的模式,J 微降、Energy 降。
- **POPC**(部分观测 + 部分控制):部分观测让 joint 过度保守(多解都被平均掉了)。**γ < 1** flatten 一下鼓励探索更激进的 $w$,跳出保守解。

`notes_baseline_summary.md §3` 已经在 DDPM 下实测验证了这个趋势。FM 这边要复刻同样的行为。

### §1.3 DDPM 已经实现的版本(代码 `diffusion_1d_burgers.py:409`)

```python
model_output = model_output - (1 - self.prior_beta) * eta * model_w_output
#              ε_joint       - (1 - γ) · η(t)     · ε_prior
```

- `prior_beta` = $\gamma$(scalar CLI arg)
- `eta = w_scheduler(t)` = 时间 schedule(默认 1,paper 用 `sigmoid_flip`)
- 真实系数 = $(1-\gamma)\cdot \eta(t)$ 不是单纯 $(1-\gamma)$

### §1.4 我们要做的事

**把上面 DDPM 的 ε-空间公式翻译成 FM 的 velocity-空间公式**,这样可以直接代进 ODE Euler 采样。

---

## §2 数学准备

### §2.1 符号约定

| 符号 | 含义 |
|:---|:---|
| $\tau \in [0, 1]$ | FM 时间。$\tau = 0$ = pure noise,$\tau = 1$ = clean data |
| $x = (u, w)$ | 联合状态(channel-wise 拼接);Burgers 里 shape $[2, 16, 128]$ |
| $\alpha_\tau, \beta_\tau$ | Gaussian path 的 noise schedule。$x_\tau = \alpha_\tau\, x_1 + \beta_\tau\, x_0$ |
| $c = (u_0, u_T^*)$ | conditioning |
| $u_\tau^{\text{target}}(x \mid c)$ | velocity field;无 tilde 的版本是 **joint** $p(u,w\|c)$ 的 velocity |
| $u_\tau^{\text{prior}}(w \mid c)$ | **prior** $p(w\|c)$ 的 velocity field(嵌入到 $(u,w)$ 联合空间,u-block = 0)|
| $\tilde u_\tau^{\text{rw}}(x\|c)$ | 我们要推的 **reweighted** velocity field |
| $\gamma$ | reweighting scalar(`prior_beta`)|
| $\tilde\eta(\tau)$ | FM 时间方向的 schedule(对应 DDPM 的 `w_scheduler`)|

### §2.2 选定 path:Gaussian CondOT

最简单的 path(`flow_matching_diffusion.md` Example 13):

$$
\alpha_\tau = \tau, \quad \beta_\tau = 1 - \tau
$$

- $\tau = 0$:$x_0 = 0 \cdot x_1 + 1 \cdot \varepsilon = \varepsilon \sim \mathcal{N}(0, I)$ ✓ pure noise
- $\tau = 1$:$x_1 = 1 \cdot x_1 + 0 \cdot \varepsilon = x_1 \sim p_{\text{data}}$ ✓ clean data

导数:$\dot\alpha_\tau = 1$,$\dot\beta_\tau = -1$。

### §2.3 关键公式:Score ↔ Velocity 转换(Proposition 1)

`flow_matching_diffusion.md` line 310-313 给出:

$$
u_\tau^{\text{target}}(x) \;=\; a_\tau\,\nabla \log p_\tau(x) \;+\; b_\tau\, x
$$

其中

$$
a_\tau \;=\; \beta_\tau^2 \,\frac{\dot\alpha_\tau}{\alpha_\tau} \;-\; \dot\beta_\tau\,\beta_\tau, \qquad b_\tau \;=\; \frac{\dot\alpha_\tau}{\alpha_\tau}
$$

代入 CondOT 数值:

$$
a_\tau \;=\; (1-\tau)^2 \cdot \frac{1}{\tau} \;-\; (-1) \cdot (1-\tau) \;=\; \frac{(1-\tau)^2 + \tau(1-\tau)}{\tau} \;=\; \frac{1-\tau}{\tau}
$$

$$
b_\tau \;=\; \frac{1}{\tau}
$$

**重点**:**反过来**(score 用 velocity 表达):

$$
\nabla \log p_\tau(x) \;=\; \frac{1}{a_\tau}\,\Big[\,u_\tau^{\text{target}}(x) \;-\; b_\tau\, x\,\Big]
$$

这就是「score 和 velocity 的线性关系」,后面 Step 4 要用。

### §2.4 关键观察:联合空间中,prior 的 score 只在 w-block 非零

`p(w | c)` 这个分布**完全不依赖** $u$。所以对联合状态 $(u, w)$ 求梯度:

$$
\nabla_{(u,w)} \log p(w \mid c) \;=\; \Big[\,\underbrace{\nabla_u \log p(w|c)}_{= 0,\, \text{因为 } p(w|c) \text{ 不含 } u}\,,\; \nabla_w \log p(w \mid c)\,\Big]
\;=\; \big[\,\mathbf{0}_u,\; \nabla_w \log p(w \mid c)\,\big]
$$

**记号 $[0, \cdot]$ 的意思就是「u-block 补零,只保留 w-block」**。这是后面公式里 $b_\tau \cdot [0, w]$ 的由来。

这一点 DiffPhyCon 代码里也是这么实现的(`diffusion_1d_burgers.py:402`):

```python
model_w_output[..., 0, :, :] = 0  # only output w, not trained on u
```

prior 模型训练时就强制 u-block 输出 0,跟数学上 $\nabla_u \log p(w|c) = 0$ 完全对应。

---

## §3 推导 Step by Step

目标:从 $\tilde p_\gamma$ 反推 $\tilde u_\tau^{\text{rw}}$。

### Step 1 — 套 Prop 1 到 reweighted 分布

直接对 $\tilde p_\gamma(u, w \mid c)$ 应用 §2.3 的公式:

$$
\tilde u_\tau^{\text{rw}}(u, w \mid c) \;=\; a_\tau\,\nabla \log \tilde p_\gamma(u, w \mid c) \;+\; b_\tau\,(u, w)
$$

这步只是「Prop 1 是普适的」,任何概率分布的 velocity 都能这样表达。✓

### Step 2 — 拆 score

$\tilde p_\gamma$ 的定义两边取 log:

$$
\log \tilde p_\gamma(u, w \mid c) \;=\; (\gamma - 1)\,\log p(w \mid c) \;+\; \log p(u, w \mid c) \;+\; C
$$

($C$ 是 normalization 常数,跟 $(u, w)$ 无关,求梯度时消掉。)

对 $(u, w)$ 求梯度:

$$
\nabla_{(u,w)} \log \tilde p_\gamma \;=\; \nabla_{(u,w)} \log p(u, w \mid c) \;+\; (\gamma - 1)\,\nabla_{(u,w)} \log p(w \mid c)
$$

代回 Step 1:

$$
\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; \underbrace{a_\tau \nabla \log p(u,w|c) + b_\tau(u,w)}_{= u_\tau^{\text{target}}(u,w|c) \text{ by Prop 1}} \;+\; a_\tau(\gamma-1)\,\nabla_{(u,w)} \log p(w \mid c)
$$

简化:

$$
\boxed{\;\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{target}}(u,w|c) \;+\; a_\tau(\gamma-1)\,\nabla_{(u,w)} \log p(w|c)\;}
$$

✓ 这就是你写的第二行。注意 **$a_\tau$ 还在里面**,下面 Step 4 会消掉。

### Step 3 — 把联合空间中的 score 拆开

应用 §2.4 的观察:

$$
\nabla_{(u,w)} \log p(w|c) \;=\; \big[\,0, \,\nabla_w \log p(w|c)\,\big]
$$

代回 Step 2:

$$
\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{target}}(u,w|c) \;+\; a_\tau(\gamma-1)\,\big[\,0,\,\nabla_w \log p(w|c)\,\big]
$$

### Step 4 — 用 prior velocity 反代 score

对 prior 分布 $p_\tau(w|c)$ 同样套 Prop 1(**关键假设:joint 和 prior 用同一条 path** $\alpha_\tau, \beta_\tau$):

$$
u_\tau^{\text{prior}}(w|c) \;=\; a_\tau\,\nabla_w \log p_\tau(w|c) \;+\; b_\tau\, w
$$

反解:

$$
\nabla_w \log p_\tau(w|c) \;=\; \frac{1}{a_\tau}\big[\,u_\tau^{\text{prior}}(w|c) \;-\; b_\tau\, w\,\big]
$$

嵌入到联合空间(u-block 补零):

$$
\big[\,0,\, \nabla_w \log p(w|c)\,\big] \;=\; \frac{1}{a_\tau}\,\Big[\,\big[\,0,\, u_\tau^{\text{prior}}(w|c)\,\big] \;-\; b_\tau\,[\,0,\, w\,]\,\Big]
$$

**约定**:把 $u_\tau^{\text{prior}}(w|c)$ 默认理解为「嵌在联合空间里、u-block 强制为 0 的那个向量」(就是 DiffPhyCon 训练 trick 产物)。这样上面写成:

$$
\big[\,0,\, \nabla_w \log p(w|c)\,\big] \;=\; \frac{1}{a_\tau}\,\Big[\,u_\tau^{\text{prior}}(w|c) \;-\; b_\tau\,[\,0,\, w\,]\,\Big]
$$

代回 Step 3:

$$
\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{target}}(u,w|c) \;+\; a_\tau(\gamma-1) \cdot \frac{1}{a_\tau}\,\Big[\,u_\tau^{\text{prior}}(w|c) - b_\tau\,[\,0,\, w\,]\,\Big]
$$

**$a_\tau$ 完全约掉**(这是这个推导漂亮的地方),剩下:

$$
\boxed{\;\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{target}}(u,w|c) \;+\; (\gamma - 1)\,\Big[\,u_\tau^{\text{prior}}(w|c) \;-\; b_\tau\,[\,0,\, w\,]\,\Big]\;}
$$

✓ 这就是最终公式,跟你推导的一致。

### Step 5(可选)— 加上经验性 schedule

数学推导给出的是 $\gamma$ 不带时间变化的「干净版」。但 DDPM 经验上发现:在噪声大的时段做强 reweight 会让 sampling 不稳定(`notes_baseline_summary.md §4.2` 实测:γ=0.3 不带 schedule J=0.0607 灾难,带了 J=0.0083)。

所以工程上把 $(\gamma - 1)$ 这个 scalar 乘上一个 schedule $\tilde\eta(\tau)$:

$$
\boxed{\;\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{target}}(u,w|c) \;+\; (\gamma - 1)\,\tilde\eta(\tau)\,\Big[\,u_\tau^{\text{prior}}(w|c) \;-\; b_\tau\,[\,0,\, w\,]\,\Big]\;}
$$

要求 $\tilde\eta(\tau)$:
- $\tau \to 0$(噪声端):$\tilde\eta \to 0$,关掉 reweight
- $\tau \to 1$(clean 端):$\tilde\eta \approx 1$,完全打开
- 单调递增

最简单的实现:直接复用 DDPM `sigmoid_schedule_flip` 函数,把 FM 时间 $\tau$ 换算成 DDPM 时间 $t$:

```python
def w_scheduler_fm(tau):
    """
    DDPM t=999 ↔ FM τ=0 (噪声端)
    DDPM t=0   ↔ FM τ=1 (clean 端)
    """
    t_ddpm = round((1 - tau) * 999)
    return sigmoid_schedule_flip(t_ddpm)
```

这样画出来的 $\tilde\eta(\tau)$ 曲线跟 DDPM 那条 $\eta(t)$ 是镜像翻过来的(x 轴翻转),数值含义一致。

---

## §4 跟 DDPM 互检:确认没翻车

DDPM 代码里:

$$
\varepsilon^{\text{rw}} \;=\; \varepsilon^{\text{joint}} \;-\; (1-\gamma)\,\eta(t)\,\varepsilon^{\text{prior}}
\;=\; \varepsilon^{\text{joint}} \;+\; (\gamma - 1)\,\eta(t)\,\varepsilon^{\text{prior}}
$$

用 $\varepsilon \leftrightarrow \nabla \log p$ 的关系(`flow_matching_diffusion.md` Example 23,line 372):

$$
\varepsilon_t^\theta(x) \;=\; -\beta_t\,\nabla \log p_t(x)
$$

即 $\nabla \log p = -\varepsilon / \beta_t$。代入:

$$
-\beta_t\,\nabla \log \tilde p_\gamma \;=\; -\beta_t\,\nabla \log p_{\text{joint}} \;+\; (\gamma - 1)\,\eta(t)\,(-\beta_t)\,\nabla \log p_{\text{prior}}
$$

两边除 $-\beta_t$:

$$
\nabla \log \tilde p_\gamma \;=\; \nabla \log p_{\text{joint}} \;+\; (\gamma - 1)\,\eta(t)\,\nabla \log p_{\text{prior}}
$$

✓ **跟我们 Step 3 推出来的 score-空间公式完全一致**(只是 DDPM 多了 $\eta(t)$ schedule)。

所以从 score 空间看,DDPM 和 FM 是同一个数学操作的两种 parametrization,只是工程实现层面 DDPM 直接在 $\varepsilon$ 上加权,FM 要先转成 velocity 才能加权(就是 §3 Step 4 那一步)。

---

## §5 跟 `notes_diffphycon_flow_bridge.md §6.4` 旧公式的差异

旧公式(我之前写的):

$$
u^{\text{rw}}_\tau \;=\; u^{(u,f),\theta}_\tau \;-\; (1-\gamma)\,\tilde\eta(\tau)\,u^{(f),\theta}_\tau
$$

新公式(本文 Step 5):

$$
\tilde u^{\text{rw}}_\tau \;=\; u^{(u,f),\theta}_\tau \;+\; (\gamma-1)\,\tilde\eta(\tau)\,\big[\,u^{(f),\theta}_\tau \;-\; b_\tau\,[\,0, w\,]\,\big]
$$

**差异**:新公式多了 $-\,b_\tau\,[\,0, w\,]$ 那项。

### 为什么我之前漏了?

我那时候是「DDPM ε-空间公式硬抄到 FM velocity 空间」,把 $\varepsilon$ 直接换成 $u$。但 $\varepsilon$ 跟 $u$ 不是简单替换:

| 量 | 跟 score 的关系 |
|:---|:---|
| DDPM $\varepsilon$ | $\varepsilon = -\beta_t \nabla \log p$(纯 score 比例) |
| FM $u^{\text{target}}$ | $u = a_t \nabla \log p + b_t x$(score 加 $b_t x$ 偏移)|

**FM velocity 多了 $b_t x$ 这个 offset**。所以在做 "joint - α·prior" 这种线性组合时,DDPM 的 $\varepsilon$ 因为是纯 score 的标量倍,可以直接相减;但 FM 的 $u$ 减一减,$b_t x$ 那个 offset 不会自动抵消,需要手动减掉。

具体看一下:

$$
u_{\text{joint}}(x) \;=\; a_\tau\,\text{score}_{\text{joint}}(x) \;+\; b_\tau\, x
$$

$$
u_{\text{prior}}(x) \;=\; a_\tau\,\text{score}_{\text{prior}}(x) \;+\; b_\tau\, [\,0, w\,] \quad \text{(注意 prior 的 } b_\tau x \text{ 项只对 w-block)}
$$

我们要的是 score 空间的线性组合:

$$
\text{score}_{\text{rw}} \;=\; \text{score}_{\text{joint}} \;+\; (\gamma - 1)\,\eta\,\text{score}_{\text{prior}}
$$

转 velocity:

$$
u_{\text{rw}} \;=\; a_\tau\,\text{score}_{\text{rw}} \;+\; b_\tau\, x
\;=\; \underbrace{a_\tau\,\text{score}_{\text{joint}} + b_\tau\, x}_{= u_{\text{joint}}} \;+\; (\gamma-1)\,\eta\,\underbrace{a_\tau\,\text{score}_{\text{prior}}}_{= u_{\text{prior}} - b_\tau[0,w]}
$$

→ **必须从 $u_{\text{prior}}$ 里减回 $b_\tau[0, w]$ 才能拿到纯 score 项**。我之前公式跳过了这一步,直接用 $u_{\text{prior}}$,所以错。

### 数值上差多少?

新公式 - 旧公式 = $-(\gamma-1)\,\eta(\tau)\,b_\tau\,[\,0, w\,]$

在 Burgers 上估计一下(取 $\gamma = 2.5$,$\tau = 0.5$,$b_\tau = 1/0.5 = 2$,$\eta \approx 0.5$,$w$ 值大概 $|w| \sim 1$):

差异 $\approx (2.5 - 1) \times 0.5 \times 2 \times 1 = 1.5$

**不小**。velocity 量级一般在 $O(1)$ 到 $O(10)$,差 1.5 是显著的。所以旧公式确实是 bug,新公式必须用。

---

## §6 实现骨架

### §6.1 公式回顾

$$
\tilde u_\tau^{\text{rw}}(u,w|c) \;=\; u_\tau^{\text{joint}}(u,w|c) \;+\; (\gamma - 1)\,\tilde\eta(\tau)\,\Big[\,u_\tau^{\text{prior}}(w|c) \;-\; b_\tau\,[\,0,\, w\,]\,\Big]
$$

### §6.2 Python 代码

```python
# flow/burgers_fm.py

class BurgersFM:
    """Flow Matching with prior reweighting for Burgers control."""

    def __init__(self, model_uw, model_w, gamma=1.0, w_scheduler=None):
        self.model_uw = model_uw        # joint velocity network: p(u,w|c)
        self.model_w  = model_w         # prior velocity network: p(w|c) — u-block 训练时强制 0
        self.gamma    = gamma           # prior_beta
        self.w_scheduler = w_scheduler  # callable: τ → η̃(τ); None = constant 1

    def predict_velocity(self, x, tau, c):
        """
        Args:
          x:    [B, 2, T, X]    联合状态; channel-0 = u, channel-1 = w
          tau:  [B]             FM 时间 ∈ (0, 1)
          c:    tuple           (u_0, u_T_star) conditioning
                                (通过 inpainting overwrite 注入到 x,不直接进 model)

        Returns:
          v:    [B, 2, T, X]    reweighted velocity field
        """
        # 1. Joint velocity
        v_joint = self.model_uw(x, tau, c)         # [B, 2, T, X]

        # 2. Prior velocity (u-channel 输入 zero out)
        x_for_prior = x.clone()
        x_for_prior[:, 0] = 0                       # zero u channel before forward
        v_prior = self.model_w(x_for_prior, tau, c) # [B, 2, T, X]
        v_prior[:, 0] = 0                           # safety: 强制 u-block = 0(model 训练 trick 应该已经如此)

        # 3. 构造 [0, w] —— u-channel 置零的版本
        x_w_only = x.clone()
        x_w_only[:, 0] = 0                          # [B, 2, T, X],只剩 w channel

        # 4. b_τ for CondOT path: α=τ, β=1-τ → b_τ = 1/τ
        tau_safe = tau.clamp(min=1e-4)              # 避免 τ→0 奇点
        b_tau = (1.0 / tau_safe).view(-1, 1, 1, 1)  # broadcastable

        # 5. 时间 schedule(默认 1 = pure math version)
        if self.w_scheduler is not None:
            eta = self.w_scheduler(tau).view(-1, 1, 1, 1)
        else:
            eta = 1.0

        # 6. 应用 §3 推出来的公式
        correction = v_prior - b_tau * x_w_only
        v_reweighted = v_joint + (self.gamma - 1) * eta * correction

        return v_reweighted


# flow/simulator.py

def euler_sample_with_inpainting(burgers_fm, x_init, c, n_steps=100):
    """
    Euler ODE sampling τ: 0 → 1, with row-0/row-T inpainting overwrite.

    Args:
      x_init:    [B, 2, T, X]   起点 ~ N(0, I)
      c:         (u_0, u_T_star) — 每个是 [B, X] tensor

    Returns:
      x_final:   [B, 2, T, X]   最终样本,从中取 channel-1 = 预测的 w(t,x)
    """
    u_0, u_T_star = c
    T_idx = 10                                       # row 10 = terminal time(11 行总)
    x = x_init.clone()
    dtau = 1.0 / n_steps

    for step in range(n_steps):
        tau = torch.full((x.shape[0],), step * dtau, device=x.device)

        # Inpainting overwrite: 强制 row 0 = u_0, row T = u_T_star (clean overwrite, 不加噪)
        x[:, 0, 0, :] = u_0                          # row 0 of u channel = u_0
        x[:, 0, T_idx, :] = u_T_star                 # row T of u channel = u_T_star

        # Predict reweighted velocity
        v = burgers_fm.predict_velocity(x, tau, c)

        # Euler step
        x = x + v * dtau

    # 最后一次 overwrite,保证 x_final 满足 boundary
    x[:, 0, 0, :] = u_0
    x[:, 0, T_idx, :] = u_T_star

    return x
```

### §6.3 训练侧补充

Inpainting overwrite trick 在训练时也要配套:行 0 和 T 的 target velocity 强制 = 0(教模型「看到 clean 值就别动」)。

```python
# flow/trainer.py:

def compute_cfm_loss(model, x_1, c, alpha_fn, beta_fn, alpha_dot_fn, beta_dot_fn):
    """
    Conditional Flow Matching loss with row-0/row-T inpainting training trick.
    """
    B = x_1.shape[0]
    tau = torch.rand(B, device=x_1.device)           # τ ~ Unif(0, 1)
    x_0 = torch.randn_like(x_1)                       # noise

    # Build noisy sample
    alpha = alpha_fn(tau).view(-1, 1, 1, 1)
    beta  = beta_fn(tau).view(-1, 1, 1, 1)
    x_tau = alpha * x_1 + beta * x_0

    # Target velocity = α̇ * x_1 + β̇ * x_0
    alpha_dot = alpha_dot_fn(tau).view(-1, 1, 1, 1)
    beta_dot  = beta_dot_fn(tau).view(-1, 1, 1, 1)
    v_target = alpha_dot * x_1 + beta_dot * x_0

    # Inpainting trick: row 0 / row T 的 target velocity = 0
    v_target[:, 0, 0, :] = 0
    v_target[:, 0, 10, :] = 0

    # Forward
    v_pred = model(x_tau, tau, c)

    # MSE loss(row 0/T 的 prediction 也算进去,但 target=0,所以模型学到 identity map)
    loss = ((v_pred - v_target) ** 2).mean()
    return loss
```

---

## §7 工程细节 / 数值稳定性

### §7.1 必须 clamp $\tau$

CondOT path 下 $b_\tau = 1/\tau$,$a_\tau = (1-\tau)/\tau$,**在 $\tau = 0$ 都是 $\infty$**。

实际跑的时候不能用纯 $\tau \in [0, 1]$,要用 $\tau \in [\tau_{\min}, 1 - \tau_{\min}]$,$\tau_{\min} \sim 10^{-3}$ 或 $10^{-4}$。

具体在三个地方:
1. **训练时**:`tau = torch.rand(B) * (1 - 2*tau_min) + tau_min`
2. **采样时**:`tau ∈ linspace(tau_min, 1 - tau_min, n_steps)`
3. **公式里 $b_\tau$ 计算**:`tau.clamp(min=tau_min)`

### §7.2 schedule 的好消息

$\tilde\eta(\tau)$ 在 $\tau \to 0$ 时趋于 0(`sigmoid_flip` 性质),所以即使 $b_\tau \to \infty$,乘积 $\tilde\eta(\tau) \cdot b_\tau \cdot w$ 是被 schedule 的衰减压住的。但保险起见还是要 clamp。

### §7.3 必须 joint 和 prior 同 path

整个推导依赖「joint 和 prior 用同一条 $\alpha_\tau, \beta_\tau$」,这样 Step 4 里 $a_\tau$ 才能约掉。

实现上:**两个模型(`model_uw`, `model_w`)训练时用完全相同的 `alpha_fn`, `beta_fn`**,绝不能一个用 CondOT 另一个用 cosine。

### §7.4 prior 模型的「u-block 训练为 0」trick

DDPM 那边代码 `diffusion_1d_burgers.py:402`:

```python
model_w_output[..., 0, :, :] = 0  # only output w, not trained on u
```

FM 等价 = 训练 prior 模型时:

1. 输入 $x$ 时 u-channel 清零
2. target velocity 的 u-channel 也强制 0(loss 不在 u-channel 上传梯度)
3. forward 完后输出再 mask 一次 u-channel = 0(防御性)

这样 prior 模型完全只学 $w$ 的 dynamics,自然嵌入到联合空间里 u-block = 0,跟数学上 $\nabla_u \log p(w|c) = 0$ 一致。

---

## §8 Sanity Checks

写完代码、跑训练之前,先做这几个数学层面的检查:

### §8.1 $\gamma = 1$ 退化

代入 $\gamma = 1$:

$$
\tilde u_\tau^{\text{rw}} \;=\; u_\tau^{\text{joint}} \;+\; 0 \cdot \tilde\eta \cdot [\,\cdots\,] \;=\; u_\tau^{\text{joint}}
$$

→ 完全退化成 joint sampling,跟 baseline 一致。代码里跑 `--gamma 1.0`,应该 reweighting 这段完全不发力。

### §8.2 形状对得上

| 项 | shape |
|:---|:---:|
| `x` | $[B, 2, 16, 128]$ |
| `v_joint` | $[B, 2, 16, 128]$ |
| `v_prior`(u-block 强制 0) | $[B, 2, 16, 128]$ |
| `x_w_only`(u-block 强制 0) | $[B, 2, 16, 128]$ |
| `b_tau`(broadcastable) | $[B, 1, 1, 1]$ |
| `eta`(broadcastable) | $[B, 1, 1, 1]$ |
| `correction = v_prior - b_tau * x_w_only` | $[B, 2, 16, 128]$ |
| `v_reweighted` | $[B, 2, 16, 128]$ |

✓ 全部一致。

### §8.3 u-block 不动

`v_prior` u-block = 0,`x_w_only` u-block = 0,所以 `correction` 的 u-block:

$$
0 - b_\tau \cdot 0 = 0
$$

→ **correction 整个 u-block 都是 0**。

所以 reweighting **只影响 w-channel**,完全不动 u-channel。这跟数学上「prior reweighting 只 reshape $p(w)$,不动 $p(u|w)$」一致。

这个性质可以在代码里加 assert 验证:

```python
assert correction[:, 0].abs().max() < 1e-6, "correction should be zero on u-channel"
```

### §8.4 跟 DDPM 同 setting 同 γ=1 应该数值一致

把 FM γ=1 的 J 跟 DDPM γ=1 的 J(`notes_baseline_summary.md §3.1`:0.0082)直接比。如果差太多(比如 FM 出 0.02+),说明 joint 模型训得不对,跟 reweighting 公式无关 — 先 debug `model_uw` 训练。

### §8.5 跟 DDPM γ=2.5 比

这是 paper FOPC 最优配置(`notes_baseline_summary.md §5.2`)。DDPM 那边 J = 0.00796。FM γ=2.5 + 同 schedule 应该数字接近(差 30% 以内)。

如果 FM γ=2.5 比 γ=1.0 显著好(像 DDPM 那样降 3%)→ reweighting 公式跑对了 ✓
如果 FM γ=2.5 比 γ=1.0 更糟 → 公式可能符号搞反了,检查 `+ (gamma - 1)` 还是 `- (gamma - 1)`(本文是前者)

---

## §9 一句话总结

> 在 FM 框架下,DiffPhyCon 的 prior reweighting trick 可以严格地写成:
> $$\tilde u_\tau^{\text{rw}}(u,w|c) = u_\tau^{\text{joint}}(u,w|c) + (\gamma - 1)\,\tilde\eta(\tau)\,\big[\,u_\tau^{\text{prior}}(w|c) - b_\tau\,[\,0,\,w\,]\,\big]$$
> 关键修正是 $b_\tau\,[\,0,\, w\,]$ 这一项——FM 跟 DDPM 不一样的地方是 velocity 跟 score 的转换公式里多了一个 $b_t x$ offset,所以做线性组合时必须手动减回去,a_t 才能约掉,公式才漂亮地化简成 $(\gamma - 1)$ 不带 $a_t$ 的形式。

---

## §10 这份文档跟其他 notes 的关系

- 取代 `notes_diffphycon_flow_bridge.md §6.4` 那个简化(有 bug)的公式
- 推导细节用到 `flow_matching_diffusion.md` Prop 1(line 310-318)和 Example 23(line 372)
- 实测数字目标用 `notes_baseline_summary.md §5.2` 的 4 行 baseline
- 工程实现可以直接对照 DDPM 代码 `diffusion/diffusion_1d_burgers.py:397-409`(`model_predictions` 函数)

下一步:把 §6.2 的代码骨架写进 `flow/burgers_fm.py`,跟 plan Phase 2A 对齐。
