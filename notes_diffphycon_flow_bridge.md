# DiffPhyCon ↔ Flow Matching:从你 MIT 视角看通整个 1D Burgers' 控制实验

写这份的目的:把整个 DiffPhyCon 1D Burgers 控制流程,**用你 MIT flow matching 笔记的语言重述一遍**,然后明确指出 FO/PO 训练设定在两种框架下的**相同与不同**。

读完应该能回答:
- DDPM 训练 loss 和 flow matching velocity loss 是什么关系?
- p_sample_loop 等价于 flow matching 里的哪个积分?
- inpainting 替换在 FM 框架下怎么写?
- γ(prior reweighting)的数学在 FM 下长什么样?
- FO 训出来的模型和 PO 训出来的模型,在 FM 下到底学的是什么?

---

## §1 大图:DDPM 和 Flow Matching 是同一件事的两种写法

两者**都在干这件事**:
- 定义一条**概率路径** $\{p_t\}_{t \in [0, T]}$,从简单分布 $p_0$(标准高斯)连续过渡到数据分布 $p_T$
- 在这条路径上**生成**(等价地:训练一个能在路径上前进的模型)

差异只在**记法**和**离散化**:

| 维度 | DDPM 写法 | Flow Matching 写法(你 MIT 视角)|
|------|----------|----------------------------------|
| 时间方向 | $t: T \to 0$(reverse,采样时走) | $\tau: 0 \to 1$(forward,采样时走) |
| 起点(高斯) | $x_T \sim \mathcal{N}(0, I)$ | $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I)$ |
| 终点(数据) | $x_0 \sim p_{\text{data}}$ | $X_1 \sim p_{\text{data}}$ |
| 时间换算 | $t_{\text{DDPM}}$ | $\tau_{\text{FM}} = 1 - t_{\text{DDPM}} / T$ |
| 中间状态 | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$ | $X_\tau = \alpha_\tau\,X_1 + \sigma_\tau\,X_0$ |
| 离散化 | 1000 步固定离散 Markov 链 | 连续 SDE / ODE,任意步数 Euler |
| 训练目标 | 预测噪声 $\varepsilon$ | 预测 velocity $u^\theta_\tau$(或等价对象) |

**对应关系**:
$$
\boxed{\;\varepsilon \;\longleftrightarrow\; X_0,\quad \sqrt{\bar\alpha_t} \;\longleftrightarrow\; \alpha_\tau,\quad \sqrt{1-\bar\alpha_t} \;\longleftrightarrow\; \sigma_\tau\;}
$$

记住这三对换符,后面所有公式就能相互翻译。

---

## §2 训练:学的到底是什么

### §2.1 DDPM 训练写法(代码里就是这个)

每一次梯度更新:

1. 从 $p_{\text{data}}$ 取一个干净样本 $x_0$
2. 取一个时间步 $t \sim \text{Uniform}\{1, ..., T\}$
3. 取噪声 $\varepsilon \sim \mathcal{N}(0, I)$
4. 拼出加噪样本 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$
5. 把 $x_t$ 和 $t$ 丢进 U-Net → 输出 $\varepsilon_\theta(x_t, t)$
6. **loss = $\|\varepsilon_\theta(x_t, t) - \varepsilon\|^2$**(MSE)

→ 模型学的是:**给我加噪图,我告诉你"加了什么噪声"**。

### §2.2 Flow Matching 训练写法(你 MIT 视角)

每一次梯度更新:

1. 从 $p_{\text{data}}$ 取 $X_1$
2. 取时间 $\tau \sim \text{Uniform}[0, 1]$
3. 取源噪声 $X_0 \sim \mathcal{N}(0, I)$
4. 拼出中间样本 $X_\tau = \alpha_\tau X_1 + \sigma_\tau X_0$
5. 真实 velocity:$u_\tau^{\text{target}}(X_\tau) = \dot\alpha_\tau X_1 + \dot\sigma_\tau X_0$
6. 模型输出 $u_\tau^\theta(X_\tau)$
7. **loss = $\|u_\tau^\theta(X_\tau) - u_\tau^{\text{target}}\|^2$**(conditional flow matching loss)

→ 模型学的是:**给我路径上的点,我告诉你"流速方向"**。

### §2.3 两者等价

任意时刻,**只要你知道 $\varepsilon$,就能推出 velocity**(代数恒等式):
$$
u_\tau^{\text{target}}(X_\tau) \;=\; \dot\alpha_\tau X_1 + \dot\sigma_\tau X_0
\;=\; \frac{\dot\alpha_\tau}{\alpha_\tau}\,X_\tau + \left(\dot\sigma_\tau - \frac{\dot\alpha_\tau\,\sigma_\tau}{\alpha_\tau}\right) X_0
$$

注意右边只剩下 $X_\tau$(已知)和 $X_0$(= DDPM 的 $\varepsilon$)。所以 **预测 $\varepsilon$ 和预测 velocity 数学等价**,选哪个只是 parametrization 习惯。

代码里 burgers 用的是 `--objective pred_noise`(就是 $\varepsilon$)。

---

## §3 推理:p_sample_loop 等价于什么 FM 操作

### §3.1 DDPM 写法(代码 `p_sample_loop`)

```
img = 纯噪声
for t in [999, 998, ..., 1, 0]:
    pred_noise = U-Net(img, t)
    x_start_hat = (img - σ_t · pred_noise) / α_t         # predict_start_from_noise
    img = q_posterior(x_start_hat, img, t) + 小扰动      # 走一步
```

### §3.2 FM 写法

```
X = X_0 ~ N(0, I)
for τ in [0, dτ, 2dτ, ..., 1]:
    velocity = u^θ_τ(X)
    X = X + velocity · dτ            # Euler step,forward
```

### §3.3 对应

**两者都是一阶迭代去噪**。差别在:
- DDPM 每一步用 closed-form posterior 公式(混合了 deterministic + stochastic)
- FM 每一步用纯 Euler ODE step(deterministic;加噪是 SDE 版,你笔记 §4)

关键认识:**两者在 1000 步以上的离散化下,生成的样本分布几乎一致**。论文用 DDPM 就是历史原因 + 兼容现有 codebase,跟 FM 互通。

→ 你 flow matching 实现里,**直接用 Euler 积分代替 p_sample_loop** 就行,理论保证一样的样本分布。

---

## §4 Hard conditioning(inpainting)在两个框架下

### §4.1 这件事的本质

我们已知 $u_0$ 和 $u_T^*$,想从条件分布 $p(u_{1:T-1}, f \mid u_0, u_T^*)$ 采样。

但模型只学了无条件 $p(u, f)$,怎么办?**inpainting trick**:每一步采样**强行把已知的两行覆盖回 clean 值**。

> ⚠️ 注意:**这里"两行"指的是时间轴的第 0 行(t=0)和第 10 行(t=T)**。覆盖整行,不管空间维度。
> 不要跟 **partial observation**(把 u 中间空间维度清零)混。两者**正交独立**,各管各的:
> - partial observation 是数据本身的样子(中间空间不可见)
> - inpainting 是采样时强制时间端点等于已知值

### §4.2 DDPM 实现(代码里是 `set_condition`)

```python
for t in [999, ..., 0]:
    img[:, 0, 0,  :] = u_0_clean       # 直接 clean 覆盖,不加噪
    img[:, 0, 10, :] = u_T_star_clean
    # 然后正常 p_sample 一步
```

**关键**:DiffPhyCon **直接用 clean 值覆盖,完全不加噪**。简单粗暴,实测有效。

> 旁注:经典 RePaint 论文(Lugmayr 2022)的做法是把覆盖值**也加噪到当前噪声水平**($\sqrt{\bar\alpha_t}\, u_0 + \sqrt{1-\bar\alpha_t}\,\text{noise}$)。**DiffPhyCon 没用这个**,直接 clean 覆盖。理解 DiffPhyCon 时**只看 clean 覆盖**就行,RePaint 加噪那套不参与。

### §4.3 FM 等价写法(**一字不变**)

```python
for τ in [0, dτ, ..., 1]:
    X[..., 0, 0,  :] = u_0_clean       # 直接 clean 覆盖
    X[..., 0, 10, :] = u_T_star_clean
    # 然后正常 Euler 一步
```

→ **FM 实现 inpainting 直接复制 DDPM 那行代码就完事**,无需任何 $\alpha_\tau / \sigma_\tau$ 调整。

### §4.4 配套训练 trick:`is_condition_uT_zero_pred_noise=True`

训练时把 row 0 / row 10 的 target noise 设为 0(model 学到"看到 clean 值就别动它"),inference 时配合 set_condition 才效率最高。FM 等价:训练时 row 0 / row 10 的 target velocity 设为 0(模型学到那两个位置 identity map)。

**这部分的代码改动非常小**,你 FM 实现需要复刻。

---

## §5 Classifier guidance(soft conditioning)在两个框架下

### §5.1 DDPM 公式(论文 + 代码)

每一步:
$$
\varepsilon_\theta^{\text{guided}} = \varepsilon_\theta^{\text{unguided}} + \lambda(t) \cdot \nabla_{\hat x_0} J(\hat x_0)
$$
其中 $\hat x_0$ 是当前预测的 clean 样本,$J$ 是你定义的控制目标。

### §5.2 FM 等价

每一步:
$$
u_\tau^{\theta,\text{guided}} = u_\tau^{\theta,\text{unguided}} + \tilde\lambda(\tau) \cdot \nabla_{X_\tau} \log p(c \mid X_\tau)
$$
或者用 $\hat X_1$(预测的 clean 样本)对 $J$ 算梯度:
$$
u_\tau^{\theta,\text{guided}} = u_\tau^{\theta,\text{unguided}} + \tilde\lambda(\tau) \cdot \nabla_{\hat X_1} J(\hat X_1)
$$

**形式完全一样**,只是 $\lambda$ 的调度函数 $\tilde\lambda(\tau)$ 要重新选(因为你 FM 时间方向跟 DDPM 反着)。

→ 我们之前发现 cosine_beta_J_schedule 在 DDPM 里搞错了方向(末段 guidance 静默),你 FM 实现要**反过来**用 sigmoid_flip 或常数。

---

## §6 Prior Reweighting(γ,论文核心创新)在两个框架下

> 重要澄清:**论文里的 γ 和代码里的 `prior_beta` 是同一个东西,同名同向**。之前我混着叫被自己搞糊涂了,现在统一用 γ。

### §6.1 数学(论文 Eq 9 原文)

$$
p_\gamma(\mathbf{u}, \mathbf{w} | \mathbf{c}) = p(\mathbf{w}|\mathbf{c})^{\gamma-1} \cdot p(\mathbf{u}, \mathbf{w}|\mathbf{c}) / Z
$$

对应 score(论文 Eq 10):
$$
\nabla E^{(\gamma)} = (\gamma - 1)\,\nabla E^{(p)}(\mathbf{w}, \mathbf{c}) + \nabla E_\theta(\mathbf{u}, \mathbf{w}, \mathbf{c})
$$

代码实现(`prior_beta = γ`):
$$
\varepsilon^{\text{reweighted}} = \varepsilon_\theta + (\gamma - 1)\,\eta(t)\,\varepsilon_\phi
$$

| γ | 系数 (γ−1) | 效果 | Theorem 3.1 适用条件 |
|---|------------|------|---------------------|
| < 1 | 负 | **flatten** prior(常见,论文 Remark 默认)| F(1) < 0 |
| = 1 | 0 | 无 reweight,等于 joint p(u,w\|c) | baseline |
| > 1 | 正 | **sharpen** prior(把控制信号拉回训练分布常见模式)| F(1) > 0 |

→ **两边都合法**,选哪边取决于任务的 $F(1)$ 符号(Theorem 3.1)。

### §6.2 论文实际配置(关键参考)

| Setting | 论文 γ | 方向 | Thm 3.1 适用条件 |
|---------|--------|------|------------------|
| **FOPC** | **1.5** | γ > 1, sharpen prior | F(1) > 0 |
| **POFC** | **2.5** | γ > 1, sharpen prior | F(1) > 0 |
| **POPC** | **0.9** | γ < 1, flatten prior | F(1) < 0 |

(代码里 `--prior_beta` 就是这个 γ,**同一个参数**。)

物理直觉:
- **FOPC / POFC**:hard conditioning + 物理学好 → J 已经很小 → 加强 prior 把 f **拉回**训练分布常见模式,可能不损 J 但让 f 更"自然" / Energy 更小
- **POPC**:部分观测让 joint 过度集中保守 → 略减 prior **鼓励探索**更激进的 f

→ **不同任务在 γ=1 两侧各占一边**。

### §6.3 DDPM 实现

需要训**两个模型**:
- joint:$\varepsilon^\theta_{\text{joint}}$,学 $\nabla \log p(u, f)$
- prior:$\varepsilon^{\theta,w}_{\text{prior}}$,学 $\nabla \log p(f)$

每一步采样(代码 `model_predictions:409`):
$$
\varepsilon^{\text{reweighted}} = \varepsilon^{\text{joint}} - (1-\gamma) \cdot \eta(t) \cdot \varepsilon^{\text{prior}}
$$

⚠️ 论文还多一个时间调度 $\eta(t)$ —— `--w_scheduler sigmoid_flip`(默认 `eta=1`,常数)。`sigmoid_flip` 让末段(t→0)reweight 力度大,首段(t→T)弱 —— 跟 J_scheduler 一个思路,**等 x_start 干净时才发力**。

### §6.4 FM 等价

同样训两个模型:joint velocity $u^{(u,f),\theta}_\tau$ 和 prior velocity $u^{(f),\theta}_\tau$。

每一步采样:
$$
\boxed{\;u^{\text{reweighted}}_\tau = u^{(u,f),\theta}_\tau - (1-\gamma) \cdot \tilde\eta(\tau) \cdot u^{(f),\theta}_\tau\;}
$$

**形式跟 DDPM 一字不差**(都是 score-matching 风格加权)。$\tilde\eta(\tau)$ 是你的 FM 时间方向上的对应调度。

### §6.5 我们的实验结论

**第一波(只扫 γ ≤ 1,且无 w_scheduler)**:
| Setting | γ=1.0 | γ=0.9 | 趋势 |
|---------|-------|-------|------|
| FOPC | J=0.0082 | J=0.0101 | γ↓ 单调更差 |
| POPC | J=0.0201 | J=0.0221 | γ↓ 也单调更差 |

→ 这只测了 reweighting 的 **γ < 1(flatten prior)** 方向。**漏掉了 FOPC/POFC 论文实际用的 γ > 1(sharpen prior)方向**。

**第二波(待跑,γ ∈ {0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.5} + sigmoid_flip)**:
- 预期 FOPC γ=1.5 应该让 J 略下降或 Energy 显著下降(Thm 3.1 的 F(1)>0 case)
- 预期 POPC γ=0.9 在加上 sigmoid_flip 后效果增强(F(1)<0 case)

→ 见 `run_gamma_sweep_FOPC_paper.sh` 和 `run_gamma_sweep_POPC_paper.sh`。

---

## §7 FO vs PO 在两个框架下:**同一件事**

**关键洞察**:FO/PO **不是 DDPM/FM 框架的差异**,它是**数据准备**的差异。

### §7.1 数据视角

| | FOPC 训练数据 | POPC 训练数据 |
|--|---------------|---------------|
| **每个样本的 u** | 完整的 11×128 矩阵 | 中间 1/2 列被强行设成 **0** |
| **每个样本的 f** | 中间 1/2 是 0(partial control,生成时定好) | 同左 |
| **模型实际看到的数据分布** | $p_{\text{data}}^{\text{FO}}(u, f)$ | $p_{\text{data}}^{\text{PO}}(u_{\text{遮挡版}}, f)$ |

→ **不同 setting = 不同 $p_{\text{data}}$**。模型架构、loss、训练循环**完全一样**,只是输入数据不同。

### §7.2 训练命令差异(回看)

| 参数 | FOPC | POPC |
|------|------|------|
| `--partially_observed` | `None` | `front_rear_quarter` |
| `--train_on_partially_observed` | `None` | `front_rear_quarter` |
| 其它 | 同 | 同 |

第一个参数:**数据加载时**把 u 中间清零
第二个参数:**loss 计算时**不要求模型预测 u 中间

→ 等于是教模型"忽略中间,只用边缘 u 推断 f"。

### §7.3 在 FM 框架下完全对应

在 FM 训练:
- FO 数据:正常 $(u_{\text{全}}, f) \sim p_{\text{data}}^{\text{FO}}$
- PO 数据:正常 $(u_{\text{遮版}}, f) \sim p_{\text{data}}^{\text{PO}}$

Conditional flow matching loss 一模一样,只是数据集换了:
$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{X_1 \sim p_{\text{data}}^{\text{FO 或 PO}}}\,\mathbb{E}_\tau\,\mathbb{E}_{X_0}\,\|u_\tau^\theta(X_\tau) - u_\tau^{\text{target}}\|^2
$$

**写完全一样的训练代码**,跑两次,数据不同,出来 FO 和 PO 两个 FM 模型。

### §7.4 为什么 γ 在 PO 有用?

因为 $p_{\text{data}}^{\text{PO}}(u_{\text{遮}}, f)$ **本身是个多峰分布**(同一个 $u_{\text{遮}}$ 能对应很多种 $f$,因为缺信息),所以 reweighting 能推动 sampling 跨峰。

而 $p_{\text{data}}^{\text{FO}}(u_{\text{全}}, f)$ 几乎是个 delta-like 分布(每个 $u_{\text{全}}$ 对应唯一最优 $f$),减 prior 推不动。

→ **同样的 γ 公式,在不同数据分布下效果天差地别**。这是论文的真正洞见。

---

## §8 一句话总结整个流程

整个 DiffPhyCon 1D Burgers' 控制流程,**用你 FM 视角**讲一遍:

> 我们有一个 Burgers 物理系统,要从 $u_0$ 控制到 $u_T^*$。
>
> **第一步**:训一个 flow matching 模型 $u_\tau^{(u,f),\theta}$,它学的是 $p_{\text{data}}(u, f)$ 联合分布的 velocity field。FO 设定:数据是完整 u;PO 设定:数据是遮挡版 u。
>
> **第二步**:同样训一个 prior model $u_\tau^{(f),\theta}$,学 $p_{\text{data}}(f)$ 的 velocity。
>
> **第三步**:采样时,从纯高斯 $X_0$ 出发,用 Euler 步从 $\tau=0$ 走到 $\tau=1$。每一步:
> - **Inpainting**:在 row 0 把 $u_\tau$ 替换成"加噪到 $\sigma_\tau$ 的 $u_0$";同样在 row 10 用 $u_T^*$
> - **Velocity 混合**:$u^{\text{out}} = u^{(u,f)} - (1-\gamma)\, u^{(f)}$
> - (可选)Classifier guidance:$u^{\text{out}} += \tilde\lambda(\tau) \nabla J(\hat X_1)$
> - Euler step:$X \leftarrow X + u^{\text{out}}\, d\tau$
>
> **第四步**:最终 $X_1$ 解出来,channel 1 就是预测的 $f(t, x)$;喂回真实 PDE solver 验证是否真到达 target。

→ **你 FM 论文的 novelty:不是改公式,是用 flow matching 代替 DDPM 实现上面这套**。代码改动相对小,主要是训练 loss 和 sampling loop 换成 FM 风格,其它(architecture、数据、conditioning、γ-mix)都直接照搬。

---

## §9 你下一步的研究路径

```
现在你有:
├── FOPC_10k(DDPM joint,J=0.0082 baseline)
├── FOPC_w_10k(DDPM prior,等训完)
└── FOPC γ ablation 完整结果(γ=1 最好,符合论文)

下一步:
├── 训 POPC_10k(DDPM joint, partial obs)
├── 跑 POPC γ ablation(预期 γ<1 有效)
└── 形成"FOPC γ 没用,POPC γ 救命"的完整 narrative

平行准备(可以并行思考):
├── FM 等价 joint 模型(代码框架重写)
├── FM 等价 prior 模型
└── 复现 FOPC/POPC γ ablation 在 FM 下

最终对比:
├── DDPM-baseline(FOPC_10k 数字)vs FM-baseline 数字
├── DDPM γ-ablation vs FM γ-ablation
└── 可能的 novelty:速度 / 精度 / 采样步数 / 灵活性
```

---

## §10 这份文档跟你之前笔记的关系

- `notes_p_sample_loop.md`:DDPM 采样 loop 细节(具体代码视角)
- `notes_inference_deep.md §02.3`:Prior reweighting 数学推导(纯论文视角)
- `notes_inference_deep.md §04.0`:$\hat x_0$ 跟你 MIT $\mathbb{E}[X_1|X_\tau]$ 的桥梁
- **本文档**:把上面三个抽象到顶层,加上 FO/PO 对比 + 总流程梳理

读完本文之后,你应该能:
1. 把 DDPM 代码每一行**翻译**成 FM 等价(训练 + 推理 + 条件采样 + γ)
2. 清楚 FO 和 PO 不是框架差异,而是数据差异
3. 心里有一份**FM 实现的设计图**:训练 loss 用 conditional flow matching;采样用 Euler ODE;conditioning 直接复制 DDPM 的覆盖逻辑;γ-mix 公式一样
