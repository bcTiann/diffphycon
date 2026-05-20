# 逐行精读 `p_sample_loop` —— 扩散模型的反向采样主循环

> 文件位置:`diffusion/diffusion_1d_burgers.py:525-584`
>
> 这一个函数 = 整个推理(inference)的核心。它从一团**纯噪声**出发,
> 经过 1000 步逐步"去噪",最终输出一对 `(u, w)` —— 即一条满足物理约束、
> 又能完成控制目标的轨迹。

---

## 0. 大局观:这函数到底在干嘛?

### ⚠️ 时间约定大坑(必须先讲清)

你 MIT 课和这份 DDPM 代码用的是**完全相反的时间约定**。

| | MIT 视角(你学的) | DDPM 视角(这份代码) |
|---|---|---|
| $t = 0$ | 噪声 $p_{\text{init}}$ | **数据 $p_{\text{data}}$** |
| $t = T$(或 $t=1$) | 数据 $p_{\text{data}}$ | **噪声 $p_{\text{init}}$** |
| **采样 / 生成** | **正向**积分 ODE/SDE,$t: 0 \to 1$ | **反向**循环,$t: T \to 0$ |
| 你 MIT 笔记里写的(Sec 2): $X_0 \sim p_{\text{init}}$,$\frac{d}{dt}X_t = u_t^\theta(X_t)$,目标 $X_1 \sim p_{\text{data}}$ | | |

换算关系(把 DDPM 的 $t$ 翻译成 MIT 的 $\tau$):
$$
\tau_{\text{MIT}} = 1 - \frac{t_{\text{DDPM}}}{T}
$$

- DDPM 的 $t = 999$(噪声态)$\leftrightarrow$ MIT 的 $\tau = 0$(`p_init`)
- DDPM 的 $t = 0$(干净数据)$\leftrightarrow$ MIT 的 $\tau = 1$(`p_data`)

### 在这两套视角下,采样在干嘛?

**MIT 视角**:你从 $p_{\text{init}}$ 抽一个 $X_0$,沿向量场 $u_t^\theta$ **正向**积分到 $\tau = 1$,得到 $X_1 \sim p_{\text{data}}$。

**DDPM 视角**:你从噪声 $x_T$ 出发,**反向**循环 $t = T \to 0$,每步去掉一点噪声,最后得到干净数据 $x_0$。

**两者描述的是同一件事**,只是 t 的方向定义不同。本质都是"从噪声到数据"。

### `p_sample_loop` 做的事

由于代码里时间是**离散**的(DDPM 约定 $t = 0, 1, \dots, 999$),且 $t=0$ 是数据,
所以采样要从 $t=999$ 一步步**退**到 $t=0$ —— 这就是为什么 for 循环用 `reversed(...)`。

从你 MIT 视角理解:**这相当于把 $\tau$ 正向地从 0 推到 1**。

---

## 1. 函数签名

```python
def p_sample_loop(self, shape, **kwargs):
```

| 参数 | 含义 |
|---|---|
| `self` | 类本身的实例(`GaussianDiffusion`),里面有所有模型参数、$\beta$ 表、各种 flag |
| `shape` | 想要生成的张量形状,比如 `(50, 2, 16, 128)`(batch=50,2 通道,16 时间步,128 空间格) |
| `**kwargs` | 所有传进来的"额外参数",像 `u_init`、`u_final`、`nablaJ`、`J_scheduler` 等等 |

> **小知识**:`**kwargs` 是 Python 把一堆关键字参数打包成一个字典 `kwargs` 的写法。
> 比如外面调用 `p_sample_loop(shape, u_init=xxx, nablaJ=yyy)`,
> 函数内部 `kwargs` 就是 `{'u_init': xxx, 'nablaJ': yyy}`。

---

## 2. 前置准备(进入循环之前)

```python
assert not self.is_ddim_sampling, 'wrong branch!'

nabla_J, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
batch, device = shape[0], self.betas.device

img = torch.randn(shape, device=device)
x_start = None
```

### 一句句拆:

**`assert not self.is_ddim_sampling, 'wrong branch!'`**
- "断言":确认当前**不是** DDIM 采样模式(DDIM 是另一种采样方式,走另一个函数 `ddim_sample`)
- 如果误进了这里,程序直接报错,防御性编程

**`nabla_J, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)`**
- 从 `kwargs` 里把 guidance 相关的 3 个东西拿出来:
  - `nabla_J`:函数,输入 $x$,返回 $\nabla J(x)$(控制目标的梯度)
  - `nablaJ_scheduler`:函数,输入 $t$,返回一个权重 $\lambda(t)$(让 guidance 随时间强弱变化)
  - `may_proj_guidance`:函数,把 $\nabla J$ 注入到 $\epsilon$ 里的方式(默认就是相加)
- 如果你没传 guidance,这 3 个会被设成"什么都不做"的占位函数

**`batch, device = shape[0], self.betas.device`**
- `batch = shape[0]`,比如 50
- `device = self.betas.device`,模型参数在哪个设备(CPU/MPS/CUDA),就在哪个设备上算

**`img = torch.randn(shape, device=device)`** ⭐ **关键一行**
- 创建一个跟 `shape` 形状一样的张量,值是从标准正态分布 $\mathcal{N}(0, I)$ 抽出来的
- 这就是 $x_T$ —— **采样的起点是纯噪声**
- 在你 MIT 视角里对应:**ODE 起点** $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I)$($\tau = 0$,而非 $\tau = 1$)
- ⚠️ 别被符号误导:DDPM 的 $x_T$ = MIT 的 $X_0$,都是"噪声端"

**`x_start = None`**
- 占位变量,用于 self-conditioning(暂时不用管,代码里这个 flag 是 False)

---

## 3. 主循环

```python
for t in tqdm(reversed(range(0, self.num_timesteps)), 
              desc='sampling loop time step', 
              total=self.num_timesteps):
```

### 拆开看:
- `range(0, self.num_timesteps)` → `[0, 1, 2, ..., 999]`(假设 1000 步)
- `reversed(...)` → `[999, 998, ..., 1, 0]`(反着来)
- `tqdm(...)` → 套一个进度条(就是终端里那个滚动的百分比)
- 整个循环跑 1000 次,每次 `t` 取一个值,**从 999 退到 0**

### 为什么反着来?
**DDPM 视角**:我们在去噪,从最噪声的 $x_{999}$ 退到干净的 $x_0$。每步把噪声去掉一点点。

**MIT 视角**:换算成 $\tau = 1 - t/T$,这其实是**正向**走 $\tau$ 从 0 到 1。
反向不反向是 t 的方向,**物理上"从噪声→数据"的方向是同一个**。

代码里单步更新对应的 DDPM 离散公式是:
$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot \xi, \quad \xi \sim \mathcal{N}(0, I)
$$

而你 MIT 笔记里的对应物是连续 SDE 的一步欧拉离散化:
$$
X_{\tau + \Delta\tau} = X_\tau + u_\tau^\theta(X_\tau) \Delta\tau + \sigma_\tau \sqrt{\Delta\tau} \cdot \xi
$$

—— 同一个东西的两种写法。

---

## 4. 循环体内部(下面是循环的"一步"在做什么)

```python
for k in range(self.recurrence_k):
```

- `recurrence_k` 是"递归采样"的次数,默认 = 1
- 简单理解:99% 情况下这个 `for` 只跑一次,**你可以当它不存在**
- 它的作用是允许某些步骤"反复横跳"(`recurrence` 是个加强技巧),不用管

---

### 4.1 三个 `if`:把已知条件硬塞进去 ⭐⭐⭐

```python
if self.is_condition_u0:
    u0 = kwargs['u_init']
    self.set_condition(img, u0, shape, 'u0')

if self.is_condition_uT:
    uT = kwargs['u_final']
    self.set_condition(img, uT, shape, 'uT')
    
if self.set_unobserved_to_zero_during_sampling:
    Nx = img.size(-1)
    img[:, 0, :, Nx // 4: (Nx * 3) // 4] = 0
```

#### 为什么需要这些 `if`?

因为**不同的实验设定开关不同**。FOPC、POFC、POPC 三种设定下,要不要给 $u_0$、要不要给 $u_T$、要不要把中间区域抹零,都不一样。代码用 flag 切换,所以一堆 `if`。

> 类比:就像你做一道菜的可选配料 —— 加不加葱、加不加辣椒,看你想要什么口味。

#### 这三个 `if` 分别做什么?

**`if self.is_condition_u0:`** —— 把已知的**初始状态** $u(t=0)$ 强行写到 `img` 里
- `kwargs['u_init']`:外面传进来的"已知初始条件",形状 `(batch, Nx)`,即 50 条样本每条 128 个空间点
- `set_condition(img, u0, shape, 'u0')`:**直接覆盖** `img[:, 0, 0, :]` 这一列(u 通道、t=0 时刻)成 `u0`

**`if self.is_condition_uT:`** —— 把已知的**末态** $u(t=T)$ 强行写到 `img` 里
- `kwargs['u_final']`:你想要的**目标终态**,形状 `(batch, Nx)`
- `set_condition(img, uT, shape, 'uT')`:覆盖 `img[:, 0, 10, :]` 这一列(u 通道、t=10 时刻)成 `uT`

**`if self.set_unobserved_to_zero_during_sampling:`** —— Partial Observation 用的
- 如果设定下"中间区域不可观测",就把那部分硬填零
- FOPC 模式下这个 flag 是 False,跳过

#### 为什么是**每一步都塞**,而不是只塞一次?

这是初学者最容易困惑的地方。我详细讲。

**问题**:`p_sample` 这一步会**整张图全部更新**(包括 `u_init` 和 `u_final` 那两列)。
如果你只在循环开始前塞一次,**第一步去噪后那两列就被模型改掉了**,后面就不再"等于" `u_init / u_final` 了。

**解决**:每一步去噪**之前**,强行把这两列再覆盖回 `u_init / u_final`。这样:
- 模型可以参考"这一行/那一列是固定的"来决定其他位置怎么去噪
- 等于在每一步都重置"你不能动这两列"的约束

> 类比:你在做填字游戏,有些格子是题目给定的(`u_init`、`u_final`),你只能填剩下的格子。
> 每写一笔,都要确认题目给定的格子没被你不小心抹掉。

这个技巧在扩散模型里叫 **replacement-based conditioning**(或叫 inpainting),
不是 classifier guidance —— 这是一种**硬约束**。

---

### 4.2 `residual` 相关(可以忽略)

```python
if self.conditioned_on_residual is not None:
    # ...
else:
    residual = None
```

- 是一个**实验性 feature**(基于物理残差做 conditioning)
- 跑 lite 版本时是 `None`,直接走 `else`
- **跳过不用管**

---

### 4.3 单步去噪 ⭐⭐⭐

```python
self_cond = x_start if self.self_condition else None
img_curr, x_start, pred_noise = self.p_sample(img, t, self_cond, residual=residual, **kwargs)
```

#### 一行拆开:

**`self_cond = x_start if self.self_condition else None`**
- self-conditioning 是一个加强 trick(把上一步的 $\hat{x}_0$ 也喂给模型)
- 这里 `self_condition=False`,所以 `self_cond=None`
- 当它不存在

**`img_curr, x_start, pred_noise = self.p_sample(img, t, ...)`** ⭐
- **核心操作**:把当前的 `img`(即 $x_t$)去噪一步,得到 $x_{t-1}$
- 返回三个东西:
  - `img_curr`: $x_{t-1}$,去噪一步后的张量
  - `x_start`: $\hat{x}_0$,模型当前对"干净版本"的估计(用来算 guidance)
  - `pred_noise`: $\epsilon_\theta$,模型预测的噪声

**`p_sample` 内部公式**(回顾):
$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot \xi, \quad \xi \sim \mathcal{N}(0, I)
$$

其中 $\mu_\theta$ 由模型预测的噪声 $\epsilon_\theta$ 推出。

---

### 4.4 Guidance 注入

```python
if self.guidance_u0:
    img = img_curr
else:
    pred_noise = may_proj_guidance(pred_noise, nabla_J(img_curr) * nablaJ_scheduler(t))
    img, x_start, _ = self.p_sample(img, t, self_cond, pred_noise=pred_noise, residual=residual, **kwargs)
```

#### 这里又是 `if`!为什么?

代码支持**两种 guidance 风格**:

| `guidance_u0=True`(我们用的) | `guidance_u0=False`(另一种) |
|---|---|
| 在 `p_sample` **内部**算 guidance(对干净估计 $\hat{x}_0$ 算梯度) | 在 `p_sample` **外部**算 guidance(对噪声态 $x_{t-1}$ 算梯度) |
| 只跑一次 `p_sample` | 跑两次 `p_sample`(第二次用修过的 ε) |

我们跑推理时 `guidance_u0=True`(看 `inference_1d_burgers.py:381`),所以走第一个分支:

**`img = img_curr`** —— 直接把 `p_sample` 的输出当作下一步的 `img`,**结束这一步**。

> guidance 实际上**已经在 `p_sample` 内部做完了**(在 `model_predictions` 里),
> 所以外面只需要把结果接收下来即可。

---

### 4.5 收尾

```python
img = img.detach()

if not self.recurrence:
    break
img = self.recurrent_sample(img, t)
```

**`img = img.detach()`**
- "切断梯度图":告诉 PyTorch 不要追踪从这里往后的梯度(节省显存,加速)
- 因为 guidance 已经算完用过了,后续步骤不再需要这部分梯度信息

**`if not self.recurrence: break`**
- 99% 情况下 `recurrence=False`,所以直接 break 退出 `for k`
- 当 `recurrence=True` 时才会反复采样(高级 trick,不用管)

---

## 5. 循环结束后

```python
img = self.unnormalize(img)
return img
```

**`img = self.unnormalize(img)`**
- 训练时模型看到的数据被归一化到 $[-1, 1]$(标准 DDPM 习惯)
- 推理结束需要**反归一化**回原始物理量纲
- 公式:`x_orig = (x + 1) / 2 * (max - min) + min`(具体看 `unnormalize` 实现)

**`return img`**
- 返回最终的 $x_0$,形状跟输入 `shape` 一样,即 `(batch, 2, 16, 128)`
- 第 0 通道是 $u$,第 1 通道是 $w$,外面的代码用切片把它们拿出来

---

## 6. `set_condition` 函数详解(被三个 `if` 调用)

```python
def set_condition(self, img, u, shape, u0_or_uT):
    if u0_or_uT == 'uT':
        if len(shape) == 4:
            img[:, 0, self.condition_idx, :] = u   # condition_idx 默认 = 10
        ...
    elif u0_or_uT == 'u0':
        if len(shape) == 4:
            img[:, 0, 0, :] = u
        ...
```

**关键操作**:`img[:, 0, 0, :] = u` 和 `img[:, 0, 10, :] = u`

| 切片 | 含义 |
|---|---|
| `img[:, 0, 0, :]` | 所有 batch,通道 0(u),时间 0,所有空间点 → **初始状态那一行** |
| `img[:, 0, 10, :]` | 所有 batch,通道 0(u),时间 10,所有空间点 → **末态那一行** |

用 `=` 赋值,**完全覆盖**原来的内容。

---

## 7. 用一张图理解整个流程

```
开始
  │
  │  img = N(0, I)        ← 纯噪声,shape = (50, 2, 16, 128)
  ▼
┌─────────────────────────────────────────────────┐
│  for t in [999, 998, ..., 1, 0]:                 │
│                                                  │
│    [覆盖] img[:, 0, 0, :]  ← u_init             │
│    [覆盖] img[:, 0, 10, :] ← u_final            │
│                                                  │
│    img, x_start, ε = p_sample(img, t)            │
│                          │                       │
│                          ├─ U-Net 预测 ε        │
│                          ├─ 算 x_0 的估计       │
│                          ├─ 算 ∇J(x_0)          │
│                          ├─ 修正 ε              │
│                          └─ 采 x_{t-1}          │
└─────────────────────────────────────────────────┘
  │
  │  img = unnormalize(img)
  ▼
返回 (u, w) 配对
```

---

## 8. 跟 MIT 框架的对照表

**关键:MIT 用 $\tau \in [0, 1]$,$\tau=0$ 是噪声;DDPM 用 $t \in \{0, \dots, T\}$,$t=0$ 是数据。
两套约定的时间方向相反,$\tau = 1 - t/T$。**

| MIT 课的连续时间视角($\tau$,正向积分) | 这里的离散 DDPM 实现($t$,反向循环) |
|---|---|
| ODE/SDE 起点 $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I)$,$\tau=0$ | `img = torch.randn(shape, device=device)`,代码里这叫 $x_T$ |
| 时间 $\tau$ 从 0 流到 1(物理上:噪声 → 数据) | `for t in reversed(range(0, 1000))`,$t$ 从 999 退到 0(同样:噪声 → 数据) |
| 单步更新 $X_{\tau+\Delta\tau} = X_\tau + u_\tau^\theta \Delta\tau + \sigma\sqrt{\Delta\tau}\xi$ | `p_sample(img, t)`,DDPM 公式 $x_{t-1} = \mu_\theta + \sigma_t \xi$ |
| Score $\nabla \log p_\tau(x)$ | $-\epsilon_\theta / \sigma_t$(差个负号和系数) |
| Classifier guidance $\nabla \log p(x\|y)$ | `pred_noise += λ(t) · ∇J(x_0_hat)` |
| 条件采样:从 $p(x \| y)$ 抽样 | replacement 或 guidance,**这里两个都用了** |
| 终点 $X_1 \sim p_{\text{data}}$,$\tau=1$ | `return img`,代码里这叫 $x_0$ |

---

## 9. 你之前提的那 4 个 if,再总结一次

> 整个函数里有很多 `if`,本质上都是**功能开关**。每个 `if` 对应一种可选行为,
> 不同实验设定开不同的开关。代码作者写成 if 是为了**一份代码跑所有变体**。

| `if` 名字 | 用途 |
|---|---|
| `if self.is_condition_u0` | 是否固定初始状态(我们这里 True) |
| `if self.is_condition_uT` | 是否固定末态(我们这里 True) |
| `if self.set_unobserved_to_zero_during_sampling` | partial observation 模式才用(我们 False) |
| `if self.conditioned_on_residual is not None` | 残差 conditioning 实验(我们 None) |
| `if self.guidance_u0` | guidance 的两种风格(我们 True) |
| `if not self.recurrence` | 是否反复采样(我们 False,直接 break) |

---

## ✅ 检查理解(自问自答)

1. **`img` 在循环开始时是什么?为什么是它?**
   - 是从 $\mathcal{N}(0, I)$ 抽出来的纯噪声。因为反向去噪的起点就是噪声分布。

2. **`for t in reversed(range(0, 1000))` 为什么要 reversed?**
   - 因为我们在做去噪(逆过程),时间从 $t=999$ 退到 $t=0$。

3. **`set_condition` 为什么每一步都要调用?**
   - 因为 `p_sample` 会更新整张图,如果不每步重新塞,条件那两列就被覆盖了。

4. **`p_sample` 返回的三个东西分别是什么?**
   - `img_curr` = $x_{t-1}$(下一步用),`x_start` = $\hat{x}_0$(模型对干净版的估计,算 guidance 用),
     `pred_noise` = $\epsilon_\theta$(模型预测的噪声)。

5. **退出循环后 `unnormalize` 干什么?**
   - 把模型用的归一化值($[-1, 1]$ 范围)反变换回物理量纲。

如果你能口答这 5 个,这个函数你就吃透了。

---

# 🔑 附录 A:**"覆盖" 到底是什么?它就是论文里的 `| c`!**

> 这是初学者最大的卡点之一。一次把它讲透。

## A.1 你的猜测 —— 全部正确 ✅

> "给模型的输入不是一个纯噪声 `img`,而是一个把 `u_init`, `u_final` 替换成真实状态的 `img`?把这个替换后的给 model 去 denoise?得到下一 timestep 的 `img`?"

**对。**

> "model 预测出来的这个其实也可以把被覆盖的 `u_init`, `u_final` 自己再覆盖一次?"

**对。事实上代码每一步都这么做。**

> "这个覆盖是不是论文中一直带的那个 `|c` 的 condition?"

**对。这就是 `|c` 的一种实现方式。**

---

## A.2 论文里的 `|c` 到底是什么?

DiffPhyCon 论文里所有概率符号都带 `|c`,例如:

$$
p(u, w \mid c), \quad p_\theta(w \mid u, c), \quad \nabla \log p(u, w \mid c)
$$

这里的 **$c$ 就是 "条件 / condition"** —— 你**事先知道**或**强制要求**的那一部分信息。

### 在 1D Burgers FOPC 实验里,$c$ 具体是:

$$
c = (u_{\text{init}},\ u_{\text{final}})
$$

- $u_{\text{init}}$ = $u(t=0, x)$,**给定的初始状态**(从测试集里取)
- $u_{\text{final}}$ = $u(t=T, x)$,**你想达到的目标末态**(从测试集里取)

### 在其他实验里,$c$ 可能是:

| 实验 | $c$ 是什么 |
|---|---|
| FOPC(我们的) | $(u_0, u_T)$ |
| POFC | $(u_0$ 的部分观测,$u_T)$ |
| POPC | $u$ 的部分时空观测 |
| Jellyfish 控制 | 速度场的边界条件 + 目标速度 |
| 一般文图生成 | 文本 prompt |

**通用规律**:$c$ = "**你不能随便采,必须满足的硬约束 / 软约束**"。

---

## A.3 为什么 "覆盖" 等于 "条件化"?数学上怎么解释?

### 朴素直觉

要让生成的 $(u, w)$ 满足"开头是 $u_{\text{init}}$、结尾是 $u_{\text{final}}$",最暴力的办法就是:

> "管你模型输出什么,**那两列我直接强制写成真值**就完事了。"

这就是 "**replacement-based conditioning**"(也叫 **inpainting-style conditioning**)。

### 类比:图像 inpainting

|  | 图像 inpainting | 我们这里 |
|---|---|---|
| 已知 | 图像的一部分像素(比如边框) | $u(t=0, \cdot)$ 和 $u(t=10, \cdot)$ |
| 未知 | 中间被挖掉的像素 | $u(t=1\ldots 9, \cdot)$ 和 全部的 $w(t, \cdot)$ |
| 做法 | 每步去噪后,把已知像素强行写回去 | 每步去噪后,把 $u_0, u_T$ 那两列强行写回去 |

**完全一样的技巧**。

---

## A.4 这个做法对应贝叶斯公式

数学上,**采样 $p(u, w \mid c)$** 等价于:
$$
p(u, w \mid c) \propto p(u, w) \cdot \mathbb{1}[u(t=0)=u_{\text{init}}] \cdot \mathbb{1}[u(t=T)=u_{\text{final}}]
$$

即:**在联合分布 $p(u, w)$ 里,只保留那些满足"开头 = $u_{\text{init}}$、结尾 = $u_{\text{final}}$"的样本**。

"硬覆盖"就是在每一步采样时,**强行把样本拽回到这个约束子空间上**。

> 你可以把"覆盖"看成一个**投影**操作:把当前的 $x_t$ 投影到"满足边界条件"的超平面上。

---

## A.5 完整流程图(条件采样)

```
开始
  │
  │  img = N(0, I)         ← 纯噪声
  ▼
┌──────────────────────────────────────────────────────────┐
│  for t = 999, 998, ..., 0:                                │
│                                                           │
│    ① 把 img[:, 0, 0,  :] 强行写成 u_init    ← |c 第 1 部分│
│    ② 把 img[:, 0, 10, :] 强行写成 u_final   ← |c 第 2 部分│
│                                                           │
│    ③ 把这个"嵌入了条件的 img"喂给 U-Net                  │
│       U-Net 在条件子空间里去噪一步                        │
│                                                           │
│    ④ 模型可能又把那两列改了一点(没关系)                │
│       下一轮迭代开始时,① ② 又把它们覆盖回去             │
│                                                           │
└──────────────────────────────────────────────────────────┘
  │
  ▼
返回 (u, w),开头结尾必然 = u_init, u_final
```

---

## A.6 一个细节:模型怎么知道"这是条件,不是变量"?

**它不知道。** 这是这种做法的一个理论缺陷。

模型只看到一个"看起来很整齐"的张量 `img`,它不知道哪几列是条件、哪几列是要去噪的。
它就按它学过的方式做去噪。

但**实践中效果很好**,因为:
1. 每步覆盖让模型"看到的"那两列总是干净的真值
2. 模型在去噪其他位置时,会**自动考虑这两列的存在**(因为训练时它学过 $u_0, u_{T}$ 跟整条轨迹的关系)
3. 物理上的连续性约束会让模型在两端附近的去噪输出"自动"靠近条件值

### 代码里的小修补

为了进一步让模型 "尊重" 条件,代码还做了一件事(`is_condition_u0_zero_pred_noise=True`):
> 当 U-Net 预测的噪声涉及到条件位置时,把那部分噪声**直接抹零**。
> 这样下一步的 $x_{t-1}$ 在那两列就更接近真值,而不是被"假噪声"扰动。

---

## A.7 ⚠️ "覆盖" ≠ 所有的 `|c`!

**DiffPhyCon 里 $c$ 实际上有两种成分**,实现方式完全不同:

| $c$ 的成分 | 实现方式 | 代码位置 |
|---|---|---|
| **边界条件** $(u_0, u_T)$ | **Replacement (硬覆盖)** | `set_condition`,每步覆盖 |
| **控制目标** $J$(最小化某个能量) | **Classifier Guidance (软引导)** | `pred_noise += λ(t) · ∇J(x_0_hat)` |

为什么分两种?

- $u_0, u_T$ 是**等式约束**(必须严格相等)→ 用硬覆盖最直接
- $J$ 是**优化目标**(越小越好,但不要求严格 = 0)→ 用梯度引导

所以论文里的 $p(u, w \mid c)$ 在代码里被分解成:
- 边界部分 → 投影 / 覆盖
- 优化部分 → guidance(对应 $\nabla \log p(y \mid x)$ 中的 $y$ = "$J$ 很小"这个事件)

---

## A.8 用 MIT 视角理解 `|c`

你笔记里 Sec 5(Guidance)讲 classifier-free guidance 时,本质上是采样:
$$
p(X_1 \mid y) \propto p(X_1) \cdot p(y \mid X_1)
$$

其中 $y$ 就是"条件"。

DiffPhyCon 把这套思想扩展到了"**条件有两种**":
- 一种是"必须满足的等式约束"($u_0 = ?$,$u_T = ?$)
- 一种是"软目标"($J$ 越小越好)

**两种都被塞进 $c$ 里**,只是实现机制不同。

---

## A.9 ✅ 一句话答复你

> **"覆盖" = 论文里 `| c` 的硬约束部分(等式条件 $u_0, u_T$)**。
> **代码每一步去噪前,把 `img` 里那两列重写为真值,等价于把样本投影到满足约束的子空间上。**
> **softer 的条件(如最小化 $J$)用另一种方式(classifier guidance)实现。**

---

## A.10 检查理解

1. 如果我把 `is_condition_u0=False, is_condition_uT=False`,采样会得到什么?
   - **答**:无条件采样,从 $p(u, w)$ 抽一个**任意**的轨迹 + 控制对。
     初末状态都是随机的,不会满足你想要的边界条件。

2. 如果只 `is_condition_u0=True`,采样会得到什么?
   - **答**:从 $p(u, w \mid u_0)$ 抽一对。开头是你指定的,
     末态由模型"想象"一个合理的。

3. 如果条件 $u_{\text{init}}$ 和 $u_{\text{final}}$ **物理上不可能连起来**,会发生什么?
   - **答**:模型仍会强行采样出 "看起来" 满足条件的 $(u, w)$,但 $w$ 可能要非常大,
     物理上不合理。然后 `burgers_numeric_solve_free` 用这个 $w$ 真实模拟时,
     得到的 $u_T'$ 跟你给的 $u_{\text{final}}$ 会差很远。
     **这就是 inference 里 `mse_deviation` 测量的"模型生成 vs 真实模拟"误差。**
