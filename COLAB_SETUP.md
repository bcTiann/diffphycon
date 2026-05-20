# Colab 部署 — Jellyfish γ Sweep

在 https://colab.research.google.com/ 新建一个 notebook,**`Runtime → Change runtime type → GPU(T4)`**,然后按顺序粘贴下面的 cell。

---

## Cell 1: 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# 验证数据快捷方式可见
import os
data_root = "/content/drive/MyDrive/DiffPhyCon-data"
assert os.path.exists(data_root), f"找不到 {data_root}!检查你是否在 Drive 加了快捷方式"
print("Drive 数据可见:", os.listdir(data_root))
```

期望输出:`['Checkpoints', 'Jellyfish-Dataset']`

---

## Cell 2: 克隆你的 GitHub repo + 安装依赖

```python
import os
os.chdir('/content')

# 改成你的 GitHub repo URL
REPO_URL = "https://github.com/你的用户名/diffphycon-mac.git"

!git clone {REPO_URL} /content/diffphycon
os.chdir('/content/diffphycon')

# 安装缺失的依赖
!pip install --quiet accelerate einops einops_exts rotary_embedding_torch tensorboardX h5py torchvision tensorboard

print("Repo 和依赖装好了")
```

---

## Cell 3: 把 Drive 数据软链到 diffphycon 目录(避免动 filepath.py)

```python
import os, shutil

# 创建本地目录
os.makedirs('/content/diffphycon/data/jellyfish', exist_ok=True)

# 软链 checkpoints
src = '/content/drive/MyDrive/DiffPhyCon-data/Checkpoints/DiffPhyCon/2D_jellyfish'
dst = '/content/diffphycon/data/jellyfish/checkpoints'
if not os.path.exists(dst):
    os.symlink(src, dst)

# 软链 test_data
src = '/content/drive/MyDrive/DiffPhyCon-data/Jellyfish-Dataset/test_data'
dst = '/content/diffphycon/data/jellyfish/test_data'
if not os.path.exists(dst):
    os.symlink(src, dst)

# 软链 normalization_max_min.pkl(在 train_data 里,只下这个小文件就行)
os.makedirs('/content/diffphycon/data/jellyfish/train_data', exist_ok=True)
src = '/content/drive/MyDrive/DiffPhyCon-data/Jellyfish-Dataset/train_data/normalization_max_min.pkl'
dst = '/content/diffphycon/data/jellyfish/train_data/normalization_max_min.pkl'
if not os.path.exists(dst):
    os.symlink(src, dst)

# 验证
!ls -la /content/diffphycon/data/jellyfish/
```

---

## Cell 4: 改 filepath.py 指向 Colab 路径

```python
filepath_content = '''import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

JELLYFISH_DATA_PATH = "/content/diffphycon/data/jellyfish/"
JELLYFISH_RESULTS_PATH = "/content/diffphycon/data/jellyfish/"
SMOKE_DATA_PATH = "/data/smoke/"
SMOKE_RESULTS_PATH = "/data/smoke/"
'''
with open('/content/diffphycon/filepath.py', 'w') as f:
    f.write(filepath_content)
print("filepath.py 已改")
```

---

## Cell 5: 创建结果 / 日志目录

```python
os.makedirs('/content/diffphycon/data/jellyfish/results/inference_full', exist_ok=True)
os.makedirs('/content/diffphycon/data/jellyfish/logs/inference_full', exist_ok=True)
print("结果目录建好")
```

---

## Cell 6: 快速 sanity test(1 个样本,~2 分钟)

```python
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONPATH'] = '/content/diffphycon'
os.chdir('/content/diffphycon')

!python -u inference/inference_2d_jellyfish.py \
    --num_batches 1 --batch_size 1 \
    --w_prob_exp 1.0 \
    --sampling_timesteps 8
```

期望看到 `[device] using cuda`(T4 GPU 在跑)+ 8 步进度条 + `id 0 saved`。

---

## Cell 7: 验证 theta 在分布内(关键!)

```python
import numpy as np
import glob

latest_dir = sorted(glob.glob('/content/diffphycon/data/jellyfish/results/inference_full/2026-*'))[-1]
th = np.load(f'{latest_dir}/thetas/0.npy')
print(f"theta range: [{th.min():.3f}, {th.max():.3f}]")
print(f"theta values: {np.array2string(th, precision=3)}")
print(f"\nTRAIN DATA range (参考): [0.36, 0.87]")
print(f"\n如果你的 theta 在 [-1, 5] 这种合理范围,说明 CUDA 修好了 MPS 的数值问题 ✅")
print(f"如果还是 [-2, 20] 这种,说明问题更深(可能不是 MPS 特有)")
```

---

## Cell 8: 跑完整 γ sweep

```python
!bash run_jellyfish_gamma_sweep.sh
```

预计 ~2~3 分钟全跑完(T4 比 V100 慢 ~2 倍,batch=4)。

---

## Cell 9: 画 trajectory 图

```python
!python plot_jellyfish_gamma.py
from IPython.display import Image
Image('/content/diffphycon/outputs/figures/jellyfish_theta_gamma_sweep.png')
```

---

## 之后再改代码怎么同步?

### 本地改完
```bash
git add -A
git commit -m "改了 XXX"
git push
```

### Colab cell 重跑(新增一个 cell)
```python
%cd /content/diffphycon
!git pull
```

→ **30 秒同步**,然后直接 rerun 你要的 cell。

---

## 注意:Colab 12h 断开问题

Colab 免费版长时间不操作会断,但**几分钟到一小时的任务无所谓**。
- 我们的 sweep 5 分钟 → 完全 OK
- 如果要跑 1000 步 DDPM(~30 min)也 OK
- 要跑训练几小时则需要 Pro($10/月)或换 RunPod 等
