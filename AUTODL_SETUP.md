# AutoDL setup — paper-scale FM Burgers training (FOPC)

Goal: train vanilla & OT-CFM FM at paper scale (100k data, dim=128, 200k steps),
compare to paper's DiffPhyCon FOPC **J=0.00037**. Est cost **~¥25-30 (~$4)** on a
4090, ~12-15 hr wall time.

## 1. Rent a GPU

- AutoDL → 租用 → pick **RTX 4090 (24GB)** (¥1.98/h). 3090 (¥1.32/h) also fine, a bit slower.
- Image: pick a **PyTorch 2.x / CUDA** base image (e.g. `PyTorch 2.1.0`).
- Disk: default is enough (data ~2GB + checkpoints ~2-4GB).

## 2. Clone repo + install deps

```bash
cd /root/autodl-tmp                      # AutoDL's data disk (persists)
git clone https://github.com/bcTiann/diffphycon.git
cd diffphycon
pip install numpy<2 h5py matplotlib tqdm einops einops_exts \
    rotary_embedding_torch accelerate ema_pytorch tensorboardX scipy pandas
# torch/torchvision already in the image; if not:
# pip install torch==2.1.0 torchvision==0.16.0
```

> If `git clone` is missing the latest scripts, make sure you pushed them:
> `flow/burgers_fm_train.py`, `flow/burgers_fm_eval.py`, `run_autodl_fopc_paper.sh`.

## 3. SMOKE first (do NOT skip — ~10 min)

```bash
cd /root/autodl-tmp/diffphycon
STAGE=smoke bash run_autodl_fopc_paper.sh
```

This generates tiny data (1k) + trains 500 steps. Watch the **s/step** printed.
- If e.g. 0.1 s/step → 200k steps ≈ 5.5 hr/model → 4 models ≈ acceptable.
- If much slower, reduce `JOINT_STEPS` in the script (150k often enough), or use a bigger GPU.

## 4. Full run (overnight)

```bash
# generate 100k + train 4 models + eval, all in one:
nohup bash run_autodl_fopc_paper.sh > run.log 2>&1 &
tail -f run.log
```

Or run stages separately:
```bash
STAGE=data  bash run_autodl_fopc_paper.sh   # ~30 min
STAGE=train bash run_autodl_fopc_paper.sh   # ~12 hr (4 models)
STAGE=eval  bash run_autodl_fopc_paper.sh   # ~30 min
```

`--ckpt_every 25000` saves intermediate checkpoints, so a disconnect mid-train
doesn't lose everything (resume = re-run, or load the latest `*_step*.pt`).

## 5. Read results + download

Eval prints a table: **vanilla J | OT-CFM J | paper 0.00037** across γ.
- **Match check**: did vanilla FM reach ~0.0004-0.001? → scale-up worked, FM is a valid DDPM replacement.
- **OT check**: is OT's J consistently below vanilla's? → OT advantage holds at scale.

Download to your Mac (from your Mac, replace `<host>`/`<port>` with AutoDL's SSH info):
```bash
scp -P <port> -r root@<host>:/root/autodl-tmp/diffphycon/checkpoints/paper_fopc ./
scp -P <port> -r root@<host>:/root/autodl-tmp/diffphycon/flow/results/paper_fopc ./
```

## 6. Shut down

⚠️ AutoDL bills while the instance is **on**. After downloading, **关机/释放** the
instance so you stop paying.

## Notes / gotchas

- `generate_burgers.py` is deterministic (`torch.manual_seed(0)`), so the dataset
  is reproducible.
- If `--device cuda` errors, the scripts auto-fall back to cpu/mps (but you want cuda).
- Prior trained 50k steps (vs joint 200k): prior is heavily constrained by c, converges faster.
- batch=16 matches paper; if OT seems weak, try `--batch_size 64` (bigger batch → better OT matching).
