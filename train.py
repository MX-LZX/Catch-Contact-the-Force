# -*- coding: utf-8 -*-
import os, sys, time, copy, json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(script_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)

from config import SEED, DEVICE, DATA, CHECKPOINTS, MODEL, OPTIM, SCHEDULER, TRAINING, LOSSES, SPRING, PHYSICS
from models.pinn_model import PINNForceModel
from utils.dataset import HallTimeDataset
from physics.loss_terms import magnetic_physics_loss_xyz

def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class EMAHelper:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=1.0 - d)
    def eval_model(self): return self.ema_model

def sup_loss_fn(pred, target, fx_w, fy_w, fz_w, smooth_w,
                use_smoothl1_fz=False, huber_beta_fz=0.1,
                fz_peak_cfg=None, fz_rel_w=0.0, return_details=False):
    # xy
    mse_xy = torch.nn.functional.mse_loss(pred[:, :2], target[:, :2])

    # Fx/Fy
    mse_fx = torch.nn.functional.mse_loss(pred[:, 2], target[:, 2])
    mse_fy = torch.nn.functional.mse_loss(pred[:, 3], target[:, 3])

    # —— Fz 基础项（Huber 可选）——
    if use_smoothl1_fz:
        fz_err = torch.nn.functional.smooth_l1_loss(
            pred[:, 4], target[:, 4], beta=huber_beta_fz, reduction='none'
        )
    else:
        fz_err = (pred[:, 4] - target[:, 4])**2  # per-sample

    # —— 峰值样本加权 —— 
    # w_peak = 1 + alpha * (|Fz_true| / ref) ** gamma
    w_peak = 1.0
    if fz_peak_cfg is not None:
        alpha = float(fz_peak_cfg.get("alpha", 1.5))
        gamma = float(fz_peak_cfg.get("gamma", 1.0))
        ref   = fz_peak_cfg.get("ref", "p90")
        abs_fz = target[:, 4].abs()
        if isinstance(ref, str):
            qmap = {"p50": 0.50, "p75": 0.75, "p90": 0.90}
            q = qmap.get(ref.lower(), 0.90)
            ref_val = torch.quantile(abs_fz.detach(), q)
        else:
            ref_val = torch.tensor(float(ref), device=abs_fz.device)
        ref_val = torch.clamp(ref_val, min=1e-6)
        w_peak = 1.0 + alpha * (abs_fz / ref_val).pow(gamma)
        w_peak = torch.clamp(w_peak, max=10.0)  # 防爆

    fz_loss_weighted = (w_peak * fz_err).mean()

    # —— 相对误差项（对付幅值偏差/顶峰偏小）——
    fz_rel = torch.tensor(0.0, device=pred.device)
    if fz_rel_w > 0:
        fz_rel = torch.mean((pred[:, 4] - target[:, 4]).abs() / (target[:, 4].abs() + 1e-6))

    base_loss = mse_xy + fx_w*mse_fx + fy_w*mse_fy + fz_w*fz_loss_weighted + fz_rel_w * fz_rel

    # 批内平滑
    smooth_fx = smooth_fy = smooth_fz = torch.tensor(0.0, device=pred.device)
    if pred.size(0) > 1:
        smooth_fx = ((pred[1:,2]-pred[:-1,2])**2).mean()
        smooth_fy = ((pred[1:,3]-pred[:-1,3])**2).mean()
        smooth_fz = ((pred[1:,4]-pred[:-1,4])**2).mean()
    total = base_loss + smooth_w["fx"]*smooth_fx + smooth_w["fy"]*smooth_fy + smooth_w["fz"]*smooth_fz

    if return_details:
        return total, {
            "mse_xy": mse_xy.item(), "mse_fx": mse_fx.item(), "mse_fy": mse_fy.item(),
            "fz_loss_weighted": fz_loss_weighted.item(), "fz_rel": fz_rel.item(),
            "fx_smooth": smooth_fx.item(), "fy_smooth": smooth_fy.item(), "fz_smooth": smooth_fz.item(),
        }
    return total


def make_optimizer(model, cfg):
    if cfg.get("name","adamw").lower()=="adamw":
        return optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    return optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups: pg["lr"] = lr

def main():
    set_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    dataset = HallTimeDataset(data_dir=DATA["data_dir"])
    n = len(dataset); train_n = int((1 - DATA["val_split"]) * n)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, n - train_n])

    train_loader = DataLoader(train_ds, batch_size=TRAINING["batch_size"], shuffle=True,
                              num_workers=DATA["num_workers"], pin_memory=DATA["pin_memory"])
    val_loader   = DataLoader(val_ds,   batch_size=TRAINING["batch_size"], shuffle=False,
                              num_workers=DATA["num_workers"], pin_memory=DATA["pin_memory"])

    model = PINNForceModel(**MODEL).to(device)
    optimizer = make_optimizer(model, OPTIM)

    use_plateau = SCHEDULER["type"].lower()=="plateau"
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=SCHEDULER["plateau_factor"], patience=SCHEDULER["plateau_patience"])
                 if use_plateau else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, TRAINING["epochs"]-SCHEDULER["warmup_epochs"]), eta_min=SCHEDULER["min_lr"]))
    scaler = GradScaler(enabled=OPTIM["amp"] and torch.cuda.is_available())

    os.makedirs(CHECKPOINTS["dir"], exist_ok=True)
    ckpt_dir = CHECKPOINTS["dir"]; best_val = float('inf')

    use_ema = OPTIM.get("ema_decay",0) and 0 < OPTIM["ema_decay"] < 1.0
    ema = EMAHelper(model, decay=OPTIM["ema_decay"]) if use_ema else None

    hist = {"train_sup": [], "train_phy": [], "train_cons": [], "train_total": [], "val": [], "lr": []}

    PHY_ARGS = dict(
        time_steps=PHYSICS["time_steps"], k_spring=SPRING["SPRING_K"],
        weights_dir=PHYSICS["weights_dir"], weights_rate=PHYSICS["weights_rate"],
        weight_spring_z=PHYSICS["weight_spring_z"], edges=PHYSICS["edges"],
        eps=PHYSICS["eps"], use_second_order=PHYSICS["use_second_order"], w_first_mean=PHYSICS["w_first_mean"]
    )
    lambda_div = PHYSICS["lambda_div"]

    base_lr = OPTIM["lr"]; warmup_epochs = SCHEDULER["warmup_epochs"] if not use_plateau else 0

    for epoch in range(1, TRAINING["epochs"]+1):
        model.train(); t0 = time.time()

        if not use_plateau and epoch <= warmup_epochs:
            set_lr(optimizer, base_lr * float(epoch)/float(max(1,warmup_epochs)))

        run_sup = run_phy = run_cons = 0.0

        for x_seq, x_flat, y in train_loader:
            x_seq = x_seq.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=OPTIM["amp"] and torch.cuda.is_available()):
                y_pred = model(x_seq)

                sup_loss, sup_stats = sup_loss_fn(
                    y_pred, y,
                    fx_w=LOSSES["fx_weight"], fy_w=LOSSES["fy_weight"], fz_w=LOSSES["fz_weight"],
                    smooth_w=LOSSES["smooth_weight"],
                    use_smoothl1_fz=LOSSES["use_smoothl1_for_fz"],
                    huber_beta_fz=LOSSES["smoothl1_beta_fz"],
                    fz_peak_cfg=LOSSES.get("fz_peak_weight", None),   # ★
                    fz_rel_w=LOSSES.get("fz_rel_loss_weight", 0.0),  # ★
                    return_details=True
                )

                phy_loss, _ = magnetic_physics_loss_xyz(x_seq, **PHY_ARGS)

                cons_loss = torch.tensor(0.0, device=device)
                if LOSSES["z_consistency_weight"]>0 and LOSSES["z_jitter_std"]>0:
                    B,T,F = x_seq.shape
                    noise = torch.zeros_like(x_seq)
                    z_cols = [2,5,8,11]
                    noise[:,:,z_cols] = torch.randn(B,T,len(z_cols), device=device) * LOSSES["z_jitter_std"]
                    y_pred_j = model(x_seq + noise)
                    cons_loss = ((y_pred[:,4] - y_pred_j[:,4])**2).mean()

                total_loss = sup_loss + lambda_div*phy_loss + LOSSES["z_consistency_weight"]*cons_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if OPTIM["grad_clip"]>0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=OPTIM["grad_clip"])
            scaler.step(optimizer); scaler.update()

            if use_ema: ema.update(model)

            bsz = x_seq.size(0)
            run_sup += sup_loss.item()*bsz; run_phy += phy_loss.item()*bsz; run_cons += cons_loss.item()*bsz

        avg_sup = run_sup/train_n; avg_phy = run_phy/train_n; avg_cons = run_cons/train_n
        avg_total = avg_sup + lambda_div*avg_phy + LOSSES["z_consistency_weight"]*avg_cons
        hist["train_sup"].append(avg_sup); hist["train_phy"].append(avg_phy)
        hist["train_cons"].append(avg_cons); hist["train_total"].append(avg_total)

        eval_model = ema.eval_model() if use_ema else model
        eval_model.eval(); val_sum = 0.0
        with torch.no_grad():
            for x_seq, x_flat, y in val_loader:
                x_seq = x_seq.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                y_pred = eval_model(x_seq)
                val_sum += torch.nn.functional.mse_loss(y_pred, y).item()*x_seq.size(0)
        avg_val = val_sum/(n-train_n); hist["val"].append(avg_val)

        if use_plateau: scheduler.step(avg_val)
        else:
            if epoch>warmup_epochs: scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]; hist["lr"].append(cur_lr)

        print(f"Epoch {epoch}/{TRAINING['epochs']} | Sup:{avg_sup:.6f} | Phy:{avg_phy:.6f} | Cons:{avg_cons:.6f} "
              f"| Total:{avg_total:.6f} | Val(EMA:{bool(use_ema)}):{avg_val:.6f} | LR:{cur_lr:.2e} "
              f"| Time:{time.time()-t0:.2f}s")

        if avg_val < best_val:
            best_val = avg_val
            out = os.path.join(ckpt_dir, "best_ema.pth" if (use_ema and CHECKPOINTS["save_best_ema"]) else "best.pth")
            torch.save((ema.eval_model().state_dict() if (use_ema and CHECKPOINTS["save_best_ema"]) else model.state_dict()), out)
            print("  >> New best model saved to", out)

    plt.figure()
    plt.plot(hist["train_sup"], label='Train Sup')
    plt.plot(hist["train_phy"], label='Train Phy')
    plt.plot(hist["train_cons"], label='Train Z-Cons')
    plt.plot(hist["train_total"], label='Train Total')
    plt.plot(hist["val"], label='Val (EMA)' if use_ema else 'Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    fig_path = os.path.join(ckpt_dir, 'loss_curve.png'); plt.savefig(fig_path, dpi=300); plt.close()

    with open(os.path.join(ckpt_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

    print(f"\nTraining finished. Best Val Loss: {best_val:.6f}")
    print("Curves saved to:", fig_path)

if __name__ == '__main__':
    main()
