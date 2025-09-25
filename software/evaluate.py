# evaluate.py — safe excel save, EMA-aware, CLI, diagnostics
import os, argparse, time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from models.pinn_model import PINNForceModel
from config import DEVICE, MODEL

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='data/inputs.npy', help='(N,48) or (N,4,12)')
    ap.add_argument('--targets', default='data/targets.npy', help='(N,5)')
    ap.add_argument('--ckpt-dir', default='checkpoints')
    ap.add_argument('--legacy-ckpt-dir', default='data/checkpoints')
    ap.add_argument('--out', default='pred_results')  # 输出文件前缀
    ap.add_argument('--seq', action='store_true', help='use [N,4,12] as input')
    ap.add_argument('--window', default='3,3,5,5,9', help='smooth window. int or "w0,w1,w2,w3,w4"')
    ap.add_argument('--no-plot', action='store_true')
    return ap.parse_args()

def parse_window(s):
    if ',' in s:
        nums = [int(x) for x in s.split(',')]
        if len(nums) != 5:
            raise ValueError('window list must have 5 ints')
        return {i: w for i, w in enumerate(nums)}
    else:
        return int(s)

def ensure_ckpt(file_name, main_dir, legacy_dir):
    p1 = os.path.join(main_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(legacy_dir, file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(f"Missing {file_name} in {main_dir} or {legacy_dir}")

def find_model_ckpt(ckpt_dir, legacy_ckpt_dir):
    for name in ['best_ema.pth','best.pth']:
        try:
            return ensure_ckpt(name, ckpt_dir, legacy_ckpt_dir)
        except FileNotFoundError:
            continue
    raise FileNotFoundError('No best_ema.pth or best.pth found')

def moving_average_1d(a: np.ndarray, n: int) -> np.ndarray:
    if n is None or n <= 1 or a.shape[0] < n: return a.copy()
    c = np.cumsum(a, axis=0, dtype=float)
    c[n:] = c[n:] - c[:-n]
    return c[n-1:] / n

def apply_smoothing(Y: np.ndarray, windows) -> (np.ndarray, int):
    if isinstance(windows, int):
        w = max(1, windows)
        return moving_average_1d(Y, w), w-1
    # dict per column
    w_by_col = [int(windows.get(i,1)) for i in range(Y.shape[1])]
    w_max = max(1, max(w_by_col))
    start = w_max - 1
    Y_s = np.zeros((Y.shape[0]-start, Y.shape[1]), dtype=Y.dtype)
    for i in range(Y.shape[1]):
        wi = max(1, w_by_col[i])
        Yi = moving_average_1d(Y[:, [i]], wi)
        need = Y.shape[0] - start
        Y_s[:, i] = Yi[-need:, 0]
    return Y_s, start

def medfilt1d(a: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k));  k = k if k % 2 == 1 else k + 1
    n = a.shape[0]
    if n < k: return a.copy()
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0,0)), mode='edge')  # [N+2p, C]
    out = np.empty_like(a)
    for i in range(n):
        window = ap[i:i+k]
        out[i] = np.median(window, axis=0)
    return out

def smooth_combo(Y: np.ndarray, median_k=5, ma_k=9):
    # 先中值（去毛刺保峰），再小窗口均值（压微抖）
    Y1 = medfilt1d(Y, median_k)
    Y2 = moving_average_1d(Y1, ma_k)
    # 对齐起点
    start = (median_k//2) + (ma_k-1)
    return Y2, start


def mae_rmse(y_true, y_pred):
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return mae, rmse

def pearson_r(y_true, y_pred):
    r = []
    for i in range(y_true.shape[1]):
        a = y_true[:, i] - y_true[:, i].mean()
        b = y_pred[:, i] - y_pred[:, i].mean()
        denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()))
        r.append(float((a*b).sum()/denom) if denom>0 else 0.0)
    return np.array(r)

def r2_score(y_true, y_pred):
    r2 = []
    for i in range(y_true.shape[1]):
        ss_res = ((y_true[:,i]-y_pred[:,i])**2).sum()
        ss_tot = ((y_true[:,i]-y_true[:,i].mean())**2).sum()
        r2.append(float(1 - ss_res/ss_tot) if ss_tot>0 else 0.0)
    return np.array(r2)

def safe_save_excel(df, path_base):
    xlsx = f'{path_base}.xlsx'
    try:
        df.to_excel(xlsx, index=False)
        print(f"[INFO] Saved Excel to {xlsx}")
    except PermissionError:
        ts = time.strftime('%Y%m%d_%H%M%S')
        xlsx_ts = f'{path_base}_{ts}.xlsx'
        df.to_excel(xlsx_ts, index=False)
        print(f"[WARN] '{xlsx}' is locked by another program. Saved to '{xlsx_ts}' instead.")

def main():
    args = parse_args()
    names = ['x','y','Fx','Fy','Fz']
    windows = parse_window(args.window)

    # load arrays
    X_raw = np.load(args.inputs).astype(np.float32)
    Y_raw = np.load(args.targets).astype(np.float32)
    if X_raw.ndim == 2 and X_raw.shape[1] == 48:
        pass
    elif X_raw.ndim == 3 and X_raw.shape[1:] == (MODEL["time_steps"], MODEL["feature_dim"]):
        X_raw = X_raw.reshape(X_raw.shape[0], -1).astype(np.float32)
    else:
        raise ValueError(f"inputs shape must be (N,48) or (N,{MODEL['time_steps']},{MODEL['feature_dim']}), got {X_raw.shape}")
    assert Y_raw.shape[1] == 5

    # load scalers
    x_mean  = np.load(ensure_ckpt('x_mean.npy', args.ckpt_dir, args.legacy_ckpt_dir))
    x_scale = np.load(ensure_ckpt('x_scale.npy', args.ckpt_dir, args.legacy_ckpt_dir))
    y_mean  = np.load(ensure_ckpt('y_mean.npy', args.ckpt_dir, args.legacy_ckpt_dir))
    y_scale = np.load(ensure_ckpt('y_scale.npy', args.ckpt_dir, args.legacy_ckpt_dir))
    print("[INFO] y_scale:", np.round(y_scale, 4))
    print("[INFO] y_mean :", np.round(y_mean, 4))

    # normalize
    X_norm = (X_raw - x_mean) / x_scale

    # model
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    model = PINNForceModel(
        feature_dim=MODEL["feature_dim"], time_steps=MODEL["time_steps"],
        xy_dim=MODEL["xy_dim"], force_dim=MODEL["force_dim"],
        dropout_p=MODEL["dropout_p"], norm=MODEL["norm"]
    ).to(device)
    ckpt_path = find_model_ckpt(args.ckpt_dir, args.legacy_ckpt_dir)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state); model.eval()
    print(f"[INFO] Loaded model weights from: {ckpt_path}")

    # inference
    with torch.no_grad():
        if args.seq:
            X_seq = X_norm.reshape(-1, MODEL["time_steps"], MODEL["feature_dim"])
            X_tensor = torch.from_numpy(X_seq).float().to(device)
        else:
            X_tensor = torch.from_numpy(X_norm).float().to(device)
        Y_pred_norm = model(X_tensor).cpu().numpy()
    Y_pred = Y_pred_norm * y_scale + y_mean

    # metrics raw
    mae0, rmse0 = mae_rmse(Y_raw, Y_pred)
    r_raw = pearson_r(Y_raw, Y_pred)
    r2_raw = r2_score(Y_raw, Y_pred)
    print("\n[METRIC] Raw predictions:")
    for i,n in enumerate(names):
        print(f"  {n:<2} | MAE={mae0[i]:.4f} | RMSE={rmse0[i]:.4f} | r={r_raw[i]:.3f} | R2={r2_raw[i]:.3f}")

    # smoothing
    Y_pred_smooth, start = apply_smoothing(Y_pred, windows)
    Y_true_aligned = Y_raw[start:]
    mae1, rmse1 = mae_rmse(Y_true_aligned, Y_pred_smooth)
    r_sm = pearson_r(Y_true_aligned, Y_pred_smooth)
    r2_sm = r2_score(Y_true_aligned, Y_pred_smooth)
    print("\n[METRIC] Smoothed (aligned):")
    for i,n in enumerate(names):
        print(f"  {n:<2} | MAE={mae1[i]:.4f} | RMSE={rmse1[i]:.4f} | r={r_sm[i]:.3f} | R2={r2_sm[i]:.3f}")

    # export
    cols_true   = [f'True_{i+1}' for i in range(5)]
    cols_pred   = [f'Pred_{i+1}' for i in range(5)]
    cols_smooth = [f'Pred_smooth_{i+1}' for i in range(5)]
    df_true_pred = pd.DataFrame(np.hstack([Y_true_aligned, Y_pred[start:]]), columns=cols_true + cols_pred)
    df_smooth = pd.DataFrame(Y_pred_smooth, columns=cols_smooth)
    df_out = pd.concat([df_true_pred, df_smooth], axis=1)

    # CSV always ok
    csv_path = f'{args.out}.csv'
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] Saved CSV to {csv_path}")

    # Excel: safe save (handle PermissionError)
    safe_save_excel(df_out, args.out)

    # optional plot
    if not args.no_plot:
        try:
            t = np.arange(Y_true_aligned.shape[0])
            plt.figure(figsize=(10,4))
            plt.plot(t, Y_true_aligned[:,4], label='True Fz', linewidth=1.2)
            plt.plot(t, Y_pred[start:,4],     label='Pred Fz', linewidth=1.0)
            plt.plot(t, Y_pred_smooth[:,4],   label='Smooth Fz', linewidth=1.2)
            plt.xlabel('t'); plt.ylabel('Fz'); plt.grid(True); plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(args.ckpt-dir if False else '', f'{args.out}_Fz.png')
            plt.savefig(plot_path, dpi=300); plt.close()
            print(f"[INFO] Plot saved to {plot_path}")
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")

    print(f"[INFO] X_raw: {X_raw.shape}, Y_raw: {Y_raw.shape}, USE_SEQ_IN={args.seq}, window={args.window}, start={start}")

if __name__ == '__main__':
    main()
