import os
import numpy as np
import torch
import pandas as pd
from models.pinn_model import PINNForceMode

# ==== 配置 ====
INPUT_PATH   = 'data/inputs_raw.npy'
TARGET_PATH  = 'data/targets.npy'      # 只用来展示真实值，无需标准化
CKPT_MODEL   = 'checkpoints/best.pth'
CKPT_X_MEAN  = 'checkpoints/x_mean.npy'
CKPT_X_SCALE = 'checkpoints/x_scale.npy'
CKPT_Y_MEAN  = 'checkpoints/y_mean.npy'
CKPT_Y_SCALE = 'checkpoints/y_scale.npy'
OUTPUT_XLS   = 'pred_results.xlsx'
WINDOW       = 5

# ==== 加载原始数据 ====
X_raw = np.load(INPUT_PATH).astype(np.float32)    # shape (N,8)
Y_raw = np.load(TARGET_PATH).astype(np.float32)   # shape (N,5)

# ==== 载入标准化参数 ====
x_mean  = np.load(CKPT_X_MEAN)
x_scale = np.load(CKPT_X_SCALE)
y_mean  = np.load(CKPT_Y_MEAN)
y_scale = np.load(CKPT_Y_SCALE)

# ==== 标准化输入 ====
X_norm = (X_raw - x_mean) / x_scale

# ==== 加载模型 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PINNForceMode(input_dim=8, output_dim=5).to(device)
checkpoint = torch.load(CKPT_MODEL, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# ==== 推理（标准化空间） ====
with torch.no_grad():
    X_tensor = torch.from_numpy(X_norm).float().to(device)
    Y_pred_norm = model(X_tensor).cpu().numpy()   # shape (N,5)

# ==== 反标准化预测输出 ====
Y_pred = Y_pred_norm * y_scale + y_mean        # shape (N,5)

# ==== （可选）滑动平均平滑 ====
def moving_average(a, n=WINDOW):
    c = np.cumsum(a, axis=0, dtype=float)
    c[n:] = c[n:] - c[:-n]
    return c[n-1:] / n

Y_pred_smooth = moving_average(Y_pred)

# ==== 拼表并保存 Excel ====
# 真实值 + 原始预测
df = pd.DataFrame({
    **{f'True_{i+1}': Y_raw[:, i]      for i in range(5)},
    **{f'Pred_{i+1}': Y_pred[:, i]     for i in range(5)},
})

# 平滑后的预测
df_s = pd.DataFrame(Y_pred_smooth, columns=[f'Pred_smooth_{i+1}' for i in range(5)])

# 对齐并合并
df = pd.concat([df.iloc[WINDOW-1:].reset_index(drop=True), df_s], axis=1)
df.to_excel(OUTPUT_XLS, index=False)

print(f"[INFO] Evaluation complete. Results saved to {OUTPUT_XLS}")
