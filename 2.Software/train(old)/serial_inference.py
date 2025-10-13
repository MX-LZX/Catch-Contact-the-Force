#!/usr/bin/env python3
# File: serial_inference.py

import os
import re
import time
import serial
import torch
import numpy as np
from models.pinn_model import PINNForceMode  # 确保在 models/pinn_model.py 中
from config import BATCH_SIZE  # 如果不使用可删除

# 输入/输出维度
INPUT_DIM  = 8
OUTPUT_DIM = 5

# 串口配置
SERIAL_PORT = 'COM10'      # Windows 示例；Linux 上可能是 '/dev/ttyUSB0'
BAUD_RATE   = 115200

# 模型与 scaler 路径
MODEL_PATH     = 'checkpoints/best.pth'
X_MEAN_PATH    = 'checkpoints/x_mean.npy'
X_SCALE_PATH   = 'checkpoints/x_scale.npy'
Y_MEAN_PATH    = 'checkpoints/y_mean.npy'
Y_SCALE_PATH   = 'checkpoints/y_scale.npy'

def load_scaler(mean_path, scale_path):
    """加载 scaler（mean, scale）"""
    mean  = np.load(mean_path).astype(np.float32)
    scale = np.load(scale_path).astype(np.float32)
    return mean, scale

def main():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型
    model = PINNForceMode(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 加载标准化参数
    x_mean, x_scale = load_scaler(X_MEAN_PATH, X_SCALE_PATH)
    y_mean, y_scale = load_scaler(Y_MEAN_PATH, Y_SCALE_PATH)

    # 3. 打开串口
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Serial port {SERIAL_PORT} opened at {BAUD_RATE} baud.")
    except Exception as e:
        print(f"[ERROR] Failed to open serial port: {e}")
        return

    print("[INFO] Waiting for data... Press Ctrl+C to exit.")

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode('utf-8', errors='ignore').strip()
            parts = re.split(r'[,\s]+', line)

            if len(parts) != INPUT_DIM:
                print(f"[WARN] Invalid input length: {line}")
                continue

            # 转浮点
            try:
                x = np.array([float(p) for p in parts], dtype=np.float32)
            except ValueError:
                print(f"[WARN] Non-numeric input: {line}")
                continue

            # 4. 输入标准化
            x_norm = (x - x_mean) / x_scale

            # 5. 模型推理
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                y_pred_norm = model(x_tensor).cpu().numpy().squeeze()  # 标准化后的输出

            # 6. 输出反归一化
            y_real = y_pred_norm * y_scale + y_mean

            # 7. 打印结果（保留3位小数）
            #print(f"[INPUT] { [f'{v:.3f}' for v in x.tolist()] }")
            print(f"[PRED]  { [f'{v:.3f}' for v in y_real.tolist()] }")
            #print("-" * 50)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        ser.close()
        print("[INFO] Serial port closed.")

if __name__ == '__main__':
    main()
