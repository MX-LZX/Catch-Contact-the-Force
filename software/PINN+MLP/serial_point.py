import os
import re
import time
import serial
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from models.pinn_model import PINNForceModel

# 参数配置
SEQ_LEN        = 4
FEATURE_DIM    = 8
OUTPUT_DIM     = 5
SERIAL_PORT    = 'COM8'
BAUD_RATE      = 115200
SAMPLE_INTERVAL = 1.0 / 100

MODEL_PATH     = 'checkpoints/best.pth'
X_MEAN_PATH    = 'data/checkpoints/x_mean.npy'
X_SCALE_PATH   = 'data/checkpoints/x_scale.npy'
Y_MEAN_PATH    = 'data/checkpoints/y_mean.npy'
Y_SCALE_PATH   = 'data/checkpoints/y_scale.npy'

def load_scaler(mean_path, scale_path):
    mean = np.load(mean_path).astype(np.float32)
    scale = np.load(scale_path).astype(np.float32)
    return mean, scale

def prepare_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINNForceModel(feature_dim=FEATURE_DIM, time_steps=SEQ_LEN).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

def main():
    model, device = prepare_model()
    x_mean, x_scale = load_scaler(X_MEAN_PATH, X_SCALE_PATH)
    y_mean, y_scale = load_scaler(Y_MEAN_PATH, Y_SCALE_PATH)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"[INFO] Serial port {SERIAL_PORT} opened.")
    except Exception as e:
        print(f"[ERROR] Cannot open serial port: {e}")
        return

    buffer = []

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title("Force Visualization (High Speed)")

    circle = Circle((0, 0), radius=0.3, edgecolor='b', facecolor='none')
    ax.add_patch(circle)
    arrow_fx = Line2D([], [], color='r')
    arrow_fy = Line2D([], [], color='g')
    arrow_fz = Line2D([], [], color='b')
    ax.add_line(arrow_fx)
    ax.add_line(arrow_fy)
    ax.add_line(arrow_fz)

    def update(frame):
        while ser.in_waiting:
            raw = ser.readline()
            try:
                line = raw.decode('utf-8', errors='ignore').strip()
                parts = re.split(r'[,\s]+', line)
                if len(parts) != FEATURE_DIM:
                    continue
                data = [float(p) for p in parts]
            except:
                continue

            buffer.append(data)
            if len(buffer) > SEQ_LEN:
                buffer.pop(0)

        if len(buffer) < SEQ_LEN:
            return

        x = np.array(buffer, dtype=np.float32).reshape(-1)
        x_norm = (x - x_mean) / x_scale
        x_tensor = torch.from_numpy(x_norm).view(1, SEQ_LEN, FEATURE_DIM).to(device)

        with torch.no_grad():
            y_pred_norm = model(x_tensor).cpu().numpy().squeeze()

        y_real = y_pred_norm * y_scale + y_mean
        x_pos, y_pos, fx, fy, fz = y_real.tolist()

        circle.set_center((x_pos, y_pos))
        scale_arrow = 0.3
        arrow_fx.set_data([x_pos, x_pos + fx * scale_arrow], [y_pos, y_pos])
        arrow_fy.set_data([x_pos, x_pos], [y_pos, y_pos + fy * scale_arrow])
        arrow_fz.set_data([x_pos, x_pos + fz * scale_arrow], [y_pos, y_pos + fz * scale_arrow])

    ani = FuncAnimation(fig, update, interval=1)
    plt.show()

    ser.close()
    print("[INFO] Serial port closed.")

if __name__ == '__main__':
    main()