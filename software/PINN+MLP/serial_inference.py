# -*- coding: utf-8 -*-
"""
serial_inference.py — 极低延迟 CPU 实时推理 + 多种输出管道（CSV/二进制/UDP/共享内存）

用法示例：
  # 仅做推理，不在控制台打印；把结果写入 UDP 与 CSV
  python serial_inference.py --port COM8 --baud 115200 --print-hz 0 --udp 127.0.0.1:5005 --csv out.csv

  # 仅用共享内存发布最新一帧（另一进程可 1kHz 轮询）
  python serial_inference.py --print-hz 0 --shm pinn_f_out --profile

  # 记录二进制日志（最快），后处理再读回
  python serial_inference.py --bin out.bin --print-hz 0
"""



import os
# —— 小模型单帧：多线程反而拖慢延迟；需在 import torch 之前设置 —— #
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import argparse
from collections import deque
from datetime import datetime
import struct
import socket

import numpy as np
import torch
import serial

# 限制 PyTorch 线程（进一步降低调度开销）
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ===== 模型/配置 =====
try:
    from config import MODEL  # 仅取结构，设备强制CPU
except Exception:
    MODEL = {
        "feature_dim": 12,
        "time_steps": 4,
        "xy_dim": 2,
        "force_dim": 3,
        "dropout_p": 0.15,
        "norm": "layer",
        "use_z_residual": True,
        "z_residual_scale": 1.0,
    }

from models.pinn_model import PINNForceModel

# ===== 路径与标定文件 =====
CKPT_DIR_MAIN   = "checkpoints"
CKPT_DIR_LEGACY = "data/checkpoints"
CKPT_CANDIDATES = ["best_ema.pth"]
X_MEAN_NAME, X_SCALE_NAME = "x_mean.npy", "x_scale.npy"
Y_MEAN_NAME, Y_SCALE_NAME = "y_mean.npy", "y_scale.npy"

# ===== 串口默认参数/平滑 =====
DEF_PORT = "COM4"
DEF_BAUD = 115200
DEF_PRINT_HZ = 0        # 默认不打印
DEF_TARGET_HZ = 0       # 0 不限速；>0 则轻节流推理

EMA_ALPHA_F = 0.2       # 极轻量 EMA
FZ_MEDIAN_K = 5         # strong 平滑时才启用
FZ_MA_K     = 9
Z_COLS = [2, 5, 8, 11]
Z_JITTER_STD = 0.01     # 不建议实时启用 jitter，会降速

# ===== sinks =====
class CsvSink:
    def __init__(self, path, flush_every=50):
        self.path = path
        self.flush_every = max(1, int(flush_every))
        self._buf = []  # list of tuples
        # 写表头（若新文件）
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("timestamp,x,y,Fx,Fy,Fz\n")
        # 以追加方式打开（大缓冲）
        self._fh = open(path, "a", encoding="utf-8", newline="")
    def write(self, ts, y5):
        self._buf.append((ts, *y5))
        if len(self._buf) >= self.flush_every:
            self.flush()
    def flush(self):
        if not self._buf:
            return
        self._fh.writelines(f"{t:.6f},{x:.6f},{y:.6f},{fx:.6f},{fy:.6f},{fz:.6f}\n"
                            for (t,x,y,fx,fy,fz) in self._buf)
        self._fh.flush()
        self._buf.clear()
    def close(self):
        self.flush()
        try: self._fh.close()
        except: pass

class BinSink:
    """二进制日志：<double timestamp, 5*float32> 按小端写入"""
    REC_FMT = "<d5f"
    REC_SIZE = struct.calcsize(REC_FMT)
    def __init__(self, path, flush_every=200):
        self.path = path
        self.flush_every = max(1, int(flush_every))
        self._fh = open(path, "ab", buffering=1024*1024)
        self._buf = bytearray()
    def write(self, ts, y5):
        self._buf += struct.pack(self.REC_FMT, float(ts),
                                 float(y5[0]), float(y5[1]), float(y5[2]), float(y5[3]), float(y5[4]))
        if len(self._buf) >= self.flush_every * self.REC_SIZE:
            self.flush()
    def flush(self):
        if not self._buf:
            return
        self._fh.write(self._buf)
        self._fh.flush()
        self._buf.clear()
    def close(self):
        self.flush()
        try: self._fh.close()
        except: pass

class UdpSink:
    """UDP 单播/广播：默认小端 <d5f>；另一端 struct.unpack('<d5f', data) 即得"""
    def __init__(self, addr):
        host, port = addr.split(":")
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pack = struct.Struct("<d5f").pack
    def write(self, ts, y5):
        self.sock.sendto(self.pack(float(ts),
                                   float(y5[0]), float(y5[1]), float(y5[2]), float(y5[3]), float(y5[4])),
                         self.addr)
    def close(self):
        try: self.sock.close()
        except: pass

class ShmSink:
    """共享内存：写入 <int64 seq, double ts, 5*float32>，另一进程可零拷贝读取最新值"""
    def __init__(self, name):
        from multiprocessing import shared_memory
        self.struct = struct.Struct("<q d 5f")  # 8 + 8 + 20 = 36B，凑整分配64B
        self.size = 64
        self.name = name
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.size)
        except FileExistsError:
            # 已存在则复用
            self.shm = shared_memory.SharedMemory(name=name, create=False, size=self.size)
        self.buf = self.shm.buf
        self.seq = 0
    def write(self, ts, y5):
        self.seq += 1
        self.buf[:self.struct.size] = self.struct.pack(self.seq, float(ts),
                                                       float(y5[0]), float(y5[1]), float(y5[2]), float(y5[3]), float(y5[4]))
    def close(self):
        try:
            self.shm.close()
        except: pass
        # 不 unlink，让读端持续使用。需要时手动清理：SharedMemory(name).unlink()

# ===== 其他工具 =====
def ensure_ckpt(file_name, main_dir=CKPT_DIR_MAIN, legacy_dir=CKPT_DIR_LEGACY):
    p1 = os.path.join(main_dir, file_name)
    if os.path.exists(p1): return p1
    p2 = os.path.join(legacy_dir, file_name)
    if os.path.exists(p2): return p2
    raise FileNotFoundError(f"找不到 {file_name} 于 {main_dir} 或 {legacy_dir}")

def find_model_ckpt():
    for name in CKPT_CANDIDATES:
        try:
            return ensure_ckpt(name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("未找到 best_ema.pth 或 best.pth")

def load_scaler(name_mean, name_scale):
    mean = np.load(ensure_ckpt(name_mean)).astype(np.float32)
    scale = np.load(ensure_ckpt(name_scale)).astype(np.float32)
    return mean, scale

def parse_numbers_fast(line: str):
    return np.fromstring(line.replace(',', ' '), sep=' ', dtype=np.float32)

class EMA:
    def __init__(self, alpha=0.2, dim=3):
        self.alpha = float(alpha); self.y = None; self.dim = dim
    def reset(self): self.y = None
    def step(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.y is None: self.y = x.copy()
        else: self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y.copy()

def medfilt1d(a: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k < 3: return a.copy()
    if k % 2 == 0: k += 1
    n = a.shape[0]
    if n < k: return a.copy()
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(a)
    for i in range(n):
        out[i] = np.median(ap[i:i+k], axis=0)
    return out

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEF_PORT)
    ap.add_argument("--baud", type=int, default=DEF_BAUD)
    ap.add_argument("--print-hz", type=int, default=DEF_PRINT_HZ, help="打印频率上限(0=不打印)")
    ap.add_argument("--target-hz", type=int, default=DEF_TARGET_HZ, help="推理限速(0=不节流)")
    ap.add_argument("--smooth", choices=["off","ema","strong"], default="ema",
                    help="平滑级别：off/ema/strong(中值->均值)")
    ap.add_argument("--jitter-k", type=int, default=0, help=">0 启用Z抖动小集成（会降速）")
    ap.add_argument("--map", default="", help="列重排，如 '0,1,2,...,11'")
    ap.add_argument("--profile", action="store_true", help="每200帧打印耗时")

    # 输出管道（可多选）
    ap.add_argument("--csv", default="", help="把结果写入 CSV 文件")
    ap.add_argument("--bin", default="", help="把结果写入二进制日志")
    ap.add_argument("--udp", default="", help="把结果通过 UDP 发送，格式 host:port")
    ap.add_argument("--shm", default="", help="共享内存名，发布最新一帧 <seq,ts,5f>")
    ap.add_argument("--flush-every", type=int, default=50, help="CSV/BIN 每多少帧刷新一次")
    return ap

def parse_map(mapping_str, F):
    if not mapping_str: return None
    idx = [int(i) for i in mapping_str.split(",")]
    assert len(idx) == F, f"--map 长度需等于 {F}"
    return np.array(idx, dtype=np.int64)

# ---------- 主逻辑 ----------
def main():
    args = build_argparser().parse_args()

    # —— 强制 CPU —— #
    device = torch.device("cpu")
    print(f"[INFO] Inference device: CPU (forced), print-hz={args.print_hz}, target-hz={args.target_hz}")

    T = int(MODEL.get("time_steps", 4))
    F = int(MODEL.get("feature_dim", 12))
    flat_dim = T * F

    # 模型
    model = PINNForceModel(
        feature_dim=F, time_steps=T,
        xy_dim=MODEL.get("xy_dim", 2), force_dim=MODEL.get("force_dim", 3),
        dropout_p=MODEL.get("dropout_p", 0.15), norm=MODEL.get("norm", "layer"),
        use_z_residual=MODEL.get("use_z_residual", True),
        z_residual_scale=MODEL.get("z_residual_scale", 1.0),
    ).to(device)
    state = torch.load(find_model_ckpt(), map_location=device)
    model.load_state_dict(state); model.eval()

    # 标定
    x_mean, x_scale = load_scaler(X_MEAN_NAME, X_SCALE_NAME)
    y_mean, y_scale = load_scaler(Y_MEAN_NAME, Y_SCALE_NAME)
    assert x_mean.shape[0] == flat_dim and x_scale.shape[0] == flat_dim
    assert y_mean.shape[0] == 5 and y_scale.shape[0] == 5

    # —— 预分配：环形缓冲（numpy）+ 零拷贝张量 —— #
    ring = np.zeros((T, F), dtype=np.float32)
    ring_ptr = 0; filled = 0
    x_flat = np.empty((flat_dim,), dtype=np.float32)      # 每帧复用
    x_torch = torch.from_numpy(x_flat).view(1, T, F)      # 常驻零拷贝视图（CPU）
    ema_f = EMA(alpha=EMA_ALPHA_F, dim=3)                 # 持久 EMA
    fz_hist = deque(maxlen=max(FZ_MA_K, FZ_MEDIAN_K))
    map_idx = parse_map(args.map, F)

    # 串口（短超时，减少阻塞）
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.01, inter_byte_timeout=0.005)
        ser.reset_input_buffer()
        print(f"[INFO] Serial opened: {args.port} @ {args.baud}")
    except Exception as e:
        print(f"[ERROR] Open serial failed: {e}")
        return

    # 输出管道
    sinks = []
    try:
        if args.csv: sinks.append(CsvSink(args.csv, flush_every=args.flush_every))
        if args.bin: sinks.append(BinSink(args.bin, flush_every=args.flush_every))
        if args.udp: sinks.append(UdpSink(args.udp))
        if args.shm: sinks.append(ShmSink(args.shm))
        if not sinks and args.print_hz == 0:
            print("[WARN] 未选择任何输出管道且 print-hz=0：推理进行中，但不会有可见输出。")
    except Exception as e:
        print(f"[ERROR] 初始化输出管道失败: {e}")
        for s in sinks:
            try: s.close()
            except: pass
        return

    # 打印/推理节流
    print_dt  = 1.0 / max(1, int(args.print_hz)) if args.print_hz > 0 else 0.0
    last_print = 0.0
    infer_dt = 1.0 / args.target_hz if args.target_hz and args.target_hz > 0 else 0.0
    last_infer = 0.0

    # 性能剖析
    rd_t=pa_t=infer_t=sm_t=pr_t=0.0; frames=0; t0=time.time()

    print("[INFO] streaming... Ctrl+C to stop.")
    try:
        while True:
            # 读一行
            t_a = time.perf_counter()
            raw = ser.readline()
            rd_t += time.perf_counter() - t_a
            if not raw:
                continue

            # 解析
            line = raw.decode("utf-8", errors="ignore").strip()
            t_b = time.perf_counter()
            nums = parse_numbers_fast(line)
            pa_t += time.perf_counter() - t_b
            if nums.shape[0] < F:
                continue
            frame = nums[-F:]
            if map_idx is not None:
                frame = frame[map_idx]

            # 写入环形缓冲
            ring[ring_ptr] = frame
            ring_ptr = (ring_ptr + 1) % T
            filled = min(T, filled + 1)
            if filled < T:
                continue

            # 推理限速（可选）
            if infer_dt > 0.0:
                now = time.perf_counter()
                if now - last_infer < infer_dt:
                    continue
                last_infer = now

            # 将环形缓冲按时间顺序摊平成 x_flat（复用内存）
            head = T - ring_ptr
            if ring_ptr == 0:
                x_flat[:] = ring.ravel()
            else:
                x_flat[:head*F] = ring[ring_ptr:].ravel()
                x_flat[head*F:] = ring[:ring_ptr].ravel()

            # 标准化（就地）
            np.subtract(x_flat, x_mean, out=x_flat)
            np.divide(x_flat, x_scale, out=x_flat)

            # —— 零拷贝：直接用 x_torch 做前向（共享 x_flat 内存）——
            t_c = time.perf_counter()
            with torch.inference_mode():
                y_pred_norm = model(x_torch).squeeze().numpy(force=True)
            infer_t += time.perf_counter() - t_c

            # 反标定
            y_pred = y_pred_norm * y_scale + y_mean  # (5,)
            ts = time.time()

            # 平滑
            t_d = time.perf_counter()
            if args.smooth != "off":
                y_pred[2:5] = ema_f.step(y_pred[2:5])  # EMA 几乎零成本
                if args.smooth == "strong":
                    # 仅在需要时才做中值->均值（小窗口，额外开销很小）
                    fz_hist.append(y_pred[4])
                    if len(fz_hist) >= 3:
                        fz_arr = np.array(fz_hist, dtype=np.float32).reshape(-1,1)
                        # 中值
                        k_med = min(FZ_MEDIAN_K, len(fz_hist))
                        if k_med >= 3:
                            pad = k_med // 2
                            ap = np.pad(fz_arr, ((pad,pad),(0,0)), mode="edge")
                            fz_med = np.median(ap[-(k_med+pad*2):], axis=0)[0]  # 只算末段
                        else:
                            fz_med = fz_arr[-1,0]
                        # 均值
                        k_ma = min(FZ_MA_K, len(fz_hist))
                        if k_ma >= 2:
                            fz_ma = np.mean(fz_arr[-k_ma:, 0])
                        else:
                            fz_ma = fz_arr[-1,0]
                        y_pred[4] = 0.5 * fz_med + 0.5 * fz_ma
            sm_t += time.perf_counter() - t_d

            # 输出到各管道（极轻）
            for s in sinks:
                s.write(ts, y_pred)

            # 可选打印（节流）
            t_e = time.perf_counter()
            if print_dt > 0.0:
                now = time.time()
                if now - last_print >= print_dt:
                    #ts_s = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]
                    dt = datetime.fromtimestamp(ts)
                    ts_s = f"{dt.hour}:{dt.strftime('%M:%S')}.{int(ts*1000)%1000:03d}"
                    vals = [f"{v:.3f}" for v in y_pred.tolist()]
                    print(f"[{ts_s}] PRED  x={vals[0]}  y={vals[1]}  Fx={vals[2]}  Fy={vals[3]}  Fz={vals[4]}")
                    last_print = now
            pr_t += time.perf_counter() - t_e

            # 统计
            frames += 1
            if args.profile and frames % 200 == 0:
                dt = time.time() - t0
                hz = frames / dt if dt > 0 else 0.0
                print(f"[PROF] {hz:.1f} Hz | read:{rd_t:.3f}s parse:{pa_t:.3f}s "
                      f"infer:{infer_t:.3f}s smooth:{sm_t:.3f}s print:{pr_t:.3f}s / {frames} frames")
                rd_t=pa_t=infer_t=sm_t=pr_t=0.0; frames=0; t0=time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except serial.SerialException as e:
        print(f"[ERROR] Serial exception: {e}")
    finally:
        # 关闭 sinks
        for s in sinks:
            try: s.close()
            except: pass
        try:
            ser.close()
        except Exception:
            pass
        print("[INFO] Serial closed.")

if __name__ == "__main__":
    main()
