import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class HallTimeDataset(Dataset):
    """
    时序数据集（48维 = 4帧 × 每帧12通道(HX/HY/HZ × 4传感器)）：
    - 加载 inputs.npy (N, 48) 和 targets.npy (N, 5)
    - 标准化输入和标签
    - __getitem__ 返回：
        x_seq:  [4,12]  时序展开 (time_steps, features_per_step)
        x_flat: [48]    扁平输入
        y:      [5]     目标输出
    """
    def __init__(self, data_dir, feature_dim=12, time_steps=4, ckpt_dir_rel='../checkpoints'):
        super().__init__()
        self.feature_dim = feature_dim   # 每帧 12 列
        self.time_steps  = time_steps    # 帧数 4

        inputs_path = os.path.join(data_dir, 'inputs.npy')   # (N, 48)
        targets_path = os.path.join(data_dir, 'targets.npy') # (N, 5)

        X = np.load(inputs_path).astype(np.float32)
        Y = np.load(targets_path).astype(np.float32)

        # 标准化
        self.X_scaler = StandardScaler().fit(X)
        self.Y_scaler = StandardScaler().fit(Y)
        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.Y_scaler.transform(Y)

        # 保存归一化参数到项目根下的 checkpoints（相对 data/）
        ckpt_dir = os.path.normpath(os.path.join(data_dir, ckpt_dir_rel))
        os.makedirs(ckpt_dir, exist_ok=True)
        np.save(os.path.join(ckpt_dir, 'x_mean.npy'),  self.X_scaler.mean_)
        np.save(os.path.join(ckpt_dir, 'x_scale.npy'), self.X_scaler.scale_)
        np.save(os.path.join(ckpt_dir, 'y_mean.npy'),  self.Y_scaler.mean_)
        np.save(os.path.join(ckpt_dir, 'y_scale.npy'), self.Y_scaler.scale_)

        # 转 torch Tensor
        self.X = torch.from_numpy(X_scaled)  # [N,48]
        self.Y = torch.from_numpy(Y_scaled)  # [N,5]

    def __len__(self):
        return self.X.shape[0]   # 确保返回 int

    def __getitem__(self, idx):
        x_flat = self.X[idx]                                   # [48]
        x_seq  = x_flat.view(self.time_steps, self.feature_dim) # [4,12]
        y      = self.Y[idx]                                    # [5]
        return x_seq, x_flat, y

    # 供训练时拆分 X/Y/Z：返回三个索引列表（与表头/列顺序严格一致）
    @staticmethod
    def xyz_indices(time_steps=4):
        # 每个时刻块 12 列：X列(0,3,6,9), Y列(1,4,7,10), Z列(2,5,8,11)
        X_IDX = [12*b + off for b in range(time_steps) for off in (0, 3, 6, 9)]
        Y_IDX = [12*b + off for b in range(time_steps) for off in (1, 4, 7, 10)]
        Z_IDX = [12*b + off for b in range(time_steps) for off in (2, 5, 8, 11)]
        return X_IDX, Y_IDX, Z_IDX
