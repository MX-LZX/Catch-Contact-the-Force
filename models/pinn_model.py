import torch
import torch.nn as nn

def make_norm1d(kind: str, dim: int):
    k = (kind or "batch").lower()
    if k == "batch":  return nn.BatchNorm1d(dim)
    if k == "layer":  return nn.LayerNorm(dim)
    return nn.Identity()

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_p=0.2, norm="batch"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim); self.n1 = make_norm1d(norm, dim)
        self.act = nn.LeakyReLU(0.2, inplace=True); self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(dim, dim); self.n2 = make_norm1d(norm, dim)
    def forward(self, x):
        out = self.fc1(x); out = self.n1(out); out = self.act(out); out = self.drop(out)
        out = self.fc2(out); out = self.n2(out)
        return self.act(x + out)

class PINNForceModel(nn.Module):
    def __init__(self, feature_dim=12, time_steps=4, xy_dim=2, force_dim=3,
                 dropout_p=0.2, norm="layer",
                 use_z_residual=True, z_residual_scale=1.0):
        super().__init__()
        self.feature_dim = feature_dim; self.time_steps = time_steps
        self.use_z_residual = use_z_residual
        self.z_residual_scale = z_residual_scale

        flat_dim = feature_dim * time_steps
        self.fc_in = nn.Linear(flat_dim, 512); self.n_in = make_norm1d(norm, 512)
        self.act = nn.LeakyReLU(0.2, inplace=True); self.drop0 = nn.Dropout(dropout_p)
        self.res1 = ResidualBlock(512, dropout_p=dropout_p, norm=norm)
        self.res2 = ResidualBlock(512, dropout_p=dropout_p, norm=norm)

        self.xy_head = nn.Sequential(
            nn.Linear(512, 256), make_norm1d(norm, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p), nn.Linear(256, xy_dim)
        )
        self.f_head = nn.Sequential(
            nn.Linear(512, 256), make_norm1d(norm, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_p), ResidualBlock(256, dropout_p=dropout_p, norm=norm),
            nn.Linear(256, force_dim)
        )

        # —— Z 专家支路：仅用每帧的 Z 列（索引 2,5,8,11）——
        if self.use_z_residual:
            z_in = 4 * time_steps  # 每帧 4 个 Hz，共 T 帧
            self.z_residual_head = nn.Sequential(
                nn.Linear(z_in, 64), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, 32),  nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, 1)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x_is_seq = (x.dim() == 3)
        if x_is_seq:
            B, T, F = x.shape
            assert T == self.time_steps and F == self.feature_dim
            x_flat = x.reshape(B, -1)
        elif x.dim() == 2:
            B = x.size(0); T = self.time_steps; F = self.feature_dim
            x_flat = x
            x = x.reshape(B, T, F)  # 便于抽取 Z 通道
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}")

        h = self.fc_in(x_flat); h = self.n_in(h); h = self.act(h); h = self.drop0(h)
        h = self.res1(h); h = self.res2(h)
        xy = self.xy_head(h)
        f  = self.f_head(h)   # [B, 3] -> (Fx,Fy,Fz_base)

        if self.use_z_residual:
            # 只取每帧的 Z 列（2,5,8,11），沿时间拼在一起
            Z_idx = [2, 5, 8, 11]
            z_seq = x[:, :, Z_idx].reshape(B, -1)  # [B, T*4]
            z_res = self.z_residual_head(z_seq).squeeze(1)  # [B]
            f[:, 2] = f[:, 2] + self.z_residual_scale * z_res  # 叠加到 Fz

        return torch.cat([xy, f], dim=1)
