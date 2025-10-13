import torch
import torch.nn as nn

mse = nn.MSELoss()

def _time_diff(x):
    return x[:, 1:, :] - x[:, :-1, :]

def _time_diff2(x):
    return _time_diff(_time_diff(x))

def split_xyz_from_flat_or_seq(x, time_steps=4):
    if x.dim() == 2:
        B = x.size(0)
        x = x.view(B, time_steps, 12)
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(f"unexpected shape {x.shape}")

    X_cols = [0, 3, 6, 9]
    Y_cols = [1, 4, 7, 10]
    Z_cols = [2, 5, 8, 11]

    Hx_seq = x[:, :, X_cols]
    Hy_seq = x[:, :, Y_cols]
    Hz_seq = x[:, :, Z_cols]
    return Hx_seq, Hy_seq, Hz_seq

def directional_consistency_loss_axis(seq, eps=1e-4):
    delta = _time_diff(seq)
    prod  = delta[:, 1:, :] * delta[:, :-1, :]
    return torch.relu(-(prod + eps)).mean()

def directional_consistency_loss_xyz(x, time_steps=4, eps=1e-4):
    Hx, Hy, Hz = split_xyz_from_flat_or_seq(x, time_steps)
    Lx = directional_consistency_loss_axis(Hx, eps=eps)
    Ly = directional_consistency_loss_axis(Hy, eps=eps)
    Lz = directional_consistency_loss_axis(Hz, eps=eps)
    return Lx, Ly, Lz, (Lx + Ly + Lz)

def temporal_rate_similarity_loss_axis(seq, edges=None):
    if edges is None:
        edges = [(0,1),(2,3),(0,2),(1,3)]
    delta = _time_diff(seq)
    diffs = [(delta[:,:,i] - delta[:,:,j])**2 for i,j in edges]
    return torch.stack(diffs, dim=0).mean() if diffs else delta.new_tensor(0.0)

def temporal_rate_similarity_loss_xyz(x, time_steps=4, edges=None):
    # 只对 Z
    _, _, Hz = split_xyz_from_flat_or_seq(x, time_steps)
    Lz = temporal_rate_similarity_loss_axis(Hz, edges=edges)
    Lx = Hz.new_tensor(0.0)
    Ly = Hz.new_tensor(0.0)
    return Lx, Ly, Lz, Lz

def spring_force_loss_z(x, time_steps=4, k_spring=7.64, use_second_order=True, w_first=0.0):
    _, _, Hz = split_xyz_from_flat_or_seq(x, time_steps)
    if use_second_order:
        d2 = _time_diff2(Hz)
        L2 = (d2**2).mean()
    else:
        d1 = _time_diff(Hz)
        L2 = (d1**2).mean()
    if w_first > 0.0:
        d1m = _time_diff(Hz).mean(dim=2)
        L1 = (d1m**2).mean()
        return L2 + w_first * (k_spring * L1)
    return L2

def magnetic_physics_loss_xyz(
    x, time_steps=4, k_spring=7.64,
    weights_dir=(1.0,1.0,1.0),
    weights_rate=(0.0,0.0,1.0),  # 只对 Z
    weight_spring_z=0.1,
    edges=None, eps=1e-4,
    use_second_order=True, w_first_mean=0.0
):
    Ldx,Ldy,Ldz,_ = directional_consistency_loss_xyz(x, time_steps=time_steps, eps=eps)
    Lrx,Lry,Lrz,_ = temporal_rate_similarity_loss_xyz(x, time_steps=time_steps, edges=edges)
    Lz_spring = spring_force_loss_z(x, time_steps=time_steps, k_spring=k_spring,
                                    use_second_order=use_second_order, w_first=w_first_mean)
    L = (weights_dir[0]*Ldx + weights_dir[1]*Ldy + weights_dir[2]*Ldz) + \
        (weights_rate[2]*Lrz) + \
        (weight_spring_z * Lz_spring)
    return L, {"dir": (Ldx.item(),Ldy.item(),Ldz.item()),
               "rate": (Lrx.item(),Lry.item(),Lrz.item()),
               "spring_z": Lz_spring.item()}
