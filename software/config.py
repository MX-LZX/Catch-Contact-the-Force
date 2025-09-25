# -*- coding: utf-8 -*-
SEED = 42
DEVICE = "cuda"
#DEVICE = "cpu"

DATA = {
    "data_dir": "data",
    "val_split": 0.2,
    "num_workers": 0,
    "pin_memory": True,
}

CHECKPOINTS = {"dir": "checkpoints", "save_best_ema": True}

MODEL = {
    "feature_dim": 12,
    "time_steps": 4,
    "xy_dim": 2,
    "force_dim": 3,
    "dropout_p": 0.15,
    "norm": "layer",          # "batch" | "layer" | "none"
}

OPTIM = {
    "name": "adamw",          # "adam" | "adamw"
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "amp": True,              # 混合精度
    "ema_decay": 0.999,       # 0 关闭；建议0.999~0.9995
}

SCHEDULER = {
    "type": "cosine",         # "plateau" | "cosine"
    "warmup_epochs": 5,
    "min_lr": 5e-6,
    "plateau_factor": 0.5,
    "plateau_patience": 5,
}

TRAINING = {"epochs": 300, "batch_size": 512}

# 监督损失与平滑
LOSSES = {
    "fx_weight": 1.0,
    "fy_weight": 1.0,
    "fz_weight": 4.0,
    "smooth_weight": {"fx": 0.05, "fy": 0.05, "fz": 0.2},
    "use_smoothl1_for_fz": True,
    "smoothl1_beta_fz": 0.1,
    # Z 一致性（抗抖动）
    "z_consistency_weight": 0.05,  # 设0可关闭
    "z_jitter_std": 0.01,          # 按数据量纲微调
}

GEOMETRY = {"R_HEMISPHERE": 12.0, "R_SENSOR": 12.0}
SPRING  = {"SPRING_K": 15.593, "SPRING_L0": 28.0}

PHYSICS = {
    "lambda_div": 0.10,              # 物理项总权重（再乘各子项）
    "weights_dir": (1.0, 1.0, 1.0),  # 方向一致性 X/Y/Z
    "weights_rate": (0.0, 0.0, 1.0), # 变化率一致性（Z）
    "weight_spring_z": 0.25,         # Z 弹簧二阶平滑
    "time_steps": 4,
    "edges": [(0,1),(2,3),(0,2),(1,3)],
    "eps": 1e-6,
    "use_second_order": True,
    "w_first_mean": 0.0,
}

MODEL.update({
    "use_z_residual": True,     # 开启 Z 专家支路（更会抓峰）
    "z_residual_scale": 1.0,    # Z 支路残差缩放
})

LOSSES.update({
    # 峰值样本加权：让 |Fz| 大/变化剧烈的样本更「重要」
    "fz_peak_weight": {"alpha": 1.5, "gamma": 1.0, "ref": "p90"},  # ref: "p50"/"p75"/"p90" or 数值
    # 相对误差（抑制幅值偏差，配合上面峰值权重）
    "fz_rel_loss_weight": 0.10,
})

# —— 推理端（evaluate.py 用）——
EVAL = {
    "smooth_kind": "combo",   # "ma" | "median" | "combo"
    "median_window": 5,       # 中值滤波窗口（奇数）
    "ma_window": 9,           # 移动平均窗口（奇数）
    "calibrate_linear": False # True=用最小二乘 a*x+b 做一次线性校准
}
