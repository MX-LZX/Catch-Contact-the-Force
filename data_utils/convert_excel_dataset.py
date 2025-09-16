import pandas as pd
import numpy as np
import os

def convert_excel_to_npy_axisaware(
    excel_file="output_48dim.xlsx",
    sheet_name=0,
    output_dir="../data"
):
    """
    从 Excel 读 48维特征 + 5维标签，并导出到:
      - ../data/inputs.npy   (N, 48)
      - ../data/targets.npy  (N, 5)

    列顺序严格匹配你现在的表头：
      对每个时刻块 lag ∈ {t-3, t-2, t-1, t}，顺序为（每块12列）：
        [HX_1, HY_1, HZ_1,  HX_2, HY_2, HZ_2,  HX_3, HY_3, HZ_3,  HX_4, HY_4, HZ_4]_lag
      四个时刻块依次拼接，共 48 维。
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    sensors = [1, 2, 3, 4]
    axes    = ["X", "Y", "Z"]
    lags    = [3, 2, 1, 0]  # t-3, t-2, t-1, t

    # 构造 48 维列名（严格按：每个时刻块内 1..4 号传感器 × (X,Y,Z)）
    feat_cols = []
    for lag in lags:
        tag = f"t-{lag}" if lag > 0 else "t"
        for sid in sensors:
            for ax in axes:
                feat_cols.append(f"H{ax}_{sid}_{tag}")

    label_cols = ["x", "y", "Fx", "Fy", "Fz"]

    # 校验列是否齐全
    missing = [c for c in feat_cols + label_cols if c not in df.columns]
    if missing:
        raise KeyError("Excel 中缺少这些列，请检查表头是否一致：\n" + ", ".join(missing))

    # 取值并保存
    inputs  = df[feat_cols].to_numpy(dtype=np.float32)   # (N, 48)
    targets = df[label_cols].to_numpy(dtype=np.float32)  # (N, 5)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "inputs.npy"),  inputs)
    np.save(os.path.join(output_dir, "targets.npy"), targets)

    # —— 训练阶段按轴切片索引（与你的列顺序一一对应）——
    # 每个时刻块大小 = 12；块内位置：X=[0,3,6,9], Y=[1,4,7,10], Z=[2,5,8,11]
    X_IDX = [12*b + off for b in range(4) for off in (0, 3, 6, 9)]
    Y_IDX = [12*b + off for b in range(4) for off in (1, 4, 7, 10)]
    Z_IDX = [12*b + off for b in range(4) for off in (2, 5, 8, 11)]

    print("✅ 转换完成！")
    print(f"inputs.npy:  {inputs.shape}")
    print(f"targets.npy: {targets.shape}")
    print(f"已保存到目录: {os.path.abspath(output_dir)}")
    print("\n—— 训练阶段按轴切片索引（复制到你的训练脚本即可）——")
    print("X indices:", X_IDX)
    print("Y indices:", Y_IDX)
    print("Z indices:", Z_IDX)

    # 若需要在其他脚本 import 使用，也可以返回
    return np.array(X_IDX), np.array(Y_IDX), np.array(Z_IDX)

if __name__ == "__main__":
    convert_excel_to_npy_axisaware()
