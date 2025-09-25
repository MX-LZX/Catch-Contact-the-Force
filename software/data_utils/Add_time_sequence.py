import pandas as pd

# 1) 读取原始 Excel（替换为你的文件名/路径）
df = pd.read_excel("data.xlsx")

# 2) 定义基础霍尔列（4个传感器 × 3轴）
sensors = [1, 2, 3, 4]
axes = ["X", "Y", "Z"]
base_cols = [f"H{ax}_{sid}" for sid in sensors for ax in axes]
# 结果顺序: HX_1, HY_1, HZ_1, HX_2, HY_2, HZ_2, ..., HZ_4

# 3) 检查列是否齐全
missing = [c for c in base_cols + ["x","y","Fx","Fy","Fz"] if c not in df.columns]
if missing:
    raise KeyError("以下列在 Excel 中找不到，请检查表头是否匹配：\n" + ", ".join(missing))

# 4) 生成时序滞后特征
lag_steps = [3, 2, 1, 0]   # t-3, t-2, t-1, t
df_lag = pd.DataFrame()

for lag in lag_steps:
    shifted = df[base_cols].shift(lag)
    # 重命名：HX_1 -> HX_1_t-3 / HX_1_t
    shifted.columns = [f"{c}_t-{lag}" if lag > 0 else f"{c}_t" for c in shifted.columns]
    df_lag = pd.concat([df_lag, shifted], axis=1)

# 5) 拼接标签列
df_out = pd.concat([df_lag, df[["x", "y", "Fx", "Fy", "Fz"]]], axis=1)

# 6) 丢掉由于 shift 产生 NaN 的开头几行
df_out = df_out.dropna().reset_index(drop=True)

# 7) 保存
df_out.to_excel("output_48dim.xlsx", index=False)
print("✅ 处理完毕，已保存为 output_48dim.xlsx")
