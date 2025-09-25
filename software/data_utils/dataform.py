import pandas as pd
import numpy as np

# 设置文件路径
file_path = 'output_32dim.xlsx'  # 替换成你的文件名
sheet_name = 0  # 或指定工作表名

# 读取数据
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 确保行数是5的整数倍，否则截断多余的行
num_groups = len(df) // 2000
df = df.iloc[:num_groups * 2000]

# 重塑为组的形式 (每组5行)
groups = np.split(df, num_groups)

# 打乱组的顺序
np.random.seed(42)  # 可复现结果
np.random.shuffle(groups)

# 重新合并为一个DataFrame
shuffled_df = pd.concat(groups, ignore_index=True)

# 保存到新的Excel文件
shuffled_df.to_excel('output.xlsx', index=False)
