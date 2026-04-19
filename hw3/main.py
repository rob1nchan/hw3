# ============================================================
#  《人工智能编程语言》第3次作业 - 公交IC卡刷卡数据分析
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Patch

# 设置中文字体，防止图表乱码
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# ════════════════════════════════════════════════════════════
# 任务1：数据预处理
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("任务1：数据预处理")
print("=" * 60)

# 读取数据，分隔符是制表符 \t
df = pd.read_csv('ICData.csv', sep='\t')

# 打印前5行
print("\n前5行数据：")
print(df.head())

# 打印行数、列数、各列数据类型
print(f"\n数据集大小：{df.shape[0]} 行  ×  {df.shape[1]} 列")
print("\n各列数据类型：")
print(df.dtypes)

# 将「交易时间」列转换为 datetime 类型，pandas 会自动识别格式
df['交易时间'] = pd.to_datetime(df['交易时间'])

# 从 datetime 中提取小时数字（0~23），新增为 hour 列
df['hour'] = df['交易时间'].dt.hour.astype(int)

print("\n时间解析完成，新增 hour 列示例：")
print(df[['交易时间', 'hour']].head(3))

# 计算搭乘站点数 = |下车站点 - 上车站点|，新增为 ride_stops 列
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()


# 删除 ride_stops == 0 的异常记录
before_len = len(df)
df = df[df['ride_stops'] != 0].reset_index(drop=True)
after_len = len(df)
print(f"\n删除 ride_stops=0 的异常记录：共删除 {before_len - after_len} 行，剩余 {after_len} 行")

# 检查各列缺失值
print("\n各列缺失值数量：")
missing = df.isnull().sum()
print(missing)

# 若有缺失值则删除整行，本数据集通常无缺失值
if missing.any():
    df = df.dropna().reset_index(drop=True)
    print("发现缺失值，已删除含缺失值的行")
else:
    print("无缺失值，无需处理")

print("\n✅ 任务1完成！")