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
df = pd.read_csv('ICData.csv', sep=',')

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

# ════════════════════════════════════════════════════════════
# 任务2：时间分布分析
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("任务2：时间分布分析")
print("=" * 60)

# 只统计刷卡类型=0（上车刷卡）的记录
df_board = df[df['刷卡类型'] == 0]

# 将 hour 列转为 numpy 数组（任务硬性要求必须用 numpy）
hours_np = df_board['hour'].to_numpy()

# 用 numpy 布尔索引统计早峰前时段（hour < 7）的刷卡量
early_count = int(np.sum(hours_np < 7))

# 用 numpy 布尔索引统计深夜时段（hour >= 22）的刷卡量
late_count = int(np.sum(hours_np >= 22))

# 全天总上车刷卡量
total_board = len(hours_np)

print(f"\n早峰前（00:00~06:59）刷卡量：{early_count} 次，"
      f"占全天 {early_count / total_board * 100:.2f}%")
print(f"深夜时段（22:00~23:59）刷卡量：{late_count} 次，"
      f"占全天 {late_count / total_board * 100:.2f}%")
print(f"全天总上车刷卡量：{total_board} 次")

# 统计每小时刷卡量，没有数据的小时填0
hourly_counts = (df_board.groupby('hour').size()
                          .reindex(range(24), fill_value=0))

# 为每个小时设置颜色：早峰前红色、深夜橙色、其余蓝色
bar_colors = []
for h in range(24):
    if h < 7:
        bar_colors.append('#FF6B6B')   # 早峰前：红色
    elif h >= 22:
        bar_colors.append('#FFA500')   # 深夜：橙色
    else:
        bar_colors.append('#4A90D9')   # 日间：蓝色

# 用 matplotlib 绘制柱状图
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(24), hourly_counts.values, color=bar_colors,
       edgecolor='white', linewidth=0.6)

# x轴刻度：步长为2显示标签
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([str(h) for h in range(0, 24, 2)], fontsize=11)

# 中文标题和坐标轴标签
ax.set_title('公交IC卡全天24小时刷卡量分布', fontsize=16, fontweight='bold', pad=12)
ax.set_xlabel('小时（时）', fontsize=13)
ax.set_ylabel('刷卡量（次）', fontsize=13)

# 添加水平网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# 添加图例
ax.legend(handles=[
    Patch(facecolor='#FF6B6B', label='早峰前（0~6时）'),
    Patch(facecolor='#4A90D9', label='日间（7~21时）'),
    Patch(facecolor='#FFA500', label='深夜（22~23时）'),
], fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
plt.close()
print("\n图像已保存：hour_distribution.png")
print("\n✅ 任务2完成！")