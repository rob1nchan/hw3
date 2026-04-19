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


# ════════════════════════════════════════════════════════════
# 任务3：线路站点分析
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("任务3：线路站点分析")
print("=" * 60)

# 定义函数（函数名和参数名严格按照题目要求，不能改）
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 按线路号分组，同时计算均值和标准差
    result = (df.groupby(route_col)[stops_col]
                .agg(mean_stops='mean', std_stops='std')  # 聚合计算均值和标准差
                .reset_index()                             # 把线路号还原为普通列
                .sort_values('mean_stops', ascending=False)  # 按均值降序排列
                .reset_index(drop=True))                   # 重置行序号
    return result

# 调用函数，打印前10行
route_stats = analyze_route_stops(df)
print("\n各线路平均搭乘站点数（前10条）：")
print(route_stats.head(10).to_string(index=False))

# 取均值最高的前15条线路的线路号列表
top15_route_ids = route_stats.head(15)['线路号'].tolist()

# 从原始数据里筛选出这15条线路的所有记录
# seaborn 需要原始数据才能自动计算误差棒，不能直接用聚合结果
df_top15 = df[df['线路号'].isin(top15_route_ids)].copy()
df_top15['线路号'] = df_top15['线路号'].astype(str)          # 转字符串方便显示
top15_order = [str(r) for r in top15_route_ids]             # 保持降序排列顺序

# 用 seaborn 绘制水平条形图
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(
    data=df_top15,
    x='ride_stops',
    y='线路号',
    hue='线路号',          # 加上这一行
    order=top15_order,
    errorbar='sd',
    capsize=0.3,
    palette='Blues_d',
    orient='h',
    legend=False,          # 加上这一行
    ax=ax
)
ax.set_xlim(left=0)        # x轴从0开始
ax.set_title('各线路平均搭乘站点数 Top 15', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('平均搭乘站点数', fontsize=13)
ax.set_ylabel('线路号', fontsize=13)
plt.tight_layout()
plt.savefig('route_stops.png', dpi=150)
plt.close()
print("\n图像已保存：route_stops.png")
print("\n✅ 任务3完成！")


# ════════════════════════════════════════════════════════════
# 任务4：高峰小时系数计算
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("任务4：高峰小时系数计算")
print("=" * 60)

# 统计全天每个小时的刷卡总次数
hourly_total = df.groupby('hour').size()

# 自动找出刷卡量最大的那个小时
peak_hour  = int(hourly_total.idxmax())   # 高峰小时数字（如 8）
peak_count = int(hourly_total[peak_hour]) # 高峰小时总刷卡量

print(f"\n高峰小时为 {peak_hour}:00-{peak_hour+1}:00，刷卡量 {peak_count} 次")

# 筛选高峰小时内的数据，把交易时间设为索引（resample必须用时间索引）
df_peak = (df[df['hour'] == peak_hour]
             .set_index('交易时间')
             .sort_index())

# 按5分钟粒度统计，找出最大5分钟刷卡量
# resample('5min') 把数据按每5分钟分成一组，size()统计每组记录数
resample_5  = df_peak.resample('5min').size()
max_5min    = int(resample_5.max())        # 最大5分钟刷卡量
t5_start    = resample_5.idxmax()          # 最大5分钟的起始时间
t5_end      = t5_start + pd.Timedelta(minutes=5)   # 结束时间

# PHF5公式：高峰小时量 ÷ (12 × 最大5分钟量)
# 1小时=60分钟，60÷5=12个区间
phf5 = peak_count / (12 * max_5min)

# 按15分钟粒度统计，找出最大15分钟刷卡量
# resample('15min') 把数据按每15分钟分成一组
resample_15 = df_peak.resample('15min').size()
max_15min   = int(resample_15.max())       # 最大15分钟刷卡量
t15_start   = resample_15.idxmax()         # 最大15分钟的起始时间
t15_end     = t15_start + pd.Timedelta(minutes=15) # 结束时间

# PHF15公式：高峰小时量 ÷ (4 × 最大15分钟量)
# 1小时=60分钟，60÷15=4个区间
phf15 = peak_count / (4 * max_15min)

# 格式化输出结果
print(f"\n高峰小时：{peak_hour:02d}:00 ~ {peak_hour+1:02d}:00，刷卡量：{peak_count} 次")
print(f"最大5分钟刷卡量（{t5_start.strftime('%H:%M')}~{t5_end.strftime('%H:%M')}）：{max_5min} 次")
print(f"PHF5  = {peak_count} / (12 × {max_5min}) = {phf5:.4f}")
print(f"最大15分钟刷卡量（{t15_start.strftime('%H:%M')}~{t15_end.strftime('%H:%M')}）：{max_15min} 次")
print(f"PHF15 = {peak_count} / ( 4 × {max_15min}) = {phf15:.4f}")
print("\n✅ 任务4完成！")


# ════════════════════════════════════════════════════════════
# 任务5：线路驾驶员信息批量导出
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("任务5：线路驾驶员信息批量导出")
print("=" * 60)

# 创建输出文件夹，exist_ok=True 表示文件夹已存在也不报错
output_dir = '线路驾驶员信息'
os.makedirs(output_dir, exist_ok=True)

# 筛选线路号在 1101~1120 范围内的所有记录
df_sel = df[(df['线路号'] >= 1101) & (df['线路号'] <= 1120)]

# 对1101到1120的每条线路逐一处理
for route_id in range(1101, 1121):
    # 筛选该线路，取车辆编号和驾驶员编号两列，去重，按车辆编号排序
    df_r = (df_sel[df_sel['线路号'] == route_id][['车辆编号', '驾驶员编号']]
            .drop_duplicates()
            .sort_values('车辆编号')
            .reset_index(drop=True))

    # 生成文件路径，如 线路驾驶员信息/1101.txt
    filepath = os.path.join(output_dir, f'{route_id}.txt')

    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'线路号: {route_id}\n')
        f.write('车辆编号\t驾驶员编号\n')
        for _, row in df_r.iterrows():
            f.write(f'{int(row["车辆编号"])}\t\t{int(row["驾驶员编号"])}\n')

    print(f'已生成：{os.path.abspath(filepath)}')

print("\n20个文件全部生成成功！")
print("\n✅ 任务5完成！")