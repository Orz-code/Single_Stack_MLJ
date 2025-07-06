import pandas as pd
import numpy as np
from keys import Cols
from tqdm import tqdm
import re
import os
import pandas as pd
from collections import defaultdict

def safe_relative_diff(current, neighbor):
    """安全计算相对差异，处理零值问题"""
    # 创建有效分母（避免除零错误）
    denominator = np.where(neighbor != 0, neighbor, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(current - neighbor) / denominator
    
    # 处理特殊情况：
    # 1. 前后值均为零视为无差异
    # 2. 前值零但当前值非零视为无限差异
    both_zero = (current == 0) & (neighbor == 0)
    current_nonzero = (current != 0) & (neighbor == 0)
    
    diff = np.where(both_zero, 0, diff)
    diff = np.where(current_nonzero, np.inf, diff)
    return diff

def process_midnight_data(df):
    target_cols = [
        Cols.temp_O, Cols.temp_H, Cols.voltage,
        Cols.current, Cols.lye_flow, Cols.lye_temp,
        Cols.HTO, Cols.OTH
    ]
    
    # 预处理确保数据完整性和顺序
    df = df.sort_values(Cols.date_time).reset_index(drop=True)
    df[Cols.date_time] = pd.to_datetime(df[Cols.date_time])
    
    # 生成有效索引掩码（排除首尾行）
    is_midnight = df[Cols.date_time].dt.time == pd.Timestamp('00:00:00').time()
    valid_mask = is_midnight & (df.index > 0) & (df.index < len(df)-1)
    midnight_indices = df.index[valid_mask]
    
    # 批量处理所有午夜时间点
    for idx in midnight_indices:
        prev = df.iloc[idx-1][target_cols].values.astype(float)
        current = df.iloc[idx][target_cols].values.astype(float)
        next_ = df.iloc[idx+1][target_cols].values.astype(float)
        
        # 计算前后相对差异
        diff_prev = safe_relative_diff(current, prev)
        diff_next = safe_relative_diff(current, next_)
        
        # 标记需要修正的列（任意方向差异超过50%）
        need_correction = (diff_prev > 0.5) | (diff_next > 0.5)
        
        if need_correction.any():
            # 计算修正值（前后行平均值）
            avg_values = (prev + next_) / 2
            # 仅更新异常列
            corrected = np.where(need_correction, avg_values, current)
            df.loc[idx, target_cols] = corrected
    
    return df

def process_abnormal_intervals(df, res_list, replace_cols):
    """
    处理异常区间数据，用前20个数据的众数替换
    
    参数:
    df (pd.DataFrame): 原始数据框
    res_list (list): find_consecutive_intervals返回的区间列表
    condition_cols (dict): 判定条件列及阈值
    replace_cols (list): 需要替换的列名列表
    
    返回:
    tuple: (处理后的DataFrame, 异常区间记录列表)
    """
    # 创建数据副本避免修改原始数据
    df_processed = df.copy()
    abnormal_records = []
    
    # 遍历所有检测到的区间
    for length, start_idx, end_idx in res_list:
        # 获取区间数据
        interval_data = df_processed.loc[start_idx:end_idx]
        
        # 检查判定条件
        condition2 = interval_data[Cols.lye_temp].mean() < 35 # 目前发现似乎碱液温度不会异常
        
        if condition2:
            # 记录异常区间
            abnormal_records.append((length, start_idx, end_idx))
            
            # 计算前20行的众数（考虑边界情况）
            start_pos = df_processed.index.get_loc(start_idx)
            window_start = max(0, start_pos - 20)
            mode_window = df_processed.iloc[window_start:start_pos][replace_cols]
            
            # 计算各列众数（处理多众数情况）
            mode_values = mode_window.mode().iloc[0] if not mode_window.empty else None
            
            if mode_values is not None:
                # 替换区间数据
                df_processed.loc[start_idx:end_idx, replace_cols] = mode_values.values
            else:
                print(f"警告：区间 {start_idx}-{end_idx} 前20行数据不足，无法计算众数")

    return df_processed, abnormal_records

def set_negative_to_zero(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    将指定列中的负数值替换为零
    
    参数:
    df (pd.DataFrame): 原始数据框
    column (str): 需要处理的列名
    
    返回:
    pd.DataFrame: 处理后的数据副本
    
    示例:
    >>> df = pd.DataFrame({'A': [1, -2, 3], 'B': [-4, 5, -6]})
    >>> set_negative_to_zero(df, 'A')
       A   B
    0  1  -4
    1  0   5
    2  3  -6
    """
    # 创建数据副本避免修改原始数据
    df_copy = df.copy()
    
    # 验证列是否存在
    if column not in df_copy.columns:
        raise ValueError(f"列 '{column}' 不存在于数据框中")
    
    # 使用向量化操作替换负值
    df_copy.loc[df_copy[column] < 0, column] = 0
    
    return df_copy

def find_consecutive_intervals(df, col1, col2):
    """
    检测两列各自连续保持相同值（≥max*10%）至少6次的共同段落
    返回格式为（长度, 起始索引, 结束索引）的元组列表
    
    参数:
    df (pd.DataFrame): 输入数据框
    col1 (str): 第一列列名
    col2 (str): 第二列列名
    
    返回:
    list: 包含元组（长度, 起始索引, 结束索引）的列表
    """
    def get_valid_mask(series):
        """生成有效连续段掩码（长度≥6且≥max*10%）"""
        # 计算阈值
        max_val = series.max()
        threshold = max_val * 0.1
        
        # 标记变化点（值改变或低于阈值时）
        change_point = (series != series.shift()) | (series < threshold)
        groups = change_point.cumsum()
        
        # 计算每组长度
        group_sizes = series.groupby(groups).transform('size')
        
        # 生成有效掩码：长度≥6且≥阈值
        return (group_sizes >= 6) & (series >= threshold)

    # 生成两列有效掩码并取交集
    mask1 = get_valid_mask(df[col1])
    mask2 = get_valid_mask(df[col2])
    combined_mask = mask1 & mask2

    # 寻找连续True段落边界
    padded = np.pad(combined_mask.values.astype(int), (1,1), 'constant')
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    # 生成结果列表
    intervals = []
    for s, e in zip(starts, ends):
        length = e - s
        if length >= 6:
            start_idx = df.index[s]
            end_idx = df.index[e-1]
            intervals.append((length, start_idx, end_idx))
    
    return intervals

def keep_quantile_range(arr):
    """
    此函数的主要目的是对输入的 numpy 数组进行筛选，只保留位于 10 分位数到 90 分位数范围内的数据元素。

    参数:
    arr (numpy.ndarray): 输入的 numpy 数组，包含待处理的数据。

    返回:
    numpy.ndarray: 一个新的 numpy 数组，仅包含原数组中处于 10 分位数到 90 分位数范围内的数据元素。

    实现步骤：
    1. 使用 numpy 的 percentile 函数计算输入数组的 10 分位数（q10）和 90 分位数（q90）。
    2. 使用布尔索引筛选出满足 (arr >= q10) 且 (arr <= q90) 的元素，将结果存储在 result 中。
    3. 最终返回存储筛选结果的 result 数组。
    """
    # 计算 10 分位数和 90 分位数
    q10 = np.percentile(arr, 10)
    q90 = np.percentile(arr, 90)
    # 使用布尔索引筛选出在 10-90 分位数范围内的数据
    result = arr[(arr >= q10) & (arr <= q90)]
    return result


def iter_function(iterable, verbose=0):
    """
    此函数根据 verbose 参数的设置，决定是否对迭代器进行进度条包装。

    参数:
    iterable: 可迭代对象，例如列表、元组、range 对象等。
    verbose (int, 默认为 0): 控制是否显示进度条。
        - 当 verbose > 0 时，使用 tqdm 对迭代器进行包装，显示进度条，方便观察迭代进度。
        - 当 verbose <= 0 时，直接返回原始迭代器，不显示进度条。

    返回:
    迭代器: 根据 verbose 参数的设置，可能是原始迭代器或 tqdm 包装过的迭代器。
    """
    if verbose > 0:
        iterator = tqdm(iterable)
    else:
        iterator = iterable
    return iterator


def detect_temperature_anomalies_difference(df, temperature_columns, window_size=10, std_threshold=150, verbose=0):
    """
    此函数用于检测 DataFrame 中指定温度列的数据异常点。
    异常检测的逻辑是计算每个数据点与前后 window_size 个数据点的平均差的绝对值，并根据标准差和阈值判断是否为异常。

    参数:
    df (pandas.DataFrame): 输入的 DataFrame，包含温度数据。
    temperature_columns (list): 一个列表，包含需要检测异常的温度列的列名。
    window_size (int, 默认为 10): 前后考虑的数据点数量，用于计算平均差。
    std_threshold (float, 默认为 150): 标准差的阈值，用于判断异常。
    verbose (int, 默认为 0): 控制是否显示进度条，同 iter_function 函数。

    返回:
    pandas.DataFrame: 一个新的 DataFrame，包含与输入 df 相同的索引，其中异常数据点在相应列标记为 True，正常数据点标记为 False。

    实现步骤：
    1. 创建一个新的 DataFrame anomalies，其索引与输入的 df 相同，用于存储异常标记结果。
    2. 对于 temperature_columns 中的每一列：
        a. 初始化 is_anomaly 和 stds 列表，用于存储异常标记和标准差信息。
        b. 首先处理前 window_size 个数据点，将其标记为正常，标准差设为 0。
        c. 对于中间部分的数据点（不包括前后 window_size 个数据点）：
            i. 使用 iter_function 迭代中间部分的索引，根据 verbose 决定是否显示进度条。
            ii. 计算当前数据点前后 window_size 个数据点的范围（start 和 end），并使用 iloc 切片获取邻居数据。
            iii. 调用 keep_quantile_range 函数筛选邻居数据，并计算筛选后数据的中位数和标准差。
            iv. 计算当前数据点与中位数的平均差的绝对值和标准差差异。
            v. 根据标准差和平均差的大小，判断当前数据点是否为异常，并将结果添加到 is_anomaly 列表，将标准差添加到 stds 列表。
        d. 最后处理后 window_size 个数据点，将其标记为正常，标准差设为 0。
        e. 将 is_anomaly 和 stds 列表存储在 anomalies 中对应的列和'stds' 列。
    3. 最终返回存储异常标记和标准差信息的 anomalies DataFrame。
    """
    anomalies = pd.DataFrame(index=df.index)

    for column in temperature_columns:
        is_anomaly = []
        stds = []
        # 首先将前 window_size 个数据点标记为正常，标准差为 0
        for i in range(window_size):
            is_anomaly.append(False)
            stds.append(0)
        for i in iter_function(range(window_size, len(df) - window_size), verbose=verbose):
            start = max(0, i - window_size)
            end = min(len(df), i + window_size + 1)
            # 获取前后数据点
            neighbors = df.iloc[start:end][column]

            # 计算当前点与前后邻居的平均差的绝对值

            quantile_neighbors = keep_quantile_range(neighbors)
            prev_median = quantile_neighbors.median()
            prev_std = quantile_neighbors.std()

            average_difference = np.abs(df[column].iloc[i] - prev_median)
            std_difference = average_difference / prev_std

            if prev_std < 1E-2:
                if average_difference > abs(prev_median) * 3:  # 即便前后的方差是 0，还有可能有异常值
                    is_anomaly.append(True)
                    stds.append(1000)
                else:  # 确实都是不变的情况
                    is_anomaly.append(False)
                    stds.append(0)
            else:  # 方差不是零的情况下，可以自动判断是否为异常值
                is_anomaly.append(std_difference > std_threshold)
                stds.append(min(10000, std_difference))

        # 最后将后 window_size 个数据点标记为正常，标准差为 0
        for i in range(window_size):
            is_anomaly.append(False)
            stds.append(0)
        anomalies[column] = is_anomaly
        anomalies[column+'_stds'] = stds
    return anomalies

def resample_dataframe(df, minutes):
    # 确保 'Cols.date_time' 列被转换为 datetime 类型
    df[Cols.date_time] = pd.to_datetime(df[Cols.date_time])

    # 将 'Cols.date_time' 列设置为索引
    df.set_index(Cols.date_time, inplace=True)

    # 根据输入的分钟数进行重采样
    df_resampled = df.resample(f'{minutes}T').mean()  # 计算均值，或者可以根据需要使用其他聚合方法

    # 恢复原始索引（如果需要的话）
    df_resampled.reset_index(inplace=True)

    return df_resampled

def process_column(df, column_name):

    # 将中文和英文的问号替换为 np.nan
    df[column_name] = df[column_name].replace(['?', '？'], np.nan, regex=False)
    
    # 转换为 float 类型，确保所有的数值和 NaN 都能处理
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # 如果整列全是 NaN，填充为 0
    if df[column_name].isna().all():
        return df[column_name].fillna(0)
    
    # 找出 NaN 的区间并逐一处理
    filled_column = df[column_name].copy()
    nan_groups = filled_column.isna().astype(int).groupby((filled_column.notna()).cumsum(), group_keys=False).cumsum()
    
    # 处理每个 NaN 区间
    group_start = None
    for idx in range(len(filled_column)):
        if pd.isna(filled_column.iloc[idx]):
            if group_start is None:
                group_start = idx
        else:
            if group_start is not None:
                group_end = idx - 1
                nan_length = group_end - group_start + 1

                # 小于等于 3 个的 NaN，使用前后插值
                if nan_length <= 20:
                    start_value = filled_column.iloc[group_start - 1] if group_start - 1 >= 0 else np.nan
                    end_value = filled_column.iloc[group_end + 1] if group_end + 1 < len(filled_column) else np.nan
                    if not np.isnan(start_value) and not np.isnan(end_value):
                        filled_column.iloc[group_start:group_end + 1] = np.linspace(start_value, end_value, nan_length)
                    else:
                        filled_column.iloc[group_start:group_end + 1] = 0
                else:
                    # 大于等于 4 个的 NaN，填充为 0
                    filled_column.iloc[group_start:group_end + 1] = 0
                group_start = None

    # 如果最后还有未处理的 NaN 区间
    if group_start is not None:
        group_end = len(filled_column) - 1
        nan_length = group_end - group_start + 1

        if nan_length <= 20:
            start_value = filled_column.iloc[group_start - 1] if group_start - 1 >= 0 else np.nan
            end_value = np.nan
            if not np.isnan(start_value):
                filled_column.iloc[group_start:group_end + 1] = np.linspace(start_value, start_value, nan_length)
            else:
                filled_column.iloc[group_start:group_end + 1] = 0
        else:
            filled_column.iloc[group_start:group_end + 1] = 0

    return filled_column

def resample_dataframe(df, minutes):
    # 确保 'Cols.date_time' 列被转换为 datetime 类型
    df[Cols.date_time] = pd.to_datetime(df[Cols.date_time])

    # 将 'Cols.date_time' 列设置为索引
    df.set_index(Cols.date_time, inplace=True)

    # 根据输入的分钟数进行重采样
    df_resampled = df.resample(f'{minutes}T').mean()  # 计算均值，或者可以根据需要使用其他聚合方法

    # 恢复原始索引（如果需要的话）
    df_resampled.reset_index(inplace=True)

    return df_resampled

def remove_duplicate_columns(df):
    columns = []
    unique_columns = []
    for col in df.columns:
        if col.endswith('_x') or col.endswith('_y'):
            base_col = col[:-2]
            if base_col in unique_columns:
                continue
            df = df.rename(
                columns = {
                    col:base_col
                }
            )
            columns.append(base_col)
            unique_columns.append(base_col)
        else:
            columns.append(col)
            unique_columns.append(col)
    return df[columns]

def column_length(column_name):
    # 检测如果包含中文字符，就给出一个较长的长度
    if re.search(r'[\u4e00-\u9fff]', column_name):
        return 550
    elif column_name == Cols.date_time:
        return 1
    elif 'Unnamed' in column_name:
        return 560
    else:
        return len(column_name)
    
def Hyzenis_clean_excel_file(input_file_path):
    """对实验数据进行数据清洗（主要为将-9999值转为平均值）

    Args:
        input_file_path (string): 原始数据路径

    Returns:
        dataframe: 清洗后转为dataframe
    """
    try:
        # 读取 Excel 文件
        df = pd.read_excel(input_file_path, engine='openpyxl')

        # 去除列名中的空格
        df.columns = df.columns.str.strip()

        # 去除数据中的前后空格
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # 去除重复行
        df = df.drop_duplicates()

        # 将 -9999 替换为 nan
        df = df.replace(-9999, np.nan)

        # 按列处理缺失值，将 nan 和前后第一个非 nan 数形成等差数列
        for col in df.columns:
            series = df[col]
            nan_indices = series[series.isna()].index
            for idx in nan_indices:
                prev_non_nan = series[:idx][series[:idx].notna()]
                next_non_nan = series[idx:][series[idx:].notna()]
                if not prev_non_nan.empty and not next_non_nan.empty:
                    prev_val = prev_non_nan.iloc[-1]
                    next_val = next_non_nan.iloc[0]
                    num_missing = (next_non_nan.index[0] - prev_non_nan.index[-1]) - 1
                    step = (next_val - prev_val) / (num_missing + 1)
                    for i in range(1, num_missing + 1):
                        series[prev_non_nan.index[-1] + i] = prev_val + i * step

        return df

    except FileNotFoundError:
        print(f"错误：未找到文件 {input_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")