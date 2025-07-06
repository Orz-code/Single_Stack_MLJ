from keys import Cols
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# 计算典型光伏发电曲线
def calculate_typical_curve(df_hade, window=14):
    typical_curve = df_hade.groupby('hour')['solar'].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    return typical_curve

# 重新分配异常高值并平滑数据（仅对有异常值的天进行处理）
def redistribute_and_smooth_anomalies(df_hade, rated_value, smooth_window=3):
    df_hade['processed'] = False
    anomaly_indices = df_hade.index[df_hade['solar'] > rated_value].tolist()
    
    for idx in anomaly_indices:
        anomaly_date = df_hade.loc[idx, 'date']
        if df_hade.loc[df_hade['date'] == anomaly_date, 'processed'].any():
            continue
        
        anomaly_value = df_hade.loc[idx, 'solar']
        missing_segment = []
        
        # 向前寻找缺失段
        for i in range(idx-1, -1, -1):
            if df_hade.loc[i, 'solar'] == 0:
                missing_segment.append(i)
            else:
                break
        
        # 向后寻找缺失段
        for i in range(idx+1, len(df_hade)):
            if df_hade.loc[i, 'solar'] == 0:
                missing_segment.append(i)
            else:
                break
        
        if missing_segment:
            total_typical_value = df_hade.loc[missing_segment, 'typical_curve'].sum()
            for i in missing_segment:
                df_hade.loc[i, 'solar'] = anomaly_value * df_hade.loc[i, 'typical_curve'] / total_typical_value
            
            # 平滑分配后的数据
            df_hade.loc[missing_segment, 'solar'] = df_hade.loc[missing_segment, 'solar'].rolling(smooth_window, min_periods=1, center=True).mean()
        
            # 将原异常值设为合理值
            df_hade.loc[idx, 'solar'] = rated_value
            df_hade.loc[df_hade['date'] == anomaly_date, 'processed'] = True
    
    return df_hade

# 只平滑处理有异常值的日期的数据
def smooth_only_anomalies(df_hade, smooth_window=3):
    anomaly_dates = df_hade['date'][df_hade['processed']].unique()
    
    for date in anomaly_dates:
        indices = df_hade[df_hade['date'] == date].index
        df_hade.loc[indices, 'solar'] = df_hade.loc[indices, 'solar'].rolling(smooth_window, min_periods=1, center=True).mean()
    
    return df_hade

def solar_data_preprocess(df_hade,rated_value = 1500):
    
    # 转换时间格式
    df_hade['time'] = pd.to_datetime(df_hade['time'])
    df_hade['hour'] = df_hade['time'].dt.hour
    df_hade['date'] = df_hade['time'].dt.date

    df_hade['typical_curve'] = calculate_typical_curve(df_hade)

    df_hade = redistribute_and_smooth_anomalies(df_hade, rated_value)

    df_hade = smooth_only_anomalies(df_hade)
    
    return df_hade


def solar_abnormal_low(df_hade,df_weather):
    # 这里的df_hade已经经过前面的异常峰值的处理，这里处理异常的低发电量问题
    df_hade[Cols.time] = df_hade[Cols.time].apply(pd.to_datetime)
    df_weather[Cols.time] = df_weather[Cols.date_time].apply(pd.to_datetime)
    df_weather[Cols.time] = df_weather[Cols.time] + pd.Timedelta(hours=12)
    df_weather['temp_max'] = df_weather['temp_max'].apply(
        lambda x : float(x.split('℃')[0])
    )
    df_weather['temp_min'] = df_weather['temp_min'].apply(
        lambda x : float(x.split('℃')[0])
    )
    df_weather['temp_delta'] = df_weather['temp_max'] - df_weather['temp_min']
    df_solar_date = df_hade.groupby(by = 'date')[[Cols.solar]].sum().reset_index()
    df_date = pd.merge(
        left = df_solar_date[['date',Cols.solar]].rename(
            columns = {
                'date':Cols.date_time
            }
        ),
        right = df_weather[[Cols.date_time,'temp_delta']],
        how = 'left',
        on=Cols.date_time
    )

    model = LinearRegression()
    x = df_date.iloc[15:][['temp_delta']]
    y = df_date.iloc[15:][[Cols.solar]]
    model.fit(x,y)

    df_date['predict'] = df_date['temp_delta'] * model.coef_[0] + model.intercept_
    df_date['ratio'] = df_date[Cols.solar]/ df_date['predict']

    res_list = []
    for date,df_cur in df_hade.groupby(by = 'date'):
        ratio_cur = df_date.loc[df_date[Cols.date_time]==date]['ratio'].values[0]
        if ratio_cur <= 0.3:
            df_cur[Cols.solar] /= ratio_cur
        res_list.append(df_cur)
        pass
    df_solar = pd.concat(res_list)[[Cols.time,Cols.solar]]
    return df_solar

# 定义方程
def polar(current_density, r1, r2, r3):
    """这里为系统设计所需要使用的极化方程，不考虑热电耦合问题，只考虑电的部分

    Args:
        current_density (_type_): 电流密度 A/m2
        r1 (float): 极化参数
        r2 (float): 极化参数
        r3 (float): 极化参数

    Returns:
        float: 小室电压
    """
    return 1.48 + (r1 * current_density) + r2 * np.log(np.abs(r3 * current_density) + 1)

def polar_function_fit(df_polar_raw):
    """自动对送入的dataframe进行参数拟合，以适合不同的电解槽数据

    Args:
        df_polar_raw (_type_):需要包含Cols.current_density,Cols.cell_voltage

    Returns:
        _type_: r1, r2, r3
    """
    # 提取数据
    current_density = df_polar_raw['current_density'].values
    cell_voltage = df_polar_raw['cell_voltage'].values

    # 拟合曲线
    initial_guess = [1, 1, 1]  # 初始猜测值
    params, covariance = curve_fit(polar, current_density, cell_voltage, p0=initial_guess)

    r1, r2, r3 = params

    # 输出结果
    # print(f"优化后的参数: r1 = {r1}, r2 = {r2}, r3 = {r3}")
    return r1, r2, r3