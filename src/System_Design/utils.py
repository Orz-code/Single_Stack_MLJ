from keys import Cols
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from constants import Constants

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

def convert_power_to_current_density(AWES_power,prod_cap,rated_current_density,min_rate):
    """计算当前功率情况下电解槽的工作电流密度
    这里只是考虑集群工作情况下的功率分配后电流密度，并不检测超负荷工作的情况    

    Args:
        AWES_power (_type_): 当前的功率
        prod_cap (_type_): 额定的电解槽功率
        rated_current_density (_type_): 额定的电流密度
        min_rate (_type_): 电解槽的最低负荷情况

    Returns:
        _type_: 当前的电流密度A/m2
    """
    if AWES_power > prod_cap * min_rate:
        current_density = AWES_power / prod_cap * rated_current_density
    else:
        current_density = rated_current_density * min_rate

    return current_density

def calculate_current_density_to_energy_cost(current_density,r1,r2,r3,discount_rate=1):
    """结合当前的电流密度计算极化后的整体制氢能耗

    Args:
        current_density (_type_): 电流密度，这里应当和极化曲线本身对应
        r1 (_type_): 极化参数
        r2 (_type_): 极化参数
        r3 (_type_): 极化参数
        discount_rate (int, optional): 电解槽在额定点折合能耗的比例. Defaults to 1.

    Returns:
        _type_: 单位制氢能耗kWh/kg
    """
    cell_voltage = polar(current_density,r1,r2,r3)
    energy_cost = cell_voltage * Constants.cell_voltage_to_energy_consumption * discount_rate
    return energy_cost

def calculate_AWES_power_to_hydrogen_production(AWES_power,energy_cost):
    """计算当前功率、能耗下的电解槽制氢量

    Args:
        AWES_power (_type_): 电解槽的当前功率，MW
        energy_cost (_type_): 制氢能耗，kWh/kg

    Returns:
        _type_: t
    """
    hydrogen_production_cur = AWES_power * 1000. / energy_cost / 1000. # t
    return hydrogen_production_cur