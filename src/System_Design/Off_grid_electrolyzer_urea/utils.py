# 可能会重复用到的方法
from System_Design.Off_grid_electrolyzer_urea.configs import Project_Config
import pandas as pd
import numpy as np
from keys import Cols,Files
import gzip, pickle

# Project_Config.initiate_config() # 暂时还不需要在这里初始化

# 由于更新的时候都是更新当前小时的结果，所以这里检查的时候需要检查上一个小时的数据
# 判断储量为空
def check_hydrogen_storage_empty(simu_idx, df_simulation):
    # 判断氢气是否已经空，t
    return df_simulation.iloc[simu_idx-1][Cols.hydrogen_storage]<=0

def check_ammonia_storage_empty(simu_idx, df_simulation):
    # 判断合成氨是否已经空，t
    return df_simulation.iloc[simu_idx-1][Cols.ammonia_storage]<=0

def check_ammonia_storage_sufficient(simu_idx, df_simulation):
    # 判断氨储量是否够设定值，t
    return df_simulation.iloc[simu_idx-1][Cols.ammonia_storage] > Project_Config.Continuous_operation.Ammonia_storage

def check_hydrogen_storage_sufficient(simu_idx, df_simulation):
    # 判断氨储量是否够设定值，t
    return df_simulation.iloc[simu_idx-1][Cols.ammonia_storage] > Project_Config.Continuous_operation.Hydrogen_storage

def check_energy_storage_empty(simu_idx, df_simulation):
    # 判断储能电池是否已经空，MWh
    return df_simulation.iloc[simu_idx-1][Cols.energy_storage]<=0

def check_hydrogen_storage_full(simu_idx, df_simulation):
    # 判断氢气是否已经储满，t
    return df_simulation.iloc[simu_idx-1][Cols.hydrogen_storage]>=Project_Config.Prod_Cap.Hydrogen_storage_capacity

def check_hydrogen_storage_half(simu_idx, df_simulation):
    # 判断氢气是否已经达到一半，t
    return df_simulation.iloc[simu_idx-1][Cols.hydrogen_storage]>=Project_Config.Prod_Cap.Hydrogen_storage_capacity/2

def check_ammonia_storage_full(simu_idx, df_simulation):
    # 判断合成氨是否已经储满，t
    return df_simulation.iloc[simu_idx-1][Cols.ammonia_storage]>=Project_Config.Prod_Cap.Ammonia_storage_capacity

def check_energy_storage_full(simu_idx, df_simulation):
    # 判断储能电池是否已经充满，MWh
    return df_simulation.iloc[simu_idx-1][Cols.energy_storage]>=Project_Config.Prod_Cap.Energy_storage_capacity

def check_urea_was_operating(simu_idx,df_simulation):
    # 如果上一时刻尿素生产还在工作，就返回true
    return df_simulation.iloc[simu_idx-1][Cols.urea_power] > 0.

def check_ammonia_was_operating(simu_idx,df_simulation):
    # 如果上一时刻合成氨还在工作，就返回true
    return df_simulation.iloc[simu_idx-1][Cols.ammonia_power] > 0.

def initialize_simulation_df(df_solar_resample):
    # 声明计算数据带
    df_simulation = df_solar_resample
    df_simulation[Cols.AWES] = 0. # MW, 默认初始化功率为0
    df_simulation[Cols.urea_production] = 0. # 每小时的尿素产量，默认为0
    df_simulation[Cols.urea_power] = 0. # MW，默认0
    df_simulation[Cols.ammonia_power] = 0. # MW，默认0
    df_simulation[Cols.energy_storage_power] = 0. # MW, 储能充放电功率，默认为0
    df_simulation[Cols.curtailment_solar] = 0. # MW, 默认的弃光功率为0
    df_simulation[Cols.ammonia_production] = 0. # t
    df_simulation[Cols.ammonia_consumption] = 0. # t
    df_simulation[Cols.hydrogen_production] = 0. # t
    df_simulation[Cols.hydrogen_consumption] = 0. # t
    df_simulation[Cols.energy_storage] = Project_Config.Prod_Cap.Energy_storage_capacity # MWh，初始状态，储能一半
    df_simulation[Cols.hydrogen_storage] = Project_Config.Prod_Cap.Hydrogen_storage_capacity # t, 初始状态，储氢量的一半
    df_simulation[Cols.ammonia_storage] = Project_Config.Prod_Cap.Ammonia_storage_capacity / 5 # t, 总共可以储存10天的量，一开始有5天的量
    
    return df_simulation

def initialize_simulation(Project_Config):
    # 根据模式读取光伏文件并进行仿真，未来可以加入实地的光伏文件
    fp_solar = Project_Config.fp_solar
    df_solar = pd.read_csv(fp_solar) # 原始的额定功率为1500kW，这里直接当作MW计算

    df_solar.loc[:, Cols.time] = pd.to_datetime(df_solar[Cols.time])
    df_solar.loc[:, Cols.solar] = df_solar[Cols.solar] / 1500 * Project_Config.Prod_Cap.Solar
    df_solar.index = df_solar[Cols.time]
    df_solar_resample = df_solar.drop(columns=[Cols.time]).resample('h').mean().reset_index()
    df_simulation = initialize_simulation_df(df_solar_resample)
    return df_simulation

def initialize_variables():
    # 初始化每小时所需的变量
    AWES_cur = 0
    urea_power_cur = 0
    ammonia_power_cur = 0
    hydrogen_production_cur = 0
    hydrogen_consumption_cur = 0
    urea_production_cur = 0
    ammonia_production_cur = 0
    ammonia_consumption_cur = 0
    energy_storage_power_cur = 0
    return (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 充电时的储能模块功率由储能更新模块指定
    )

def hydrogen_production_discount(AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur):
    # 如果本身氢气已经储满，则如果合成氨消耗氢气量不如生产量，需要缩减此时的AWES功率
    if hydrogen_production_cur == 0:
        AWES_cur, hydrogen_production_cur
    hydrogen_ratio = min(
        hydrogen_consumption_cur / hydrogen_production_cur,
        1
    ) # 合成氨生产消耗不了这么多氢气，则制氢功率需要降低
    AWES_cur *= hydrogen_ratio
    hydrogen_production_cur *= hydrogen_ratio
    return AWES_cur, hydrogen_production_cur

def ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation):
    # 合成氨部分如果消耗的氢气超出现在的可用氢气，则需要对其功率进行折扣
    if hydrogen_consumption_cur <= 1E-5 :
        return 1.
    total_hydrogen_available = sum([
        df_simulation.at[simu_idx-1,Cols.hydrogen_storage], # 之前的氢气储量
        hydrogen_production_cur # 这一回合生产的氢气，这部分不会被ratio修正
    ])
    ammonia_ratio = min(
        total_hydrogen_available/hydrogen_consumption_cur,
        1
    ) # 如果total更大，就是1，意味着合成氨可以满功率运行；如果total较小，则需要乘上系数
    return ammonia_ratio

def print_config_attr():
    # 在主程序中初始化config后，再在这里引用Project_Config也已经是初始化后的结果
    # 因此在以后放在子文件中的函数，不需要再额外输入config了
    # 并且也可以在主程序中修改config中单独的属性，可以在这里读到新的结果
    Project_Config.print_all_attributes()