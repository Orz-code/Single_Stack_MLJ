# 可能会重复用到的计算方法
from System_Design.Off_grid_electrolyzer_urea.configs import Project_Config
from keys import Cols
from constants import Constants
from System_Design.Off_grid_electrolyzer_urea.data_preprocess import polar

def calculate_AWES_energy_cost(AWES_power,Project_Config):
    # 给定电解系统的功率，返回当前工作的能耗
    if AWES_power >= Project_Config.Prod_Cap.AWES * Project_Config.AWES.Electrolyzer_min_power_rate:
        # 整体功率高于30%时，可以让所有的电解槽都工作在一样的工况
        current_density = AWES_power/Project_Config.Prod_Cap.AWES * Project_Config.AWES.Rated_current_density
    else:    
        # 当功率小于整体的30%时，可以通过不同电解槽间开关机、功率分配，使得整体的功耗保持在30%的水平
        current_density = Project_Config.AWES.Electrolyzer_min_power_rate * Project_Config.AWES.Rated_current_density
    cell_voltage = polar(current_density,Project_Config.Polar.r1,Project_Config.Polar.r2,Project_Config.Polar.r3)
    energy_cost = cell_voltage * Constants.cell_voltage_to_energy_consumption * Project_Config.AWES.Energy_discount
    return energy_cost

def calculate_hydrogen_production(
        AWES_power,
        Project_Config=Project_Config,
    ):
    # TODO:这里需要根据不同的能耗来估算
    # TODO:还需要判断电解槽集群是否是全开的，如果不是全开的，则需要有不同的电压计算
    # 制氢能量一小时能够生产的氢气，t
    if Project_Config.AWES.Energy_cost_calculation == 'fixed':
        energy_cost = Project_Config.Consume.Hydrogen_electricity # 传统的固定氢耗
    elif Project_Config.AWES.Energy_cost_calculation == 'polar':
        energy_cost = calculate_AWES_energy_cost(AWES_power,Project_Config) # 氢耗跟随工况变化
    else:
        raise KeyError('Wrong Project_Config.AWES.Energy_cost_calculation')
    return AWES_power/energy_cost # 根据氢耗计算产氢量

def calculate_urea_production(urea_power):
    # urea_power, 实际的生产功率，MW，由于单次计算为1小时，所以这里其实是MWh
    # 一小时的尿素生产量，t
    # 以及一小时的合成氨消耗量，t
    urea_production = urea_power / Project_Config.Consume.Urea_electricity
    ammonia_consumption = urea_production / Project_Config.Consume.Urea_ammonia
    return urea_production,ammonia_consumption

def calculate_ammonia_production(ammonia_power):
    # 合成氨的实际功率取决于安排功率和实际能够消耗的氢气量
    # 出于计算方便，如果当前的功率超过了实际能够生产的氨量，则过剩的功率也不会被计入弃光
    # 氨产量与氢气消耗量均为t，且均为假设氢气重组情况下的结果，实际上的还需要按照氢气量计算
    ammonia_production = ammonia_power / Project_Config.Consume.Ammonia_electricity
    hydrogen_consumption = ammonia_production / Project_Config.Consume.Ammonia_hydrogen
    return ammonia_production, hydrogen_consumption

def calculate_overnight_hours(hour_cur):
    # 计算到第二天有光伏的时间差，用于估算储能还需要坚持多少个小时
    sun_rise_hour = Project_Config.Scenario_3.sun_rise_hour
    if hour_cur > sun_rise_hour and hour_cur <= 16:
        return 1 # 如果是白天光伏不足，应该也是暂时情况，只需要按照下一个小时会有光伏即可
    elif hour_cur > 16: #下午
        hour_cur = hour_cur - 24
    return sun_rise_hour - hour_cur + 1
    
def calculate_power_available(solar,simu_idx,df_simulation,Project_Config,urea = None):
    # 打包用于计算当前小时允许的除了最低尿素生产功率以外的功率
    if not urea:
        # 这里是为了适配之前的版本函数，scenarios——old
        urea = Project_Config.Prod_Cap.Urea_electricity_rated * Project_Config.Prod_Cap.Urea_min_rate
    hour_cur = df_simulation.iloc[simu_idx][Cols.time].hour
    overnight_hours = calculate_overnight_hours(hour_cur)
    energy_storage_cur = df_simulation.iloc[simu_idx-1][Cols.energy_storage]
    energy_storage_available = energy_storage_cur - Project_Config.Scenario_3.energy_storage_min
    energy_storage_for_urea = urea * overnight_hours # 要保证尿素生产的预留能量
    energy_storage_left = energy_storage_available - energy_storage_for_urea # 在剩余的小时中均匀消耗
    power_available = solar + max(
        0, # 如果出现了负值，就只能最多给0
        min(
            energy_storage_left / overnight_hours,
            Project_Config.Prod_Cap.Energy_storage_capacity * Project_Config.Consume.Discharge_speed # 放电允许的最大功率
        ) # 当前状态下允许的用于制氢、合成氨的功率
    )
    return power_available