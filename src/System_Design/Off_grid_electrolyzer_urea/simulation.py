# 分成四大类
from System_Design.Off_grid_electrolyzer_urea.configs import Project_Config
from keys import Cols
from System_Design.Off_grid_electrolyzer_urea.utils import *
from System_Design.Off_grid_electrolyzer_urea.update import *
from System_Design.Off_grid_electrolyzer_urea.calculate import *
from System_Design.Off_grid_electrolyzer_urea.economics import *

from System_Design.Off_grid_electrolyzer_urea.report import *
from tqdm import tqdm


# ——————————————————————————————————————————单步仿真入口——————————————————————————————————————————————————————
def simulation_step_useless(simu_idx,df_simulation,Project_Config):
    # 单次仿真时长约为5.2ms
    # 应当注意要从simu_idx = 1开始，0是不能被仿真的
    # 区分工况场景
    # 在储氢总体容量比交大的时候，这个方法比较好
    from System_Design.Off_grid_electrolyzer_urea.scenarios_old import (
        scenario_1,scenario_2,scenario_3,scenario_4
    )
    solar = df_simulation.iloc[simu_idx][Cols.solar] 
    if solar >= Project_Config.Scenario_1.Threshold_3 and not check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 光伏功率大于最小阈值，且储能未满，则工况1
        df_simulation = scenario_1(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
    elif solar >= Project_Config.Scenario_2.Threshold_2 and check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 如果储能已经满，并且光伏功率大于工况2 的最小阈值
        df_simulation = scenario_2(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
        
    else: # 如果光伏功率低于阈值，则进入工况3、4，需要消耗储能
        # 此时还存在可能光伏还有<100MW的功率，也可以被利用
        # 根据控制策略，入夜时的储能应该是基本保持满电量的，应当肯定可以只少支撑尿素的夜间生产
        if not check_hydrogen_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): # 如果储氢未满，则为工况3
            df_simulation = scenario_3(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
        else: # 如果储氢满了，则为工况4
            df_simulation = scenario_4(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
    return df_simulation

def simulation_step_old(simu_idx,df_simulation,Project_Config):
    # 单次仿真时长约为5.2ms
    # 应当注意要从simu_idx = 1开始，0是不能被仿真的
    # 区分工况场景
    # 和基础版的主要不同，在于提高了场景4的优先级，只要还有氢气储量，就优先进行合成氨
    # 但仍然未解决连续几日光伏不稳定情况下的氨储量减少问题
    from System_Design.Off_grid_electrolyzer_urea.scenarios_old import (
        scenario_1,scenario_2,scenario_3,scenario_4
    )
    solar = df_simulation.iloc[simu_idx][Cols.solar] 
    if solar >= Project_Config.Scenario_1.Threshold_3 and not check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 光伏功率大于最小阈值，且储能未满，则工况1
        df_simulation = scenario_1(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
    elif solar >= Project_Config.Scenario_2.Threshold_2 and check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 如果储能已经满，并且光伏功率大于工况2 的最小阈值
        df_simulation = scenario_2(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
        
    else: # 光伏比较低的时候，只要还有氢气，就先制氨
        if not check_hydrogen_storage_empty(simu_idx=simu_idx,df_simulation=df_simulation): # 如果储氢达到一半，就优先进行合成氨
            df_simulation = scenario_4(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
        else: # 如果储氢空了，则制取氢气
            df_simulation = scenario_3(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
    return df_simulation

def simulation_step_v2(simu_idx,df_simulation,Project_Config):
    # 单次仿真时长约为5.2ms
    # 应当注意要从simu_idx = 1开始，0是不能被仿真的
    # 区分工况场景
    # 和基础版的主要不同，在于提高了场景4的优先级，只要还有氢气储量，就优先进行合成氨
    # 但仍然未解决连续几日光伏不稳定情况下的氨储量减少问题
    from System_Design.Off_grid_electrolyzer_urea.scenarios_v2 import (
        scenario_1,scenario_2,scenario_3,scenario_4
    )
    solar = df_simulation.iloc[simu_idx][Cols.solar] 
    if solar >= Project_Config.Scenario_1.Threshold_3 and not check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 光伏功率大于最小阈值，且储能未满，则工况1
        df_simulation = scenario_1(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
    elif solar >= Project_Config.Scenario_2.Threshold_2 and check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 如果储能已经满，并且光伏功率大于工况2 的最小阈值
        df_simulation = scenario_2(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
        
    else: # 光伏比较低的时候，只要还有氢气，就先制氨
        if not check_hydrogen_storage_empty(simu_idx=simu_idx,df_simulation=df_simulation): # 如果储氢达到一半，就优先进行合成氨
            df_simulation = scenario_4(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
        else: # 如果储氢空了，则制取氢气
            df_simulation = scenario_3(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
    return df_simulation

def simulation_step_v3(simu_idx,df_simulation,Project_Config):
    # 单次仿真时长约为5.2ms
    # 应当注意要从simu_idx = 1开始，0是不能被仿真的
    # 区分工况场景
    # 和基础版的主要不同，在于提高了场景4的优先级，只要还有氢气储量，就优先进行合成氨
    # 但仍然未解决连续几日光伏不稳定情况下的氨储量减少问题
    from System_Design.Off_grid_electrolyzer_urea.scenarios_v3 import (
        scenario_1,scenario_2,scenario_3,scenario_4
    )
    solar = df_simulation.iloc[simu_idx][Cols.solar] * Project_Config.Efficiency.DC_DC # 考虑电源效率，直接对可用电量进行打折
    if solar >= Project_Config.Scenario_1.Threshold_3 and not check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 光伏功率大于最小阈值，且储能未满，则工况1
        df_simulation = scenario_1(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
    elif solar >= Project_Config.Scenario_2.Threshold_2 and check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation): 
        # 如果储能已经满，并且光伏功率大于工况2 的最小阈值
        df_simulation = scenario_2(
            solar = solar,
            simu_idx = simu_idx,
            df_simulation = df_simulation,
            Project_Config = Project_Config
        )
        
    else: # 光伏比较低的时候，只要还有氢气，就先制氨
        if not check_hydrogen_storage_empty(simu_idx=simu_idx,df_simulation=df_simulation): # 如果储氢达到一半，就优先进行合成氨
            df_simulation = scenario_4(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
        else: # 如果储氢空了，则制取氢气
            df_simulation = scenario_3(
                solar = solar,
                simu_idx = simu_idx,
                df_simulation = df_simulation,
                Project_Config = Project_Config
            )
    return df_simulation

# ——————————————————————————————————————————整体仿真入口——————————————————————————————————————————————————————
def simulation_sequence(df_simulation,Project_Config,mode = 'old',verbose = 1):
    # 进行整段序列的仿真，并输出结果
    if mode  == 'old':
        if verbose == 1:
            for simu_idx in tqdm(range(1,len(df_simulation))):
                df_simulation = simulation_step_old(
                    simu_idx = simu_idx,
                    df_simulation = df_simulation,
                    Project_Config = Project_Config
                )
        else:
            for simu_idx in range(1,len(df_simulation)):
                df_simulation = simulation_step_old(
                    simu_idx = simu_idx,
                    df_simulation = df_simulation,
                    Project_Config = Project_Config
                )
    elif mode == 'v2':
        if verbose == 1:
            for simu_idx in tqdm(range(1,len(df_simulation))):
                df_simulation = simulation_step_v2(
                    simu_idx = simu_idx,
                    df_simulation = df_simulation,
                    Project_Config = Project_Config
                )
        else:
            for simu_idx in range(1,len(df_simulation)):
                df_simulation = simulation_step_v2(
                    simu_idx = simu_idx,
                    df_simulation = df_simulation,
                    Project_Config = Project_Config
                )
    elif mode == 'v3':
        if verbose == 1:
            for simu_idx in tqdm(range(1,len(df_simulation))):
                df_simulation = simulation_step_v3(
                    simu_idx = simu_idx,
                    df_simulation = df_simulation,
                    Project_Config = Project_Config
                )
        else:
            for simu_idx in range(1,len(df_simulation)):
                df_simulation = simulation_step_v3(
                    simu_idx = simu_idx,
                    df_simulation = df_simulation,
                    Project_Config = Project_Config
                )
    else:
        raise KeyError('Mode only support: old, continuous')
    (
        utilization_urea,
        utilization_AWES,
        utilization_ammonia,
        shut_down_urea,
        curtailment_solar_rate,
        total_electricity_consumed,
        hydrogen_production_total,
        urea_electricity_cost,
        hydrogen_energy_cost,
    ) = simulation_material_metrics(df_simulation,Project_Config,verbose = verbose)
    # 计算氢气生产能耗
    df_simulation[Cols.hydrogen_energy_cost] = df_simulation[Cols.AWES]/df_simulation[Cols.hydrogen_production]
    df_simulation[Cols.hydrogen_energy_cost] = df_simulation[Cols.hydrogen_energy_cost].fillna(0.)
    return (
        df_simulation,
        utilization_urea,
        utilization_AWES,
        utilization_ammonia,
        shut_down_urea,
        curtailment_solar_rate,
        total_electricity_consumed,
        hydrogen_production_total,
        urea_electricity_cost,
        hydrogen_energy_cost,
    )