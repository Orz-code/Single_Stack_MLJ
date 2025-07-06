from System_Design.Offshore_wind.calculate import *
from System_Design.Offshore_wind.scenarios import *
from keys import *
from tqdm import tqdm

import warnings
# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


def power_allocation(wind_cur,ES_last,project_config):
    # scenario这里只进行功率分配，
    if wind_cur > project_config.threshold_1:
        # 同时满足大量制氢、充电
        water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur = scenario_1(
            wind_cur,ES_last,project_config
        )
    elif wind_cur > project_config.threshold_2:
        if ES_last == project_config.Prod_Cap.ES:
            # 储能满电，直接制氢
            water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur = scenario_2(
                wind_cur,ES_last,project_config
            )
        else:
            # 储能电不满，优先充电
            water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur = scenario_3(
                wind_cur,ES_last,project_config
            )
    elif ES_last > project_config.threshold_3:
        # 利用剩余风电和储能制氢
        water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur = scenario_4(
            wind_cur,ES_last,project_config
        )
    else:
        # 没有储能，风电也不多，有多少算多少全用来制氢
        water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur = scenario_5(
            wind_cur,ES_last,project_config
        )
    return water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur

def update_simulation(
        df_simulation,
        simu_idx,
        water_purification_power_cur,
        AWES_power_cur,
        ES_power_cur,
        curtailment_wind_cur,
        project_config
    ):
    

    df_simulation.loc[simu_idx,Cols.water_purification] = float(water_purification_power_cur)
    df_simulation.loc[simu_idx,Cols.AWES] = float(AWES_power_cur)
    df_simulation.loc[simu_idx,Cols.energy_storage_power] = float(ES_power_cur)
    df_simulation.loc[simu_idx,Cols.curtailment_wind] = float(curtailment_wind_cur)
    df_simulation.loc[simu_idx,Cols.energy_storage] = float(df_simulation.loc[simu_idx-1,Cols.energy_storage] + ES_power_cur)
    hydrogen_production = calculate_AWES_energy_cost(AWES_power_cur,project_config)
    df_simulation.loc[simu_idx,Cols.hydrogen_production] = float(hydrogen_production)

    return df_simulation

def simulate(
        df_simulation,
        project_config,
        verbose = 0
    ):
    # 对整个序列进行仿真
    if verbose ==1 :
        for simu_idx in tqdm(range(1,len(df_simulation))):
            wind_cur,ES_last = observe(df_simulation,simu_idx,project_config)
            (
                water_purification_power_cur,
                AWES_power_cur,ES_power_cur,
                curtailment_wind_cur
            ) = power_allocation(wind_cur,ES_last,project_config)
            df_simulation = update_simulation(
                df_simulation,
                simu_idx,
                water_purification_power_cur,
                AWES_power_cur,
                ES_power_cur,
                curtailment_wind_cur,
                project_config
            )
    else:
        for simu_idx in range(1,len(df_simulation)):
            wind_cur,ES_last = observe(df_simulation,simu_idx,project_config)
            (
                water_purification_power_cur,
                AWES_power_cur,ES_power_cur,
                curtailment_wind_cur
            ) = power_allocation(wind_cur,ES_last,project_config)
            df_simulation = update_simulation(
                df_simulation,
                simu_idx,
                water_purification_power_cur,
                AWES_power_cur,
                ES_power_cur,
                curtailment_wind_cur,
                project_config
            )
    return df_simulation

def analyze_sequence(df_simulation):
    # 整体分析整个序列的氢气产量以及弃风率
    hydrogen_production_total = df_simulation[Cols.hydrogen_production].sum()
    curtailment_rate = df_simulation[Cols.curtailment_wind].sum()/df_simulation[Cols.wind].sum()
    return hydrogen_production_total,curtailment_rate