from System_Design.Offshore_wind.calculate import *

def observe(df_simulation,simu_idx,project_config):
    # 读取用于决策的状态
    wind_cur = df_simulation.at[simu_idx,Cols.wind] * project_config.wind_efficiency # 风力发电要打一个电源效率的折扣
    ES_last = df_simulation.at[simu_idx-1,Cols.energy_storage]
    return wind_cur,ES_last

def scenario_1(wind_cur,ES_last,project_config):
    # 风电充沛，全力制氢，并给储能充电
    water_purification_power_cur = project_config.water_purification # MW
    AWES_power_cur = min(
        project_config.Prod_Cap.AWES*project_config.AWES.max_rate,
        wind_cur - water_purification_power_cur    
    )# MW, 按照最大功率制氢，包含了压缩部分的能耗
    ES_power_cur = available_ES_charge_power(
        wind_cur - water_purification_power_cur,
        ES_last,
        project_config
    )# MW，储能的充电功率，当前场景下只能充电，为正
    curtailment_wind_cur = max(
        0,
        wind_cur - water_purification_power_cur - ES_power_cur - AWES_power_cur
    )
    return water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur

def scenario_2(wind_cur,ES_last,project_config):
    # 风力相对不足，储能电量已满，全部功率制氢
    water_purification_power_cur = project_config.water_purification # MW
    AWES_power_cur = max(
        min(
            project_config.Prod_Cap.AWES*project_config.AWES.max_rate,
            wind_cur - water_purification_power_cur   
        ),
        project_config.AWES.single_cap * project_config.AWES.min_rate # 单台电解槽的最低功率
    )# MW, 剩余的功率用来制氢，包含了压缩部分的能耗
    ES_power_cur = available_ES_charge_power(
        wind_cur - water_purification_power_cur,
        ES_last,
        project_config
    ) # MW，储能的充电功率，当前储能已满
    curtailment_wind_cur = max(
        0,
        wind_cur - water_purification_power_cur - ES_power_cur - AWES_power_cur
    )
    return water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur

def scenario_3(wind_cur,ES_last,project_config):
    # 风力相对不足，需要全力对储能充电，剩下的功率制氢
    water_purification_power_cur = project_config.water_purification # MW
    ES_power_cur = available_ES_charge_power(
        wind_cur - water_purification_power_cur,
        ES_last,
        project_config
    )# MW，储能的充电功率，当前场景下只能充电，为正
    AWES_power_cur = max(
        wind_cur - water_purification_power_cur-ES_power_cur,
        0
    )
    curtailment_wind_cur = max(
        0,
        wind_cur - water_purification_power_cur - ES_power_cur - AWES_power_cur
    )
    return water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur

def scenario_4(wind_cur,ES_last,project_config):
    # 剩下的功率用来制氢，尽可能保证threshold-2运行
    water_purification_power_cur = project_config.water_purification # MW
    require_power_cur = project_config.threshold_2 - wind_cur # MW，除去风电后，还需要的功率
    ES_power_cur = actual_ES_discharge_power(
        require_power_cur,
        ES_last,
        project_config
    ) # fuzhi
    AWES_power_cur = wind_cur - ES_power_cur - water_purification_power_cur
    curtailment_wind_cur = 0 # 肯定是0了
    return water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur

def scenario_5(wind_cur,ES_last,project_config):
    # 剩下的功率用来制氢，尽可能保证threshold-2运行
    ES_power_cur = 0 # 储能肯定是不能充不能放了
    if wind_cur < project_config.water_purification:
        # 功率特别小的时候，也无法制氢了
        water_purification_power_cur = 0
        AWES_power_cur = 0
        curtailment_wind_cur = wind_cur
    elif wind_cur > project_config.threshold_4:
        # 能够满足至少一个开机
        water_purification_power_cur = project_config.water_purification
        AWES_power_cur = wind_cur - water_purification_power_cur
        curtailment_wind_cur = 0
    else:
        # 只能全都关机
        water_purification_power_cur = project_config.water_purification
        AWES_power_cur = 0
        curtailment_wind_cur = wind_cur - water_purification_power_cur
    return water_purification_power_cur,AWES_power_cur,ES_power_cur,curtailment_wind_cur
