from System_Design.utils import *

def calculate_AWES_energy_cost(AWES_power,project_config):
    """对utils中的电流密度与能耗计算进行二次包装以适应当前的configs

    Args:
        AWES_power (_type_): 当前的电解槽功率
        project_config (_type_): 远海风电的相关设定

    Returns:
        _type_: 当前的制氢单位能耗kWh/kg
    """
    current_density = convert_power_to_current_density(
        AWES_power = AWES_power,
        prod_cap = project_config.Prod_Cap.AWES,
        rated_current_density = project_config.AWES.rated_current_density,
        min_rate = project_config.AWES.min_rate
    )
    energy_cost = calculate_current_density_to_energy_cost(
        current_density=current_density,
        r1 = project_config.AWES.r1,
        r2 = project_config.AWES.r2,
        r3 = project_config.AWES.r3,
        discount_rate = project_config.AWES.energy_cost_discount_rate        
    )
    energy_cost =  energy_cost * (1+project_config.AWES.compressor_power_ratio) # 压缩机的功率线性缩放
    return energy_cost

def available_ES_charge_power(residual_power_supply,ES_last,project_config):
    # 计算当前剩余功率与容量情况下的储能模块的充电功率
    ES_power_cur = min(
        residual_power_supply, # 剩余可以用于充电功率
        min(
            project_config.Energy_Storage.charge_power_max, #最大允许充电功率
            project_config.Prod_Cap.ES - ES_last # 目前剩余的容量
        )
    )
    return ES_power_cur

def actual_ES_discharge_power(require_power_cur,ES_last,project_config):
    # 计算当前储能功率需求情况下，剩余电量情况下的实际功率输出
    ES_power_cur = - min(
        require_power_cur, # 功率需求
        min(
            project_config.Energy_Storage.discharge_power_max, # 对外放电的最大功率
            ES_last - project_config.Energy_Storage.min_capacity # 剩余的电量
        )
    )
    return ES_power_cur