from System_Design.Off_grid_electrolyzer_urea.configs import Project_Config
from keys import Cols

def update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation):
    # 更新尿素生产，一定是按照给定功率运行
    ammonia_available = df_simulation.iloc[simu_idx][Cols.ammonia_storage] + ammonia_production_cur * ammonia_ratio # 当前总体可用的氨
    if urea_power_cur < 1E-5:
        # 如果没有合成尿素，则不需要折扣
        urea_ratio = 1.
    else:
        # 如果当前的氨总量不足消耗总量，则需要对氨进行折扣
        urea_ratio = min(
            ammonia_available / ammonia_consumption_cur,
            1
        )    
    df_simulation.loc[
        simu_idx,
        [
            Cols.urea_power,
            Cols.urea_production
        ]
    ] = [
        urea_power_cur * urea_ratio,
        urea_production_cur * urea_ratio
    ]
    return df_simulation,ammonia_consumption_cur * urea_ratio

def update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ratio,df_simulation):
    # 更新合成氨，可能存在不能按照给定功率满负荷运行的情况
    ammonia_power_cur = ammonia_power_cur * ratio # 如果功率被限制，先复写
    ammonia_production_cur = ammonia_production_cur * ratio # 如果功率被限制，先复写
    ammonia_storage_delta = ammonia_production_cur - ammonia_consumption_cur # 消耗是不会被修改的
    df_simulation.loc[
        simu_idx,[
            Cols.ammonia_power,
            Cols.ammonia_production,
            Cols.ammonia_consumption
        ]
    ] = [
        ammonia_power_cur,
        ammonia_production_cur,
        ammonia_consumption_cur
    ]
    df_simulation.loc[simu_idx,[Cols.ammonia_storage]] = max(
        min(
            df_simulation.loc[simu_idx-1,[Cols.ammonia_storage]].item()+ammonia_storage_delta,
            Project_Config.Prod_Cap.Ammonia_storage_capacity
        ), # 超出容量的部分会被直接抛弃，功率也不会被折扣
        0 # 不能小于0
    )
    return df_simulation

def update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ratio,df_simulation):
    # 更新电解槽功率、氢气生产量与消耗量
    hydrogen_consumption_cur = hydrogen_consumption_cur * ratio # 如果合成氨消耗的氢气太多，最多也只能消耗之前的储量+现在的量
    hydrogen_storage_delta = hydrogen_production_cur - hydrogen_consumption_cur # 正则氢气储量增加
    df_simulation.loc[simu_idx,[
        Cols.AWES, # 电解槽功率
        Cols.hydrogen_production,
        Cols.hydrogen_consumption,
    ]] = [
        AWES_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur
    ]
    df_simulation.loc[simu_idx,[Cols.hydrogen_storage]] =  max(
        min(
            df_simulation.loc[simu_idx-1,[Cols.hydrogen_storage]].item()+hydrogen_storage_delta,
            Project_Config.Prod_Cap.Hydrogen_storage_capacity
        ), # 超出容量的部分会被直接抛弃，功率也不会被折扣
        0
    ) # 更新氢气储量
    return df_simulation

def update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ratio,df_simulation):
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光
    # 充电时候的情况和放电不同，需要分开来写方法
    if not energy_storage_power_cur > 0:
        residual_power_cur = solar / Project_Config.Efficiency.DC_DC - sum([
            AWES_cur,urea_power_cur,
            ammonia_power_cur * ratio
        ])
        energy_storage_power_cur = min(
            min(
                residual_power_cur,
                Project_Config.Prod_Cap.Energy_storage_capacity * Project_Config.Consume.Charge_speed # 最大不能超过充电的最大倍率
            ),
            Project_Config.Prod_Cap.Energy_storage_capacity - df_simulation.loc[simu_idx-1,Cols.energy_storage].item() # 不能超过储能的总体能量上限
        )
        curtailment_solar_cur = residual_power_cur - energy_storage_power_cur
    else:
        # 如果过在外部已经制定了储能功率，则意味着储能已经吃掉了所有剩余功率
        # 但是由于可能储能这时候充满，所以还是需要计算
        curtailment_solar_cur = solar / Project_Config.Efficiency.DC_DC  - sum([AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur])
    if energy_storage_power_cur > 0.:
        # 充电时不计算效率
        pass
    else:
        # 如果是对外放电，则需要增加消耗的电能
        energy_storage_power_cur = energy_storage_power_cur / Project_Config.Efficiency.Energy_storage
    df_simulation.loc[
        simu_idx,[
            Cols.energy_storage_power,
            Cols.curtailment_solar
        ]
    ] = [
        energy_storage_power_cur,
        curtailment_solar_cur
    ]
    df_simulation.loc[simu_idx,Cols.energy_storage] = df_simulation.loc[simu_idx-1,Cols.energy_storage] + energy_storage_power_cur
    return df_simulation

def update_energy_storage_discharge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ratio,df_simulation):
    # 在储能放电时，只有当弃光小于2.5MW才会被记录，因此忽略不计
    # 输入的energy_storage_power_cur一定是0
    energy_storage_power_cur = solar - sum([
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur * ratio # 合成氨的功率可能还是会受到限制
    ]) # 储能的功率应当是负值，意味着对外放电
    df_simulation.loc[
        simu_idx,[
            Cols.energy_storage_power,
            Cols.curtailment_solar
        ]
    ] = [
        energy_storage_power_cur,
        0. # 此时的弃光默认是0
    ]
    df_simulation.loc[simu_idx,Cols.energy_storage] = df_simulation.loc[simu_idx-1,Cols.energy_storage] + energy_storage_power_cur
    return df_simulation