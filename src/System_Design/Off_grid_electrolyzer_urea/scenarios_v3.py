# 分成四大类
from System_Design.Off_grid_electrolyzer_urea.configs import Project_Config
from keys import Cols
from System_Design.Off_grid_electrolyzer_urea.utils import *
from System_Design.Off_grid_electrolyzer_urea.update import *
from System_Design.Off_grid_electrolyzer_urea.calculate import *


# ——————————————————————————————————————————power determination——————————————————————————————————————————————————————
# 确定功率时，先确定尿素生产功率，再确定合成氨功率，最后确定制氢功率
def determine_urea_power(simu_idx,df_simulation,Project_Config,mode = 'rated'):
    # 根据当前情况判断合适的尿素生产功率
    operation_mode_dict = {
        'rated':Project_Config.Prod_Cap.Urea_electricity_rated ,
        'min' : Project_Config.Prod_Cap.Urea_electricity_rated * Project_Config.Prod_Cap.Urea_min_rate # 这里的尿素生产按照最低功率
    } # 可供选择的模式

    # 如果氨储量不足或为零，就只能是0了
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = 0,0,0

    if check_urea_was_operating(simu_idx=simu_idx,df_simulation=df_simulation):
        # 如果之前已经在工作，就检查一下还有没有氨
        if not check_ammonia_storage_empty(simu_idx=simu_idx,df_simulation=df_simulation):
            # 如果还有氨，就按照这里设定的模式来给功率
            urea_power_cur = operation_mode_dict[mode]
            urea_production_cur, ammonia_consumption_cur = calculate_urea_production(urea_power=urea_power_cur)
        else:
            pass
    else:
        # TODO: 当现在的氨储量不够时，应当增加合成氨的功率
        # 如果之前没在工作，就要判断一下氨储量够不够，如果不够当前就不启动，等合成氨再积累
        if check_ammonia_storage_sufficient(simu_idx, df_simulation):
            # 如果氨储量较高，就按照这里设定的模式来给功率
            urea_power_cur = operation_mode_dict[mode]
            urea_production_cur, ammonia_consumption_cur = calculate_urea_production(urea_power=urea_power_cur)
        else:
            pass
    return urea_power_cur,urea_production_cur,ammonia_consumption_cur


def determine_ammonia_power(simu_idx,df_simulation,Project_Config,mode = 'rated'):
    # 根据当前情况判断合适的合成氨生产功率
    operation_mode_dict = {
        'rated':Project_Config.Prod_Cap.Ammonia_electricity_rated ,
        'min' : Project_Config.Prod_Cap.Ammonia_electricity_rated * Project_Config.Prod_Cap.Ammonia_min_rate # 这里的合成氨生产按照最低功率
    } # 可供选择的模式

    # 如果氨储量不足或为零，就只能是0了
    ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur = 0,0,0

    if check_ammonia_was_operating(simu_idx=simu_idx,df_simulation=df_simulation):
        # 如果之前已经在工作，就检查一下还有没有氨
        if not check_hydrogen_storage_empty(simu_idx=simu_idx,df_simulation=df_simulation):
            # 如果还有氨，就按照这里设定的模式来给功率
            ammonia_power_cur = operation_mode_dict[mode]
            ammonia_production_cur, hydrogen_consumption_cur =  calculate_ammonia_production(ammonia_power=ammonia_power_cur)
    else:
        # 如果之前没在工作，就要判断一下氨储量够不够
        if check_hydrogen_storage_sufficient(simu_idx, df_simulation):
            # 如果氨储量较高，就按照这里设定的模式来给功率
            ammonia_power_cur = operation_mode_dict[mode]
            ammonia_production_cur, hydrogen_consumption_cur =  calculate_ammonia_production(ammonia_power=ammonia_power_cur)

    return ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur

def determine_AWES_power(simu_idx,df_simulation,solar,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,Project_Config,mode = 'full'):
    # 确定了尿素和合成氨功率后，再来确定电解槽的功率
    if mode == 'full':
        # 如果允许全功率运行
        if Project_Config.AWES.Overload:
            AWES_cur = min(
                solar - urea_power_cur - ammonia_power_cur, # 如果超过光伏剩余功率，则只能按照剩余功率
                Project_Config.Prod_Cap.AWES * ( 1 + Project_Config.AWES.Overload_rate) # 剩余功率较多，则可以全部跑满超负荷部分
            )
        else:
            AWES_cur = Project_Config.Prod_Cap.AWES
    elif mode == 'conditional':
        # 仍有光伏，但是不能全负荷工作
        AWES_cur = solar - sum(
            [
            urea_power_cur,
            ammonia_power_cur,
            energy_storage_power_cur
        ]
    )
    hydrogen_production_cur = calculate_hydrogen_production(
        AWES_cur,
        Project_Config=Project_Config,
    ) # 计算当前小时的氢气产量，这部分可以和满的氢气储存叠加
    return AWES_cur,hydrogen_production_cur


# ——————————————————————————————————————————Scenario 1——————————————————————————————————————————————————————
def scenario_1_over_threshold_1(solar, simu_idx, df_simulation,Project_Config):
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()

    # 确定尿素生产功率
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    # 确定合成氨功率
    ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur = determine_ammonia_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    # 确定了尿素和合成氨功率后，再来确定电解槽的功率
    AWES_cur,hydrogen_production_cur = determine_AWES_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        solar=solar,
        urea_power_cur=urea_power_cur,
        ammonia_power_cur=ammonia_power_cur,
        energy_storage_power_cur=energy_storage_power_cur, # 此时不管储能是多少，都按0来给
        Project_Config=Project_Config,
        mode='full'
    )

    # 按照氢气生产更新数据
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 制氢功率
    # 如果本身氢气已经储满，则如果合成氨消耗氢气量不如生产量，需要缩减此时的AWES功率
    if df_simulation.loc[simu_idx-1,Cols.hydrogen_storage] == Project_Config.Prod_Cap.Hydrogen_storage_capacity:
        AWES_cur, hydrogen_production_cur = hydrogen_production_discount(AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur)

    # 得到可以攻击给储能的最大功率
    energy_storage_cur = solar - sum(
        [
            AWES_cur,
            urea_power_cur,
            ammonia_power_cur
        ]
    )
    # 更新尿素生产，一定是按照给定功率运行
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨，可能存在不能按照给定功率满负荷运行的情况
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率、氢气生产量与消耗量
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光，充电
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation


def scenario_1_over_threshold_2(
        solar,
        simu_idx,
        df_simulation,
        Project_Config
    ):
    # 马上要进入夜晚，或者暂时功率不足，应当尽可能先充满储能，以保证所有生产可以正常运行
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()
    # 确定尿素生产功率
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    if not check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation):
        # 如果储能不满，就优先供给储能
        # 储能如果满了，剩下的功率就都给制氢
        residual_power_cur = solar - urea_power_cur
        energy_storage_power_cur = min(
            min(
                residual_power_cur,
                Project_Config.Prod_Cap.Energy_storage_capacity * Project_Config.Consume.Charge_speed # 最大不能超过充电的最大倍率
            ),
            Project_Config.Prod_Cap.Energy_storage_capacity - df_simulation.loc[simu_idx-1,Cols.energy_storage].item() # 不能超过储能的总体能量上限
        )
    # 确定了尿素和合成氨功率后，再来确定电解槽的功率
    AWES_cur,hydrogen_production_cur = determine_AWES_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        solar=solar,
        urea_power_cur=urea_power_cur,
        ammonia_power_cur=ammonia_power_cur,
        energy_storage_power_cur=energy_storage_power_cur, # 此时已经确定了储能的功率，消耗之后才能保留给制氢
        Project_Config=Project_Config,
        mode='conditional'
    )
    # 如果本身氢气已经储满，则如果合成氨消耗氢气量不如生产量，需要缩减此时的AWES功率
    if df_simulation.loc[simu_idx-1,Cols.hydrogen_storage] == Project_Config.Prod_Cap.Hydrogen_storage_capacity:
        AWES_cur, hydrogen_production_cur = hydrogen_production_discount(AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur)

    # 由于没有氨的生产，所以不存在合成氨的功率折扣
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 更新尿素生产
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率，没有合成氨工作，所以氢气消耗量为0
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光，充电
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation


def scenario_1_over_threshold_3(
        solar,
        simu_idx,
        df_simulation,
        Project_Config
    ):
    # 光伏功率已经不支持全功率储能充电+100%尿素
    # 所以这里赶紧给储能充电，剩下一些功率支持最底的尿素合成
    # 制氢和合成氨不工作
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()
    # 确定尿素生产功率，额定工作
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    if not check_energy_storage_full(simu_idx=simu_idx,df_simulation=df_simulation):
        # 剩余的功率优先供给储能
        residual_power_cur = solar - urea_power_cur
        energy_storage_power_cur = min(
            min(
                residual_power_cur,
                Project_Config.Prod_Cap.Energy_storage_capacity * Project_Config.Consume.Charge_speed # 最大不能超过充电的最大倍率
            ),
            Project_Config.Prod_Cap.Energy_storage_capacity - df_simulation.loc[simu_idx-1,Cols.energy_storage].item() # 不能超过储能的总体能量上限
        )

    # 由于没有氨的生产，所以不存在合成氨的功率折扣
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 更新尿素生产
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率，没有合成氨工作，所以氢气消耗量为0
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光，充电
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation

# scenario 1, 有光伏功率，储能可以充电
def scenario_1(solar, simu_idx, df_simulation,Project_Config):

    if solar >= Project_Config.Scenario_1.Threshold_1:
        df_simulation = scenario_1_over_threshold_1(
            solar,
            simu_idx,
            df_simulation,
            Project_Config
        )
    elif solar >= Project_Config.Scenario_1.Threshold_2:
        df_simulation = scenario_1_over_threshold_2(
            solar,
            simu_idx,
            df_simulation,
            Project_Config
        )
    else:
        # 这里一定大于 scenario_1.Threshold_3
        df_simulation = scenario_1_over_threshold_3(
            solar,
            simu_idx,
            df_simulation,
            Project_Config
        )
    return df_simulation

# ——————————————————————————————————————————Scenario 2——————————————————————————————————————————————————————
def scenario_2_over_threshold_1(
        solar, simu_idx, df_simulation,Project_Config):
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()

    # 确定尿素生产功率
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    # 确定合成氨功率
    ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur = determine_ammonia_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    # 确定了尿素和合成氨功率后，再来确定电解槽的功率
    AWES_cur,hydrogen_production_cur = determine_AWES_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        solar=solar,
        urea_power_cur=urea_power_cur,
        ammonia_power_cur=ammonia_power_cur,
        energy_storage_power_cur=energy_storage_power_cur, # 此时储能功率只能是0
        Project_Config=Project_Config,
        mode='full'
    )

    # 按照氢气生产更新数据
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 制氢功率
    # 如果本身氢气已经储满，则如果合成氨消耗氢气量不如生产量，需要缩减此时的AWES功率
    if df_simulation.loc[simu_idx-1,Cols.hydrogen_storage] == Project_Config.Prod_Cap.Hydrogen_storage_capacity:
        AWES_cur, hydrogen_production_cur = hydrogen_production_discount(AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur)

    # 得到可以攻击给储能的最大功率
    energy_storage_cur = solar - sum(
        [
            AWES_cur,
            urea_power_cur,
            ammonia_power_cur
        ]
    )
    # 更新尿素生产，一定是按照给定功率运行
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨，可能存在不能按照给定功率满负荷运行的情况
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率、氢气生产量与消耗量
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光，充电
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation

def scenario_2_over_threshold_2(
        solar,
        simu_idx,
        df_simulation,
        Project_Config
    ):
    # 快进入夜晚，或者光伏功率不足，此时储能已经充满，因此可以降低合成氨功率，同时保证氢气生产
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()
    # 确定尿素生产功率
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'rated'
    )
    # 确定合成氨功率
    if check_hydrogen_storage_half(simu_idx=simu_idx,df_simulation=df_simulation):
        # 如果氢气的储量比较多，就开满合成氨功率，以多储存氨
        ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur = determine_ammonia_power(
            simu_idx=simu_idx,
            df_simulation=df_simulation,
            Project_Config=Project_Config,
            mode = 'rated'
        )
    else:
        # 如果氢气储量不算多，这里采用最小功率
        ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur = determine_ammonia_power(
            simu_idx=simu_idx,
            df_simulation=df_simulation,
            Project_Config=Project_Config,
            mode = 'min'
        )

    # 确定了尿素和合成氨功率后，再来确定电解槽的功率
    AWES_cur,hydrogen_production_cur = determine_AWES_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        solar=solar,
        urea_power_cur=urea_power_cur,
        ammonia_power_cur=ammonia_power_cur,
        energy_storage_power_cur=energy_storage_power_cur, # 此时储能已经给满了，功率按照0来
        Project_Config=Project_Config,
        mode='conditional'
    )

    # 按照氢气生产更新数据
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 制氢功率
    # 如果本身氢气已经储满，则如果合成氨消耗氢气量不如生产量，需要缩减此时的AWES功率
    if df_simulation.loc[simu_idx-1,Cols.hydrogen_storage] == Project_Config.Prod_Cap.Hydrogen_storage_capacity:
        AWES_cur, hydrogen_production_cur = hydrogen_production_discount(AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur)
    # 更新尿素生产
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率，没有合成氨工作，所以氢气消耗量为0
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光，充电
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation

# scenario 2, 有光伏功率，储能已满，无法再充电
def scenario_2(solar, simu_idx, df_simulation,Project_Config):

    if solar >= Project_Config.Scenario_2.Threshold_1:
        df_simulation = scenario_2_over_threshold_1(
            solar,
            simu_idx,
            df_simulation,
            Project_Config
        )
    else:
        # 一定大于Threshold 2 的功率
        df_simulation = scenario_2_over_threshold_2(
            solar,
            simu_idx,
            df_simulation,
            Project_Config
        )
    return df_simulation

# ——————————————————————————————————————————Scenario 3——————————————————————————————————————————————————————
def scenario_3(solar, simu_idx, df_simulation,Project_Config):
    # 剩余功率优先进行生产氢气
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()

    # 确定尿素生产功率，按照最低允许功率运行
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'min'
    )

    power_available = calculate_power_available(
        solar = solar,
        urea=urea_production_cur,
        simu_idx = simu_idx,
        df_simulation = df_simulation,
        Project_Config = Project_Config
    )

    if power_available > Project_Config.Prod_Cap.Electrolyzer_power_single_min:
        AWES_cur = power_available
        hydrogen_production_cur = calculate_hydrogen_production(
            AWES_cur,
            Project_Config=Project_Config,
        ) # 计算当前小时的氢气产量，这部分可以和满的氢气储存叠加 # 计算当前小时的氢气产量，这部分可以和满的氢气储存叠加

    # 此时只消耗氨，生产氢气和尿素
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 更新尿素生产
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率，没有合成氨工作，所以氢气消耗量为0
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，弃光默认为0
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation


# ——————————————————————————————————————————Scenario 4——————————————————————————————————————————————————————
def scenario_4(solar, simu_idx, df_simulation,Project_Config):
    # 当前有一定的储氢量，可以用来进行合成氨
    (
        AWES_cur,
        urea_power_cur,
        ammonia_power_cur,
        hydrogen_production_cur,
        hydrogen_consumption_cur,
        urea_production_cur,
        ammonia_production_cur,
        ammonia_consumption_cur,
        energy_storage_power_cur # 这里用不到，充电时的功率由更新函数指定
    ) = initialize_variables()

    # 确定尿素生产功率，按照最低允许功率运行
    urea_power_cur,urea_production_cur,ammonia_consumption_cur = determine_urea_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'min'
    )

    power_available = calculate_power_available(
        solar = solar,
        urea=urea_production_cur,
        simu_idx = simu_idx,
        df_simulation = df_simulation,
        Project_Config = Project_Config
    ) # 可以利用的剩余功率应当高于合成氨的需求

    # 确定合成氨功率，这里还需要对最小功率进行修正
    ammonia_power_cur,ammonia_production_cur,hydrogen_consumption_cur = determine_ammonia_power(
        simu_idx=simu_idx,
        df_simulation=df_simulation,
        Project_Config=Project_Config,
        mode = 'min'
    )
    if power_available > Project_Config.Prod_Cap.Ammonia_electricity_rated * Project_Config.Prod_Cap.Ammonia_min_rate: 
        # 只要剩余功率大于合成氨最低功率
        # 如果剩余可用功率较小，就不进行合成氨制备，这部分功率就不使用，保留在储能中以防意外情况
        ammonia_power_cur = min(
            Project_Config.Prod_Cap.Ammonia_electricity_rated, # 允许按照最大功率运行，这样消耗的氢气可以制取很久
            power_available
        )
        ammonia_production_cur, hydrogen_consumption_cur = calculate_ammonia_production(ammonia_power=ammonia_power_cur)
    # 虽然理论上这时候氢气储量应该是满的，还是按照流程进行这算
    # 合成氨功率根据可用氢气量折扣
    ammonia_ratio = ammonia_power_discount( simu_idx,hydrogen_production_cur,hydrogen_consumption_cur,df_simulation)
    # 更新尿素生产
    df_simulation,ammonia_consumption_cur = update_urea(simu_idx,urea_power_cur,urea_production_cur,ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新合成氨
    df_simulation = update_ammonia(simu_idx, ammonia_power_cur, ammonia_production_cur,ammonia_consumption_cur,ammonia_ratio,df_simulation)
    # 更新电解槽功率，没有制氢，但是消耗了氢气
    df_simulation = update_hydrogen(simu_idx,AWES_cur,hydrogen_production_cur,hydrogen_consumption_cur,ammonia_ratio,df_simulation)
    # 更新储能和弃光数据，由于上面的氢不足产生的合成氨部分的剩余功率还会被计算入弃光，充电
    df_simulation = update_energy_storage_curtailment_charge(simu_idx,solar,AWES_cur,urea_power_cur,ammonia_power_cur,energy_storage_power_cur,ammonia_ratio,df_simulation)
    return df_simulation
