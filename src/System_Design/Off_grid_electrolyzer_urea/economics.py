import matplotlib.pyplot as plt
from keys import Cols
import pandas as pd

def simulation_material_metrics(df_simulation,Project_Config,verbose = 1):
    """计算simulation结果中的各种重要metrics，并且根据需要进行输出

    Args:
        df_simulation (_type_): 仿真结果
        Project_Config (_type_): 当前设定
        verbose (int, optional): 是否打印结果. Defaults to 1.
    """
    urea_production = df_simulation[Cols.urea_production].sum() # 尿素总生产量
    utilization_urea = df_simulation[Cols.urea_production].sum()/Project_Config.Prod_Cap.Urea_year*100 # 尿素产能利用率
    utilization_AWES = df_simulation[Cols.AWES].mean()/Project_Config.Prod_Cap.AWES*100 # 制氢平均负荷
    utilization_ammonia = df_simulation[Cols.ammonia_power].mean()/Project_Config.Prod_Cap.Ammonia_electricity_rated*100
    shut_down_urea = df_simulation.loc[
        df_simulation[Cols.urea_power] == 0
    ][Cols.AWES].count()/len(df_simulation)*100 # 尿素生产停机时长
    curtailment_solar_rate = (
        df_simulation[Cols.solar].sum() - sum(
            [
                df_simulation[Cols.AWES].sum(),
                df_simulation[Cols.urea_power].sum(),
                df_simulation[Cols.ammonia_power].sum()
            ]
        )
    ) / df_simulation[Cols.solar].sum() * 100.# 总弃光率（%），已经考虑了效率，但是可能有点出入，
    total_electricity_consumed = sum(
        [
            df_simulation[Cols.AWES].sum(),
            df_simulation[Cols.urea_power].sum(),
            df_simulation[Cols.ammonia_power].sum()
        ]
    ) / Project_Config.Efficiency.DC_DC # 总能耗，考虑能耗
    urea_electricity_cost = total_electricity_consumed/urea_production # WMh/t，尿素生产的能耗
    hydrogen_production_total = df_simulation[Cols.hydrogen_production].sum() # t，总制氢量
    hydrogen_energy_cost = df_simulation[Cols.AWES].sum() / Project_Config.Efficiency.DC_DC /hydrogen_production_total # MWh/t，kWh/kg制氢能耗

    if verbose == 1:
        print('尿素产能利用率：\t{:.2F}%'.format(utilization_urea))
        print('尿素停机占比：\t{:.2F}%'.format(shut_down_urea))
        print('系统光伏弃光率：\t{:.2F}%'.format(curtailment_solar_rate))
        print('尿素生产耗能：\t{:.2f} WWh/t'.format(total_electricity_consumed/urea_production))
        print('氢气生产能耗：\t{:.2f} MWh/t'.format(hydrogen_energy_cost))
    return (
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

def get_project_income_cost(
    cur_project,
    Project_Config
):
    """计算当前项目的收入、投资、支出

    Args:
        cur_project (pd.Series): 当前项目的仿真结果信息
        Project_Config (_type_): 通用的整体Config，economics部分不会变，尿素产能也不会变

    Returns:
        _type_: 生产、光伏、整体的收入、投资、支出
    """
    # 总体投资
    investment_production = cur_project[Cols.investment_production] # ￥，生产部份总投资
    investment_solar = cur_project[Cols.investment_solar] # ￥，光伏部份总投资
    investment_total = investment_production + investment_solar # ￥，系统总投资
    (
        income_production, income_solar, income_total,
        cost_production, cost_solar, cost_total
    ) = calculate_income_cost(
        urea_production = cur_project[Cols.utilization_urea] * Project_Config.Prod_Cap.Urea_year / 100,  # 吨，尿素产量
        total_electricity_consumed = cur_project[Cols.total_electricity_consumed],
        urea_price = Project_Config.Economics.Urea_sale,
        electricity_price = Project_Config.Economics.Electricity_price,
        interest_rate = Project_Config.Economics.Interest_rate,
        maintenance_rate = Project_Config.Economics.Maintenance,
        investment_production = investment_production, 
        investment_solar = investment_solar, 
        investment_total = investment_total,
    )
    return (
        income_production, income_solar, income_total,
        investment_production, investment_solar, investment_total,
        cost_production, cost_solar, cost_total
    )

def calculate_income_cost(
    urea_production,
    total_electricity_consumed,
    urea_price,
    electricity_price,
    interest_rate,
    maintenance_rate,
    investment_production, 
    investment_solar,
    investment_total,
):
    """根据项目的投资、生产计算其收入与支出

    Args:
        urea_production (float): 尿素年产量，吨
        urea_price (float): 尿素售价，￥/吨
        electricity_price (float): 电价，￥/MWh
        interest_rate (float): 资金的年利率，%/100
        maintenance_rate (float): 项目的运维年费率，%/100

    Returns:
        _type_: 生产、光伏、总体的收入、总投资、总支出
    """
    # 尿素销售收入
    income_production = urea_production * urea_price # ￥，尿素销售总收入
    income_solar = total_electricity_consumed * electricity_price # ￥，光伏电力成本/收入
    income_total = income_production # ￥，尿素销售总收入
    # 光伏费用/收入
    cost_electricity = total_electricity_consumed * electricity_price # ￥，光伏电力成本/收入
    # 运维成本
    maintenance_production = investment_production * maintenance_rate # ￥，生产部分每年的运维费用
    maintenance_solar = investment_solar * maintenance_rate # ￥，光伏部分每年的运维费用
    maintenance_total = maintenance_production + maintenance_solar # ￥，每年的总运维费用
    # 资金成本
    interest_production = investment_production * interest_rate # ￥，每年生产部分利息
    interest_solar = investment_solar * interest_rate # ￥，每年生产部分利息
    interest_total = interest_production + interest_solar # ￥，项目的整体利息
    # 一年总成本支出
    cost_production = cost_electricity + interest_production + maintenance_production # ￥，生产的年成本
    cost_solar = interest_solar + maintenance_solar # ￥，光伏的年成本
    cost_total = interest_total + maintenance_total # ￥，系统的整体年成本
    return (
        income_production, income_solar, income_total,
        cost_production, cost_solar, cost_total
    )

def calculate_profit_payback(
    income_production, income_solar, income_total,
    investment_production, investment_solar, investment_total,
    cost_production, cost_solar, cost_total
):

    # 净利润
    net_profit_production = income_production - cost_production # ￥，生产部分净利润
    net_profit_solar = income_solar - cost_solar # ￥，光伏部分净利润
    net_profit_total = income_total - cost_total # ￥，系统整体净利润
    # 净利率
    NPM_production = net_profit_production / investment_production * 100 # %，整体的年净利率
    NPM_solar = net_profit_solar / investment_solar * 100 # %，整体的年净利率
    NPM_total = net_profit_total / investment_total * 100 # %，整体的年净利率
    # 投资回报周期
    PBP_production = investment_production / net_profit_production # 年，投资回报周期
    PBP_solar = investment_solar / net_profit_solar # 年，投资回报周期
    PBP_total = investment_total / net_profit_total # 年，投资回报周期
    return (
            net_profit_production,net_profit_solar,net_profit_total,
            NPM_production,NPM_solar,NPM_total,
            PBP_production,PBP_solar,PBP_total,
        )
def get_project_economics(cur_project,Project_Config):
    (
        income_production, income_solar, income_total,
        investment_production, investment_solar, investment_total,
        cost_production, cost_solar, cost_total
    ) = get_project_income_cost(
        cur_project,
        Project_Config
    )
    (
        net_profit_production,net_profit_solar,net_profit_total,
        NPM_production,NPM_solar,NPM_total,
        PBP_production,PBP_solar,PBP_total,
    ) = calculate_profit_payback(
        income_production, income_solar, income_total,
        investment_production, investment_solar, investment_total,
        cost_production, cost_solar, cost_total
    )
    return (
        income_production, income_solar, income_total,
        cost_production, cost_solar, cost_total,
        net_profit_production,net_profit_solar,net_profit_total,
        NPM_production,NPM_solar,NPM_total,
        PBP_production,PBP_solar,PBP_total,
    )

def calculate_irr(initial_investment, annual_cash_flow, years):
    """
    Calculate the Internal Rate of Return (IRR).
    
    Parameters:
    initial_investment (float): The initial investment amount.
    annual_cash_flow (float): The annual cash flow.
    years (int): The number of years the project will generate cash flows.
    
    Returns:
    float: The calculated IRR.
    """
    # Create a list of cash flows
    cash_flows = [-initial_investment] + [annual_cash_flow] * years
    
    # Calculate IRR using numpy_financial's irr function
    irr = npf.irr(cash_flows)
    
    return irr