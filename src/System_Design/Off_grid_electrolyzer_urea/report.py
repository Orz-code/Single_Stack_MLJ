import matplotlib.pyplot as plt
from keys import Cols
import pandas as pd


def simulation_report_annual(df_simulation, Project_Config):
    """完整展示全年的仿真工况结果，这里默认采用散点图来展现全年的结果

    Args:
        df_simulation (pd.DataFrame): 仿真的结果数据
        Project_Config (Project_Config): 仿真的config，可能会用到
    """
    xlim = [
        df_simulation[Cols.time].min(),
        df_simulation[Cols.time].max(),
    ]
    alpha_ = 0.1
    plt.figure(figsize=(16,24))
    plt.subplot(4,2,1)
    
    col = Cols.AWES
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    col = Cols.solar
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.legend()

    plt.subplot(4,2,2)
    col = Cols.urea_power
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.title(col)
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    plt.subplot(4,2,3)
    col = Cols.energy_storage_power
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_
    )
    plt.title(col)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    plt.subplot(4,2,6)
    col = Cols.ammonia_power
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_
    )
    plt.title(col)
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    plt.subplot(4,2,5)
    col = Cols.energy_storage
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title(col)

    plt.subplot(4,2,4)
    col = 'Ammonia balance'
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[Cols.ammonia_production]-df_simulation[Cols.ammonia_consumption],
        alpha = alpha_,
        label = 'Ammonia balance'
    )
    plt.xlim(xlim)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('Ammonia balance')

    plt.subplot(4,2,7)
    col = Cols.ammonia_storage
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title(col)

    plt.subplot(4,2,8)
    col = Cols.hydrogen_storage
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.title(col)
    plt.xticks(rotation = 15)
    plt.show()

def simulation_report_period(
    df_simulation, 
    Project_Config, 
    season = None,
    xlim = [
        pd.to_datetime('2021-10-13'),
        pd.to_datetime('2021-11-03'),
    ] 
):
    year = df_simulation[Cols.time].min().year
    if year == 2021:
        season_dict = {
            'spring':[
                pd.to_datetime('2021-02-15'),
                pd.to_datetime('2021-03-02'),
            ],
            'summer':[
                pd.to_datetime('2021-08-05'),
                pd.to_datetime('2021-08-20'),
            ],
            'fall':[
                pd.to_datetime('2021-10-05'),
                pd.to_datetime('2021-10-20'),
            ],
            'winter':[
                pd.to_datetime('2021-12-05'),
                pd.to_datetime('2021-12-20'),
            ]
        }
    elif year == 2023:
        season_dict = {
            'spring':[
                pd.to_datetime('2024-02-15'),
                pd.to_datetime('2024-03-02'),
            ],
            'summer':[
                pd.to_datetime('2023-08-05'),
                pd.to_datetime('2023-08-20'),
            ],
            'fall':[
                pd.to_datetime('2023-10-05'),
                pd.to_datetime('2023-10-20'),
            ],
            'winter':[
                pd.to_datetime('2023-12-05'),
                pd.to_datetime('2023-12-20'),
            ]
        }
    if season:
        try:
            xlim = season_dict[season]
        except KeyError:
            raise KeyError('季节输入有误，目前支持：'+ ', '.join(season_dict.keys()))
            
    alpha_ = 1
    plt.figure(figsize=(16,24))

    plt.subplot(4,2,1)
    col = Cols.AWES
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    col = Cols.solar
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    col = Cols.curtailment_solar
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.legend()

    plt.subplot(4,2,2)
    col = Cols.urea_power
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.title(col)
    plt.xlim(xlim)
    plt.xticks(rotation = 15)

    plt.subplot(4,2,3)
    col = Cols.energy_storage_power
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_
    )
    plt.title(col)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xlim(xlim)
    plt.xticks(rotation = 15)

    plt.subplot(4,2,6)
    col = Cols.ammonia_power
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_
    )
    plt.title(col)
    plt.xlim(xlim)
    plt.xticks(rotation = 15)

    plt.subplot(4,2,5)
    col = Cols.energy_storage
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.axhline(y = Project_Config.Prod_Cap.Energy_storage_capacity,c='grey',linestyle = '-.')
    plt.xlim(xlim)
    plt.title(col)
    plt.xticks(rotation = 15)

    plt.subplot(4,2,4)
    col = 'Ammonia balance'
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[Cols.ammonia_production]-df_simulation[Cols.ammonia_consumption],
        alpha = alpha_,
        label = 'Ammonia balance'
    )
    plt.xlim(xlim)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.title('Ammonia balance')
    plt.xticks(rotation = 15)

    plt.subplot(4,2,7)
    col = Cols.ammonia_storage
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.axhline(y = Project_Config.Prod_Cap.Ammonia_storage_capacity,c='grey',linestyle = '-.')
    plt.xlim(xlim)
    plt.title(col)
    plt.xticks(rotation = 15)

    plt.subplot(4,2,8)
    col = Cols.hydrogen_storage
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.axhline(y = Project_Config.Prod_Cap.Hydrogen_storage_capacity,c='grey',linestyle = '-.')
    plt.xlim(xlim)
    plt.title(col)
    plt.xticks(rotation = 15)


def simulation_report_annual_9grid(df_simulation, Project_Config):
    """完整展示全年的仿真工况结果，这里默认采用散点图来展现全年的结果

    Args:
        df_simulation (pd.DataFrame): 仿真的结果数据
        Project_Config (Project_Config): 仿真的config，可能会用到
    """
    # 改成9图
    xlim = [
        df_simulation[Cols.time].min(),
        df_simulation[Cols.time].max(),
    ]
    alpha_ = 0.1
    plt.figure(figsize=(24,18))

    # 第一行 光伏+AWES，尿素，合成氨
    plt.subplot(3,3,1)

    col = Cols.AWES
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    ) # AWES
    col = Cols.solar
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    ) # solar
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('光伏与制氢')
    plt.ylabel('功率（MW）')
    plt.legend()

    plt.subplot(3,3,2)
    col = Cols.urea_production
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.title('尿素生产')
    plt.ylabel('尿素生产速率（t/h）')
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    plt.subplot(3,3,3)
    col = Cols.ammonia_production    
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.title('氨生产')
    plt.ylabel('合成氨速率（t/h）')
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    # 第二行储能功率，氨平衡，氢能耗
    plt.subplot(3,3,4)
    col = Cols.energy_storage_power
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_
    )
    plt.title(col)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('储能充放电')
    plt.ylabel('储能功率（MW）')
    plt.xlim(xlim)

    plt.subplot(3,3,5)
    col = 'Ammonia balance'
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[Cols.ammonia_production]-df_simulation[Cols.ammonia_consumption],
        alpha = alpha_,
        label = 'Ammonia balance'
    )
    plt.xlim(xlim)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('氨生产/消耗平衡')
    plt.ylabel('氨储存平衡变化速率（t/h）')

    plt.subplot(3,3,6)
    col = Cols.hydrogen_energy_cost
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    if df_simulation[Cols.hydrogen_energy_cost].max() <47:
        plt.ylim([35,47])
    elif df_simulation[Cols.hydrogen_energy_cost].max() <48:
        plt.ylim([35,48])
    plt.axhline(y=45.,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('氢气生产能耗')
    plt.ylabel('系统单位制氢能耗（kWh/kg）')

    # 第三行，储能，储氨，储氢
    plt.subplot(3,3,7)
    col = Cols.energy_storage
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('储能工况')
    plt.ylabel('储能电量（MWh）')

    plt.subplot(3,3,8)
    col = Cols.ammonia_storage
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('储氨工况')
    plt.ylabel('氨储量（t）')

    plt.subplot(3,3,9)
    col = Cols.hydrogen_storage
    plt.scatter(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('储氢工况')
    plt.ylabel('氢储量（t）')

    plt.show()


def simulation_report_period_9grid(
    df_simulation, 
    Project_Config, 
    season = None,
    xlim = [
        pd.to_datetime('2021-10-13'),
        pd.to_datetime('2021-11-03'),
    ] 
):
    year = df_simulation[Cols.time].min().year
    if year == 2021:
        season_dict = {
            'spring':[
                pd.to_datetime('2021-02-15'),
                pd.to_datetime('2021-03-02'),
            ],
            'summer':[
                pd.to_datetime('2021-08-05'),
                pd.to_datetime('2021-08-20'),
            ],
            'fall':[
                pd.to_datetime('2021-10-05'),
                pd.to_datetime('2021-10-20'),
            ],
            'winter':[
                pd.to_datetime('2021-12-05'),
                pd.to_datetime('2021-12-20'),
            ]
        }
    elif year == 2023:
        season_dict = {
            'spring':[
                pd.to_datetime('2024-02-15'),
                pd.to_datetime('2024-03-02'),
            ],
            'summer':[
                pd.to_datetime('2023-08-05'),
                pd.to_datetime('2023-08-20'),
            ],
            'fall':[
                pd.to_datetime('2023-10-05'),
                pd.to_datetime('2023-10-20'),
            ],
            'winter':[
                pd.to_datetime('2023-12-05'),
                pd.to_datetime('2023-12-20'),
            ]
        }
    if season:
        try:
            xlim = season_dict[season]
        except KeyError:
            raise KeyError('季节输入有误，目前支持：'+ ', '.join(season_dict.keys()))
            
    alpha_ = 1
    plt.figure(figsize=(36,18))

    # 第一行 光伏+AWES，尿素，合成氨
    plt.subplot(3,3,1)

    col = Cols.AWES
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    ) # AWES
    col = Cols.solar
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    ) # solar
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('光伏与制氢')
    plt.ylabel('功率（MW）')
    plt.legend()

    plt.subplot(3,3,2)
    col = Cols.urea_production
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.title('尿素生产')
    plt.ylabel('尿素生产速率（t/h）')
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    plt.subplot(3,3,3)
    col = Cols.ammonia_production    
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.title('氨生产')
    plt.ylabel('合成氨速率（t/h）')
    plt.xticks(rotation = 15)
    plt.xlim(xlim)

    # 第二行储能功率，氨平衡，氢能耗
    plt.subplot(3,3,4)
    col = Cols.energy_storage_power
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_
    )
    plt.title(col)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('储能充放电')
    plt.ylabel('储能功率（MW）')
    plt.xlim(xlim)

    plt.subplot(3,3,5)
    col = 'Ammonia balance'
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[Cols.ammonia_production]-df_simulation[Cols.ammonia_consumption],
        alpha = alpha_,
        label = 'Ammonia balance'
    )
    plt.xlim(xlim)
    plt.axhline(y=0,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('氨生产/消耗平衡')
    plt.ylabel('氨储存平衡变化速率（t/h）')

    plt.subplot(3,3,6)
    col = Cols.hydrogen_energy_cost
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.ylim([35,47])
    plt.axhline(y=45.,c='r',linestyle = '-.')
    plt.xticks(rotation = 15)
    plt.title('氢气生产能耗')
    plt.ylabel('系统单位制氢能耗（kWh/kg）')

    # 第三行，储能，储氨，储氢
    plt.subplot(3,3,7)
    col = Cols.energy_storage
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('储能工况')
    plt.ylabel('储能电量（MWh）')

    plt.subplot(3,3,8)
    col = Cols.ammonia_storage
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('储氨工况')
    plt.ylabel('氨储量（t）')

    plt.subplot(3,3,9)
    col = Cols.hydrogen_storage
    plt.plot(
        df_simulation[Cols.time],
        df_simulation[col],
        alpha = alpha_,
        label = col
    )
    plt.xlim(xlim)
    plt.xticks(rotation = 15)
    plt.title('储氢工况')
    plt.ylabel('氢储量（t）')

    plt.show()

# 过滤函数
def filter_dataframe(df, **kwargs):
    """
    过滤数据框，根据传入的关键字参数和默认值筛选数据。
    未指定值的变量将使用默认值。
    
    参数：
    df (pd.DataFrame): 要过滤的数据框
    kwargs (dict): 关键字参数，对应自变量的筛选条件
    
    返回：
    pd.DataFrame: 筛选后的数据框
    """
    
    # 预设的默认自变量值
    default_values = {
        'solar': 3000.,
        'AWES_ratio': 0.5,
        'ammonia_ratio': 1.,
        'ammonia_storage': 2.,
        'hydrogen_storage': 250.,
        'energy_storage': 1000.,
        'overload_rate': 0.1
    }
    # 使用默认值更新kwargs中未提供的值
    filter_params = {**default_values, **kwargs}
    
    filter_condition = pd.Series([True] * len(df))
    
    for key, value in filter_params.items():
        if key in df.columns:
            if value is not None:
                filter_condition &= (df[key] == value)
    
    return df[filter_condition]
