CACHE_DIR = "../.cache"
MODELS_DIR = "../models"
REPORTS_DIR = "../reports"
FIGURES_DIR = "../figures"
NOTEBOOKS_DIR = "../notebooks"
LOGGER_DIR = '../logs'
CACHE_CHECKPOINT_DIR = '../.cache/checkpoint'
CACHE_TRAIN_DIR = '../.cache/train'
CACHE_DATA_DIR = '../.cache/data'

FIG_SEQUENCE_DICT = {
    0:'(a)',
    1:'(b)',
    2:'(c)',
    3:'(d)',
    4:'(e)',
    5:'(f)'
} # 绘图时的索引方法

class DataDir:
    """一些常用的数据路径"""
    Raw = "../data/raw"
    Int = "../data/interim"
    Prc = "../data/processed"


class Files:
    """一些常用的，不会改变的数据文件"""
    aquarel_theme_scientific = "../configs/aquarel_theme_scientific.json"
    solar_wind_pred_20s = '../data/raw/Solar Wind power 2021 20s with prediction gzip'
    hade_history_weather = '../data/raw/哈得历史天气_2023_06_2024_06.csv'
    hade_solar_1h = '../data/interim/Hade_solar_202306_2024_06.csv'
    polar_raw_jlm_0627 = '../data/raw/Polar raw data jlm 20240627.csv'


    
class PlotterOffset:
    # 画图时需要进行标注，这里需要进行相应的偏执，才能保证位置正确
    class Marker:
        class Cross:
            class Subplot4:
                current_density = -85
                lye_temperature = -1
                font_size = 16
            class Infrared:
                x = -4.5
                y = 2.5
                font_size = 14
                color = 'r'

class Cols:
    solar = 'solar' # 光伏发电功率，MW
    wind = 'wind' # 风力发电功率，MW
    time = 'time' # 应当是复合timestamp标准的时间
    date = 'date' # 标准日期格式
    hour = 'hour' # 一天中小时
    month = 'month' # 标准月份
    season = 'season' # 季节，符合constant中的标准
    date_time = 'date_time' # 非标准的时间格式
    
    energy_storage = 'energy_storage' # 储能，MWh
    energy_storage_power = 'energy_storage_power' # 储能充放电功率，MW
    hydrogen_storage = 'hydrogen_storage' # 储氢，t
    ammonia_storage = 'ammonia_storage' # 储氨，t
    AWES = 'AWES' # AWES系统功率，MW
    BoP = 'BoP' # BoP部分的功率，MW

    water_purification = 'water_purification' # 海水淡化的功率，MW
    urea_power = 'urea_power' # 尿素生产电耗，MW
    ammonia_power = 'ammonia_power' # 合成氨电耗，MW
    curtailment_solar = 'curtailment_solar' # 弃光功率，MW
    curtailment_wind = 'curtailment_wind' # 弃风功率，MW
    urea_production = 'urea_production' # t,当前小时的尿素产量，t
    ammonia_production = 'ammonia_production' # t, 氨合成量
    ammonia_consumption = 'ammonia_consumption' # t, 氨消耗量
    hydrogen_production = 'hydrogen_production' # t, 氢气生产量
    hydrogen_consumption = 'hydrogen_consumption' # t, 氢气消耗量
    hydrogen_energy_cost = 'hydrogen_energy_cost' # 单位生产能耗

    AWES_ratio = 'AWES_ratio' # AWES/光伏比例
    ammonia_ratio = 'ammonia_ratio' # 合成氨装机/额定需求比例
    overload_rate = 'overload_rate' # AWES超负荷比例
    utilization_urea = 'utilization_urea' # 尿素产能利用率
    utilization_AWES = 'utilization_AWES' # 制氢负荷率
    utilization_ammonia = 'utilization_ammonia' # 合成氨负荷率
    shut_down_urea = 'shut_down_urea' # 尿素停机时长 
    curtailment_solar = 'curtailment_solar' # 弃光量
    curtailment_solar_rate = 'curtailment_solar_rate' # 弃光率（%）
    total_income = 'total_income' # 尿素销售收入
    total_electricity_cost = 'total_electricity_cost' # 光伏电力采购成本
    total_electricity_consumed = 'total_electricity_consumed' # MWh，消耗的总电量
    urea_electricity_cost = 'urea_electricity_cost' # MWh/t，生产1吨尿素消耗的电能
    urea_cost = 'urea_cost' # ￥/吨，尿素生产时的电力消耗折算成本（暂时不考虑其他的成本）
    hydrogen_production_total = 'hydrogen_production_total' # 吨，氢气总生产量
    
    # 收入
    income_production = 'income_production' # 经济指标
    income_solar = 'income_solar' # 经济指标
    income_total = 'income_total' # 经济指标
    # 投资
    investment_production = 'investment_production' # 经济指标
    investment_solar = 'investment_solar' # 经济指标
    investment_total = 'investment_total' # 经济指标
    # 支出
    cost_production = 'cost_production' # 经济指标
    cost_solar = 'cost_solar' # 经济指标
    cost_total = 'cost_total' # 经济指标
    # 净利润
    net_profit_production = 'net_profit_production' # 经济指标
    net_profit_solar = 'net_profit_solar' # 经济指标
    net_profit_total = 'net_profit_total' # 经济指标
    # 净利润率
    NPM_production = 'NPM_production' # 经济指标
    NPM_solar = 'NPM_solar' # 经济指标
    NPM_total = 'NPM_total' # 经济指标
    # 投资回收周期
    PBP_production = 'PBP_production' # 经济指标
    PBP_solar = 'PBP_solar' # 经济指标
    PBP_total = 'PBP_total' # 经济指标

    current = 'current' # A
    current_density = 'current_density' # A/m2
    voltage = 'voltage' # V 整体电压，一正两副则*2
    cell_voltage = 'cell_voltage' # V
    pressure = 'pressure' # MPa 工作压力
    lye_flow = 'lye_flow' # m3/h 碱液流量
    lye_temp = 'lye_temp' # ℃ 碱液入口温度
    temp_H = 'temp_H' # ℃，氢侧槽温
    temp_O = 'temp_O' # ℃，氧侧槽温
    temp_out = 'temp_out' # ℃，出口温度平均
    temp_environment = 'temp_environment' # ℃，环境温度
    hydrogen_flow = 'hydrogen_flow' # Nm3/h，氢气流量计的度数
    HTO = 'HTO' # %, 氧中氢
    OTH = 'OTH' # %，氢中氧
    delta_temp_out = 'delta_temp_out' # ℃/s 出口温度变化率
    cooling_valve = 'cooling_valve' # % 冷却阀门开度