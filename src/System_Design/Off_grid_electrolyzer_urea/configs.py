from keys import Files
import pandas as pd
from System_Design.Off_grid_electrolyzer_urea.data_preprocess import polar_function_fit

class Project_Config:
    # 离网绿电制氢合成氨、尿素项目的设定
    fp_solar = Files.hade_solar_1h
    class Prod_Cap: # 规划产能
        Solar_margin = 10. # MW, 计算方便，低于这个量的光伏在判断时按照0计算
        # 二氧化碳捕集等，是整个项目设计的起点，这部分的功率暂时不考虑
        CCUS_year = 900000 # t/a 吨/年，二氧化碳的总产能
        CCUS_day = CCUS_year / 365 # t/D
        CCUS_rated = CCUS_day / 24 # t/h，吨/小时，最后也可以看这部分的利用率

        # 尿素产能
        Urea_year = 1200000. # t/a 吨/年，尿素总体产能
        Urea_day = Urea_year / 365 # t/D 吨/天
        Urea_rated = Urea_day / 24 # t/h 吨/小时，由于这里计算的时候，都是以小时为单位进行仿真，所以这里最终折算到小时
        Urea_min_rate = 1. # 最低的尿素合成功率为额定点的0.75，且不允许停机

        # 合成氨产能
        Ammonia_rated = Urea_rated * 0.6 * 1  # t/h 吨/小时，额定生产速率，这里不考虑氨的状态变化的物质及能量损失，合成氨功率至少得是尿素额定消耗的两倍
        Ammonia_rated_year = Ammonia_rated * 365 * 24 # 全年的产能
        Ammonia_min_rate = 0.5 # 合成氨允许的最小功率比例，可以停机，目前还暂时不考虑最小功率的问题
        
        # 空分部分
        Air_rated = Ammonia_rated * 0.82 / 0.78 * 1.2 # t/h，氨-氮-空气-裕度
        Air_rated_year = Ammonia_rated_year * 0.82 / 0.78 * 1.2 # t/a，氨-氮-空气-裕度

        # Solar
        Solar = 3*1000 # MW 光伏发电的额定装机容量

        # AWES
        AWES = 1.5*1000 # WW 电解制氢的额定功率
        
        Electrolyzer_num = 300. # 300台槽
        Electrolyzer_power_single = 0. # 单台的功率
        Electrolyzer_power_single_min = 0. # 单台的最低功率

        # Storage
        Ammonia_storage_days = 4 # 总计可以储存10天的氨
        Ammonia_storage_capacity = Ammonia_rated * 24 * Ammonia_storage_days # t 
        Hydrogen_storage_capacity = 200. # t 总计的氢气储存量，可以修改
        Energy_storage_capacity = 2000. # MWh 储能模块的总体储能能力，可以修改

        # 尿素和合成氨的额定电耗
        Urea_electricity_rated = None
        Ammonia_electricity_rated = None
    class AWES:
        Rated_current_density = 3000. # A/m2，一般按照3000电密作为标准的额定电密
        Energy_discount = 0.8814 # 需要将额定点的系统能耗映射到45kWh/kg，这个需要根据电解槽随动调整
        Electrolyzer_min_power_rate = 0.3 # 每台功率最低为额定点0.3
        Overload = True # 是否允许超负荷
        Overload_rate = 0.5 # 功率最多可以超过额定值多少，0.5为50%
        Energy_cost_calculation = 'polar' # fixed、polar，按照哪种规则计算氢耗

    class Efficiency:
        DC_DC = 0.96 # DC/DC 电源效率为96%，光伏进来就得这个折扣，并且和弃光率怎么算是个问题
        Energy_storage = 0.95 # 储能充100度电，只能对外放95度电

    class Consume: # 生产的消耗
        Prod_Cap = None 
        # 碳捕集消耗
        CCUS_electricity = 0.666 # MWh/t CO2, 文献中第二页https://www.cup.edu.cn/sykxtb/docs//2023-09/2a65551f1d45470784156caa0aa2d9e3.pdf

        # 尿素消耗
        Urea_ccus =0.733333 # t/t 生产1吨尿素消耗约0.733吨二氧化碳
        Urea_ammonia = 1 / 0.6 # t/t 一般而言，生产1吨尿素需消耗0.57—0.62吨液氨，这里按照0.6吨计算
        Urea_electricity = 200/1000 # MWh/t

        # 氨消耗
        Ammonia_hydrogen = 1 / 0.176 # t/t，目前技术条件下，合成氨耗氢量约176kg/t，生产耗电量约1000kW·h/t？这里的耗电量可能和下面的对不上（https://www.escn.com.cn/20231012/7ce6620f6b2a4d50a9f23846ed3d899f/c.html）
        Ammonia_electricity = 300/1000 # MWh/t
        Hydrogen_electricity = 45000/1000 # MWh/t，生产1t氢气消耗4.5万度电，系统电耗，目前仍有难度

        # Energy_Storage
        Charge_speed = 0.5 # C 充电倍率
        Discharge_speed = 0.5 # C 放电倍率

    class Unit_Price: # 各类东西的价格
        # 能源来源
        # 生产部分
        CCUS = 0 # 这部分已经包含在生产尿素中
        Urea = 689570000/800000 # ￥/t年产能，80万吨尿素/6.8957亿：https://mp.weixin.qq.com/s/znpF7k-yj7kHYOOCUJiNFw
        Air = 4802300 # ￥/（吨/小时）产能空分投资：https://mp.weixin.qq.com/s/jPALxNmWNlrw93TReL8f8w
        Ammonia = 530 # ￥/t，根据合成氨消耗功率计算的系统总体成本，大概在1.5亿元/万吨产能，包括空分：https://pdf.dfcfw.com/pdf/H3_AP202402191622381924_1.pdf?1708332198000.pdf；这里有一个1.25亿元/万吨：https://mp.weixin.qq.com/s/nHgVhqihZRIrKZw-mHFF1Q；这里有一个成本特别高的https://mp.weixin.qq.com/s/DT9nyVVgbe5mL_4b_5_AnA
        AWES = 8000000/5 # ￥/MW, 电解水系统的单位价格，示例价格为氢气产能1000标方/小时的电解槽的价格
        # 储存部分
        Ammonia_Storage =  7180000 / 904 / 0.617 # ￥/t，1立方液氨=0.617吨，招标信息：https://ggzy.qz.gov.cn/art/2024/5/28/art_1229683630_691442.html
        Energy_Storage = 700000000/1000 # ￥/WWh, 采购储能设施的单位价格
        Hydrogen_Storage = 2500*1000 # ￥/t, 这个数据是老张给的，每公斤氢气储存设施的价格，来源：https://finance.sina.cn/2023-04-17/detail-imyqsxhc1515207.d.html?from=wap

    class Cost:# 各类型总投资
        System = None # 系统成本
        System_production = None # 不含光伏的系统成本

        # 能源来源部分
        Solar = None # 光伏系统成本
        # 生产部分
        CCUS = 0 # 这部分成本已经计算在尿素设备投资中
        Urea = None # 生产尿素的设备总投资
        Ammonia = None # 合成氨设备总投资，包括压缩机，但不包含空分
        Air = None # 空气分离总投资
        AWES = None # 制氢部分总投资
        # 储存部分
        Ammonia_storage = None # 液氨储存投资
        Hydrogen_storage = None # 储氢投资
        Energy_storage = None # 储能部分投资

    class Economics:# 最终经济性计算时的数据
        # 销售部分
        Urea_sale = 2000. # ￥/t，每吨尿素售价
        # 电力部分
        Electricity_price = 0.15*1000 # ￥/MWh，采购光伏发电的电力价格
        # 利息
        Interest_rate = 3.75/100 # %，贷款年化利率
        # 运维
        Maintenance = 1.5/100 # %，每年的其他运维费用相对于总投资的比例
        # 项目持有周期
        Years = [10,15,20,25] # 计算项目资本内部回报率时，使用的项目持有年限

    class Polar:
        data_file = Files.polar_raw_jlm_0627
        r1 = None
        r2 = None
        r3 = None
        
    class Continuous_operation:
        Ammonia_storage = None # 当氨储量大于多少时，才可以启动尿素生产
        Hydrogen_storage = None # 当氢储量大于多少时，才可以启动合成氨生产

    class Scenario_1: 
        # 光伏充足、储能可以充电
        Threshold_1 = 0. # MW，电解制氢、尿素、合成氨额定功率总和
        Threshold_2 = 0. # MW，50%单台电解槽+尿素额定功率+储能最大功率之和
        Threshold_3 = 0. # MW, 75%尿素合成功率，小于此功率则使用无光伏场景

    class Scenario_2:
        # 光伏充足，储能不能充电
        Threshold_1 = 0. # MW，满足制氢、尿素、合成氨功率之和
        Threshold_2 = 0. # MW，50%单台电解槽、100%尿素、50%合成氨功率之和

    class Scenario_3:
        # 光伏很少，要消耗储能，这里不需要区分储氢满不满什么的，合并成一个好了
        sun_rise_hour = 7 # 在早上七点后，可以认为有光照
        energy_storage_min_ratio = 0.1 # 储能最低能量不能低于此阈值
        energy_storage_min = 0. # MWh，储能的最低容量
    
    @classmethod
    def calculate_AWES_single(cls):
        # 计算单台套电解槽相关功率
        cls.Prod_Cap.Electrolyzer_power_single = cls.Prod_Cap.AWES / cls.Prod_Cap.Electrolyzer_num # 单台的功率
        cls.Prod_Cap.Electrolyzer_power_single_min = cls.Prod_Cap.Electrolyzer_power_single * cls.AWES.Electrolyzer_min_power_rate # 单台最低功率
    
    @classmethod
    def calculate_urea_electricity_rated(cls):
        # 计算合成尿素的电耗
        cls.Prod_Cap.Urea_electricity_rated =  cls.Consume.Urea_electricity * cls.Prod_Cap.Urea_rated# MWh/h 额定工况下生产尿素环节的电能消耗，在估算储能部分能量的时候，需要按照额定工况给尿素制造预留功率

    @classmethod
    def calculate_ammonia_electricity_rated(cls):
        # 计算合成氨的电耗
        cls.Prod_Cap.Ammonia_electricity_rated =  cls.Consume.Ammonia_electricity * cls.Prod_Cap.Ammonia_rated# MWh/h 额定工况下生产合成氨需要预留的功率，用于储能系统的能量预留
    
    @classmethod
    def initialize_power(cls):
        # 部分功率存在相互依赖，需要进行初始化
        cls.calculate_AWES_single()
        cls.calculate_urea_electricity_rated()
        cls.calculate_ammonia_electricity_rated()

    @classmethod
    def initialize_continuous_threshold(cls):
        cls.Continuous_operation.Ammonia_storage = cls.Prod_Cap.Ammonia_storage_capacity * 0.2 # 剩余氨20%容量才能开始连续合成尿素
        cls.Continuous_operation.Hydrogen_storage = cls.Prod_Cap.Hydrogen_storage_capacity * 0.2 # 剩余氢20%容量以上，才能开始连续的氨合成

    @classmethod
    def fit_polar(cls):
        df_polar_raw = pd.read_csv(cls.Polar.data_file)
        cls.Polar.r1, cls.Polar.r2, cls.Polar.r3 = polar_function_fit(df_polar_raw)

    @classmethod
    def calculate_project_cost(cls):
        # 能源来源
        cls.Cost.Solar = cls.Prod_Cap.Solar * cls.Unit_Price.Solar
        # 生产部分
        cls.Cost.CCUS = cls.Prod_Cap.CCUS_year * cls.Unit_Price.CCUS # 已经计算在尿素生产之内
        cls.Cost.Urea = cls.Prod_Cap.Urea_year * cls.Unit_Price.Urea # 生产尿素
        cls.Cost.Ammonia = cls.Prod_Cap.Ammonia_rated_year * cls.Unit_Price.Ammonia # 合成氨
        cls.Cost.Air = cls.Prod_Cap.Air_rated * cls.Unit_Price.Air # 这里是按照每小时产能计算
        cls.Cost.AWES = cls.Prod_Cap.AWES * cls.Unit_Price.AWES # 制氢
        # 储存部分
        cls.Cost.Ammonia_storage = cls.Prod_Cap.Ammonia_storage_capacity * cls.Unit_Price.Ammonia_Storage 
        cls.Cost.Hydrogen_storage = cls.Prod_Cap.Hydrogen_storage_capacity * cls.Unit_Price.Hydrogen_Storage
        cls.Cost.Energy_storage = cls.Prod_Cap.Energy_storage_capacity * cls.Unit_Price.Energy_Storage
        # 总体
        cls.Cost.System = sum([
            cls.Cost.Solar,
            cls.Cost.CCUS,
            cls.Cost.Urea,
            cls.Cost.Ammonia,
            cls.Cost.Air,
            cls.Cost.AWES,
            cls.Cost.Ammonia_storage,
            cls.Cost.Hydrogen_storage,
            cls.Cost.Energy_storage
        ])
        cls.Cost.System_production = sum([
            cls.Cost.CCUS,
            cls.Cost.Urea,
            cls.Cost.Ammonia,
            cls.Cost.Air,
            cls.Cost.AWES,
            cls.Cost.Ammonia_storage,
            cls.Cost.Hydrogen_storage,
            cls.Cost.Energy_storage
        ])
    @classmethod
    def calculate_scenario_thresholds(cls):
        cls.Scenario_1.Threshold_1 = sum([
            cls.Prod_Cap.AWES, # 电解系统的额定功率
            cls.Prod_Cap.Urea_electricity_rated, # 额定的尿素生产功率
            cls.Prod_Cap.Ammonia_electricity_rated # 额定的合成氨生产功率
        ]) # 可以光伏功率较高时的阈值功率

        cls.Scenario_1.Threshold_2 = sum([
            cls.Prod_Cap.Urea_electricity_rated, # 额定的尿素生产功率
            cls.Prod_Cap.Energy_storage_capacity * cls.Consume.Charge_speed, # 储能允许的最高充电功率
            cls.Prod_Cap.AWES / cls.Prod_Cap.Electrolyzer_num * cls.AWES.Electrolyzer_min_power_rate # 单电解槽允许的最小功率
        ]) # 全力进行储能充电的阈值功率
        cls.Scenario_1.Threshold_3 = cls.Prod_Cap.Urea_electricity_rated * cls.Prod_Cap.Urea_min_rate # 合成尿素允许的最低功率

        cls.Scenario_2.Threshold_1 = sum([
            cls.Prod_Cap.AWES, # 电解系统的额定功率
            cls.Prod_Cap.Urea_electricity_rated, # 额定的尿素生产功率
            cls.Prod_Cap.Ammonia_electricity_rated # 额定的合成氨生产功率
        ]) # 可以光伏功率较高时的阈值功，这部分对于scenario 1、2来说阈值是一样的
        cls.Scenario_2.Threshold_2 = sum([
            cls.Prod_Cap.Urea_electricity_rated, # 额定的尿素生产功率
            cls.Prod_Cap.AWES / cls.Prod_Cap.Electrolyzer_num * cls.AWES.Electrolyzer_min_power_rate, # 单电解槽允许的最小功率
            cls.Prod_Cap.Ammonia_electricity_rated * cls.Prod_Cap.Ammonia_min_rate # 合成氨允许的最小功率
        ])
        
        cls.Scenario_3.energy_storage_min = cls.Prod_Cap.Energy_storage_capacity * cls.Scenario_3.energy_storage_min_ratio # MWH，储能允许的最低电量
    
    # 初始化计算结果，避免每次调用时重复计算
    @classmethod
    def initiate_config(cls):
        cls.initialize_power()
        cls.calculate_scenario_thresholds()
        cls.calculate_project_cost()
        cls.initialize_continuous_threshold()
        cls.fit_polar()
    
    @classmethod
    def re_initiate_config(cls):
        # 在已经初始化之后，随着参数的修改，需要重新初始化
        cls.calculate_AWES_single()
        cls.calculate_ammonia_electricity_rated()
        cls.initialize_continuous_threshold()
        cls.calculate_project_cost()
        cls.calculate_scenario_thresholds()
        cls.fit_polar()
        
    @classmethod
    def get_project_investment_detail(cls):
        return (
            cls.Cost.System,
            cls.Cost.System_production,
            cls.Cost.Solar,
            cls.Cost.CCUS,
            cls.Cost.Urea,
            cls.Cost.Ammonia,
            cls.Cost.Air,
            cls.Cost.AWES,
            cls.Cost.Ammonia_storage,
            cls.Cost.Hydrogen_storage,
            cls.Cost.Energy_storage,
        )

    @classmethod
    def get_project_investment(cls):
        return (
            cls.Cost.System,
            cls.Cost.System_production,
            cls.Cost.Solar,
        )
    
    @classmethod
    def draw_cost_breakdown(cls):
        import matplotlib.pyplot as plt
        (
            Cost_System,
            Cost_System_production,
            Cost_Solar,
            Cost_CCUS,
            Cost_Urea,
            Cost_Ammonia,
            Cost_Air,
            Cost_AWES,
            Cost_Ammonia_storage,
            Cost_Hydrogen_storage,
            Cost_Energy_storage,
        ) = Project_Config.get_project_investment_detail()
        total = sum([
            Cost_Urea,
            Cost_Ammonia,
            Cost_Air,
            Cost_AWES,
            Cost_Ammonia_storage,
            Cost_Hydrogen_storage,
            Cost_Energy_storage,
        ])
        plt.pie(
            x=(
                Cost_Urea,
                Cost_Ammonia,
                Cost_Air,
                Cost_AWES,
                Cost_Ammonia_storage,
                Cost_Hydrogen_storage,
                Cost_Energy_storage,
            ),
            autopct='%1.1f%%',
            startangle= 90, # y轴开始,
            counterclock = False
        )
        labels = [
            '尿素（含碳捕集）: {:.1f}亿'.format(Cost_Urea/1E8),
            '合成氨: {:.1f}亿'.format(Cost_Ammonia/1E8),
            '空分: {:.1f}亿'.format(Cost_Air/1E8),
            'AWES系统: {:.1f}亿'.format(Cost_AWES/1E8),
            '储氨: {:.1f}亿'.format(Cost_Ammonia_storage/1E8),
            '储氢: {:.1f}亿'.format(Cost_Hydrogen_storage/1E8),
            '储能: {:.1f}亿'.format(Cost_Energy_storage/1E8)
        ]

        # 调整图例位置到下方
        plt.legend(
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),  # 图例位置设置为正下方
            ncol=3  # 图例列数设置为3列，可以根据需要调整
        )
        plt.title(
            '系统总投资：{:.1f}亿元，其中光伏{:.1f}亿元，绿氢-氨-尿素{:.1f}亿元'.format(
                Cost_System/1E8,
                Cost_Solar/1E8,
                Cost_System_production/1E8
            )
        )

    @classmethod
    def print_all_attributes(cls):
        def print_attrs(obj, obj_name):
            attrs = {attr: getattr(obj, attr) for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")}
            print(f"{obj_name} 属性:")
            for attr, value in attrs.items():
                print(f"  {attr}: {value}")

        def traverse_and_print(obj, obj_name):
            print_attrs(obj, obj_name)
            for attr in dir(obj):
                if attr.startswith("__"):
                    continue
                value = getattr(obj, attr)
                if isinstance(value, type) and issubclass(value, object) and value is not object:
                    traverse_and_print(value, f"{obj_name}.{attr}")

        traverse_and_print(cls, cls.__name__)