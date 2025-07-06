from abc import ABC
from keys import Files
import pandas as pd
from System_Design.utils import polar_function_fit,polar

class Project_config(ABC):

    class Prod_Cap:
        AWES = 1005 # MW
        AWES_num = 67 # 电解槽台数
        Wind = 0 # MW
        ES = 0 # MWh

    class Energy_Storage:
        margin_rate = 0.1 # 最后的无法利用的部分
        charge_efficiency = 0.95 # 充电时效率，假设充放电效率在一起考虑
        charge_rate = 0.5 # C
        discharge_rate = 0.5 # C
        min_capacity = 0 # MWh
        charge_power_max = 0 # MW
        discharge_power_max = 0 # MW

    class AWES:
        polar_data_file = Files.polar_raw_jlm_0627
        r1 = None
        r2 = None
        r3 = None
        energy_cost_discount_rate = 50/51.065163917624595 # 目前的额定能耗是51
        rated_current_density = 3000. # 假设电密是3000 A/m2
        min_rate = 0.4 # 最低负荷率
        max_rate = 1.1 # 最大负荷率
        compressor_power_ratio = 700./10000. # 每2000方氢气压缩与电解的耗电量比例，原始为700kW/2000方
        single_cap = 0 # MW，单台的额定功率

    def fit_polar(self):
        df_polar_raw = pd.read_csv(self.AWES.polar_data_file)
        self.AWES.r1, self.AWES.r2, self.AWES.r3 = polar_function_fit(df_polar_raw)

    def __init__(
        self,
        wind_cap = 200,
        ES_cap = 20
    ):
        self.Prod_Cap.Wind = wind_cap
        self.Prod_Cap.ES = ES_cap
        self.AWES.single_cap = self.Prod_Cap.AWES / self.Prod_Cap.AWES_num
        self.Energy_Storage.charge_power_max = self.Prod_Cap.ES * self.Energy_Storage.charge_rate
        self.Energy_Storage.discharge_power_max = self.Prod_Cap.ES * self.Energy_Storage.discharge_rate
        self.Energy_Storage.min_capacity = self.Prod_Cap.ES * self.Energy_Storage.margin_rate
        self.fit_polar()
        self.water_purification = 750/1000 # MW，海水淡化的固定功率
        self.wind_efficiency = 0.95 # 风力发电到电解制氢的电源效率
        self.threshold_1 = self.Prod_Cap.AWES*self.AWES.max_rate
        self.threshold_2 = self.AWES.single_cap * self.AWES.min_rate * 3 + self.water_purification
        self.threshold_3 = self.Energy_Storage.min_capacity # 储能最低可利用值
        self.threshold_4 = self.AWES.single_cap * self.AWES.min_rate + self.water_purification