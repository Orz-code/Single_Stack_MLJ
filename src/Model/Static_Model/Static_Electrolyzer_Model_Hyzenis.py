import math
import torch
import pickle
import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import root_scalar

import scipy.constants


Faraday_constant = 96485  # 法拉第常数
R_constant = scipy.constants.R  # 理想气体常数
Heat_capacity_lye = 3.35  # 30%KOH溶液比热容 J/(kg・K)

class AWE_Electrolyzer_Static():
    # 初始化函数
    def __init__(self, Diameter_Electrode, Width_Cell, Num_Cells, Lye_flow_min, Lye_flow_max, Static_Electrochemical_params_file_path, Static_Thermal_params_file_path):
        """电解槽初始化函数

        Args:
            Diameter_Electrode (float): 电极活性直径 m
            Width_Cell (float): 小室宽度 m
            Num_Cells (float): 小室数
            Lye_flow_min (float): 碱液流量最小值 ℃
            Lye_flow_max (float): 碱液流量最大值 ℃
            Static_Electrochemical_params_file_path (string): 存放稳态电化学模型参数的文件路径
            Static_Thermal_params_file_path (string): 存放稳态热模型参数的文件路径 
        """
        # 定义电解槽常量
        self.Diameter_Electrode = Diameter_Electrode
        self.Width_Cell = Width_Cell
        self.Num_Cells = Num_Cells
        self.Lye_flow_min = Lye_flow_min
        self.Lye_flow_max = Lye_flow_max

        # 定义电解槽常量引申出的计算值
        self.Area_Electrode = math.pi * (self.Diameter_Electrode / 2) ** 2
        self.Volume_active = math.pi * (self.Diameter_Electrode / 2) ** 2 * self.Width_Cell * self.Num_Cells

        # 读取稳态模型参数
        # 电化学模型参数
        self.Electrochemical_params = torch.load(Static_Electrochemical_params_file_path)
        Params = self.Electrochemical_params
        self.t1=Params['t1'].item()
        self.t2=Params['t2'].item()
        self.t3=Params['t3'].item()

        self.r1=Params['r1'].item()
        self.r2=Params['r2'].item()

        self.s1=Params['s1'].item()

        self.c1=Params['c1'].item()
        self.c2=Params['c2'].item()

        # 热模型
        self.Static_Thermal_params = torch.load(Static_Thermal_params_file_path)
        self.Heat_capacity_lye = self.Static_Thermal_params['Heat_capacity_lye'].item()
        self.Heat_capacity_stack = self.Static_Thermal_params['Heat_capacity_stack'].item()
        self.Surface_heat_transfer_coefficient = self.Static_Thermal_params['Surface_heat_transfer_coefficient'] .item()
        self.tg1 = self.Static_Thermal_params['tg1'].item()
        self.tc1 = self.Static_Thermal_params['tc1'].item()
        self.tc2 = self.Static_Thermal_params['tc2'].item()

        # 法拉第经验公式参数
        self.f11 = 1.067e4
        self.f12 = 101.1
        self.f21 = 0.989
        self.f22 = 7.641e-5

    # 热中性电压、可逆电压的计算函数
    def Vrev_Vtn_cal(self, Lye_temp, Temp_out, Pressure):
        """可逆电压的计算公式
        Args:
            Lye_temp(float): 进口温度，即碱液温度 ℃
            Temp_out (float): 电解槽工作温度（取进出口与温度的算数平均数）℃
            Pressure (float): 电解槽工作压力 MPa
        Returns:
            Vtn(float): 热中性电压 V
            Vrev(float): 可逆电压 V
        """
        Temp_work = (Lye_temp + Temp_out) / 2 + 273
        Tref = 298  # 参考点温度: K
        Pref = 1  # 参考点压力: bar
        Pressure = Pressure * 10  # 公式中压力单位bar，Mpa → bar
        z = 2  # 单位摩尔反应转移的电子数: mol

        Delta_H0_H2O = -2.86E5  # 参考点状态下的焓变(单位：J / mol)
        Delta_H0_H2 = 0  # 参考点状态下的焓变(单位：J / mol)
        Delta_H0_O2 = 0  # 参考点状态下的焓变(单位：J / mol)

        S0_H2O = 70  # 参考点状态下的熵值(单位：J / (K * mol))
        S0_H2 = 131  # 参考点状态下的熵值(单位：J / (K * mol))
        S0_O2 = 205  # 参考点状态下的熵值(单位：J / (K * mol))

        Cp0_H2O = 75  # 参考点状态下的水热容(单位：J / (K * mol))
        Cp0_H2 = 29  # 参考点状态下的氢气热容(单位：J / (K * mol))
        Cp0_O2 = 29  # 参考点状态下的氧气热容(单位：J / (K * mol))

        Delta_H_H2 = Cp0_H2 * (Temp_work - Tref) + Delta_H0_H2
        Delta_H_O2 = Cp0_O2 * (Temp_work - Tref) + Delta_H0_O2
        Delta_H_H2O = Cp0_H2O * (Temp_work - Tref) + Delta_H0_H2O
        Delta_H = Delta_H_H2 + 0.5 * Delta_H_O2 - Delta_H_H2O

        Vtn_cell = Delta_H / (z * Faraday_constant)

        S_H2 = Cp0_H2 * math.log(Temp_work / Tref) - R_constant * math.log(Pressure / Pref) + S0_H2
        S_O2 = Cp0_O2 * math.log(Temp_work / Tref) - R_constant * math.log(Pressure / Pref) + S0_O2
        S_H2O = Cp0_H2O * math.log(Temp_work / Tref) + S0_H2O
        
        Delta_S = S_H2 + 0.5 * S_O2 - S_H2O

        Delta_G = Delta_H - Temp_work * Delta_S

        Vrev_cell = Delta_G / (z * Faraday_constant)  # 可逆电压

        return Vrev_cell, Vtn_cell
    
    # 稳态电化学模型
    def Electrolytic_voltage_cal(self, Lye_temp, Temp_out, Current_density, Lye_flow, Pressure):
        """电解电压的计算模型

        Args:
            Lye_temp (float): 进口温度，即碱液温度 ℃
            Temp_out (float): 出口温度℃
            Current_density (float): 电流密度 A/m2
            Lye_flow (float): 碱液流量 m3/h
            Pressure (float): 压力 MPa

        Returns:
            float: 电解小室电压 V
        """

        Vrev_cell = self.Vrev_Vtn_cal(Lye_temp = Lye_temp,
                                      Temp_out = Temp_out,
                                      Pressure = Pressure)[0]

        # 避免 T_in 或 T_out 为零
        Lye_temp = max(Lye_temp, 1e-10)
        Temp_out = max(Temp_out, 1e-10)

        lambda_flow = Lye_flow / self.Volume_active / 25 + 0.1

        # temp_work = math.sqrt(max((1 - lambda_flow) * Lye_temp ** 2 * self.c1 + lambda_flow * Temp_out ** 2 * self.c2, 1E-10))
        temp_work = (1 - lambda_flow) * Lye_temp * self.c1 + lambda_flow * Temp_out * self.c2

        # 避免 log 函数的负数输入
        log_input = max((self.t1 + self.t2 / temp_work + self.t3 / temp_work ** 2) * Current_density + 1, 1e-10)
        
        Vcell = (self.r1 + self.r2 * temp_work) * Current_density + self.s1 * math.log(log_input) + Vrev_cell

        return Vcell
    
    # 稳态热模型
    def Delta_temp_cal(self, Lye_flow, Pressure, Cell_voltage, Current_density, Temp_out, Temp_environment, Lye_temp, Sample_time):
        """根据当前状态计算下一时刻的电解槽出口温度的稳态模型

        Args:
            Pressure (float): 压力 MPa
            Cell_voltage (float): 小室电压 V
            Current_density (float): 电流密度A/m2
            Temp_out (float): 出口温度
            Lye_flow (float): 碱液流量 m3/h
            Lye_temp (float): 碱液温度，即进口温度
            Temp_environment (float): 环境温度
            Sample_time (float): 采样时间

        Returns:
            float: 出口温度变化率
        """

        lye_flow = Lye_flow * 1.328 * 1E6 / 3600 # m3/h → g/s

        Vtn_cell = self.Vrev_Vtn_cal(Lye_temp=Lye_temp,
                                     Temp_out=Temp_out,
                                     Pressure=Pressure)[1]

        Qdot_gen = self.Num_Cells * (Cell_voltage - Vtn_cell) * Current_density * self.Area_Electrode * self.tg1 # 单位 W

        Qdot_loss = (Temp_out - Temp_environment) * self.Surface_heat_transfer_coefficient # 单位 W

        Qdot_lye_in = Heat_capacity_lye * lye_flow * Lye_temp * self.tc1 # 单位 W

        Qdot_lye_out = Heat_capacity_lye * lye_flow * Temp_out * self.tc2 # 单位 W

        Qdot_cool = Qdot_lye_out - Qdot_lye_in # 单位 W

        delta_Temp_H = (Qdot_gen - Qdot_loss - Qdot_cool) / self.Heat_capacity_stack # 单位 ℃/s

        delta_Temp_H_sample = delta_Temp_H * Sample_time

        return delta_Temp_H_sample
    
    # 稳态电热耦合模型
    def Static_Electrothermal_Coupling_Model(self, Current_density, Lye_temp, Temp_out, Temp_environment, Lye_flow, Pressure, Sample_time):
        """_summary_

        Args:
            Current_density (float): 电流密度 A/m2
            Lye_temp (float): 碱液温度，即进口温度
            Temp_out (float): 出口温度
            Temp_environment (float): 环境温度
            Lye_flow (float): 碱液流量 m3/h
            Pressure (float): 压力 MPa
            Sample_time (float): 采样时间

        Returns:
            float: 出口温度变化率
        """

        voltage_cell = self.Electrolytic_voltage_cal(Lye_temp = Lye_temp,
                                                     Temp_out = Temp_out,
                                                     Current_density = Current_density,
                                                     Lye_flow = Lye_flow,
                                                     Pressure = Pressure)

        Delta_Temp_out = self.Delta_temp_cal(Lye_flow=Lye_flow,
                                             Pressure=Pressure,
                                             Cell_voltage=voltage_cell,
                                             Current_density=Current_density,
                                             Temp_out=Temp_out,
                                             Temp_environment=Temp_environment,
                                             Lye_temp=Lye_temp,
                                             Sample_time=Sample_time)

        return Delta_Temp_out
    
    # 定义法拉第效率计算函数
    def Empirical_Faraday_efficiency_cal(self, Current_density, Temp_out, Lye_temp):
        """根据电流密度和温度计算当前的电流效率

        Args:
            Current_density (float): 电流密度 A/m2
            Temp_out (float): 电解槽工作温度 ℃
            Lye_temp (float): 碱液温度 ℃

        Returns:
            float: 电流效率 %
        """
        Temp_work = (Temp_out + Lye_temp) / 2
        yita_faraday = (Current_density ** 2 / (self.f11 + self.f12 * Temp_work + (Current_density ** 2))) * (self.f21 + self.f22 * Temp_work) * 0.97

        return yita_faraday
    
    # 定义产氢量计算函数
    def H_production_cal(self, Current_density, Temp_out, Lye_temp):
        """计算产氢量

        Args:
            current_density (float): 电流密度 A/m2
            temp_out (float): 出口温度 ℃
            lye_temp (float): 碱液温度 ℃

        Returns:
            float: 产氢量 Nm3/h
        """
        yita_faraday = self.Empirical_Faraday_efficiency_cal(Current_density = Current_density,
                                                    Temp_out = Temp_out,
                                                    Lye_temp = Lye_temp)
        H_production = yita_faraday * Current_density * self.Area_Electrode * self.Num_Cells * 3600 / Faraday_constant / 2 * 22.4 * 1E-3

        return H_production
    
    # 定义碱液泵耗电功率计算函数
    def Power_Lye_Pump_cal(self, Lye_flow):
        """碱液泵耗电功率

        Args:
            Lye_flow (float): 碱液流量 m3/h

        Returns:
            float: 碱泵电耗 W
        """
        Lye_pump_power = (Lye_flow / 0.5) **3 * 2.2 * 1E3

        return Lye_pump_power
    
    # 定义冷却功率需求计算函数
    def Power_Cooling_cal(self, Lye_flow, Temp_out, Lye_temp):
        """计算碱液的散热需求功率

        Args:
            Lyeflow (float): 碱液流量 m3/h
            Temp_out (float): 电解槽工作温度 ℃
            Lye_temp (float): 碱液温度 ℃

        Returns:
            float: 冷却需求 W
        """

        Lye_flow = Lye_flow * 1.328 * 1E6 / 3600 # 碱液流速单位换算Nm3/h → g/s
        Cooling_power = Heat_capacity_lye * Lye_flow * (Temp_out - Lye_temp)

        return Cooling_power
    
    # 定义单位产氢能耗计算函数
    def Power_per_H_cal(self, Lye_temp, Current_density, Temp_out, Pressure, Lye_flow):
        """计算单位产氢能耗

        Args:
            Lye_temp (float): 碱液温度
            Current_density (float): 电流密度
            Temp_out (float): 出口温度
            Pressure (float): 工作压力
            Lye_flow (float): 碱液流量

        Returns:
            flaot: _description_
        """

        power_lye_pump = self.Power_Lye_Pump_cal(Lye_flow = Lye_flow) # 碱液泵耗电功率 W

        power_cooling = self.Power_Cooling_cal(Lye_flow = Lye_flow,
                                          Temp_out = Temp_out,
                                          Lye_temp = Lye_temp) * 0.3 # 冷却功率 W
        
        voltage_cell = self.Electrolytic_voltage_cal(Current_density = Current_density,
                                              Temp_out = Temp_out,
                                              Lye_temp = Lye_temp,
                                              Lye_flow = Lye_flow,
                                              Pressure = Pressure)
        
        power_electrolysis = Current_density * voltage_cell * self.Area_Electrode * self.Num_Cells # 电功率

        power_system = power_electrolysis + power_lye_pump + power_cooling # 总耗电功率 W

        power_system = power_system * 1E-3 # 耗电功率单位换算W → kW

        H_production = self.H_production_cal(Current_density = Current_density,
                                        Temp_out = Temp_out,
                                        Lye_temp = Lye_temp) # 制氢速率 Nm3/h
        
        Power_per_H = power_system / H_production # 单位产氢耗电功率

        return Power_per_H
    
    # 定义在一定电解电流、一定出口温度、一定碱液流量、一定工作压力下计算对应的碱液温度的函数
    def Lye_temp_cal(self, Current_density, Temp_out, Temp_environment, Lye_flow, Pressure):
        """在一定电解电流、一定出口温度、一定碱液流量、一定工作压力下计算对应的碱液温度

        Args:
            Current_density (float): 电流密度 A/m2
            Temp_out (float): 出口温度 ℃
            Temp_environment (float): 环境温度 ℃
            Lye_flow (float): 碱液流速 m3/h
            Pressure (float): 工作压力 Mpa

        Raises:
            ValueError: 碱液温度上下限错误

        Returns:
            _type_: 碱液温度的近似值 ℃
        """

        # 定义目标函数
        def Lye_temp_cal_objective(Lye_temp):
            delta_temp_out_prediction = self.Static_Electrothermal_Coupling_Model(Current_density=Current_density,
                                                                   Lye_temp=Lye_temp,
                                                                   Temp_out=Temp_out,
                                                                   Temp_environment=Temp_environment,
                                                                   Lye_flow=Lye_flow,
                                                                   Pressure=Pressure,
                                                                   Sample_time=1)
            return delta_temp_out_prediction
        
        delta_temp_out_best = float('inf')
        for lye_temp in np.arange(Temp_out - 40, Temp_out, 0.01):
            delta_temp_out = abs(Lye_temp_cal_objective(Lye_temp=lye_temp))
            if delta_temp_out < delta_temp_out_best:
                delta_temp_out_best = delta_temp_out
                lye_temp_fitting = lye_temp
        
        lye_temp_fitting = round(number = lye_temp_fitting, ndigits = 2)
            
        return lye_temp_fitting
    
    # 定义一定输入电流密度下，电解槽最优工作状态的寻优函数
    def Working_Optimization(self, Current_density, Pressure):
        """_summary_

        Args:
            Current_density (float): 电流密度 A/m2
            Pressure (float): 压力 MPa

        Raises:
            ValueError: 电解槽关机

        Returns:
            float: 出口温度、碱液流量、碱液温度目标值
        """

        # 维持电解槽工作在高温区间
        temp_out_target = 85

        # 定义计算一定碱液流量下的单位制氢能耗计算函数
        def Power_consumption_Lye_flow(Lye_flow):
            Lye_flow = Lye_flow
            lye_temp = self.Lye_temp_cal(Current_density = Current_density,
                              Temp_out = temp_out_target,
                              Temp_environment = 25,
                              Lye_flow = Lye_flow,
                              Pressure = Pressure)

            power_consumption_Lye_flow = self.Power_per_H_cal(Lye_temp = lye_temp,
                                                     Current_density = Current_density,
                                                     Temp_out = temp_out_target,
                                                     Pressure = Pressure,
                                                     Lye_flow = Lye_flow)
            
            return lye_temp, power_consumption_Lye_flow
        
        if temp_out_target != None:
            # 穷举方法搜寻单位制氢能耗最低时对应的碱液流量
            power_consumption_best = float('inf')
            for lye_flow in np.arange(self.Lye_flow_min, self.Lye_flow_max + 0.01, 0.01):
                lye_temp, power_consumption = Power_consumption_Lye_flow(Lye_flow=lye_flow)
                if power_consumption < power_consumption_best:
                    power_consumption_best = power_consumption
                    lye_flow_target = lye_flow
                    lye_temp_target = lye_temp
        else:
                raise ValueError("电解槽关机")
            
        return lye_flow_target, lye_temp_target

if __name__ == '__main__':
    AWE_Electrolyzer = AWE_Electrolyzer_Static(Diameter_Electrode = 560 * 1E-3,
                                    Width_Cell = 5 * 1E-3,
                                    Num_Cells = 31,
                                    Lye_flow_min = 0.3,
                                    Lye_flow_max = 0.5,
                                    Static_Electrochemical_params_file_path = r'D:\Devs\Single_Stack_MLJ\src\Model\Static_Model\Static_Electrochemical_params_Hyzenis.pth',
                                    Static_Thermal_params_file_path = r'D:\Devs\Single_Stack_MLJ\src\Model\Static_Model\Static_Thermal_params_Hyzenis.pth')