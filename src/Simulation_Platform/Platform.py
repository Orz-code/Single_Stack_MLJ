import numpy as np
import pandas as pd
from keys import Cols

import matplotlib.pyplot as plt

class Simulation_Platform:
    def __init__(self,awe_static_model, awe_dynamic_model, awe_state_df, controller, time_step, total_time):
        self.awe_static_model = awe_static_model  # 电解槽稳态模型，搜寻电解最优工况
        self.awe_dynamic_model = awe_dynamic_model  # 电解槽动态模型，仿真电解槽实时状态
        self.awe_state_df = awe_state_df  # 存储电解槽状态的dataframe,已有20s电解槽运行状态
        self.controller = controller  # 控制器模型
        self.time_step = time_step  # 每个时间步的长度
        self.total_time = total_time  # 总仿真时间


    def load_current_data(self, current_data):
        """
        加载输入电流数据

        Args:
            current_data (_type_): 电流数据
        """
    
        self.awe_state_df[Cols.current_density] =  current_data[Cols.current] / self.awe_static_model.Area_Electrode

    def run_simulation(self):
        """
        运行仿真，并实时更新电解槽状态
        """
        
        for step in range(self.total_time):
            if step % self.time_step == 0:
                #控制器更新电解槽系统参数设置
                lye_flow_cal, lye_temp_target= self.controller.lye_flow_update(self.awe_state_df[Cols.current_density].iloc[step + 10], self.awe_state_df[Cols.temp_out])

            # 预测电解槽运行状态
            temp_out_next = self.awe_dynamic_model.awe_state_next_cal(self.awe_state_df.iloc[step:step+10])
            voltage_next = self.awe_static_model.Electrolytic_voltage_cal(Lye_temp=lye_temp_target,
                                                                     Temp_out=temp_out_next,
                                                                     Current_density=self.awe_state_df[Cols.current_density].iloc[step+10],
                                                                     Lye_flow=lye_flow_cal,
                                                                     Pressure=self.awe_state_df[Cols.pressure].iloc[step+10])
            # 在新索引位置添加值
            self.awe_state_df.loc[10 + step] = {Cols.voltage: voltage_next, Cols.temp_out: temp_out_next}
        
        print("Simulation completed.")

    def plot_result(self):
        plt.plot(self.awe_state_df)