import math

class Feedforward_PID_Controller:
    def __init__(self, setpoint, Kp, Ki, Kd, integral_min, integral_max, awe_model):
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.awe_model = awe_model
        self.prev_error = 0
        self.integral = 0

    # def para_update(self, current_density):
    #     """更新控制器的相关参数
    #     (当前只能手动更新
    #     后期应该构建更新的规则函数)
    #     """
    #     # self.Kp = Kp
    #     # self.Ki = Ki
    #     # self.Kd = Kd
    #     self.setpoint = setpoint_cal(current_density)

    def feedforward_lye_flow_cal(self, current_density):
        """
        前馈值的计算函数

        Args:
            current_density (float): 输入电流密度

        Returns:
            float: 前馈模型计算出的在某电流下达到目标温度对应的流速
        """

        lye_flow_target, lye_temp_target = self.awe_model.Working_Optimization(current_density=current_density,
                                                           Pressure=1.6)

        return lye_flow_target, lye_temp_target


    def lye_flow_update(self, current_density, Temperature):
        """
        前馈PID输出控制量的核心函数
        根据前馈和反馈数据计算应调节的碱液流量

        Args:
            current_density (float): 输入电流密度
            Temperature (float): 电解槽的当前出口温度

        Returns:
            float: 控制量，即碱液流速
        """
        feedforward_lye_flow, lye_temp_target = self.feedforward_lye_flow_cal(current_density)

        # 比例项
        error = self.setpoint - Temperature
        
        # 积分项，并规定积分项上下限
        self.integral += error
        if self.integral > self.integral_max:
            self.integral = self.integral_max
        elif self.integral < self.integral_min:
            self.integral = self.integral_min

        #微分项
        derivative = error - self.prev_error

        self.prev_error = error

        lye_flow_cal = feedforward_lye_flow + self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        return lye_flow_cal, lye_temp_target