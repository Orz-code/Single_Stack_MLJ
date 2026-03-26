import pandas as pd
import numpy as np
from time import time
from utils.keys import Cols

# 数据管理基类
class BaseDataFrameHolder:
    """
    输入数据或状态数据的基类
    封装 DataFrame 的常用功能
    """
    def __init__(self, columns):
        """
        columns: 列名称字符串列表，例如 ["current", "lye_temp"]
        """
        self.columns = [Cols.date_time] + columns
        self.df = pd.DataFrame(columns=self.columns)

    def _validate_columns(self, kwargs):
        """校验传入的列名是否都在 DataFrame 中"""
        for key in kwargs:
            if key not in self.columns:
                raise ValueError(f"Unknown column '{key}'. Valid columns: {self.columns}")

    def add_row(self, timestamp=None, **kwargs):
        """
        添加一行数据：

        如果不提供 timestamp，会自动使用当前时间戳
        """
        self._validate_columns(kwargs)

        if timestamp is None:
            timestamp = time()

        row = {col: None for col in self.columns}
        row[Cols.date_time] = timestamp

        for k, v in kwargs.items():
            row[k] = v

        self.df.loc[len(self.df)] = row

    def update_last(self, **kwargs):
        """更新最后一行，可以用于控制器操作等"""
        self._validate_columns(kwargs)
        idx = self.df.index[-1]
        for k, v in kwargs.items():
            self.df.at[idx, k] = v

    def get(self, columns):
        """读取部分列"""
        if isinstance(columns, str):
            columns = [columns]
        return self.df[columns]

    def get_last(self, column):
        """读取最新值"""
        return self.df[column].iloc[-1]

    def interpolate(self, column, t):
        """
        按时间插值（对仿真器很有用）
        若 t 不在范围内，会根据 numpy.interp 的规则外推
        """
        if "time" not in self.df.columns:
            raise ValueError("Data must contain a 'time' column to interpolate.")
        return np.interp(t, self.df["time"], self.df[column])

    def save(self, path):
        """保存为 CSV"""
        self.df.to_csv(path, index=False)

    def load(self, path):
        """从 CSV 加载"""
        self.df = pd.read_csv(path)

    def trim_to_window(self, max_seconds):
        """仅保留最近 max_seconds 秒的数据，滚动删除更早数据"""
        if self.df.empty:
            return
        cutoff = time() - max_seconds
        if Cols.date_time not in self.df.columns:
            return
        self.df = self.df[self.df[Cols.date_time] >= cutoff]

    def __repr__(self):
        return f"{self.__class__.__name__}:\n{self.df.tail()}"

# 电解系统运行输入数据类
class InputData(BaseDataFrameHolder):
    """保存输入条件的类，例如 current, ambient_temp, flow_rate 等"""
    pass

# 电解系统运行状态数据类
class StateData(BaseDataFrameHolder):
    """保存系统状态的类，例如 temperature, voltage, pressure 等"""
    pass