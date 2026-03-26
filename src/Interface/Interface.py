import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import logging
from time import time, sleep
import pandas as pd
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import matplotlib
import os
import sys
matplotlib.use("TkAgg") 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# 设置字体
matplotlib.rcParams['font.family'] = 'Microsoft YaHei', 'Times New Roman'
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False

from utils.keys import Cols
from Statistics_manager import InputData, StateData
from pymodbus.client import ModbusTcpClient

from Static_Electrolyzer_Model_Hyzenis import AWE_Electrolyzer_Static

def resource_path(relative_path):
    """在开发与 PyInstaller 运行环境下获取资源路径（相对当前文件/打包目录）"""
    try:
        base_dir = sys._MEIPASS  # PyInstaller 临时目录
    except Exception:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, relative_path))

awe_static_model = AWE_Electrolyzer_Static(Diameter_Electrode = 560 * 1E-3,
                                    Width_Cell = 5 * 1E-3,
                                    Num_Cells = 31,
                                    Lye_flow_min = 0.3,
                                    Lye_flow_max = 0.5
)

# ===================================================================
# 核心后端代码 START
# ===================================================================

class ControlSystem:    
    def __init__(self, modbus_ip="127.0.0.1", modbus_port=502, modbus_id=1):
        self.modbus_ip = modbus_ip
        self.modbus_port = modbus_port
        self.modbus_id = modbus_id
        self.client = None
        self.cycle_count = 0 # 用于绘图计数

        # 统一的 all_data DataFrame，保存时间 + 6 个实时监测项 + 2 个目标值
        cols = [
            Cols.current,
            Cols.voltage,
            Cols.lye_flow,
            Cols.lye_temp,
            Cols.temp_out,
            Cols.pressure,
            Cols.lye_flow_target,
            Cols.lye_temp_target,
        ]
        # 使用 InputData（继承自 BaseDataFrameHolder）作为通用容器
        self.all_data = InputData(columns=cols)
        # 保持向后兼容：input_data / state_data 指向同一数据结构
        self.input_data = self.all_data
        self.state_data = self.all_data

        # 待写入的最近一次计算出的目标值（在第(6n+1)步计算，在第(6n+2)步写入）
        self.pending_targets = None
        # PLC 状态寄存器最近值（地址 0），0 表示停机，1 表示运行
        self.last_status = None

    def _initialize_input_data(self):
        return self.all_data

    def _initialize_state_data(self):
        return self.all_data

    # --- Modbus 连接与断开 ---
    def connect_modbus(self):
        if self.client and self.client.is_socket_open():
            logging.info("Modbus连接已存在并打开。")
            return True
        try:
            self.client = ModbusTcpClient(self.modbus_ip, port=self.modbus_port)
            if self.client.connect():
                logging.info(f"成功连接到Modbus设备: {self.modbus_ip}:{self.modbus_port} (ID={self.modbus_id})")
                return True
            else:
                logging.error(f"无法连接到Modbus设备: {self.modbus_ip}:{self.modbus_port} (ID={self.modbus_id})")
                self.client = None
                return False
        except Exception as e:
            logging.error(f"Modbus连接异常: {e}")
            self.client = None
            return False

    def disconnect(self):
        try:
            if self.client:
                self.client.close()
                logging.info("Modbus连接已断开。")
        except Exception as e:
            logging.error(f"断开Modbus连接时发生错误: {e}")

    # --- 兼容不同 pymodbus 版本的读写封装 ---
    def _read_holding(self, address, count):
        try:
            return self.client.read_holding_registers(address=address, count=count, slave=self.modbus_id)
        except TypeError:
            try:
                return self.client.read_holding_registers(address=address, count=count, unit=self.modbus_id)
            except TypeError:
                return self.client.read_holding_registers(address=address, count=count)

    def _write_registers(self, address, values):
        try:
            return self.client.write_registers(address=address, values=values, slave=self.modbus_id)
        except TypeError:
            try:
                return self.client.write_registers(address=address, values=values, unit=self.modbus_id)
            except TypeError:
                return self.client.write_registers(address=address, values=values)

    # --- 辅助方法 ---
    def _last_nonnull(self, column):
        """返回指定列最后一个非空值，找不到返回 None"""
        if self.all_data.df.empty:
            return None
        s = self.all_data.df[column].dropna()
        return s.iloc[-1] if not s.empty else None

    # --- 数据读取与存储（每 10s 读取状态+6 个监测寄存器） ---
    def read_and_store_data(self, auto_write=True):
        if not self.client or not self.client.is_socket_open():
            logging.warning("客户端未连接或连接已断开，无法读取数据。")
            return False

        # 状态寄存器（地址 0）用于判定是否继续运行
        STATUS_ADDRESS = 0
        # 监测数据寄存器起始地址 11：current、lye_flow*100、lye_temp*10、pressure*10、voltage*10、temp_out*10
        READ_START_ADDRESS = 11
        READ_COUNT = 6
        mapping = [Cols.current, Cols.lye_flow, Cols.lye_temp, Cols.pressure, Cols.voltage, Cols.temp_out]
        
        try:
            status_result = self._read_holding(address=STATUS_ADDRESS, count=1)
            if status_result.isError():
                logging.error(f"❌ 读取状态寄存器失败: {status_result}")
                return False
            self.last_status = status_result.registers[0] if status_result.registers else None
            if self.last_status == 0:
                logging.warning("⚠️ 检测到工作状态为 0，自动停止控制循环。")
                return False

            result = self._read_holding(address=READ_START_ADDRESS, count=READ_COUNT)
            if result.isError():
                logging.error(f"❌ 读取 Modbus 数据失败: {result}")
                return False

            raw_values = result.registers
            current_timestamp = time()

            if len(raw_values) < READ_COUNT:
                logging.error(f"❌ 读取到的数据不足 {READ_COUNT} 个，跳过存储。")
                return False

            # 解码所有寄存器
            row_kwargs = {}
            for i, col in enumerate(mapping):
                raw = raw_values[i]
                if col == Cols.current:
                    value = raw
                elif col == Cols.voltage:
                    value = raw / 10.0
                elif col == Cols.lye_flow:
                    value = raw / 100.0
                elif col == Cols.lye_temp:
                    value = raw / 10.0
                elif col == Cols.temp_out:
                    value = raw / 10.0
                elif col == Cols.pressure:
                    value = raw / 10.0
                else:
                    value = None
                row_kwargs[col] = value

            # 添加新行到 all_data
            self.all_data.add_row(timestamp=current_timestamp, **row_kwargs)

            # 使用独立的循环计数计算 6 步位置，避免受滚动删除影响
            pos = self.cycle_count % 6

            # 如果是第(6n+1)行（pos==0），则计算目标值（但当前行的目标值仍沿用上一行）
            if pos == 0:
                current_val = self._last_nonnull(Cols.current)
                pressure_val = self._last_nonnull(Cols.pressure)

                if current_val is not None and pressure_val is not None:
                    computed = self.calculate_targets(current=current_val, pressure=pressure_val)
                    if computed is not None:
                        self.pending_targets = computed
                        logging.info(f"✅ 计算得到待写目标: {self.pending_targets}")
                else:
                    self.pending_targets = None
                    logging.info("ℹ️ 无足够数据计算目标，保持 pending_targets=None")
                
                # 当前行的目标值仍沿用上一行（第一行除外，第一行为N/A）
                prev_flow = self._last_nonnull(Cols.lye_flow_target)
                prev_temp = self._last_nonnull(Cols.lye_temp_target)
                if prev_flow is not None or prev_temp is not None:
                    self.all_data.update_last(**{Cols.lye_flow_target: prev_flow, Cols.lye_temp_target: prev_temp})

            # 如果是第(6n+2)行（pos==1），则将上一步计算到的 pending_targets 写入当前行并下发到 PLC
            elif pos == 1:
                if self.pending_targets is not None:
                    flow_t, temp_t = self.pending_targets
                    self.all_data.update_last(**{Cols.lye_flow_target: flow_t, Cols.lye_temp_target: temp_t})
                    if auto_write:
                        if self.write_targets(flow_target=flow_t, temp_target=temp_t):
                            logging.info(f"✅ 已将目标值写入 PLC: flow={flow_t}, temp={temp_t}")
                        else:
                            logging.error("❌ 将目标值写入 PLC 失败。")
                    else:
                        logging.info("ℹ️ 当前为手动模式，已在 DataFrame 中保存目标值但未下发 PLC。")
                else:
                    prev_flow = self._last_nonnull(Cols.lye_flow_target)
                    prev_temp = self._last_nonnull(Cols.lye_temp_target)
                    if prev_flow is not None or prev_temp is not None:
                        self.all_data.update_last(**{Cols.lye_flow_target: prev_flow, Cols.lye_temp_target: prev_temp})

            # 对于其他位置（pos in 2..5），保持与上一行一致
            else:
                prev_flow = self._last_nonnull(Cols.lye_flow_target)
                prev_temp = self._last_nonnull(Cols.lye_temp_target)
                if prev_flow is not None or prev_temp is not None:
                    self.all_data.update_last(**{Cols.lye_flow_target: prev_flow, Cols.lye_temp_target: prev_temp})

            self.cycle_count += 1
            # 仅保留最近3小时数据（10800秒）
            try:
                self.all_data.trim_to_window(3 * 3600)
            except Exception as te:
                logging.debug(f"滚动删除时发生异常: {te}")
            return True

        except Exception as e:
            logging.error(f"❌ 读取或存储数据时发生错误: {e}")
            return False

    # --- 计算目标值（基于给定的 current, pressure） ---
    def calculate_targets(self, current=None, pressure=None):
        """返回 (flow_target, temp_target) 或者 None"""
        try:
            if current is None or pressure is None:
                return None
            latest_current_density = current / awe_static_model.Area_Electrode
            lye_flow_target, lye_temp_target = awe_static_model.Working_Optimization(
                Current_density=latest_current_density,
                Pressure=pressure
            )
            lye_flow_target = round(lye_flow_target, 2)
            lye_temp_target = round(lye_temp_target, 2)
            return (lye_flow_target, lye_temp_target)
        except Exception as e:
            logging.error(f"计算目标值出错: {e}")
            return None

# --- 写入目标值 (自动模式) ---
    def write_targets(self, flow_target=None, temp_target=None):
        if not self.client or not self.client.is_socket_open():
            logging.warning("❌ 客户端未连接，无法写入数据。")
            return False

        # 优先使用传入的值，否则从表中取最后一条
        if flow_target is None or temp_target is None:
            flow_target = self._last_nonnull(Cols.lye_flow_target)
            temp_target = self._last_nonnull(Cols.lye_temp_target)

        if flow_target is None or temp_target is None or pd.isna(flow_target) or pd.isna(temp_target):
            logging.warning("⚠️ 没有可用的目标值，跳过写入。")
            return False

        flow_target_int = int(round(flow_target * 100))
        temp_target_int = int(round(temp_target * 10))

        WRITE_START_ADDRESS = 3
        write_values = [flow_target_int, temp_target_int]

        try:
            write_result = self._write_registers(address=WRITE_START_ADDRESS, values=write_values)
            if write_result.isError():
                logging.error(f"❌ 写入目标值失败: {write_result}")
                return False
            else:
                return True
        except Exception as e:
            logging.error(f"❌ 写入 PLC 时发生错误: {e}")
            return False

    # --- 写入手动目标 (手动模式) ---
    def write_manual_targets(self, flow_target, temp_target):
        if not self.client or not self.client.is_socket_open():
            logging.error("❌ 无法写入：PLC 未连接。")
            return False

        # 将手动目标写入表格的最后一行，并下发到 PLC
        if not self.all_data.df.empty:
            self.all_data.update_last(**{Cols.lye_flow_target: flow_target, Cols.lye_temp_target: temp_target})
        else:
            # 若表为空，添加一行时间戳并写入目标值（其他字段为空）
            self.all_data.add_row(timestamp=time(), **{Cols.lye_flow_target: flow_target, Cols.lye_temp_target: temp_target})

        if self.write_targets(flow_target=flow_target, temp_target=temp_target):
             logging.info(f"✅ 成功将手动目标 [流量={flow_target:.2f}, 温度={temp_target:.2f}] 写入 PLC。")
             return True
        else:
             logging.error("❌ 手动目标写入 PLC 失败。")
             return False

# ===================================================================
# 核心后端代码 END
# ===================================================================


# ===================================================================
# GUI 框架 START
# ===================================================================

# 日志面板
class GUiLogHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self._last_message = None  # 避免相同状态重复刷新

    def emit(self, record):
        msg = self.format(record)
        # 仅在状态变化时刷新，减少无效 UI 更新
        if msg == self._last_message:
            return
        self._last_message = msg
        # 必须在主线程中执行对 Tkinter 控件的修改
        def update_text():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            
            # 报警颜色高亮
            if record.levelno >= logging.ERROR:
                self.text_widget.tag_config("error", foreground="red")
                self.text_widget.tag_add("error", "end-2c linestart", "end-1c lineend")
            elif record.levelno >= logging.WARNING:
                self.text_widget.tag_config("warning", foreground="orange")
                self.text_widget.tag_add("warning", "end-2c linestart", "end-1c lineend")

            self.text_widget.configure(state='disabled')
            self.text_widget.see(tk.END)
        
        self.text_widget.after(0, update_text) # 使用 after 确保线程安全


# 趋势图与分析面板
class TrendChartPanel:
    def __init__(self, master, control_sys):
        self.control_sys = control_sys
        self.master = master
        self.MAX_HOURS = 2  # 最多显示2小时
        self.MAX_SECONDS = self.MAX_HOURS * 3600  # 转换为秒
        self.start_time = None  # 记录开始时间

        self.time_data = []  # 使用真实时间戳
        self.actual_temp_data = []
        self.target_temp_data = []
        self.actual_flow_data = []
        self.target_flow_data = []
        self.last_target_temp = None
        self.last_target_flow = None

        # 创建两个子图：温度和流量
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_temp = self.fig.add_subplot(211)  # 上面的温度图
        self.ax_flow = self.fig.add_subplot(212)  # 下面的流量图
        self.fig.tight_layout()

        # 温度图配置
        self.ax_temp.set_ylabel("温度 (°C)", color='tab:red')
        self.line_actual_temp, = self.ax_temp.plot([], [], 'r-', label="实际温度", linewidth=2)
        self.line_target_temp, = self.ax_temp.plot([], [], 'r--', label="目标温度", linewidth=2)
        self.ax_temp.tick_params(axis='y', labelcolor='tab:red')
        self.ax_temp.legend(loc='upper left')
        self.ax_temp.grid(True, alpha=0.3)

        # 流量图配置
        self.ax_flow.set_xlabel("时间")
        self.ax_flow.set_ylabel("流量 (L/min)", color='tab:blue')
        self.line_actual_flow, = self.ax_flow.plot([], [], 'b-', label="实际流量", linewidth=2)
        self.line_target_flow, = self.ax_flow.plot([], [], 'b--', label="目标流量", linewidth=2)
        self.ax_flow.tick_params(axis='y', labelcolor='tab:blue')
        self.ax_flow.legend(loc='upper left')
        self.ax_flow.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, master)
        toolbar.update()
        
    def update_chart_data(self):
        if not hasattr(self.master, 'winfo_exists') or not self.master.winfo_exists():
            return

        try:
            temp_actual = self.control_sys.state_data.get_last(Cols.lye_temp)
            temp_target = self.control_sys.input_data.get_last(Cols.lye_temp_target)
            
            flow_actual = self.control_sys.state_data.get_last(Cols.lye_flow)
            flow_target = self.control_sys.input_data.get_last(Cols.lye_flow_target)

            temp_target = self._resolve_target(temp_target, "last_target_temp", temp_actual)
            flow_target = self._resolve_target(flow_target, "last_target_flow", flow_actual)

            # 仅在获取到有效数据时才更新
            if pd.notna(temp_actual) and pd.notna(flow_actual):
                # 初始化开始时间
                if self.start_time is None:
                    self.start_time = time()
                
                current_time = time()
                elapsed_time = current_time - self.start_time
                
                # 添加数据
                self.time_data.append(current_time)
                self.actual_temp_data.append(temp_actual)
                self.target_temp_data.append(temp_target if pd.notna(temp_target) else temp_actual)
                self.actual_flow_data.append(flow_actual)
                self.target_flow_data.append(flow_target if pd.notna(flow_target) else flow_actual)
                
                # 保持最多2小时的数据窗口
                if elapsed_time > self.MAX_SECONDS:
                    cutoff_time = current_time - self.MAX_SECONDS
                    while self.time_data and self.time_data[0] < cutoff_time:
                        self.time_data.pop(0)
                        self.actual_temp_data.pop(0)
                        self.target_temp_data.pop(0)
                        self.actual_flow_data.pop(0)
                        self.target_flow_data.pop(0)
                
                self._redraw_chart()

        except Exception as e:
            logging.error(f"更新图表数据时发生错误: {e}")

    def _redraw_chart(self):
        if not self.time_data:
            return

        # 转换时间戳为相对时间（秒）用于绘图
        if self.start_time is not None:
            relative_time = [t - self.start_time for t in self.time_data]
        else:
            relative_time = self.time_data

        # 更新温度图
        self.line_actual_temp.set_data(relative_time, self.actual_temp_data)
        self.line_target_temp.set_data(relative_time, self.target_temp_data)
        
        # 更新流量图
        self.line_actual_flow.set_data(relative_time, self.actual_flow_data)
        self.line_target_flow.set_data(relative_time, self.target_flow_data)

        # 设置X轴范围
        if relative_time:
            x_min = relative_time[0]
            x_max = relative_time[-1]
            x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 1
            
            self.ax_temp.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax_flow.set_xlim(x_min - x_padding, x_max + x_padding)
        
        # 设置温度图Y轴范围
        temp_data = self.actual_temp_data + self.target_temp_data
        if temp_data:
            temp_max = max(temp_data)
            temp_min = min(temp_data)
            temp_padding = (temp_max - temp_min) * 0.1 if temp_max > temp_min else 1
            self.ax_temp.set_ylim(temp_min - temp_padding, temp_max + temp_padding)
        
        # 设置流量图Y轴范围
        flow_data = self.actual_flow_data + self.target_flow_data
        if flow_data:
            flow_max = max(flow_data)
            flow_min = min(flow_data)
            flow_padding = (flow_max - flow_min) * 0.1 if flow_max > flow_min else 0.1
            self.ax_flow.set_ylim(flow_min - flow_padding, flow_max + flow_padding)

        self.canvas.draw_idle()

    def _resolve_target(self, current_target, cache_attr, fallback_value):
        if pd.notna(current_target):
            setattr(self, cache_attr, current_target)
            return current_target
        cached = getattr(self, cache_attr, None)
        if cached is not None:
            return cached
        return fallback_value

# 主应用框架
# 模块一：主应用框架
class AECControllerApp:
    def __init__(self, master, control_system):
            self.master = master
            self.control_sys = control_system
            self.master.title("碱性电解槽工况优化控制系统")
            self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

            # 控制和更新频率
            self.control_period_ms = 60000  # 60秒
            self.chart_update_ms = 10000    # 10秒
            self.control_thread = None
            self.is_running = False # 标志控制线程是否运行
            self.last_target_update_time = None  # 记录上次更新目标值的时间

            # 界面控制变量
            self.conn_status_var = tk.StringVar(value="🔴 未连接")
            self.control_mode = tk.IntVar(value=0) # 0: 自动, 1: 手动
            self.ip_var = tk.StringVar(value=self.control_sys.modbus_ip)
            self.port_var = tk.StringVar(value=str(self.control_sys.modbus_port))
            self.unit_var = tk.StringVar(value=str(getattr(self.control_sys, "modbus_id", 1)))

            # 载入和处理公司Logo
            self.logo_path = resource_path("微信图片_20260104100405_72_5.jpg")
            self.logo_tk = None # 初始化为None
        
            # 尝试加载Logo，失败则不显示
            try:
                original_image = Image.open(self.logo_path)
                # 计算合适的缩放比例。例如，将其宽度限制在200像素左右。
                # 您需要根据实际Logo和界面大小调整这里的尺寸。
                new_width = 250 # 示例宽度
                new_height = int(new_width * (original_image.height / original_image.width))
                
                resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.logo_tk = ImageTk.PhotoImage(resized_image)
                logging.info(f"✅ 公司Logo加载成功，尺寸为 {new_width}x{new_height}。")
            except FileNotFoundError:
                logging.warning(f"⚠️ 无法找到公司Logo图片: {self.logo_path}")
            except Exception as e:
                logging.error(f"❌ 加载公司Logo时发生错误: {e}")

            # 1. 创建整体布局 (包含所有的面板和现在移动到右侧的日志)
            self._create_layout()

            # 2. 配置日志系统 (依赖于 _create_log_panel 中创建的 self.log_text)
            self._setup_logging()

            # 3. 启动定时更新循环
            self.master.after(100, self._update_gui_data) # 实时数据显示更新（首次快速启动）
            self.master.after(self.chart_update_ms, self._update_chart) # 趋势图更新

    def _create_layout(self):
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill='both', expand=True)

        # 配置主框架为双列
        self.main_frame.columnconfigure(0, weight=1)  # 左侧 (控制/数据显示) 权重
        self.main_frame.columnconfigure(1, weight=3)  # 右侧 (趋势图/日志) 权重更高，以适应图表

        # --- 左侧容器 (保持不变的控制和数据显示) ---
        left_frame = ttk.Frame(self.main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nwes')
        # 允许左侧的两块内容垂直堆叠
        left_frame.rowconfigure(0, weight=0) # 控制面板
        left_frame.rowconfigure(1, weight=0) # 数据面板
        left_frame.rowconfigure(2, weight=1) # 底部留给Logo

        # 1. 控制与状态区 (左上)
        control_frame = ttk.LabelFrame(left_frame, text="控制与状态", padding="10")
        control_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nwes')
        self._create_control_panel(control_frame)

        # 2. 实时数据区 (左下)
        data_frame = ttk.LabelFrame(left_frame, text="实时运行数据 & 目标值", padding="10")
        data_frame.grid(row=1, column=0, padx=5, pady=5, sticky='nwes')
        self._create_data_monitor_panel(data_frame)

        # 5. 公司Logo (左下角)
        logo_frame = ttk.Frame(left_frame)
        logo_frame.grid(row=2, column=0, padx=5, pady=5, sticky='sw')
        if self.logo_tk:
            ttk.Label(logo_frame, image=self.logo_tk).pack(anchor='sw')

        # --- 右侧容器 (趋势图和日志) ---
        right_frame = ttk.Frame(self.main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nwes')
        # 允许右侧的两块内容垂直堆叠，并让趋势图占据大部分垂直空间
        right_frame.rowconfigure(0, weight=3) # 趋势图
        right_frame.rowconfigure(1, weight=1) # 日志

        # 3. 趋势图区 (右上)
        chart_frame = ttk.LabelFrame(right_frame, text="关键参数趋势分析", padding="10")
        chart_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nwes')
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.trend_chart_panel = TrendChartPanel(chart_frame, self.control_sys)

        # 4. 日志区 (右下) - 调用新的日志创建方法
        self._create_log_panel(right_frame)

        # 确保左侧容器在垂直方向填满 (如果右侧内容较高)
        self.main_frame.rowconfigure(0, weight=1)

    # 模块二：数据监控面板
    def _create_data_monitor_panel(self, parent):
        self.data_vars = {}
        data_config = {
            "实时数据": [
                (Cols.current, "电流 (A)"),
                (Cols.voltage, "电压 (V)"),
                (Cols.lye_flow, "碱液流量 (L/min)"),
                (Cols.lye_temp, "碱液温度 (°C)"),
                (Cols.temp_out, "出口温度 (°C)"),
                (Cols.pressure, "压力 (bar)"),
            ],
            "计算目标": [
                (Cols.lye_flow_target, "目标流量 (L/min)"),
                (Cols.lye_temp_target, "目标温度 (°C)"),
            ]
        }
        
        row_idx = 0
        for header, items in data_config.items():
            ttk.Label(parent, text=f"--- {header} ---", font=('Arial', 10, 'bold')).grid(row=row_idx, column=0, columnspan=2, sticky='w', pady=(10, 2))
            row_idx += 1
            for i, (key, label) in enumerate(items):
                ttk.Label(parent, text=f"{label}:").grid(row=row_idx + i, column=0, sticky='w', padx=5)
                
                var = tk.StringVar(value="N/A")
                self.data_vars[key] = var
                ttk.Label(parent, textvariable=var, font=('Arial', 10, 'bold')).grid(row=row_idx + i, column=1, sticky='e', padx=5)
            row_idx += len(items)

    # 模块三：控制与模式面板
    def _create_control_panel(self, parent):
        status_label = ttk.Label(parent, textvariable=self.conn_status_var, font=('Arial', 12, 'bold'))
        status_label.grid(row=0, column=0, columnspan=2, pady=10)

        ttk.Label(parent, text="PLC IP:").grid(row=1, column=0, sticky='w', padx=5)
        ttk.Entry(parent, textvariable=self.ip_var, width=16).grid(row=1, column=1, sticky='e', padx=5, pady=2)

        ttk.Label(parent, text="Port:").grid(row=2, column=0, sticky='w', padx=5)
        ttk.Entry(parent, textvariable=self.port_var, width=16).grid(row=2, column=1, sticky='e', padx=5, pady=2)

        ttk.Label(parent, text="ID:").grid(row=3, column=0, sticky='w', padx=5)
        ttk.Entry(parent, textvariable=self.unit_var, width=16).grid(row=3, column=1, sticky='e', padx=5, pady=2)

        self.btn_connect = ttk.Button(parent, text="连接 PLC", command=self._start_connection_thread)
        self.btn_connect.grid(row=4, column=0, padx=5, pady=5, sticky='ew')
        self.btn_disconnect = ttk.Button(parent, text="断开 PLC", command=self._disconnect_modbus)
        self.btn_disconnect.grid(row=4, column=1, padx=5, pady=5, sticky='ew')
        
        ttk.Separator(parent, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=10)

        self.btn_start = ttk.Button(parent, text="▶️ 启动控制循环", command=self._start_control_loop)
        self.btn_start.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        self.btn_stop = ttk.Button(parent, text="⏹ 停止控制循环", command=self._stop_control_loop, state=tk.DISABLED)
        self.btn_stop.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky='ew')
        
        ttk.Separator(parent, orient='horizontal').grid(row=8, column=0, columnspan=2, sticky='ew', pady=10)

        mode_frame = ttk.LabelFrame(parent, text="运行模式", padding="5")
        mode_frame.grid(row=9, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        
        ttk.Radiobutton(mode_frame, text="自动优化 (AEC)", variable=self.control_mode, value=0, command=self._toggle_manual_input).pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="手动控制 (MAN)", variable=self.control_mode, value=1, command=self._toggle_manual_input).pack(anchor='w')

        self.manual_frame = ttk.LabelFrame(parent, text="手动目标输入", padding="5")
        self.manual_frame.grid(row=10, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        self.manual_frame.grid_remove()
        
        ttk.Label(self.manual_frame, text="流量 (L/min):").grid(row=0, column=0, sticky='w')
        self.entry_flow = ttk.Entry(self.manual_frame, width=10)
        self.entry_flow.grid(row=0, column=1, sticky='e', padx=5, pady=2)
        
        ttk.Label(self.manual_frame, text="温度 (°C):").grid(row=1, column=0, sticky='w')
        self.entry_temp = ttk.Entry(self.manual_frame, width=10)
        self.entry_temp.grid(row=1, column=1, sticky='e', padx=5, pady=2)
        
        ttk.Button(self.manual_frame, text="写入目标 (仅手动模式)", command=self._write_manual_targets, style='Accent.TButton').grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)


    # 模块五：日志面板
    def _create_log_panel(self, parent_frame):
        log_frame = ttk.LabelFrame(parent_frame, text="系统日志", padding="10")
        # 使用 grid 布局在右侧容器 (parent_frame) 的第二行 (row=1)
        log_frame.grid(row=1, column=0, sticky='nwes', padx=5, pady=5)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state='disabled')
        self.log_text.grid(row=0, column=0, sticky='ew')
        
    def _setup_logging(self):
        gui_handler = GUiLogHandler(self.log_text)
        gui_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(gui_handler)

    def _update_chart(self):
        if self.is_running:
            self.trend_chart_panel.update_chart_data()
        self.master.after(self.chart_update_ms, self._update_chart)

    def _extract_conn_inputs(self):
        ip = self.ip_var.get().strip() or self.control_sys.modbus_ip
        try:
            port = int(self.port_var.get())
        except ValueError:
            port = self.control_sys.modbus_port
            self.port_var.set(str(port))
        try:
            unit = int(self.unit_var.get())
        except ValueError:
            unit = getattr(self.control_sys, "modbus_id", 1)
            self.unit_var.set(str(unit))
        self.ip_var.set(ip)
        return ip, port, unit

    # --- 核心功能逻辑 ---
    def _start_connection_thread(self):
        if self.control_sys.client and self.control_sys.client.is_socket_open():
            logging.warning("已连接，无需重复操作。")
            return
        
        self.conn_status_var.set("🟡 连接中...")
        conn_thread = threading.Thread(target=self._run_connection)
        conn_thread.start()

    def _run_connection(self):
        ip, port, unit = self._extract_conn_inputs()
        self.control_sys.modbus_ip = ip
        self.control_sys.modbus_port = port
        self.control_sys.modbus_id = unit

        if self.control_sys.connect_modbus():
            self.master.after(0, lambda: self.conn_status_var.set("🟢 已连接"))
        else:
            self.master.after(0, lambda: self.conn_status_var.set("🔴 未连接"))
            
    def _disconnect_modbus(self):
        self._stop_control_loop()
        self.control_sys.disconnect()
        self.conn_status_var.set("🔴 未连接")

    def _start_control_loop(self):
        if not self.control_sys.client or not self.control_sys.client.is_socket_open():
            logging.error("无法启动：PLC 未连接。")
            return
            
        if self.is_running:
            logging.warning("控制循环已在运行。")
            return

        self.is_running = True
        self.last_target_update_time = None  # 首次启动后立即计算目标值
        
        # 重置图表状态
        self.trend_chart_panel.start_time = None
        self.trend_chart_panel.time_data = []
        self.trend_chart_panel.actual_temp_data = []
        self.trend_chart_panel.target_temp_data = []
        self.trend_chart_panel.actual_flow_data = []
        self.trend_chart_panel.target_flow_data = []
        
        self.control_thread = threading.Thread(target=self._run_control_thread, daemon=True)
        self.control_thread.start()
        
        self.btn_start['state'] = tk.DISABLED
        self.btn_stop['state'] = tk.NORMAL
        logging.info("🚀 控制循环已启动。")
        
    def _run_control_thread(self):
        """
        控制循环：
        - 每 10 秒读取一次状态寄存器+6 个监测寄存器并保存到 DataFrame
        - 目标值计算和下发逻辑：
          * 在第(6n+1)行监测时计算目标值
          * 在第(6n+2)行监测时保存目标值并根据模式下发 PLC
          * 其他行保持与前一行的目标值一致
        """
        READ_INTERVAL = 10.0  # 每读取一次间隔（秒）

        while self.is_running:
            try:
                # 根据当前模式决定是否自动将目标写入 PLC（自动模式下下发）
                auto_write = (self.control_mode.get() == 0)
                self.control_sys.read_and_store_data(auto_write=auto_write)
                if self.control_sys.last_status == 0:
                    logging.warning("⚠️ PLC 状态为 0，已停止控制循环。")
                    self.master.after(0, self._stop_control_loop)
                    break

            except Exception as e:
                logging.error(f"控制周期运行中发生未捕获的错误: {e}")
                self.master.after(0, self._stop_control_loop)
                break

            sleep(READ_INTERVAL)

    def _stop_control_loop(self):
        if self.is_running:
            self.is_running = False
            logging.info("⏸ 控制循环已停止。")
        
        self.btn_start['state'] = tk.NORMAL
        self.btn_stop['state'] = tk.DISABLED

    def _toggle_manual_input(self):
        if self.control_mode.get() == 1:
            self.manual_frame.grid()
            logging.warning("⚠️ 模式切换为 [手动控制]。目标值将不会自动计算。")
        else:
            self.manual_frame.grid_remove()
            logging.info("模式切换为 [自动优化]。目标值将由模型计算。")

    def _write_manual_targets(self):
        if self.control_mode.get() == 0:
            logging.error("❌ 当前处于自动模式，无法手动写入。请切换到 [手动控制]。")
            return
            
        try:
            flow_target = float(self.entry_flow.get())
            temp_target = float(self.entry_temp.get())
            
            threading.Thread(target=self._run_manual_write, args=(flow_target, temp_target), daemon=True).start()
            
        except ValueError:
            logging.error("❌ 输入值无效。请确保流量和温度输入的是有效数字。")

    def _run_manual_write(self, flow_target, temp_target):
        self.control_sys.write_manual_targets(flow_target, temp_target)
        
    def _update_gui_data(self):
        """GUI 界面数据更新函数 (由 Tkinter 定时调用)"""
        try:
            # 实时数据
            current_val = self._safe_get(Cols.current, self.control_sys.input_data)
            self.data_vars[Cols.current].set(f"{current_val:.1f} A" if current_val is not None else "N/A")
            
            voltage_val = self._safe_get(Cols.voltage, self.control_sys.state_data)
            self.data_vars[Cols.voltage].set(f"{voltage_val:.2f} V" if voltage_val is not None else "N/A")
            
            flow_val = self._safe_get(Cols.lye_flow, self.control_sys.state_data)
            self.data_vars[Cols.lye_flow].set(f"{flow_val:.2f} L/min" if flow_val is not None else "N/A")
            
            temp_val = self._safe_get(Cols.lye_temp, self.control_sys.state_data)
            self.data_vars[Cols.lye_temp].set(f"{temp_val:.1f} °C" if temp_val is not None else "N/A")
            
            temp_out_val = self._safe_get(Cols.temp_out, self.control_sys.state_data)
            self.data_vars[Cols.temp_out].set(f"{temp_out_val:.1f} °C" if temp_out_val is not None else "N/A")
            
            pressure_val = self._safe_get(Cols.pressure, self.control_sys.state_data)
            self.data_vars[Cols.pressure].set(f"{pressure_val:.1f} bar" if pressure_val is not None else "N/A")

            # 目标值
            flow_target = self._safe_get(Cols.lye_flow_target, self.control_sys.input_data)
            self.data_vars[Cols.lye_flow_target].set(f"{flow_target:.2f} L/min" if flow_target is not None else "N/A")

            temp_target = self._safe_get(Cols.lye_temp_target, self.control_sys.input_data)
            self.data_vars[Cols.lye_temp_target].set(f"{temp_target:.1f} °C" if temp_target is not None else "N/A")
            
        except Exception as e:
            logging.debug(f"更新GUI数据时出错: {e}")

        # 10秒更新一次，与数据读取同步
        self.master.after(10000, self._update_gui_data)

    def _safe_get(self, col, data_holder):
        try:
            val = data_holder.get_last(col)
            return val if pd.notna(val) else None
        except Exception:
            return None
        
    def on_closing(self):
        self._stop_control_loop()
        self.control_sys.disconnect()
        self.master.destroy()


# --- 主程序入口 ---
if __name__ == "__main__":
    # 您的实际 PLC/Modbus 配置
    PLC_IP = '192.168.1.1' 
    PLC_PORT = 502
    PLC_ID = 1

    # 实例化您的 ControlSystem
    control_system_instance = ControlSystem(modbus_ip=PLC_IP, modbus_port=PLC_PORT, modbus_id=PLC_ID)

    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Accent.TButton', foreground='white', background='#007BFF', font=('Arial', 10, 'bold'))

    app = AECControllerApp(root, control_system_instance)
    
    root.mainloop()