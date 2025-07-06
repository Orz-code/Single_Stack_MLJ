import matplotlib.pyplot as plt
import matplotlib

class LinePlotter:
    def __init__(self, figsize=(16, 5)):
        self.figsize = figsize
        # 设置字体为支持中文的字体，如SimHei（黑体）
        matplotlib.rcParams['font.family'] = 'SimHei'
        # 解决负号显示问题
        matplotlib.rcParams['axes.unicode_minus'] = False

    # 单变量绘图函数
    def single_variable_plot(self, x_data, y_data, title="折线图", color="blue", linewidth=2, marker=None):
        """单变量的绘图函数

        Args:
            x_data (series): 自变量.
            y_data (series): 因变量.
            title (str): 图标题. Defaults to "折线图".
            color (str): 曲线颜色. Defaults to "blue".
            linewidth (int): 曲线线宽. Defaults to 2.
            marker (str): 标记样式. Defaults to None.
        """
        plt.figure(figsize=self.figsize)
        plt.plot(x_data, y_data, color=color, linewidth=linewidth, marker=marker, label=y_data.name)
        plt.xlabel(x_data.name)
        plt.ylabel(y_data.name)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    # 多变量的绘图函数
    def multiple_variable_plot(self, x_data, Y_data, color, title="折线图", linewidth=2, marker=None, legend_bbox_to_anchor=(0.9, 0.9)):
        """多变量的绘图函数

        Args:
            X_data (series): 自变量.
            Y_data (dataframe): 因变量
            color (dictionary): 每条曲线的颜色
            title (str): 图标题. Defaults to "折线图".
            linewidth (int): 曲线线宽. Defaults to 2.
            marker (dictionary): 每条曲线的标记样式. Defaults to None.
            legend_bbox_to_anchor (Tuple):图例的锚点坐标. Defaults to (0.9, 0.9).
        """
        fig, ax1 = plt.subplots(figsize=self.figsize)
        axes = [ax1]  # 存储所有 y 轴的列表

        # 绘制第一条曲线
        first_column = Y_data.columns[0]
        ax1.plot(x_data, Y_data[first_column], color=color.get(first_column), 
                 linewidth=linewidth, marker=marker.get(first_column, None) if marker else None, 
                 label=first_column)
        ax1.set_ylabel(first_column, color=color.get(first_column))
        ax1.tick_params(axis='y', labelcolor=color.get(first_column))

        # 绘制剩余的曲线，并为每条曲线创建一个新的 y 轴
        for i, column in enumerate(Y_data.columns[1:]):
            ax = ax1.twinx()  # 创建一个新的 y 轴
            ax.spines['right'].set_position(('outward', 40 * i))  # 调整新 y 轴的位置
            ax.plot(x_data, Y_data[column], color=color.get(column), 
                   linewidth=linewidth, marker=marker.get(column, None) if marker else None, 
                   label=column)
            ax.set_ylabel(column, color=color.get(column))
            ax.tick_params(axis='y', labelcolor=color.get(column))
            axes.append(ax)

        # 设置标题和图例
        ax1.set_title(title)
        lines, labels = [], []
        for ax in axes:
            ax_lines, ax_labels = ax.get_legend_handles_labels()
            lines.extend(ax_lines)
            labels.extend(ax_labels)
        fig.legend(lines, labels, bbox_to_anchor=legend_bbox_to_anchor)

        plt.tight_layout()
        return fig, ax1
    
class ScatterPlotter:
    def __init__(self, figsize=(16, 5)):
        self.figsize = figsize
        # 设置字体为支持中文的字体，如SimHei（黑体）
        matplotlib.rcParams['font.family'] = 'SimHei'
        # 解决负号显示问题
        matplotlib.rcParams['axes.unicode_minus'] = False

    def draw_scatter_plot(self, x_data, y_data, title="散点图", color="blue", marker=None):
        """_summary_

        Args:
            x_data (series): 自变量.
            y_data (series): 因变量.
            title (str): 图标题. Defaults to "散点图".
            color (str): 曲线颜色. Defaults to "blue".
            marker (str): 标记样式. Defaults to None.

        Returns:
            _type_: _description_
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)
        # 绘制散点图
        ax.scatter(x_data, y_data, color=color, marker=marker)
        # 设置标题
        ax.set_title(title)
        # 设置 X 轴标签
        ax.set_xlabel(x_data.name)
        # 设置 Y 轴标签
        ax.set_ylabel(y_data.name)
        # 显示网格线
        ax.grid(True)
        return fig, ax