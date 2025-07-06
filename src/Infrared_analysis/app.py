import os
import re
import glob
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import dash
from dash import dcc, html, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.colors import sequential
from Infrared_analysis.utils_image import read_infrared_image,multi_gaussian
from skimage.draw import line as draw_line
import io
import gzip

ROOT_DIR = 'D:\\Devs\\Simulation-Platform\\data\\raw\\202409allOUT' # 请将此路径替换为您的实际路径

def get_folder_options():
    folders = [f.path for f in os.scandir(ROOT_DIR) if f.is_dir()]
    options = [{'label': os.path.basename(folder), 'value': folder} for folder in folders]
    return options

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    # 标题
    html.H1('红外温度检测', style={'textAlign': 'center'}),

    # 文件选择部分
    html.Div([
        html.Label('选择文件夹：'),
        dcc.Dropdown(id='folder-dropdown', options=[], placeholder='选择文件夹'),

        html.Label('选择文件：'),
        dcc.Dropdown(id='file-dropdown', options=[], placeholder='选择文件'),

        html.Div(id='file-info', style={'marginTop': '10px'}),
    ], style={'width': '80%', 'margin': '0 auto'}),

    # 添加隐藏的 dcc.Store 组件
    dcc.Store(id='image-data-store'),
    dcc.Store(id='analysis-data-store'),  # 新增

    html.Hr(),

    # 主内容区域
    html.Div([
        # 左侧：图片展示和交互
        html.Div([
            dcc.Loading(
                id='loading-image',
                children=[dcc.Graph(id='image-graph', style={'height': '100%'}, config={})],  # 添加 config={}
                type='default',
            ),
            html.Div([
                html.Button('上一张', id='prev-button'),
                html.Button('下一张', id='next-button'),
                html.Button('逆时针旋转90度', id='rotate-ccw-button'),
                html.Button('顺时针旋转90度', id='rotate-cw-button'),
            ], style={
                'display': 'flex',
                'justifyContent': 'space-around',
                'marginTop': '10px'
            }),
        ], style={
            'width': '58%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'height': 'calc(100vh - 200px)',
            'overflow': 'auto'
        }),

        # 右侧：统计分析和结果展示
        html.Div([
            # 选择模式切换控件
            html.Div([
                html.Label('选择模式：'),
                dcc.RadioItems(
                    id='selection-mode',
                    options=[
                        {'label': '矩形选择', 'value': 'rectangle'},
                        {'label': '线选择', 'value': 'line'}
                    ],
                    value='rectangle',
                    inline=True
                ),
            ], style={'textAlign': 'center', 'marginTop': '10px'}),

            # 保存数据按钮
            html.Div([
                html.Button('保存数据', id='save-data-button')
            ], style={'position': 'absolute', 'top': 250, 'right': 10, 'zIndex': 1}),

            # 分析图表和结果
            dcc.Loading(
                id='loading-analysis',
                children=[dcc.Graph(id='analysis-graph', style={'height': '100%'})],
                type='default',
            ),
            html.Div(id='analysis-results', style={'marginTop': '10px'}),
            # 隐藏的下载组件
            dcc.Download(id='download-data'),
        ], style={
            'width': '38%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'marginLeft': '4%',
            'height': 'calc(100vh - 200px)',
            'overflow': 'auto'
        }),
    ]),
], style={
    'height': '100vh',
    'display': 'flex',
    'flexDirection': 'column'
})

# 修改 update_folder_options 回调函数
@app.callback(
    [Output('folder-dropdown', 'options'),
     Output('folder-dropdown', 'value')],
    Input('folder-dropdown', 'value')
)
def update_folder_options(current_value):
    options = get_folder_options()
    if current_value is None:
        if options:
            default_value = options[0]['value']
        else:
            default_value = None
        return options, default_value
    else:
        return options, dash.no_update

@app.callback(
    [Output('file-dropdown', 'options'),
     Output('file-dropdown', 'value')],
    [Input('folder-dropdown', 'value'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('file-dropdown', 'value'),
     State('file-dropdown', 'options')]
)
def update_file_dropdown(selected_folder, prev_clicks, next_clicks, current_file, file_options):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_id == 'folder-dropdown':
        # 当文件夹变化时，更新文件列表，选取第一个文件
        if selected_folder is None:
            return [], None
        files = glob.glob(os.path.join(selected_folder, '*.txt'))
        options = [{'label': os.path.basename(f), 'value': f} for f in files]
        if options:
            default_value = options[0]['value']
        else:
            default_value = None
        return options, default_value
    elif triggered_id in ['prev-button', 'next-button']:
        # 当点击上一张或下一张时，切换文件
        if not file_options:
            return dash.no_update, current_file
        file_values = [opt['value'] for opt in file_options]
        current_index = file_values.index(current_file) if current_file in file_values else 0
        if triggered_id == 'prev-button':
            new_index = (current_index - 1) % len(file_values)
        elif triggered_id == 'next-button':
            new_index = (current_index + 1) % len(file_values)
        else:
            new_index = current_index
        return dash.no_update, file_values[new_index]
    else:
        # 对于其他情况，不更新任何内容
        return dash.no_update, dash.no_update


@app.callback(
    Output('file-info', 'children'),
    Input('file-dropdown', 'value')
)
def display_file_info(selected_file):
    if selected_file is None:
        return ''
    file_name = os.path.basename(selected_file)
    match = re.search(r'IMG(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', file_name)
    if match:
        year, month, day, hour, minute, second = match.groups()
        date_time_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
    else:
        date_time_str = '无法解析日期时间'
    return html.Div([
        html.P(f"文件名：{file_name}"),
        html.P(f"拍摄时间：{date_time_str}")
    ])

from plotly.colors import sequential

# 修改 update_image 回调函数
@app.callback(
    [Output('image-graph', 'figure'),
     Output('image-graph', 'config'),  # 新增输出
     Output('image-data-store', 'data')],
    [Input('file-dropdown', 'value'),
     Input('rotate-ccw-button', 'n_clicks'),
     Input('rotate-cw-button', 'n_clicks'),
     Input('selection-mode', 'value')],
    [State('image-graph', 'figure'),
     State('image-data-store', 'data')]
)
def update_image(selected_file, rotate_ccw_clicks, rotate_cw_clicks, selection_mode, existing_figure, stored_data):
    if selected_file is None:
        return go.Figure(), dash.no_update, None

    # 检查缓存是否已有数据
    if stored_data and stored_data.get('file') == selected_file:
        # 从缓存中获取数据
        data = np.array(stored_data.get('data'))
    else:
        # 读取新文件的数据
        data = read_infrared_image(selected_file)
    
    # 打印数据形状和范围
    print(f"Data shape: {data.shape}")
    print(f"Data min: {np.min(data)}, Data max: {np.max(data)}")
    
    ctx = callback_context
    rotation = 0
    if existing_figure and 'layout' in existing_figure and 'meta' in existing_figure['layout']:
        rotation = existing_figure['layout']['meta'].get('rotation', 0)
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'rotate-ccw-button':
            rotation -= 90
        elif triggered_id == 'rotate-cw-button':
            rotation += 90
    rotation = rotation % 360
    k = rotation // 90
    data_rotated = np.rot90(data, k=-k)
    
    # 不需要归一化，直接使用原始温度数据
    # 指定颜色映射（colormap），例如 'Inferno'
    colorscale = 'Inferno'
    
    # 设置颜色轴范围
    zmin = np.min(data_rotated)
    zmax = np.max(data_rotated)
    
    # 获取数据的形状
    height, width = data_rotated.shape

    # 创建 x 和 y 坐标
    x = np.arange(width)
    y = np.arange(height)

    # 设置拖拽模式和配置
    if selection_mode == 'rectangle':
        dragmode = 'select'
        new_config = {'modeBarButtonsToAdd': []}
    elif selection_mode == 'line':
        dragmode = 'drawline'
        new_config = {'modeBarButtonsToAdd': ['drawline'], 'editable': True}
    else:
        dragmode = 'select'
        new_config = {'modeBarButtonsToAdd': []}

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x,
        y=y,
        z=data_rotated,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title='Temperature (°C)'),
        hovertemplate='温度: %{z:.2f}°C<extra></extra>',
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x', scaleratio=1, autorange='reversed'),
        dragmode=dragmode,
        meta={'rotation': rotation}
    )
    # 返回图表对象、配置和数据
    return fig, new_config, {'file': selected_file, 'data': data.tolist()}



@app.callback(
    [Output('analysis-graph', 'figure'),
     Output('analysis-results', 'children'),
     Output('analysis-data-store', 'data')],  # 新增输出
    [Input('image-graph', 'selectedData'),
     Input('image-graph', 'relayoutData'),
     Input('selection-mode', 'value')],
    [State('image-graph', 'figure'),
     State('image-data-store', 'data')]
)
def analyze_selected_area(selectedData, relayoutData, selection_mode, figure, stored_data):
    # 函数内部逻辑
    if stored_data is None:
        return go.Figure(), '', None  # 添加第三个返回值
    
    # 从缓存中获取数据
    data = np.array(stored_data.get('data'))
    rotation = figure.get('layout', {}).get('meta', {}).get('rotation', 0)
    k = rotation // 90
    data_rotated = np.rot90(data, k=-k)

    if selection_mode == 'rectangle':
        # 检查 selectedData
        if selectedData is None or 'range' not in selectedData:
            return go.Figure(), "请在图像上选择一个区域进行分析。", None
        
        # 获取选区的范围
        x0 = int(min(selectedData['range']['x']))
        x1 = int(max(selectedData['range']['x']))
        y0 = int(min(selectedData['range']['y']))
        y1 = int(max(selectedData['range']['y']))
        
        # 确保索引在有效范围内
        x0 = max(0, min(x0, data_rotated.shape[1]))
        x1 = max(0, min(x1, data_rotated.shape[1]))
        y0 = max(0, min(y0, data_rotated.shape[0]))
        y1 = max(0, min(y1, data_rotated.shape[0]))
        
        # 选取区域
        selected_area = data_rotated[y0:y1, x0:x1]
        temperatures = selected_area.flatten()
        # 准备分析数据
        analysis_data = {
            'temperatures': temperatures.tolist()  # 将 numpy 数组转换为列表，以便序列化
        }

        # 计算平均温度
        mean_temperature = np.mean(temperatures)
        mean_temp_text = f"平均温度：{mean_temperature:.2f}°C"
        
        fig = go.Figure()
        
        # 检查数据量是否足够
        min_data_points = 50  # 您可以根据需要调整最小数据量
        if len(temperatures) < min_data_points:
            # 数据量不足，绘制直方图，显示平均温度
            bins = 30  # 您可以根据需要调整 bin 的数量
            counts, bin_edges = np.histogram(temperatures, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 绘制直方图
            fig.add_trace(go.Bar(x=bin_centers, y=counts, name='温度分布'))
            
            # 更新布局
            fig.update_layout(
                xaxis_title='温度 (°C)',
                yaxis_title='频数',
                legend_title='图例',
                title='温度分布直方图'
            )
            
            result_text = html.Div([
                html.P("选取的区域像素数量不足，无法进行拟合。"),
                html.P(mean_temp_text)
            ])
            return fig, result_text, analysis_data
        
    elif selection_mode == 'line':
        # 检查 relayoutData
        if relayoutData is None or 'shapes' not in relayoutData:
            return go.Figure(), "请在图像上绘制一条直线进行分析。", None
        
        shapes = relayoutData['shapes']
        if not shapes:
            return go.Figure(), "请在图像上绘制一条直线进行分析。", None
        line = shapes[-1]
        if line['type'] != 'line':
            return go.Figure(), "请在图像上绘制一条直线进行分析。", None
        
        x0 = line['x0']
        y0 = line['y0']
        x1 = line['x1']
        y1 = line['y1']

        # 将坐标转换为索引
        x0_idx = int(x0)
        x1_idx = int(x1)
        y0_idx = int(y0)
        y1_idx = int(y1)

        # 获取直线上所有的像素点坐标
        rr, cc = draw_line(y0_idx, x0_idx, y1_idx, x1_idx)

        # 确保索引在有效范围内
        rr = np.clip(rr, 0, data_rotated.shape[0] - 1)
        cc = np.clip(cc, 0, data_rotated.shape[1] - 1)

        # 获取直线上对应的温度值
        temperatures = data_rotated[rr, cc]

        # **先计算距离**
        distances = np.sqrt((rr - rr[0])**2 + (cc - cc[0])**2)

        # **然后准备分析数据**
        analysis_data = {
            'distances': distances.tolist(),
            'temperatures': temperatures.tolist()
        }

        # 绘制温度沿线分布图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distances,
            y=temperatures,
            mode='lines+markers',
            name='温度沿线分布'
        ))
        fig.update_layout(
            xaxis_title='距离 (像素)',
            yaxis_title='温度 (°C)',
            title='温度沿线分布'
        )
        result_text = html.Div([
            html.P(f"直线长度：{distances[-1]:.2f} 像素"),
            html.P(f"平均温度：{np.mean(temperatures):.2f}°C"),
            html.P(f"最高温度：{np.max(temperatures):.2f}°C"),
            html.P(f"最低温度：{np.min(temperatures):.2f}°C")
        ])

        # 返回分析结果
        return fig, result_text, analysis_data
    
    else:
        return go.Figure(), "未知的选择模式。", None
    
    # 生成直方图
    bins = 30  # 您可以根据需要调整 bin 的数量
    counts, bin_edges = np.histogram(temperatures, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 对 counts 进行平滑（仅用于峰检测，不再绘制）
    from scipy.ndimage import gaussian_filter1d
    counts_smoothed = gaussian_filter1d(counts, sigma=1)
    
    # 峰检测，限制峰的数量
    peaks_indices, _ = find_peaks(counts_smoothed, height=counts_smoothed.max()*0.1, distance=2)
    
    # 检查是否检测到峰值
    if len(peaks_indices) == 0:
        # 未检测到明显的峰，绘制直方图，显示平均温度
        fig.add_trace(go.Bar(x=bin_centers, y=counts, name='温度分布'))
        
        # 更新布局
        fig.update_layout(
            xaxis_title='温度 (°C)',
            yaxis_title='频数',
            legend_title='图例',
            title='温度分布直方图'
        )
        
        result_text = html.Div([
            html.P("未检测到明显的峰值，无法进行拟合。"),
            html.P(mean_temp_text)
        ])
        return fig, result_text
    
    # 按高度排序，取前 3 个峰
    peaks_indices = peaks_indices[np.argsort(counts_smoothed[peaks_indices])[::-1]]
    peaks_indices = peaks_indices[:3]  # 限制峰的数量不超过 3 个
    
    # 构建初始猜测参数
    initial_guess = []
    for idx in peaks_indices:
        amp = counts_smoothed[idx]
        cen = bin_centers[idx]
        wid = 1  # 可以根据数据调整宽度
        initial_guess.extend([amp, cen, wid])
    
    # 添加偏移量作为参数
    initial_guess.append(0)  # 偏移量初始值
    
    # 定义新的拟合函数，包含偏移量
    def multi_gaussian_with_offset(x, *params):
        y = np.zeros_like(x)
        num_peaks = (len(params) - 1) // 3
        for i in range(0, num_peaks * 3, 3):
            amp = params[i]
            cen = params[i+1]
            wid = params[i+2]
            y += amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))
        offset = params[-1]
        y += offset
        return y
    
    try:
        popt, pcov = curve_fit(multi_gaussian_with_offset, bin_centers, counts_smoothed, p0=initial_guess)
        peaks = []
        num_peaks = (len(popt) - 1) // 3
        for i in range(0, num_peaks * 3, 3):
            amp = popt[i]
            cen = popt[i+1]
            wid = popt[i+2]
            peaks.append({'amp': amp, 'cen': cen, 'wid': wid})
        # 按峰值强度排序
        peaks_sorted = sorted(peaks, key=lambda x: x['amp'], reverse=True)
        # 取最多前 3 个峰
        top_peaks = peaks_sorted[:3]
        peak_info = [f"峰{i+1}：特征温度={peak['cen']:.2f}°C" for i, peak in enumerate(top_peaks)]
        result_text = html.Div([
            html.P(info) for info in peak_info
        ] + [html.P(mean_temp_text)])
        
        # 绘制原始直方图
        fig.add_trace(go.Bar(x=bin_centers, y=counts, name='原始直方图'))
        
        # 绘制各个高斯分布组件
        x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 500)
        offset = popt[-1]
        y_total = np.zeros_like(x_fit)
        
        for i in range(0, num_peaks * 3, 3):
            amp = popt[i]
            cen = popt[i+1]
            wid = popt[i+2]
            y_component = amp * np.exp(-(x_fit - cen) ** 2 / (2 * wid ** 2))
            y_total += y_component
            # 绘制单个高斯组件曲线
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_component + offset,
                mode='lines',
                name=f'峰 {(i // 3) + 1} 组件',
                line=dict(dash='dash')
            ))
            # 标注特征峰位置
            fig.add_trace(go.Scatter(
                x=[cen, cen],
                y=[0, amp + offset],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ))
            fig.add_annotation(
                x=cen,
                y=amp + offset,
                text=f"<b>峰{(i // 3) + 1}</b>",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
                font=dict(color='red', size=12),
                align='center'
            )
        
        # 绘制拟合总曲线
        y_total += offset
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_total,
            mode='lines',
            name='拟合曲线',
            line=dict(color='red')
        ))
        
        # 更新布局
        fig.update_layout(
            xaxis_title='温度 (°C)',
            yaxis_title='频数',
            legend_title='图例'
        )
    except Exception as e:
        print(f"An error occurred during curve fitting: {e}")
        # 拟合失败，绘制直方图，显示平均温度
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=counts, name='温度分布'))
        fig.update_layout(
            xaxis_title='温度 (°C)',
            yaxis_title='频数',
            legend_title='图例',
            title='温度分布直方图'
        )
        result_text = html.Div([
            html.P("拟合失败。"),
            html.P(mean_temp_text)
        ])
    return fig, result_text, analysis_data  # 确保返回分析数据


@app.callback(
    Output('download-data', 'data'),
    Input('save-data-button', 'n_clicks'),
    [State('analysis-data-store', 'data'),
     State('file-dropdown', 'value'),
     State('selection-mode', 'value')]
)
def save_data(n_clicks, analysis_data, selected_file, selection_mode):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    if analysis_data is None:
        return

    # 准备文件名
    file_name = os.path.basename(selected_file)
    base_name, _ = os.path.splitext(file_name)
    if selection_mode == 'rectangle':
        save_file_name = f"{base_name}_rectangle_data.npz.gz"
    elif selection_mode == 'line':
        save_file_name = f"{base_name}_line_data.npz.gz"
    else:
        save_file_name = f"{base_name}_data.npz.gz"

    # 准备要保存的数据
    data_to_save = {key: np.array(value) for key, value in analysis_data.items()}

    # 保存数据到缓冲区
    npz_buffer = io.BytesIO()
    np.savez_compressed(npz_buffer, **data_to_save)
    npz_buffer.seek(0)

    # 进行 Gzip 压缩
    gz_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buffer, mode='wb') as gz_file:
        gz_file.write(npz_buffer.read())
    gz_buffer.seek(0)

    # 返回下载数据
    return dcc.send_bytes(gz_buffer.getvalue(), filename=save_file_name)

if __name__ == '__main__':

    app.run_server(debug=True)