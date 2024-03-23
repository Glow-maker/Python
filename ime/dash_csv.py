import dash
from dash import dcc, html, Input, Output, callback, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# pyinstaller --onefile dash_csv.py
# pyinstaller --onefile --add-data "C:\Users\94723\OneDrive\work\Vscode\ime\IV_TEST;." dash_csv.py
app = dash.Dash(__name__)
app.title = 'Local CSV Plotter'

app.layout = html.Div([
    # html.Img(src=r'assets\Boya.png', style={
    #     'height': '1000px',  # 或者根据您的需要调整
    #     'width': 'auto',
    #     'position': 'absolute',
    #     'right': '10px',  # 根据需要调整距离右边界的距离
    #     'top': '70px',    # 根据需要调整距离顶部的距离
    #     #'z-index': '1000',
    #     'background-color': '#808080',
    #     'opacity': '0.15'
    # }),
    # html.Img(src=r'assets\logo.png', style={
    # 'height': '100px',  # 或者根据您的需要调整
    # 'width': 'auto',
    # 'position': 'absolute',
    # 'right': '270px',  # 根据需要调整距离右边界的距离
    # 'top': '70px',    # 根据需要调整距离顶部的距离
    # #'z-index': '1000',
    # 'background-color': '#808080',
    # 'opacity': '0.03'
    # }),
    html.Div([
        "Input Folder Path: ",
        dcc.Input(id='folder-path', type='text', value='', style={'width': '400px'}),
        html.Button('Load CSV Files', id='load-files', n_clicks=0)
    ]),
    html.Div([
        html.Button('Previous', id='prev-file', n_clicks=0, 
                    style={
                        'width': '100px', 
                        'height': '50px',
                        'background-color': '#007BFF',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '5px',
                        'font-size': '18px',
                        'margin-right': '10px',  # 在两个按钮之间添加一些空间
                        'cursor': 'pointer'
                    }),
        html.Button('Next', id='next-file', n_clicks=0, 
                    style={
                        'width': '100px', 
                        'height': '50px',
                        'background-color': '#28A745',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '5px',
                        'font-size': '18px',
                        'margin-right': '10px',
                        'cursor': 'pointer'
                    }),
        html.Button('Set as Reference Plot', id='set-reference-plot', n_clicks=0,                     
                    style={
                        'width': '100px', 
                        'height': '50px',
                        'background-color': '#808080',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '5px',
                        'font-size': '18px',
                        'margin-right': '10px',
                        'cursor': 'pointer'
                    }),
    ], style={'position': 'fixed', 'bottom': '20px', 
              'right': '20px', 'display': 'flex', 
              'align-items': 'center', 'z-index': '1000'}),
    dcc.Dropdown(
        id='file-dropdown',
        options=[],
        value=None,
        placeholder="Select a CSV file"
    ),
    dcc.Store(id='reference-plot-store'),
    # 现有的 Graph 组件
    dcc.Graph(id='voltage-current-graph'),
    # 下方的参照 Graph 组件
    dcc.Graph(id='reference-voltage-current-graph')
])

@callback(
    Output('file-dropdown', 'options'),
    Input('load-files', 'n_clicks'),
    State('folder-path', 'value')
)
def list_csv_files(n_clicks, folder_path):
    if n_clicks > 0 and folder_path:
        try:
            csv_files = [os.path.join(dirpath, f)
             for dirpath, dirnames, files in os.walk(folder_path)
             for f in files if f.endswith('.csv')]
            # csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            return [{'label': f, 'value': os.path.join(folder_path, f)} for f in csv_files]
        except FileNotFoundError:
            print("Folder not found.")
            return []
    return []

@callback(
    Output('voltage-current-graph', 'figure'),
    Input('file-dropdown', 'value')
)
def update_graph(selected_file):
    if selected_file is None:
        return go.Figure()

    # 读取CSV文件并提取数据
    data = pd.read_csv(selected_file, sep=None, engine='python')
    data_values = data[data.iloc[:, 0].str.startswith('DataValue')]
    voltage_current_data = data_values.iloc[:, 0].str.split(',', expand=True)
    voltage = pd.to_numeric(voltage_current_data[1].str.strip())
    current = pd.to_numeric(voltage_current_data[2].str.strip())

    # 创建交互式图表
    fig = go.Figure(data=go.Scatter(x=voltage, y=current, mode='markers+lines', text=voltage_current_data[1] + ", " + voltage_current_data[2]))
    fig.update_layout(title='Voltage vs. Current', xaxis_title='Voltage (V)', yaxis_title='Current (A)')

    return fig
# 回调以处理按钮点击事件
@app.callback(
    Output('file-dropdown', 'value'),
    [Input('prev-file', 'n_clicks'), Input('next-file', 'n_clicks')],
    [State('file-dropdown', 'options'), State('file-dropdown', 'value')]
)
def switch_file(prev_clicks, next_clicks, options, current_value):
    ctx = dash.callback_context
    if not ctx.triggered or not options:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    options_values = [option['value'] for option in options]
    
    if current_value not in options_values:
        return options_values[0] if options_values else None

    current_index = options_values.index(current_value)
    
    if button_id == 'next-file':
        next_index = (current_index + 1) % len(options_values)
    elif button_id == 'prev-file':
        next_index = (current_index - 1) % len(options_values)
    else:
        raise PreventUpdate

    return options_values[next_index]

# 创建回调来更新 dcc.Store 的内容
"""
这个回调监听“Set as Reference Plot”按钮的点击事件，
并使用当前 graph 组件的 figure 作为状态。
点击按钮时，它将当前图表的数据保存到 dcc.Store 中。
"""
@app.callback(
    Output('reference-plot-store', 'data'),
    [Input('set-reference-plot', 'n_clicks')],
    [State('voltage-current-graph', 'figure')]
)
def update_reference_plot(n_clicks, current_figure):
    if n_clicks > 0:
        return current_figure
    return dash.no_update

# 创建回调来显示参照 plot
"""
这个回调函数监听 dcc.Store 组件的 data 属性。
当用户设置参照 plot 时，dcc.Store 中的数据会更新，
触发这个回调函数运行，并在下方的 graph 组件中显示参照 plot。
"""
@app.callback(
    Output('reference-voltage-current-graph', 'figure'),
    [Input('reference-plot-store', 'data')]
)
def display_reference_plot(stored_data):
    if stored_data is None:
        return go.Figure()  # 如果没有存储的数据，则显示空图表
    return stored_data



if __name__ == '__main__':
    app.run_server(debug=False)
