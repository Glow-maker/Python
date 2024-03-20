import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import os

# 你的CSV文件所在的目录
directory = r"IV_TEST/20240304"

# 创建Dash应用
app = dash.Dash(__name__)

# 获取CSV文件列表
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 应用布局
app.layout = html.Div([
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': f, 'value': os.path.join(directory, f)} for f in csv_files],
        value=None,
        placeholder="Select a file"
    ),
    dcc.Graph(id='voltage-current-graph')
])

# 回调函数，根据选择的文件名更新图表
@app.callback(
    Output('voltage-current-graph', 'figure'),
    [Input('file-dropdown', 'value')]
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

if __name__ == '__main__':
    app.run_server(debug=True)
