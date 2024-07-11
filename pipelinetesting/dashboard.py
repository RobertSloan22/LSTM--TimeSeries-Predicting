import re
import json
import numpy as np
import plotly.graph_objects as go
from flask import Flask
import dash
import logging
from dash import dcc, html
from dash.dependencies import Input as DashInput, Output

app = Flask(__name__)

# Dash app setup
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/', external_stylesheets=external_stylesheets)
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

dash_app.layout = html.Div([
    html.H1('Model Forecasting Dashboard'),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Model 1', 'value': 'models/model1.h5'},
            {'label': 'Model 2', 'value': 'models/model2.h5'},
            {'label': 'Model 3', 'value': 'models/model3.h5'},
            {'label': 'Model 4', 'value': 'models/model4.h5'},
            {'label': 'Model 5', 'value': 'models/model5.h5'}
        ],
        value='models/model1.h5'
    ),
    dcc.Graph(id='forecast-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*60000,  # in milliseconds
        n_intervals=0
    ),
    html.Div(id='log-output', style={'whiteSpace': 'pre-line'}),
    html.H2('Training Status'),
    html.Div(id='training-status-output')
])

@dash_app.callback(
    Output('forecast-graph', 'figure'),
    Output('log-output', 'children'),
    Output('training-status-output', 'children'),
    DashInput('model-dropdown', 'value'),
    DashInput('interval-component', 'n_intervals')
)
def update_graph(selected_model, n_intervals):
    try:
        with open('forecast.log', 'r') as f:
            log_content = f.readlines()
    except Exception as e:
        log_content = [f"Error reading log file: {e}"]

    parsed_logs = parse_logs(log_content)
    logs = '\n'.join([f"{log['timestamp']} - {log['message']}" for log in parsed_logs['INFO']])

    training_status = ''  # Placeholder for actual training status updates
    # Add logic to retrieve training status from a shared resource or through websocket updates

    # Load the latest forecast results
    predictions, dates = load_forecast_results(selected_model)

    if predictions is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Predictions'))
    else:
        avg_predictions = predictions.mean(axis=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=avg_predictions, mode='lines', name='Predictions'))

    return fig, logs, training_status

def parse_logs(logs):
    parsed_logs = {
        'INFO': [],
        'WARNING': [],
        'ERROR': []
    }
    log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):(\w+):(.*)')

    for log in logs:
        match = log_pattern.match(log)
        if match:
            timestamp, log_type, message = match.groups()
            parsed_logs[log_type].append({'timestamp': timestamp, 'message': message})

    for log_type in parsed_logs:
        parsed_logs[log_type] = parsed_logs[log_type][-10:]

    return parsed_logs

def load_forecast_results(model_path):
    try:
        with open(f'results/{model_path}_forecast_results.json', 'r') as f:
            results = json.load(f)
        predictions = np.array(results['predictions'])
        dates = np.array(results['dates'])
        return predictions, dates
    except Exception as e:
        logging.error(f"Error loading forecast results: {e}")
        return None, None

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5200, debug=True)
