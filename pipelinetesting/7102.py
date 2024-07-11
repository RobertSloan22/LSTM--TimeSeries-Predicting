import re
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import logging
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from threading import Thread, Lock
import os
from datetime import datetime, timedelta
import json
import time
from tensorflow.keras.losses import MeanSquaredError
from dash import Dash, html, dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Database connection details
db_params = {
    'dbname': 'sandbox',
    'user': 'postgres',
    'host': 'runes.csxbyr0egtki.us-east-1.rds.amazonaws.com',
    'password': 'uIPRefz6doiqQcbpM5po'
}

# Create the SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['dbname']}")

# Initialize the Flask app
server = Flask(__name__)
socketio = SocketIO(server)

# Initialize the Dash app
app = Dash(__name__, server=server, url_base_pathname='/dash/')
app.config.suppress_callback_exceptions = True

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Columns to fetch from the database
columns_to_fetch = ['price_sats']

# Global variable to store forecasting results and a lock for thread safety
forecasting_results = None
results_lock = Lock()

def get_data():
    try:
        query = f"SELECT {', '.join(columns_to_fetch)}, timestamp FROM runes_token_info_genii WHERE rune_name = 'BILLION•DOLLAR•CAT' ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql_query(query, engine)
        logging.info("Data fetched successfully")
        return df
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return None

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.sort_values('timestamp', inplace=True)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.set_index('timestamp', inplace=True)

    df = df.rename(columns={'price_sats': 'Close'})
    df['Open'] = df['Close']
    df['High'] = df['Close']
    df['Low'] = df['Close']
    df['Adj Close'] = df['Close']
    df['Volume'] = 1

    df.ffill(inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

def create_sequences(data, sequence_length=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])

    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

def load_model_and_predict():
    df = get_data()
    if df is None:
        logging.error("No data fetched, returning None.")
        return None, None

    data = preprocess_data(df)
    X, scaler = create_sequences(data)

    try:
        model = load_model('modelsats5.h5', custom_objects={'mse': MeanSquaredError()})
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None

    try:
        predictions = model.predict(X)
        predictions = predictions.reshape(predictions.shape[0], -1)
        predictions = scaler.inverse_transform(predictions)
        logging.info("Predictions made successfully.")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None, None

    return predictions, df.index[-len(predictions):]

def forecast_job():
    global forecasting_results
    while True:
        predictions, dates = load_model_and_predict()
        if predictions is not None:
            with results_lock:
                forecasting_results = (predictions, dates)
            logging.info("Forecasting completed and results updated.")
        else:
            logging.error("Error during forecasting.")
        time.sleep(300)

@server.route('/')
def index():
    global forecasting_results
    with results_lock:
        if forecasting_results is None:
            return "Forecasting results not yet available.", 503

        predictions, dates = forecasting_results

    # Convert dates to string for JSON serialization if not already strings
    dates = [date.strftime('%Y-%m-%d %H:%M:%S') if not isinstance(date, str) else date for date in dates]

    # Create the Plotly figures
    fig_line = go.Figure()
    avg_predictions = predictions.mean(axis=1)
    split_index = int(len(dates) / 2)

    # Line plot
    fig_line.add_trace(go.Scatter(
        x=dates[:split_index], 
        y=avg_predictions[:split_index],
        mode='lines',
        name='Actual Prices',
        line=dict(color='blue')
    ))
    fig_line.add_trace(go.Scatter(
        x=dates[split_index:], 
        y=avg_predictions[split_index:],
        mode='lines',
        name='Predicted Prices',
        line=dict(color='red')
    ))
    fig_line.update_layout(
        title='Price Prediction',
        xaxis_title='Time',
        yaxis_title='Price',
        legend_title='Legend',
        xaxis=dict(tickangle=45),
        template='plotly_white'
    )

    # ECDF plot
    df_predictions = pd.DataFrame({'dates': dates, 'avg_predictions': avg_predictions})
    fig_ecdf = px.ecdf(df_predictions, x='dates', y='avg_predictions', color_discrete_sequence=['blue', 'red'])

    try:
        with open('app.log', 'r') as f:
            log_content = f.readlines()
    except Exception as e:
        log_content = [f"Error reading log file: {e}"]

    parsed_logs = parse_logs(log_content)
    return render_template('model.html', logs=parsed_logs, plot_line=fig_line.to_html(full_html=False), plot_ecdf=fig_ecdf.to_html(full_html=False), dates=json.dumps(list(dates)), predictions=json.dumps(predictions.tolist()))

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

def get_options(list_stocks):
    dict_list = [{'label': i, 'value': i} for i in list_stocks]
    return dict_list

df = get_data()
if df is not None:
    df = preprocess_data(df)
    df['stock'] = 'BILLION•DOLLAR•CAT'  # Add a 'stock' column for demonstration

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('DASH - STOCK PRICES'),
                                 html.P('Visualising time series with Plotly - Dash.'),
                                 html.P('Pick one or more stocks from the dropdown below.'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='stockselector', options=get_options(df['stock'].unique()),
                                                      multi=True, value=[df['stock'].sort_values().iloc[0]],  # Use iloc for positional indexing
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'})
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True),
                                 dcc.Graph(id='ecdf', config={'displayModeBar': False}, animate=True)
                             ])
                              ])
        ]

)

@app.callback(
    [Output('timeseries', 'figure'),
     Output('ecdf', 'figure')],
    [Input('stockselector', 'value')]
)
def update_graph(selected_dropdown_value):
    global forecasting_results
    with results_lock:
        if forecasting_results is None:
            logging.info("No forecasting results available in update_graph.")
            return {}, {}

        predictions, dates = forecasting_results
        logging.info(f"Forecasting results found: {len(predictions)} predictions")

    dates = pd.to_datetime(dates)
    avg_predictions = predictions.mean(axis=1)

    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(go.Scatter(x=dates,
                                 y=avg_predictions,  # Assuming single stock selection for simplicity
                                 mode='lines',
                                 opacity=0.7,
                                 name=stock,
                                 textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure_line = {
        'data': data,
        'layout': go.Layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            template='plotly_dark',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            margin={'b': 15},
            hovermode='x',
            autosize=True,
            title={'text': 'Predicted Prices', 'font': {'color': 'white'}, 'x': 0.5},
            xaxis={'range': [dates.min(), dates.max()]},
        ),
    }

    df_predictions = pd.DataFrame({'dates': dates, 'avg_predictions': avg_predictions})
    figure_ecdf = px.ecdf(df_predictions, x='dates', y='avg_predictions', color_discrete_sequence=['blue', 'red']).update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
    )

    return figure_line

if __name__ == "__main__":
    forecast_thread = Thread(target=forecast_job)
    forecast_thread.daemon = True
    forecast_thread.start()
    socketio.run(server, host='0.0.0.0', port=5400, debug=True)
