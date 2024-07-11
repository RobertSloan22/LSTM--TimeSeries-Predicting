import re
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import logging
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from threading import Thread
import os
from datetime import datetime, timedelta
import json

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

app = Flask(__name__)
socketio = SocketIO(app)

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Columns to fetch from the database
columns_to_fetch = ['price_sats']

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
    df.sort_values('timestamp', inplace=True)  # Sort in ascending order
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

def create_sequences(data, sequence_length=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])

    X = np.array(X)
    if X.size == 0:
        logging.error("No data available to create sequences.")
        return None, None

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

def load_model_and_predict():
    df = get_data()
    if df is None or df.empty:
        logging.error("Dataframe is empty or None")
        return None, None

    data = preprocess_data(df)
    if data.empty:
        logging.error("No data available after preprocessing.")
        return None, None

    X, scaler = create_sequences(data)
    if X is None or scaler is None:
        logging.error("No sequences created.")
        return None, None

    model = load_model('seq.h5')
    predictions = model.predict(X)
    predictions = predictions.reshape(predictions.shape[0], -1)

    predictions = scaler.inverse_transform(predictions)

    return predictions, df.index[-len(predictions):]

@app.route('/')
def index():
    predictions, dates = load_model_and_predict()
    if predictions is None:
        return "Error loading data or model", 500

    plot_path = 'static/prediction_plot.png'

    # Calculate the average prediction per sample (if that makes sense for your data)
    avg_predictions = predictions.mean(axis=1)

    # Ensure dates array matches the number of samples in avg_predictions
    assert len(dates) == len(avg_predictions), "Dates and predictions length mismatch"

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dates, avg_predictions, color='red', label='Predicted Prices')
    plt.scatter(dates, avg_predictions, color='blue', marker='o', label='Prediction Points')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Price Prediction')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Create a detailed output with dates and predictions
    detailed_output = [{'date': date, 'prediction': float(prediction)} for date, prediction in zip(dates, avg_predictions)]

    try:
        with open('app.log', 'r') as f:
            log_content = f.readlines()
    except Exception as e:
        log_content = [f"Error reading log file: {e}"]

    parsed_logs = parse_logs(log_content)
    return render_template('model.html', logs=parsed_logs, plot_url=plot_path, detailed_output=json.dumps(detailed_output, indent=4))

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

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5900, debug=True)
