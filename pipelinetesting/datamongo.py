from flask import Flask, render_template, jsonify, send_from_directory, request
import json
from flask_socketio import SocketIO, emit
from threading import Thread, Lock
import os
import logging
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, AdditiveAttention, Permute, Reshape, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import schedule
import time
import re
from datetime import datetime, timedelta

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
        query = f"SELECT {', '.join(columns_to_fetch)}, timestamp FROM runes_token_info_genii WHERE rune_name = 'BILLION•DOLLAR•CAT'"
        df = pd.read_sql_query(query, engine)
        logging.info("Data fetched successfully")
        return df
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return None

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
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

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(units=50, return_sequences=True)(inputs)
    x = LSTM(units=50, return_sequences=True)(x)

    attention = AdditiveAttention(name='attention_weight')
    permuted = Permute((2, 1))(x)
    reshaped = Reshape((-1, input_shape[0]))(permuted)
    attention_result = attention([reshaped, reshaped])
    multiplied = Multiply()([reshaped, attention_result])
    permuted_back = Permute((2, 1))(multiplied)
    reshaped_back = Reshape((-1, 50))(permuted_back)

    flattened = Flatten()(reshaped_back)
    outputs = Dense(1)(flattened)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model():
    df = get_data()
    if df is None:
        return
    data = preprocess_data(df)
    X, y, scaler = create_sequences(data)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_model((X_train.shape[1], 1))

    history = model.fit(X_train, y_train, epochs=10, batch_size=25, validation_split=0.2)
    model.save('modelsats4.h5')

    with open('history.npy', 'wb') as f:
        np.save(f, history.history['loss'])
        np.save(f, history.history['val_loss'])

    test_loss = model.evaluate(X_test, y_test)
    logging.info(f"Test Loss: {test_loss}")

    y_pred = model.predict(X_test)
    logging.info("Predictions made successfully")

    # Ensure the index is a datetime index
    df.index = pd.to_datetime(df.index)

    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Prices')
    plt.plot(scaler.inverse_transform(y_pred.reshape(-1, 1)), color='red', label='Predicted Prices')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('static/prediction_plot.png')
    plt.close()

    socketio.emit('future_predictions', {
        'dates': df.index[-len(y_test):].strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'predictions': y_pred.tolist(),
        'actuals': y_test.tolist()
    })

def forecast(model, last_sequence, n_future_steps, scaler, sequence_length):
    forecasted_values = []
    for _ in range(n_future_steps):
        input_data = np.array([last_sequence])
        input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
        predicted_value = model.predict(input_data)
        forecasted_values.append(predicted_value[0][0])
        last_sequence = np.append(last_sequence[1:], predicted_value[0][0])
    forecasted_values = np.array(forecasted_values)
    forecasted_values = np.reshape(forecasted_values, (forecasted_values.shape[0], 1))
    forecasted_values = scaler.inverse_transform(forecasted_values)
    return forecasted_values

def load_model_and_predict():
    df = get_data()
    if df is None:
        return None, None, None
    data = preprocess_data(df)
    sequence_length = 30
    n_future_steps = 100
    X, y, scaler = create_sequences(data, sequence_length)
    model = load_model('modelsats2.h5')
    predictions = model.predict(X)
    
    predictions = predictions.reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    last_sequence = data['Close'].values[-sequence_length:]
    forecasted_values = forecast(model, last_sequence, n_future_steps, scaler, sequence_length)

    return predictions, data.index[-len(predictions):], forecasted_values

def forecast_job():
    global forecasting_results
    while True:
        predictions, dates, forecasted_values = load_model_and_predict()
        if predictions is not None:
            with results_lock:
                forecasting_results = (predictions, dates, forecasted_values)
            logging.info("Forecasting completed and results updated.")
        else:
            logging.error("Error during forecasting.")
        time.sleep(300)  # Reforecast every 5 minutes

def retrain_model():
    df = get_data()
    if df is None:
        logging.error("Error retrieving data for retraining.")
        return False

    data = preprocess_data(df)
    X, y, scaler = create_sequences(data)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = build_model((X_train.shape[1], 1))

    history = model.fit(X_train, y_train, epochs=50, batch_size=25, validation_split=0.2)
    model.save('modelsats2.h5')

    with open('history.npy', 'wb') as f:
        np.save(f, history.history['loss'])
        np.save(f, history.history['val_loss'])

    test_loss = model.evaluate(X_test, y_test)
    logging.info(f"Test Loss: {test_loss}")

    y_pred = model.predict(X_test)
    logging.info("Retrained predictions made successfully")

    return True

@app.route('/retrain', methods=['POST'])
def retrain():
    success = retrain_model()
    if success:
        return jsonify({'status': 'success', 'message': 'Model retrained successfully.'})
    else:
        return jsonify({'status': 'error', 'message': 'Error during retraining.'}), 500

results_lock = Lock()

def forecast_job():
    global forecasting_results
    while True:
        predictions, dates, forecasted_values = load_model_and_predict()
        if predictions is not None:
            with results_lock:
                forecasting_results = (predictions, dates, forecasted_values)
            logging.info("Forecasting completed and results updated.")
        else:
            logging.error("Error during forecasting.")
        time.sleep(300)  # Reforecast every 5 minutes

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

@app.route('/')
def index():
    global forecasting_results
    with results_lock:
        if forecasting_results is None:
            return "Forecasting results not yet available.", 503

        predictions, dates, forecasted_values = forecasting_results

    plot_path = 'static/prediction_plot.png'

    avg_predictions = predictions.mean(axis=1)
    assert len(dates) == len(avg_predictions), "Dates and predictions length mismatch"

    plt.figure(figsize=(12, 6))
    plt.plot(dates, avg_predictions, color='red', label='Predicted Prices')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Price Prediction')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    try:
        with open('app.log', 'r') as f:
            log_content = f.readlines()
    except Exception as e:
        log_content = [f"Error reading log file: {e}"]

    parsed_logs = parse_logs(log_content)
    return render_template('modelapp.html', logs=parsed_logs, plot_url=plot_path, dates=json.dumps(list(dates)), predictions=json.dumps(predictions.tolist()))

@socketio.on('connect')
def handle_connect(auth=None):
    logging.info('Client connected')
    next_run_time = schedule.next_run()
    if next_run_time is not None:
        socketio.emit('next_run', {'next_run': next_run_time.strftime('%Y-%m-%d %H:%M:%S')})
    else:
        socketio.emit('next_run', {'next_run': (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == "__main__":
    # Start the training in a separate thread
    training_thread = Thread(target=train_model)
    training_thread.start()

    # Start the forecasting job in a background thread
    forecast_thread = Thread(target=forecast_job)
    forecast_thread.daemon = True
    forecast_thread.start()

    # Start the Flask application
    socketio.run(app, host='0.0.0.0', port=5300, debug=True)

    while True:
        schedule.run_pending()
        time.sleep(1)
