import random
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import schedule
import time
import logging
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from threading import Thread
import re
import os
from datetime import datetime, timedelta
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

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
columns_to_fetch = [
    "marketcap_usd", "holders", "price_sats", "price_usd", "price_change",
    "volume_1h_btc", "volume_1d_btc", "volume_7d_btc", "volume_total_btc",
    "sales_1h", "sales_1d", "sales_7d", "sellers_1h", "sellers_1d", "sellers_7d",
    "buyers_1h", "buyers_1d", "buyers_7d", "listings_min_price", "listings_max_price",
    "listings_avg_price", "listings_percentile_25", "listings_median_price",
    "listings_percentile_75", "count_listings", "listings_total_quantity",
    "balance_change_last_1_block", "balance_change_last_3_blocks",
    "balance_change_last_10_blocks"
]

# Function to retrieve data from PostgreSQL
def get_data():
    try:
        query = f"SELECT {', '.join(columns_to_fetch)}, timestamp FROM runes_token_info_genii WHERE rune_name = 'BILLION•DOLLAR•CAT'"
        df = pd.read_sql_query(query, engine)
        logging.info("Data fetched successfully")
        return df
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return None

# Function to preprocess data
def preprocess_data(df):
    # Convert the 'timestamp' column to datetime, handling various formats
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Remove milliseconds and timezone from the 'timestamp' column
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Set the 'timestamp' column as the index of the DataFrame
    df.set_index('timestamp', inplace=True)
    
    # Fill missing values with the mean of their respective columns
    df.fillna(df.mean(), inplace=True)
    
    logging.info("Data preprocessed successfully")
    return df

# Normalize and denormalize functions
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    data_std[data_std == 0] = 1  # Prevent division by zero
    return (data - data_mean) / data_std, data_mean, data_std

def denormalize(data, mean, std):
    return (data * std) + mean

# Define LSTM model with attention mechanism
def create_lstm_attention_model(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = Attention()([x, x])
    x = TimeDistributed(Dense(64))(x)
    x = Dense(output_steps)(x)
    
    model = keras.Model(inputs, x)
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to train model
def train_model():
    logging.info("Retraining model...")
    
    # Retrieve and preprocess data
    df = get_data()
    if df is None:
        logging.error("Failed to retrieve data.")
        return
    
    df = preprocess_data(df)
    
    split_fraction = 0.715
    train_split = int(split_fraction * int(df.shape[0]))
    sequence_length = 100
    future_steps = 200

    features = df.drop(columns=['price_usd'])
    target = df['price_usd']
    
    # Normalize data
    features, data_mean, data_std = normalize(features.values, train_split)
    features = pd.DataFrame(features, columns=df.columns.drop('price_usd'))
    
    # Create time series data
    x_train = []
    y_train = []
    for i in range(len(features) - sequence_length):
        x_train.append(features.iloc[i:i+sequence_length].values)
        y_train.append(target.iloc[i+sequence_length])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Create and train model
    model = create_lstm_attention_model((x_train.shape[1], x_train.shape[2]), 1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1)

    history = model.fit(
        x_train, y_train, 
        epochs=100, 
        batch_size=64, 
        validation_split=0.2, 
        callbacks=[early_stopping, reduce_lr, tensorboard],
        verbose=1
    )
    
    # Save the model
    model.save('model7.h5')
    logging.info("Model retrained and saved.")

    # Save training history for visualization
    with open('history.npy', 'wb') as f:
        np.save(f, history.history['loss'])
        np.save(f, history.history['val_loss'])

    logging.info("Model retraining completed.")

    # Make future predictions
    latest_sequence = features.iloc[-sequence_length:].values
    latest_sequence = np.expand_dims(latest_sequence, axis=0)

    num_predictions = 200
    predicted_prices = []
    current_sequence = latest_sequence

    for _ in range(num_predictions):
        prediction = model.predict(current_sequence)
        predicted_prices.append(prediction.flatten()[0])  # Flatten and take the first element
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, features.columns.get_loc('price_sats')] = prediction.flatten()[0]  # Use the first element of prediction

    denorm_predicted_prices = denormalize(np.array(predicted_prices), data_mean[features.columns.get_loc('price_sats')], data_std[features.columns.get_loc('price_sats')]).flatten()
    future_dates = pd.date_range(start=df.index[-1], periods=num_predictions + 1, freq='H')[1:]

    denorm_latest_data = denormalize(features.iloc[-sequence_length:][["price_sats"]].values, data_mean[features.columns.get_loc('price_sats')], data_std[features.columns.get_loc('price_sats')]).flatten()

    # Emit future predictions
    socketio.emit('future_predictions', {'dates': future_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(), 'predictions': denorm_predicted_prices.tolist(), 'latest_data': denorm_latest_data.tolist()})

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
    
    # Keep only the most recent 10 entries for each log type
    for log_type in parsed_logs:
        parsed_logs[log_type] = parsed_logs[log_type][-10:]
    
    return parsed_logs

@app.route('/')
def index():
    try:
        with open('app.log', 'r') as f:
            log_content = f.readlines()
    except Exception as e:
        log_content = [f"Error reading log file: {e}"]

    parsed_logs = parse_logs(log_content)

    return render_template('index3.html', logs=parsed_logs)

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

    # Start the Flask web server in the main thread
    socketio.run(app, host='0.0.0.0', port=5300, debug=False)

    while True:
        schedule.run_pending()
        time.sleep(1)
