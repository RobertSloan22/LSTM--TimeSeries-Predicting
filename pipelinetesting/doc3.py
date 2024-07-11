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

# Define your model architecture here
def create_lstm_model(input_shape, output_steps):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dense(output_steps)
    ])
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
    
    # Assuming df is already defined as your combined dataframe
    split_fraction = 0.715
    train_split = int(split_fraction * int(df.shape[0]))
    step = 1
    past = 100
    future_steps = 50
    learning_rate = 0.002
    batch_size = 256
    epochs = 5

    features = df.drop(columns=['price_sats'])  # Replace 'price_usd' with your actual target column name
    target = df['price_sats']
    
    # Normalize data
    features, data_mean, data_std = normalize(features.values, train_split)
    features = pd.DataFrame(features, columns=df.columns.drop('price_sats'))
    
    # Create time series data
    time_steps = 100
    x_train = []
    y_train = []
    for i in range(len(features) - time_steps):
        x_train.append(features.iloc[i:i+time_steps].values)
        y_train.append(target.iloc[i+time_steps])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    # Create and train model
    model = create_lstm_model((x_train.shape[1], x_train.shape[2]), 1)
    for epoch in range(epochs):
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1)
        
        # Emit training progress
        socketio.emit('training_progress', {'epoch': epoch + 1, 'loss': history.history['loss'][0], 'val_loss': history.history['val_loss'][0]})

    # Save the model
    model.save('model.h5')
    logging.info("Model retrained and saved.")

    # Save training history for visualization
    with open('history.npy', 'wb') as f:
        np.save(f, history.history['loss'])
        if 'val_loss' in history.history:
            np.save(f, history.history['val_loss'])
        else:
            np.save(f, [])
            logging.warning("No validation loss found in history.")

    logging.info("Model retraining completed.")

# Function to make future predictions
def make_future_predictions(model, last_sequence, future_steps):
    predictions = []
    current_sequence = last_sequence

    for _ in range(future_steps):
        prediction = model.predict(current_sequence[np.newaxis, :, :])
        predictions.append(prediction[0, 0])
        
        current_sequence = np.append(current_sequence[1:], prediction, axis=0)

    return predictions

# Schedule the retraining job every hour
schedule.every().hour.do(train_model)

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
    
    return parsed_logs

@app.route('/')
def index():
    try:
        with open('app.log', 'r') as f:
            log_content = f.readlines()
    except Exception as e:
        log_content = [f"Error reading log file: {e}"]

    parsed_logs = parse_logs(log_content)

    # Load training history
    try:
        with open('history.npy', 'rb') as f:
            loss = np.load(f)
            val_loss = np.load(f)

        epochs = range(len(loss))

        plt.figure()
        plt.plot(epochs, loss, "b", label="Training loss")
        if len(val_loss) > 0:
            plt.plot(epochs, val_loss, "r", label="Validation loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('static/loss_plot.png')  # Save plot in the static directory

    except Exception as e:
        parsed_logs['ERROR'].append({'timestamp': 'N/A', 'message': f"Error loading metrics: {e}"})

    return render_template('index2.html', logs=parsed_logs)

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')

if __name__ == "__main__":
    # Start the Flask web server in a separate thread
    server_thread = Thread(target=socketio.run, args=(app,), kwargs={'host': '0.0.0.0', 'port': 4700, 'debug': False})
    server_thread.start()

    # Start the training in a separate thread
    training_thread = Thread(target=train_model)
    training_thread.start()

    while True:
        schedule.run_pending()
        time.sleep(1)
