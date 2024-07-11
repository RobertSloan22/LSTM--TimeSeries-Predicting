import pandas as pd
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Multiply, AdditiveAttention, Permute, Reshape, Flatten
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
from sqlalchemy import create_engine
import os
import time
import schedule
import json
# Set up logging
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Database connection details
db_params = {
    'dbname': 'sandbox',
    'user': 'postgres',
    'host': 'runes.csxbyr0egtki.us-east-1.rds.amazonaws.com',
    'password': 'uIPRefz6doiqQcbpM5po'
}

# Create the SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}/{db_params['dbname']}")

# Ensure the directories exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('history'):
    os.makedirs('history')

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
    try:
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
        logging.info("Data preprocessed successfully")
        return df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None

def create_sequences(data, sequence_length=30):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        X = []
        y = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        X = np.array(X)
        y = np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to (samples, sequence_length, num_features)
        logging.info("Sequences created successfully")
        return X, y, scaler
    except Exception as e:
        logging.error(f"Error creating sequences: {e}")
        return None, None, None

def build_model(input_shape):
    try:
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
        logging.info("Model built successfully")
        return model
    except Exception as e:
        logging.error(f"Error building model: {e}")
        return None

def train_and_save_model(model_name='model.h5'):
    try:
        df = get_data()
        if df is None:
            return
        data = preprocess_data(df)
        if data is None:
            return
        X, y, scaler = create_sequences(data)
        if X is None or y is None:
            return

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = build_model((X_train.shape[1], 1))
        if model is None:
            return

        history = model.fit(X_train, y_train, epochs=50, batch_size=25, validation_split=0.2)
        model.save(f'models/{model_name}')

        with open(f'history/{model_name}_history.npy', 'wb') as f:
            np.save(f, history.history['loss'])
            np.save(f, history.history['val_loss'])

        test_loss = model.evaluate(X_test, y_test)
        logging.info(f"Test Loss: {test_loss} for {model_name}")

        y_pred = model.predict(X_test)
        logging.info(f"Predictions made successfully for {model_name}")

        plt.figure(figsize=(12, 6))
        plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Prices')
        plt.plot(scaler.inverse_transform(y_pred.reshape(-1, 1)), color='red', label='Predicted Prices')
        plt.title(f'Price Prediction - {model_name}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig(f'static/{model_name}_prediction_plot.png')
        plt.close()

        logging.info(f"Model trained and saved successfully as {model_name}")
    except Exception as e:
        logging.error(f"Error in training and saving model: {e}")

def run_scheduler():
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in scheduler: {e}")

if __name__ == "__main__":
    run_scheduler()
