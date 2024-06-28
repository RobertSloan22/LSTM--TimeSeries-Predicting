import random
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import warnings
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout,BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import backend

import pandas as pd

# Read the Excel file
df = pd.read_csv('/content/drive/MyDrive/BitcoinRunes/BILLION•DOLLAR•CAT.csv')

# Display the first few rows of the DataFrame
print(df.head())

df.drop(['symbol', 'etching_time', 'turbo', 'inscription_id' ], axis=1, inplace=True)


# Load the data from a CSV file

# Assign the loaded DataFrame to a variable
final_data = df.copy()

# Convert the 'timestamp' column to datetime, handling various formats
final_data['timestamp'] = pd.to_datetime(final_data['timestamp'], errors='coerce')

# Remove milliseconds and timezone from the 'timestamp' column
final_data['timestamp'] = final_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Set the 'timestamp' column as the index of the DataFrame
final_data.set_index('timestamp', inplace=True)

# Print the final DataFrame to verify the result
print(final_data)
# Ensure random initialization of the seed
seed_value = random.randint(0, 10000)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# Assuming `df` is your combined DataFrame containing all rune data
# Define the columns to be used
titles = [
    "Market Cap (USD)", "Holders", "Price (Sats)", "Price (USD)", "Price Change",
    "Volume (1h BTC)", "Volume (1d BTC)", "Volume (7d BTC)", "Volume Total BTC",
    "Sales (1h)", "Sales (1d)", "Sales (7d)", "Sellers (1h)", "Sellers (1d)",
    "Sellers (7d)", "Buyers (1h)", "Buyers (1d)", "Buyers (7d)", "Listings Min Price",
    "Listings Max Price", "Listings Avg Price", "Listings Percentile 25",
    "Listings Median Price", "Listings Percentile 75", "Listings Total Quantity"
]

feature_keys = [
    "marketcap_usd", "holders", "price_sats", "price_usd", "price_change",
    "volume_1h_btc", "volume_1d_btc", "volume_7d_btc", "volume_total_btc",
    "sales_1h", "sales_1d", "sales_7d", "sellers_1h", "sellers_1d", "sellers_7d",
    "buyers_1h", "buyers_1d", "buyers_7d", "listings_min_price", "listings_max_price",
    "listings_avg_price", "listings_percentile_25", "listings_median_price",
    "listings_percentile_75", "listings_total_quantity"
]

date_time_key = "timestamp"

# Normalize the data
def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    data_std[data_std == 0] = 1  # Prevent division by zero
    return (data - data_mean) / data_std, data_mean, data_std

def denormalize(data, mean, std):
    return (data * std) + mean

# Assuming df is already defined as your combined dataframe
split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 1
past = 100
future_steps = 50
learning_rate = 0.002
batch_size = 256
epochs = 75

print("The selected parameters are:", ", ".join(titles))
selected_features = feature_keys
features = df[selected_features]
features.index = df[date_time_key]
features.head()

# Apply normalization
features, data_mean, data_std = normalize(features.values, train_split)
features = pd.DataFrame(features, columns=selected_features)
features.head()

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

# Ensure start and end indices are correctly calculated
start = past
end = train_split

x_train = train_data.values
y_train = features.iloc[start:end][["price_usd"]]

sequence_length = int(past / step)

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

# Verify the lengths of validation data
print("Validation data length:", len(val_data))
x_end = len(val_data) - past - future_steps
print("x_end:", x_end)

# Ensure label_start is within bounds and get the correct slice of validation data
label_start = train_split + past
label_end = len(features)  # End index for validation labels
print("label_start:", label_start)
print("label_end:", label_end)

# Debug: Check the ranges for x_val and y_val
if x_end > 0 and label_start < label_end:
    x_val = val_data.iloc[:x_end].values
    y_val = features.iloc[label_start:label_end][["price_usd"]].values  # Slice y_val up to the end

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    # Verify that dataset_val is not empty
    print("Number of batches in validation dataset:", len(dataset_val))
    for batch in dataset_val.take(1):
        inputs, targets = batch
        print("Validation Input shape:", inputs.numpy().shape)
        print("Validation Target shape:", targets.numpy().shape)
else:
    print("Insufficient validation data for the given parameters.")

# Define the model
def create_lstm_model(input_shape, output_steps):
    inputs = keras.layers.Input(shape=input_shape)
    lstm_out = keras.layers.LSTM(32)(inputs)
    # reduce overfitting with a dropout layer
    lstm_out = keras.layers.Dropout(0.02)(lstm_out)
    lstm_out = keras.layers.LSTM(32)(inputs)
    # reduce overfitting with a dropout layer
    lstm_out = keras.layers.LSTM(32)(inputs)

    outputs = keras.layers.Dense(output_steps)(lstm_out)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Get the input shape from the training data
for batch in dataset_train.take(1):
    inputs_batch, targets_batch = batch
    input_shape = inputs_batch.shape[1:]
    output_steps = targets_batch.shape[1]

model = create_lstm_model(input_shape, output_steps)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")

# Prediction and Visualization
def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    future = list(range(1, delta + 1))

    plt.figure(figsize=(10, 6))
    plt.title(title)
    for i, val in enumerate(plot_data):
        if i == 0:
            plt.plot(time_steps, val, marker[i], label=labels[i])
        else:
            plt.plot(future, val, marker[i], markersize=10, label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], delta])
    plt.xlabel("Time-Step")
    plt.ylabel("Price (USD)")
    plt.show()

# Collect predictions and actual values for the entire validation set
predictions_list = []
true_values_list = []

for batch in dataset_val:
    x, y = batch
    predictions = model.predict(x)
    predictions_list.append(predictions)
    true_values_list.append(y)

# Convert lists to numpy arrays
predictions_array = np.concatenate(predictions_list, axis=0)
true_values_array = np.concatenate(true_values_list, axis=0)

# Denormalize predictions and true values
denorm_predictions = denormalize(predictions_array, data_mean[feature_keys.index('price_usd')], data_std[feature_keys.index('price_usd')])
denorm_true_values = denormalize(true_values_array, data_mean[feature_keys.index('price_usd')], data_std[feature_keys.index('price_usd')])

# Plot the predictions versus the true values
def plot_predictions(true_values, predictions, title):
    plt.figure(figsize=(15, 5))
    plt.plot(true_values, label="True Values")
    plt.plot(predictions, label="Predictions")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

plot_predictions(denorm_true_values, denorm_predictions, "Model Predictions vs. True Values")
