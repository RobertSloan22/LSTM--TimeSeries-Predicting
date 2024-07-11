from flask import Flask, request, render_template
import pandas as pd
from pymongo import MongoClient
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import time

app = Flask(__name__)

# Establish a connection to the MongoDB server
client = MongoClient('mongodb+srv://radevai1201:szZ2HmXFRc902EeW@cluster0.b8z5ks7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
database = client['runes']
collection = database['GinidataRunes']

@app.route('/', methods=['GET', 'POST'])
def index():
    rune_name = None
    plot_url = None

    if request.method == 'POST':
        rune_name = request.form['rune_name']
        plot_url = update_forecast_plot(rune_name)

    return render_template('index.html', plot_url=plot_url)

def fetch_data(rune_name):
    # Fetch data for the specified rune name
    query = {"rune_name": rune_name}
    data = collection.find(query, {"timestamp": 1, "price_usd": 1})
    df = pd.DataFrame(list(data))
    df['ds'] = pd.to_datetime(df['timestamp'])
    df['y'] = df['price_usd'].astype(float)
    return df[['ds', 'y']]

def update_forecast_plot(rune_name):
    df = fetch_data(rune_name)

    if df.empty:
        return None

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create a DataFrame for future predictions
    future = model.make_future_dataframe(periods=365)

    # Generate predictions
    forecast = model.predict(future)

    # Plot the forecast
    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    plt.title(f"Real-time Price Forecast for {rune_name}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)

    return plot_url

if __name__ == '__main__':
    app.run(debug=True)
