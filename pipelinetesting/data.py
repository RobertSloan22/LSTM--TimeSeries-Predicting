import csv
import time
from pymongo import MongoClient

# Establish a connection to the MongoDB server
client = MongoClient('mongodb+srv://radevai1201:szZ2HmXFRc902EeW@cluster0.b8z5ks7.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

# Connect to the specific database and collection
database = client['runes']
collection = database['GinidataRunes']

# Specify the output CSV file path
output_csv_file = 'database_output.csv'

# Function to fetch data from the database and write unique entries to the CSV file
def fetch_and_write_data():
    # Retrieve data from the collection
    result = collection.find()

    # Read existing entries from the CSV file to avoid duplicates
    existing_entries = set()
    try:
        with open(output_csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                existing_entries.add(row[0])  # Assuming "_id" is the first column
    except FileNotFoundError:
        pass

    # Open the CSV file for appending new data
    with open(output_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header row if the file is empty
        if not existing_entries:
            header = ["_id", "amount", "balance_change_last_10_blocks", "balance_change_last_1_block",
                      "balance_change_last_3_blocks", "burned", "buyers_1d", "buyers_1h", "buyers_7d",
                      "count_listings", "etching", "etching_time", "holders", "inscription_id",
                      "listings_avg_price", "listings_max_price", "listings_median_price", "listings_min_price",
                      "listings_percentile_25", "listings_percentile_75", "listings_total_quantity",
                      "marketcap_usd", "minted", "mints", "premine", "price_change", "price_sats", "price_usd",
                      "rune_id", "rune_name", "rune_number", "sales_1d", "sales_1h", "sales_7d", "sales_total",
                      "sellers_1d", "sellers_1h", "sellers_7d", "supply", "symbol", "timestamp", "turbo",
                      "volume_1d_btc", "volume_1h_btc", "volume_7d_btc", "volume_total_btc"]
            writer.writerow(header)

        # Write unique documents to the CSV file
        for document in result:
            document_id = document.get("_id")
            if str(document_id) not in existing_entries:
                row = [str(document_id), document.get("amount"), document.get("balance_change_last_10_blocks"),
                       document.get("balance_change_last_1_block"), document.get("balance_change_last_3_blocks"),
                       document.get("burned"), document.get("buyers_1d"), document.get("buyers_1h"),
                       document.get("buyers_7d"), document.get("count_listings"), document.get("etching"),
                       document.get("etching_time"), document.get("holders"), document.get("inscription_id"),
                       document.get("listings_avg_price"), document.get("listings_max_price"),
                       document.get("listings_median_price"), document.get("listings_min_price"),
                       document.get("listings_percentile_25"), document.get("listings_percentile_75"),
                       document.get("listings_total_quantity"), document.get("marketcap_usd"), document.get("minted"),
                       document.get("mints"), document.get("premine"), document.get("price_change"),
                       document.get("price_sats"), document.get("price_usd"), document.get("rune_id"),
                       document.get("rune_name"), document.get("rune_number"), document.get("sales_1d"),
                       document.get("sales_1h"), document.get("sales_7d"), document.get("sales_total"),
                       document.get("sellers_1d"), document.get("sellers_1h"), document.get("sellers_7d"),
                       document.get("supply"), document.get("symbol"), document.get("timestamp"), document.get("turbo"),
                       document.get("volume_1d_btc"), document.get("volume_1h_btc"), document.get("volume_7d_btc"),
                       document.get("volume_total_btc")]
                writer.writerow(row)

# Loop to fetch and write data every 1 minute
while True:
    fetch_and_write_data()
    time.sleep(60)

# Close the MongoDB connection
client.close()
