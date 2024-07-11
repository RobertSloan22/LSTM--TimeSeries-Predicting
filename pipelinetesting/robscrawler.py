import requests
import json
from dotenv import load_dotenv
import os
import time
from pymongo import MongoClient

load_dotenv()  # Take environment variables from .env
AUTH_TOKEN = os.getenv('AUTH_TOKEN')

# MongoDB connection
client = MongoClient('mongodb://robert:spaceforce@localhost:27017/')
db = client.discord
collection = db.messages

# Include other headers as you have specified in your original code
headers = {
    "Authorization": AUTH_TOKEN,
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-CH-UA": "\"Not A(Brand\";v=\"99\", \"Microsoft Edge\";v=\"121\", \"Chromium\";v=\"121\"",
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": "\"Windows\"",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Debug-Options": "bugReporterEnabled",
    "X-Discord-Locale": "en-US",
    "X-Discord-Timezone": "America/Los_Angeles",
}

# Including all cookies as provided
cookies = {
    "__dcfduid": "0eb87a805e3611eeb0c5172c1d8a127e",
    "__sdcfduid": "0eb87a815e3611eeb0c5172c1d8a127eadb7421d9c4263db5c5f3860808fba5aa656ebd00291ecba4038993fd706d963",
    "_ga_XXP2R74F46": "GS1.2.1699745291.1.0.1699745291.0.0.0",
    "__cfruid": "275cd80707867f0c5915cff170cd9c0104c38228-1707796659",
    "_cfuvid": "XBQT9C_CZO1rFrgQgSFTDzAvLPu0QX90XJXXJmJTOes-1707796659923-0-604800000",
    "locale": "en-US",
    "cf_clearance": "ol0n0Ip0uXf3UnxtWjYf673qt_55MUuTqhzep7MorJA-1707796660-1-AUVYoky4MnxZYXLIBPVzztudKlJIHkn0yKshrAyM+MXLmoMm3wQMm8T9gOhqsHjrj3SxPYVoOJRSTmGg+/hJ8Zs=",
    "_gcl_au": "1.1.1857939141.1707796660",
    "OptanonConsent": "isIABGlobal=false&datestamp=Mon+Feb+13+2024+01%3A57%3A40+GMT-0800+(Pacific+Standard+Time)&version=6.33.0&hosts=&landingPath=https%3A%2F%2Fdiscord.com%2F&groups=C0001%3A1%2CC0002%3A1%2CC0003%3A1",
    "_ga": "GA1.1.1160875646.1699745291",
    "_ga_Q149DFWHT7": "GS1.1.1707796660.1.0.1707796662.0.0.0",
}

# List of channels to check
channels = [
    {"id": "1106390097378684983", "nickname": "TRAC: main-chat"},
    {"id": "1157429728282673264", "nickname": "TRAC: pipe"},
    {"id": "1138509209525297213", "nickname": "TRAC: tap-protocol"},
    {"id": "1236020558488141874", "nickname": "TRAC: gib"},
    {"id": "1115824966470991923", "nickname": "OnlyFarmers: alpha"},
    {"id": "1166459733075579051", "nickname": "Ordicord: ordinals coding club 4/10/2024"},
    {"id": "1224564960575623269", "nickname": "Taproot Alpha: runes"},
    {"id": "1084525778852651190", "nickname": "DogePunks: holder-chat"},
    {"id": "1010230594367655996", "nickname": "Tensor: alpha"},
    {"id": "987504378749538366", "nickname": "Ordicord: general"},
    {"id": "1069465367988142110", "nickname": "Ordicord: tech-support"},
]

def fetch_discord_messages(channel_id):
    """Fetches messages from a specified Discord channel using Discord API."""
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages?limit=50"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch messages for channel {channel_id}: HTTP {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Request exception for channel {channel_id}: {e}")
        return []

def save_messages_to_mongodb(messages, channel_nickname):
    """Saves messages to MongoDB."""
    if messages:
        for message in messages:
            message['channel_nickname'] = channel_nickname
        collection.insert_many(messages)

def main():
    """Main function to handle fetching and storing Discord messages."""
    while True:
        for channel in channels:
            print(f"Processing channel: {channel['nickname']} (ID: {channel['id']})")
            messages = fetch_discord_messages(channel["id"])
            if messages:
                print(f"Fetched {len(messages)} messages for channel: {channel['nickname']}")
                save_messages_to_mongodb(messages, channel["nickname"])
            else:
                print(f"No new messages or failed to fetch messages for channel: {channel['nickname']}")
            time.sleep(1.2)  # Respectful delay between requests
        print("Completed one loop through all channels. Restarting...")

if __name__ == "__main__":
    main()
