from flask import Flask, jsonify, render_template
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/fetch-element')
def fetch_element():
    url = 'https://magiceden.io/runes/BILLION%E2%80%A2DOLLAR%E2%80%A2CAT'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Assuming you want to extract an element with id 'specific-element'
        element = soup.find(id='specific-element')
        if element:
            return jsonify({'content': str(element)})
        else:
            return jsonify({'error': 'Element not found'}), 404
    else:
        return jsonify({'error': 'Failed to fetch the page'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
