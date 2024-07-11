from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Example of passing some logs to the template. Adjust as needed.
    logs = {"INFO": ["Log 1", "Log 2", "Log 3"]}
    return render_template('medash.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True, port=5700)