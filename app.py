# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fever = float(request.form['fever'])
    cough = int(request.form['cough'])
    breath = int(request.form['breath'])

    features = np.array([[fever, cough, breath]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "High chance of Infection"
    else:
        result = "Low chance of Infection"

    return render_template('index.html', output=result)

if __name__ == "__main__":
    app.run(debug=True)