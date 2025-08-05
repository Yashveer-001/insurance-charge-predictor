from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('insurance_model.pkl')

# Mapping for encoding
sex_map = {'female': 0, 'male': 1}
smoker_map = {'no': 0, 'yes': 1}
region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Read input from form
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # Encode for model input
    input_features = [
        age,
        sex_map[sex],
        bmi,
        children,
        smoker_map[smoker],
        region_map[region]
    ]

    # Predict
    prediction = model.predict([input_features])[0]
    prediction = round(prediction, 2)

    # Prepare readable input values
    readable_input = {
        'Age': age,
        'Sex': sex.capitalize(),
        'BMI': bmi,
        'Children': children,
        'Smoker': smoker.capitalize(),
        'Region': region.capitalize()
    }

    return render_template('index.html', prediction=prediction, inputs=readable_input)

if __name__ == '__main__':
    app.run(debug=True)
