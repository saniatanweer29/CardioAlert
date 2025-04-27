from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('cardiac_arrest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'age': int(request.form['age']) * 365,
        'gender': int(request.form['gender']),
        'height': int(request.form['height']),
        'weight': int(request.form['weight']),
        'ap_hi': int(request.form['ap_hi']),
        'ap_lo': int(request.form['ap_lo']),
        'cholesterol': int(request.form['cholesterol']),
        'gluc': int(request.form['gluc']),
        'smoke': int(request.form['smoke']),
        'alco': int(request.form['alco']),
        'active': int(request.form['active'])
    }
    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0][1]
    prediction = 'Yes' if prob >= 0.5 else 'No'
    prob_percentage = prob * 100

    if prob_percentage < 30:
        precautions = 'Maintain regular exercise and a healthy diet.'
    elif 30 <= prob_percentage < 60:
        precautions = 'Consult a doctor, monitor blood pressure and cholesterol regularly.'
    elif 60 <= prob_percentage < 80:
        precautions = 'Start prescribed medications, strict diet control, regular checkups.'
    else:
        precautions = 'Immediate medical intervention needed, lifestyle changes must be rigorous.'

    return render_template('result.html', prediction=prediction, probability=round(prob_percentage, 2), precautions=precautions)

if __name__ == "__main__":
    app.run(debug=True)