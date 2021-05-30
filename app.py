from flask import Flask,render_template,request
import joblib
import pandas as pd

app = Flask(__name__)

model_s = joblib.load('model_spain.pkl')
model_f = joblib.load('model_france.pkl')
model_g = joblib.load('model_germany.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
    country = request.form.get('country')
    values = request.form.values()
    data={
    'CreditScore':request.form.get('cscore'),
    'Gender':str(request.form.get('gender')),
    'Age':request.form.get('age'),
    'Tenure':request.form.get('tenure'),
    'Balance':request.form.get('balance'),
    'NumOfProducts':request.form.get('np'),
    'HasCrCard':request.form.get('cc'),
    'IsActiveMember':request.form.get('active'),
    'EstimatedSalary':request.form.get('salary')
    }

    data = pd.DataFrame([data])
    if country == 'Spain':
        if model_s.predict(data)==0:
            prediction_text = 'Safe not a churner'
        else:
            prediction_text = 'Going to leave!!!'
    elif country == 'Germany':
        if model_g.predict(data)==0:
            prediction_text = 'Safe not a churner'
        else:
            prediction_text = 'Going to leave!!!'
    else:
        if model_f.predict(data)==0:
            prediction_text = 'Safe not a churner'
        else:
            prediction_text = 'Going to leave!!!'

    return render_template('index.html', prediction_text= prediction_text)
if __name__ == '__main__':
    app.run(debug=True)