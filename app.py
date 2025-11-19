from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mainpage')
def homepage():
    return render_template('homepage.html')

@app.route('/learnmore')
def learnmore():
    return render_template('learnmore.html')

@app.route('/sampledata')
def table():
    return render_template('table.html')

@app.route('/predict', methods=['POST'])
def predict():

    preg = float(request.form['pregnancies'])
    glu = float(request.form['glucose'])
    bp = float(request.form['blood'])
    skin = float(request.form['skin'])
    ins = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    dpf = float(request.form['dpf'])
    age = float(request.form['age'])

    input_data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    # send everything to result.html
    return render_template(
        "result.html",
        result=result,
        preg=preg,
        glu=glu,
        bp=bp,
        skin=skin,
        ins=ins,
        bmi=bmi,
        dpf=dpf,
        age=age
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
