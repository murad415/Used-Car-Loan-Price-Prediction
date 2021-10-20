import numpy as np
from flask import Flask, request, jsonify, render_template
import dill
import pandas as pd
import datetime as dt

app = Flask(__name__)
with open("Training_Model_ Lightgbm", 'rb') as pickle_file:
    model=dill.load(pickle_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.to_dict()
    df=pd.DataFrame(int_features,index=[0])
    df.Year=pd.to_numeric(df.Year)
    output=np.round(model.predict(df),2)
    return render_template('index.html', prediction_text='Used car Price is  {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)