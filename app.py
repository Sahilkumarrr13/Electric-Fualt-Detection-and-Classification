from flask import Flask, request, render_template, url_for, redirect
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Ia=float(request.form.get('Ia')),
            Ib=float(request.form.get('Ib')),
            Ic=float(request.form.get('Ic')),
            Va=float(request.form.get('Va')),
            Vb=float(request.form.get('Vb')),
            Vc=float(request.form.get('Vc'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        
        return render_template('index.html', results = result )


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)