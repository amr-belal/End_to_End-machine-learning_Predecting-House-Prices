import pickle
import os 
from flask import Flask , request , app,jsonify , render_template, url_for
import numpy as np
import pandas as pd

# Create Flask app starting poin of the application
# This is the entry point of the application
app = Flask(__name__)

#load the model pickle file

# linear regression model
# regmodel = pickle.load(open('linear_regression_model.pkl','rb'))

# random forest model 
rfmodel = pickle.load(open('random_forest_model.pkl','rb'))

# load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the home route of the application
@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict_api" , methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1)) # reshape the data to match the model input shape
    # pass the data to the scaler and get the scaled data and then pass it to the model
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    
    rfmodel_prediction_output = rfmodel.predict(new_data)
    
    
    print("regmodel_prediction_output", rfmodel_prediction_output[0])
    
    return jsonify(rfmodel_prediction_output[0])


if __name__ == "__main__":
    
    app.run(debug=True)
    