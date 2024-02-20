

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__,static_url_path='/static')
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
accuracy1 = pickle.load(open('accuracy1.pkl', 'rb'))
accuracy2 = pickle.load(open('accuracy2.pkl', 'rb'))

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for model prediction
@app.route('/predict',methods=['POST'])
def predict():
    
    # Get the input data from the form
    features = [float(x) for x in request.form.values()]
    
    # Convert the input data into numpy array
    final_features = [np.array(features)]
    final_features = scaler.transform(final_features)    
    
    # Make prediction using the loaded model
    predict1 = model1.predict(final_features)
    predict2 = model2.predict(final_features)
    y_prob1_test = model1.predict_proba(final_features)
    y_prob2_test = model1.predict_proba(final_features)
    y_prob1_success = y_prob1_test[:, 1]
    y_prob2_success = y_prob2_test[:, 1]
    
    # print("final features",final_features)
    # print("prediction 1:",predict1)
    # print("prediction 2:",predict2)
    
    output1 = round(predict1[0], 2)
    output2 = round(predict2[0], 2)
    y_prob1=round(y_prob1_success[0], 3)
    y_prob2=round(y_prob2_success[0], 3)

    # print(output1)
    # print(output2)
    
    if output1 == 0: 
        ans1 = f'THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {y_prob1:.2f} AND ACCURACY OF {accuracy1:.2f}'
    else:
        ans1 = f'THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {y_prob1:.2f} AND ACCURACY OF {accuracy1:.2f}'
    
    if output2 == 0:
        ans2 = f'THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {y_prob2:.2f}AND ACCURACY OF {accuracy2:.2f}'
    else:
        ans2 = f'THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {y_prob2:.2f} AND ACCURACY OF {accuracy1:.2f}'
    
    # Return the prediction result
    return render_template('index.html', ans1 = ans1, accuracy1 = format(accuracy1  * 100,'.2f'), ans2 = ans2, accuracy2 = format(accuracy2  * 100,'.2f'), y_prob1 = format(y_prob1  * 100,'.2f'), y_prob2 = format(y_prob2  * 100,'.2f') )

        
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    predict1 = model1.predict([np.array(list(data.values()))])
    predict2 = model2.predict([np.array(list(data.values()))])

    output = predict1[0]
    output = predict2[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)