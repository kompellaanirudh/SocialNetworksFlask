import numpy as np
from flask import Flask, request,render_template
import pickle
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
from model import  X_train

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    age = request.args.get("age")
    estimated_salary = request.args.get("estimated_salary")

    pred = model.predict([[age, estimated_salary]])


    return str(pred)


if __name__ == "__main__":
    app.run(debug=True)