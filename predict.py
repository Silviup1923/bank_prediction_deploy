import pickle
from flask import Flask,jsonify,request

input_file = 'model1.bin'
with open(input_file,'rb') as f_in:
    model = pickle.load(f_in)

input_file2 = 'dv.bin'
with open(input_file2,'rb') as f_in:
    dv = pickle.load(f_in)

customer = {"job": "retired", "duration": 445, "poutcome": "success"}

app = Flask('bank')

@app.route('/predict',methods=['POST'])

def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    credit = y_pred >= 0.5
    result={'credit probabiliy': float(y_pred),
            'get_credit': bool(credit)
            }
    return jsonify(result)

#print(predict(customer).round(3)) = 0.902

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)


