from flask import Flask, request, jsonify
import pickle


app = Flask('credit-card-app')


def load_model():
    with open('model1.bin', 'rb') as f_in:  
        model = pickle.load(f_in)
    return model

def load_dict_vectorizer():
    with open('dv.bin', 'rb') as f_in:  
        dict_vectorizer = pickle.load(f_in)
    return dict_vectorizer


model = load_model()
dict_vectorizer = load_dict_vectorizer()


@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    
    sample = dict_vectorizer.transform(client_data)
    score = round(model.predict_proba(sample)[0][1], 3)
    
    return jsonify({'score': score})
    

