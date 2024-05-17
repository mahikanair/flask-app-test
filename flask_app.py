from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the features from the request
        data = request.get_json()
        features = np.array([data['features']])
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
