from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model
data = pd.read_csv('Crop_Recommendation.csv')
# Separate the features and the target variable
feature_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
X = data[feature_columns]
y = data['Crop']
label_encoder = LabelEncoder()

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request
    features = request.get_json()
    features_df = pd.DataFrame([features], columns=feature_columns)
    
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
