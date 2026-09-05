from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app) # Allow cross-origin requests from the frontend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def serve_frontend():
    return send_from_directory(BASE_DIR, 'index.html')

# Load model dynamically based on environment
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.joblib')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}. Run train.py first.")

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500

    try:
        data = request.json
        features = data.get('features')

        if not features or len(features) != 30:
            return jsonify({'error': 'Invalid input: Expected array of 30 clinical features.'}), 400

        # Reshape for sklearn (1 sample, n features)
        features_array = np.array(features).reshape(1, -1)
        
        # Predict class and probability
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)[0]

        result = {
            'prediction': int(prediction[0]),
            'diagnosis': 'Malignant (Anomaly Detected)' if prediction[0] == 1 else 'Benign (Normal)',
            'confidence': round(float(probability[prediction[0]]) * 100, 2)
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5001)