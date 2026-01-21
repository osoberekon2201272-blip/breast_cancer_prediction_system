from flask import Flask, render_template, request, jsonify
from model import BreastCancerModel
import os

app = Flask(__name__)

# Initialize the model
cancer_model = BreastCancerModel()

# Load the trained model (or train if not exists)
if not cancer_model.load_model():
    print("No trained model found. Training new model...")
    from model import train_and_save_model

    train_and_save_model()
    cancer_model.load_model()


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html', feature_names=cancer_model.feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle cancer prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features (expecting all 30 features)
        features = {}

        # Validate that we have all required features
        for feature_name in cancer_model.feature_names:
            if feature_name not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing feature: {feature_name}'
                })

            try:
                features[feature_name] = float(data[feature_name])
            except (ValueError, TypeError):
                return jsonify({
                    'success': False,
                    'error': f'Invalid value for feature: {feature_name}'
                })

            # Basic validation (features should be positive)
            if features[feature_name] < 0:
                return jsonify({
                    'success': False,
                    'error': f'{feature_name} must be a positive number'
                })

        # Make prediction
        prediction, probability = cancer_model.predict(features)

        # prediction: 0 = malignant, 1 = benign
        # probability: probability of being benign

        is_benign = (prediction == 1)
        confidence = probability if is_benign else (1 - probability)

        # Return result
        # Return result
        return jsonify({
            'success': True,
            'diagnosis': 'Benign' if is_benign else 'Malignant',
            'is_benign': bool(is_benign),
            'confidence': float(round(confidence * 100, 1)),
            'message': '✓ Tumor appears to be BENIGN (non-cancerous)' if is_benign
            else '⚠️ Tumor appears to be MALIGNANT (cancerous)',
            'disclaimer': 'This is a prediction tool for educational purposes only. Always consult medical professionals for proper diagnosis.'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/sample-data')
def sample_data():
    """Provide sample data for testing"""
    try:
        # Load a sample from the dataset
        import pandas as pd
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data', 'breast_cancer.csv')

        df = pd.read_csv(data_path)

        # Get one benign and one malignant sample
        benign_sample = df[df['diagnosis'] == 1].iloc[0].drop('diagnosis').to_dict()
        malignant_sample = df[df['diagnosis'] == 0].iloc[0].drop('diagnosis').to_dict()

        return jsonify({
            'success': True,
            'benign_sample': benign_sample,
            'malignant_sample': malignant_sample
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)