"""
Phishing Detection API - Flask Application
Production-ready REST API for URL phishing detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import json
from urllib.parse import urlparse
import re
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome Extension

# Load models and preprocessing components
print("Loading models and preprocessing components...")
try:
    bilstm_model = joblib.load('bilstm_model.pkl')
    xgboost_model = joblib.load('xgboost_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = ['length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_and', 'nb_or', 'nb_underscore', 'nb_slash', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'web_traffic']
    
    with open('ensemble_config.json', 'r') as f:
        ensemble_config = json.load(f)
    
    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    raise

class URLFeatureExtractor:
    """Extract features from URLs"""
    
    @staticmethod
    def calculate_entropy(text):
        """Calculate Shannon entropy"""
        if not text:
            return 0
        entropy = 0
        for x in range(256):
            p_x = float(text.count(chr(x))) / len(text)
            if p_x > 0:
                entropy += - p_x * np.log2(p_x)
        return entropy
    
    @staticmethod
    def extract_features(url):
        """Extract all features from URL"""
        features = {}
        
        try:
            parsed = urlparse(url)
            
            # Basic URL characteristics
            features['length_url'] = len(url)
            features['length_hostname'] = len(parsed.netloc)
            
            # IP address in URL
            features['ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
            
            # Character counts
            features['nb_dots'] = url.count('.')
            features['nb_hyphens'] = url.count('-')
            features['nb_at'] = url.count('@')
            features['nb_and'] = url.count('&')
            features['nb_or'] = url.count('|')
            features['nb_underscore'] = url.count('_')
            features['nb_slash'] = url.count('/')
            features['nb_colon'] = url.count(':')
            features['nb_comma'] = url.count(',')
            features['nb_semicolumn'] = url.count(';')
            features['nb_dollar'] = url.count('$')
            features['nb_space'] = url.count(' ')
            features['nb_www'] = url.count('www')
            features['nb_com'] = url.count('.com')
            
            # Web traffic (Default value)
            features['web_traffic'] = 0
            
            # --- DELETE EVERYTHING BELOW THIS LINE UNTIL 'except' ---
            # (Remove has_https, domain_length, path_length, etc.)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            features = {name: 0 for name in feature_names}
        
        return features

def predict_phishing(url):
    """Predict if URL is phishing"""
    try:
        # Extract features
        features_dict = URLFeatureExtractor.extract_features(url)
        
        # Convert to array in correct order
        features_array = np.array([[features_dict.get(name, 0) for name in feature_names]])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Get predictions from all models
        nn_prob = bilstm_model.predict_proba(features_scaled)[0, 1]
        gb_prob = xgboost_model.predict_proba(features_scaled)[0, 1]
        rf_prob = rf_model.predict_proba(features_scaled)[0, 1]
        
        # Ensemble prediction
        weights = ensemble_config['weights']
        ensemble_prob = (weights['nn'] * nn_prob + 
                        weights['gb'] * gb_prob + 
                        weights['rf'] * rf_prob)
        
        # Determine prediction
        threshold = ensemble_config.get('threshold', 0.5)
        is_phishing = ensemble_prob >= threshold
        
        # Confidence level
        confidence = ensemble_prob if is_phishing else (1 - ensemble_prob)
        
        # Risk level
        if confidence >= 0.9:
            risk_level = "HIGH"
        elif confidence >= 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'url': url,
            'prediction': 'PHISHING' if is_phishing else 'LEGITIMATE',
            'is_phishing': bool(is_phishing),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'probability_scores': {
                'phishing': float(ensemble_prob),
                'legitimate': float(1 - ensemble_prob)
            },
            'model_scores': {
                'neural_network': float(nn_prob),
                'gradient_boosting': float(gb_prob),
                'random_forest': float(rf_prob)
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print(traceback.format_exc())
        raise

# API Routes

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'service': 'Phishing Detection API',
        'version': '1.0',
        'status': 'online',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict single URL',
            '/predict/batch': 'POST - Predict multiple URLs'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'version': '1.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict single URL"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Missing URL parameter',
                'message': 'Please provide a URL in the request body: {"url": "http://example.com"}'
            }), 400
        
        url = data['url']
        
        # Validate URL
        if not url or not isinstance(url, str):
            return jsonify({
                'error': 'Invalid URL',
                'message': 'URL must be a non-empty string'
            }), 400
        
        # Make prediction
        result = predict_phishing(url)
        
        return jsonify({
            'success': True,
            'result': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict multiple URLs"""
    try:
        data = request.get_json()
        
        if not data or 'urls' not in data:
            return jsonify({
                'error': 'Missing URLs parameter',
                'message': 'Please provide URLs in the request body: {"urls": ["http://example1.com", "http://example2.com"]}'
            }), 400
        
        urls = data['urls']
        
        if not isinstance(urls, list) or len(urls) == 0:
            return jsonify({
                'error': 'Invalid URLs',
                'message': 'URLs must be a non-empty list'
            }), 400
        
        # Limit batch size
        if len(urls) > 100:
            return jsonify({
                'error': 'Batch size too large',
                'message': 'Maximum 100 URLs per batch request'
            }), 400
        
        # Make predictions
        results = []
        for url in urls:
            try:
                result = predict_phishing(url)
                results.append(result)
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e),
                    'prediction': 'ERROR'
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during batch prediction'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 PHISHING DETECTION API STARTING")
    print("="*60)
    print("Server: Flask Development Server")
    print("Host: 0.0.0.0")
    print("Port: 5000")
    print("\nEndpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Predict single URL")
    print("  POST /predict/batch - Predict multiple URLs")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
