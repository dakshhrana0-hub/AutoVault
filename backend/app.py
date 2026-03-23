import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add root folder to sys.path so model_training can be imported correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model_training.base_model import predict_price, LeakFreeTargetEncoder

# Fix for joblib looking for LeakFreeTargetEncoder in __main__
import __main__
__main__.LeakFreeTargetEncoder = LeakFreeTargetEncoder

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        brand = data.get('brand')
        model = data.get('model')
        year = int(data.get('year'))
        kms = float(data.get('kms'))
        fuel = data.get('fuel')
        trans = data.get('trans')
        
        price = predict_price(brand, model, year, kms, fuel, trans)
        
        return jsonify({"price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Start the application on port 5001
    app.run(debug=True, port=5001, host='0.0.0.0')
