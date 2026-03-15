from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"}), 200

@app.route('/api/ml/sales-analysis', methods=['GET'])
def analyze_sales():
    return jsonify({
        "message": "Testing connection",
        "seasonal": [],
        "topProducts": [],
        "clusters": [],
        "insights": ["Flask server is reachable! Now testing data logic."]
    }), 200

if __name__ == '__main__':
    print("🚀 TEST SERVER starting on port 5005")
    app.run(port=5005, host='0.0.0.0')
