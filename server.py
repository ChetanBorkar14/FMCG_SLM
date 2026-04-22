from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from mba import mba_engine
from insights import insights_generator

app = Flask(__name__, static_folder=".")
CORS(app)

print("Starting MBA Engine setup...")
mba_engine.load_and_mine(min_support=0.02, min_confidence=0.3)
print("MBA Engine ready!")

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/mba', methods=['GET'])
def get_mba_rules():
    """Returns raw association rules."""
    rules = mba_engine.get_rules()
    return jsonify({
        'status': 'success',
        'total_rules_returned': len(rules),
        'rules': rules
    })

@app.route('/insights', methods=['GET'])
def get_insights():
    """Returns processed business insights."""
    insights = insights_generator.generate_insight_report()
    return jsonify({
        'status': 'success',
        'insights': insights
    })

@app.route('/query', methods=['POST'])
def query_model():
    """Accepts a user question and returns a business insight answer."""
    req = request.json
    if not req or 'message' not in req:
        return jsonify({'message': 'Please provide a valid query.'}), 400
        
    user_input = req.get('message', '')
    
    if not user_input.strip():
        return jsonify({'message': 'Query cannot be empty.'})

    answer = insights_generator.answer_query(user_input)
    
    return jsonify({'message': answer})

# Legacy fallback for the frontend that expects /chat
@app.route('/chat', methods=['POST'])
def chat():
    return query_model()

if __name__ == '__main__':
    print("Starting Flask web server on http://127.0.0.1:6969 ...")
    app.run(host='0.0.0.0', port=6969)
