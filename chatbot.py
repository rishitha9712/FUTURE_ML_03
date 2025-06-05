import json
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk

# Ensure punkt tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load FAQ data
try:
    with open("faq_data.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
except Exception as e:
    print("Error loading JSON:", e)
    faq_data = []

# Function to find best answer
def find_answer(user_input):
    user_tokens = word_tokenize(user_input.lower())
    best_match = None
    max_overlap = 0

    for faq in faq_data:
        faq_tokens = word_tokenize(faq["question"].lower())
        overlap = len(set(user_tokens) & set(faq_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = faq["answer"]

    return best_match if best_match else "Sorry, I don't understand your question."

# Home route (for testing)
@app.route("/", methods=["GET"])
def home():
    return "Chatbot is running. Use POST /chat with JSON payload { 'question': 'your question' }."

# Chat route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("question", "")  # <-- changed from "message" to "question"
    response = find_answer(user_input)
    return jsonify({"answer": response})  # <-- returning key "answer" to match frontend

if __name__ == "__main__":
    app.run(debug=True)
