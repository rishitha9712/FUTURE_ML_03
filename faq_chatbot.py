import json
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faq_data.json", "r") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

preprocessed_questions = [preprocess(q) for q in questions]

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_questions)

print("Chatbot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    user_input_clean = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_clean])

    # Find best match
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    score = similarity[0][index]

    if score > 0.3:
        print("Chatbot:", answers[index])
    else:
        print("Chatbot: I'm sorry, I don't have an answer for that.")
