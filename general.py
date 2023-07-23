import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import pickle




# Preprocessing
with open('chatbot_model.pkl', 'rb') as file:
    model = pickle.load(file)

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

with open('chatbot_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('chatbot_intents.pkl', 'rb') as file:
    intents = pickle.load(file)

# Chat function
def chat(input_text):
    input_text = preprocess_text(input_text)
    input_vector = vectorizer.transform([input_text])
    predicted_intent = model.predict(input_vector)[0]
    
    # Find the appropriate response
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            return random.choice(intent['responses'])



