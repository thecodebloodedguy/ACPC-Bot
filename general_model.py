import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import pickle


# Load intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Creating training data
training_data = []
output = []
for intent in intents['intents']:
    for text in intent['patterns']:
        training_data.append(preprocess_text(text))
        output.append(intent['tag'])

# Vectorize the data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_data)

# Train the model
model = MultinomialNB()
model.fit(X, output)
with open('chatbot_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('chatbot_intents.pkl', 'wb') as file:
    pickle.dump(intents, file)
with open('chatbot_model.pkl', 'wb') as file:
    pickle.dump(model, file)