import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Step 1: Import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from data import sentences_A,sentences_B,sentences_C

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Assuming you have sentences_A, sentences_B, and sentences_C as lists of sentences in categories A, B, and C respectively.

# Create labels for each category (0 for A, 1 for B, 2 for C).
labels_A = np.zeros(len(sentences_A))
labels_B = np.ones(len(sentences_B))
labels_C = np.ones(len(sentences_C)) * 2

# Concatenate sentences and labels from all categories.
sentences = np.concatenate([sentences_A, sentences_B, sentences_C])
labels = np.concatenate([labels_A, labels_B, labels_C])

# Data preprocessing
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)
    
    # Lowercasing
    words = [word.lower() for word in words]
    
    # Stopword Removal
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Reassemble the words to form a preprocessed sentence
    preprocessed_text = " ".join(words)
    
    return preprocessed_text

sentences = [preprocess_text(sentence) for sentence in sentences]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Feature Extraction - TF-IDF Vectorization with n-grams
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Adding word and bigram features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = SVC(random_state=42)  # Step 2: Replace RandomForestClassifier with SVC
model.fit(X_train_tfidf, y_train)
# Assuming you have already trained the model and have the tfidf_vectorizer and model objects ready.
import pickle
with open('model_qcat.pkl','wb') as file:
    pickle.dump(model,file)
with open('q_tfidf.pkl','wb') as file:
    pickle.dump(tfidf_vectorizer,file)
# New sentences to classify
# new_sentences = ["If I have scored 90 which college will I get for Computer or Computer Engineering or CE or CSE or Computer or Computer Science Engineering?","If my merit marks are 98.234 which college will I get for CSE or Computer Science or CE or Computer Engineering or Computer?","If my rank or ACPC rank is 1200 which college will I get for CE or CSE or COmpter Engineering or Computer Science or Computer Science Engineering?",
#     "What is or was the closure rank or rank of LDCE or LD Engineering college or LD college for CE or CSE or Computer Engineering or Computer Science or Computer Science Engineering or Computer?","Will I get LJ College Electrical or EE if my rank is 12000?","Will I get LD or LDCE or LD College ECE or Electronics and Communication or EC Engineering if my merit marks is or are 65?",
#     "Will I get LD or LDCE or LD College Mechanical or Mechanical Engineering or ME if my merit marks is or are 73.54627?","What is or was the merit closure for DDIT for Civil Engineering?","What are the merit marks required or necessary or needed for DDIT Mechanical or Mechanial Engineering?","Colleges or List of colleges for Computer Science or Computer Engineering or CE or CSE in AHmedabad","Colleges in Baroda for EE or ELectrical ENgineering or Electrical","Colleges for ME or Mechanical Engineering under rank 12000",
#     "Colleges for ME or Mechanical Engineering under 69 marks or merit marks","Colleges for ME or Mechanical Engineering under 88.3882 marks or merit marks",
#     "Colleges for Civil Engineering with or having clousre rank 15000","Colleegs for IT or Information technology Engineering in Ahmedabad under or cutoff rank 1200","Colleges for IT or Information technology Engineering in Ahmedabad under or cutoff merit marks or marks 55",
#     "what will be the cutoff or required or necessary or needed rank for CSE or CE or Computer Engineering LDCE or LD College or LD Engineering for SC category?","what will be the cutoff or required or necessary or needed rank for Mechanical Engineering or ME LJ or LJ College or LJ Engineering for ST category?","what will be the cutoff or necessary or required or needed rank for Electrical Engineering or EE DDIT for SEBC category?",
#     "what will be the cutoff or necessary or needed or required rank for EEE VGEC or Vishwakarma Enginerring College for EWS category?"]

# Preprocess the new sentences (Note: Ensure to use the same preprocessing steps as the training data)
# For simplicity, we'll only apply lowercasing here, but you should apply the complete preprocessing steps as used during training.

new_sentences=["If I have scored 90 which college will I get for Computer or Computer Engineering or CE or CSE or Computer or Computer Science Engineering?"]
def pre_quest(sentence):
    new_sentences=[sentence]
    preprocessed_sentences = [sentence.lower() for sentence in new_sentences]
    new_sentences_tfidf = tfidf_vectorizer.transform(preprocessed_sentences)
    predicted_labels = model.predict(new_sentences_tfidf)
    for label in predicted_labels:
        if label == 0:
            return 0
        elif label == 1:
            return 1
        else:
            return 2
        
print(pre_quest(new_sentences[0]))

# Print the predicted categories for each sentence
# for sentence, category in zip(new_sentences, predicted_categories):
#     print(f"Sentence: {sentence}\nPredicted Category: {category}\n")


'''
# Model Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
'''