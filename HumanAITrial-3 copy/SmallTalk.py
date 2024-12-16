import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
from preprocessing import lemmatisation
from sklearn.metrics import pairwise_distances

data_path = 'Datasets/smallTalk.csv'
error_message = [
    "Sorry, I don't have an answer for that.",
    "sorry I can't understand. Type 'help' to know what I can do."
    "I'm not sure how to respond to that",
    "Try rephrasing the question.",
    "Sorry I do not understand.",
    "Unfortunately, I cannot help with that question.",
    "I don't have an answer to that question."
]

THRESHOLD = 0.5
df = pd.read_csv(data_path)

# Create and save the small talk model

def create_smalltalk_model():
    df = pd.read_csv(data_path)
    df['question'] = df['phrase'].apply(lemmatisation)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    transform = tfidf_vectorizer.fit_transform(df['question']).toarray()

    # Save the vectorizer and transformed data
    dump((tfidf_vectorizer, transform), 'models/smalltalk_model.joblib')

# Load the saved small talk model
def load_smalltalk_model():
    smalltalk_vectorizer, smalltalk_transform = load('models/smalltalk_model.joblib')
    return smalltalk_vectorizer, smalltalk_transform

# Function to handle small talk response
def talk_response(query):
    smalltalk_vectorizer, smalltalk_transform = load_smalltalk_model()

    processed_query = lemmatisation(query)
    query_transform = smalltalk_vectorizer.transform([processed_query]).toarray()

    cos_sim = 1 - pairwise_distances(query_transform, smalltalk_transform, metric='cosine').flatten()

    max_sim = cos_sim.max()
    matching_indices = np.where(cos_sim == max_sim)[0]

    if max_sim >= THRESHOLD:
        rand_index = random.choice(matching_indices)
        return df['Response'].iloc[rand_index]
    else:
        return random.choice(error_message)
