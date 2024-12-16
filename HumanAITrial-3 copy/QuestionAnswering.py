import random

import numpy as np
from joblib import dump, load
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import pairwise_distances

from preprocessing import lemmatisation_q, error_message

data_path = 'Datasets/COMP3074-CW1-Dataset.csv'
THRESHOLD=0.4
df = pd.read_csv(data_path)

# creates and saves the question answering model

def create_question_model():
    df['question'] = df['Question'].apply(lemmatisation_q)
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4))
    transform = tfidf_vectorizer.fit_transform(df['question']).toarray()
    # Save the vectorizer and transformed data
    dump((tfidf_vectorizer, transform), 'models/question_model.joblib')


# Load the saved question answering model
def load_question_model():
    question_vectorizer, question_transform = load('models/question_model.joblib')

    return question_vectorizer, question_transform


# function to handle question answering
def QuestionAnwering(query):
    question_vectorizer, question_transform = load_question_model()

    processed_query = lemmatisation_q(query)
    query_transform = question_vectorizer.transform([processed_query]).toarray()

    cos_sim = 1 - pairwise_distances(query_transform, question_transform, metric='cosine').flatten()

    max_sim = cos_sim.max()
    matching_indices = np.where(cos_sim == max_sim)[0]

    if max_sim >= THRESHOLD:
        rand_index = random.choice(matching_indices)
        return df['Answer'].iloc[rand_index]
    else:
        return random.choice(error_message)
