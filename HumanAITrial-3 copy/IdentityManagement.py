import numpy as np
from joblib import dump, load
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import pairwise_distances

from preprocessing import lemmatisation, tokenisation,name_error
import random

name=""   #this is the current extracted name
pname="" #previous extracted name
set_response = ['[name] is such a pretty name',
                'I will now call you [name]',
                '[name] is such a cool name',
                'I will remember [name]!',
                'Got it! I’ll remember your name [name] from now on',
                '[name] is such a  lovely name',
                "I'll keep [name] in mind"]

change_response = ["Sure, I’ll now call you [name] now instead of [pname]",
                   "Alright, I’ll call you [name] instead of [pname]",
                   "No problem, I will update your name from [pname] to [name] right away"]

get_response = ["Let me remember, your name is",
                "I remember you, your name is",
                "You told me your name was",
                "Your name is",
                "You like to be called"]

data_path = 'Datasets/IdentityManagement.csv'

def extract_name(user_query):
    tokens = tokenisation(user_query)
    relevant_words = [word for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]

    if relevant_words:
        return relevant_words[-1].capitalize()  # Return the last word
    else:
        return None



# Create and save the identity management model

def create_identity_model():
    df = pd.read_csv(data_path)
    df['question'] = df['phrase'].apply(lemmatisation)
    question_type = df['type']

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4))
    transform = tfidf_vectorizer.fit_transform(df['question']).toarray()

    # Save the vectorizer and transformed data
    dump((tfidf_vectorizer, transform,question_type), 'models/identity_model.joblib')


# Load the saved identity model
def load_identity_model():
    identity_vectorizer, identity_transform,question_type = load('models/identity_model.joblib')
    return identity_vectorizer, identity_transform,question_type


#handle identity management
def identity_management(query):
    global name
    identity_vectorizer, identity_transform,question_type = load_identity_model()

    processed_query = lemmatisation(query)
    query_transform = identity_vectorizer.transform([processed_query]).toarray()

    cos_sim = 1 - pairwise_distances(query_transform, identity_transform, metric='cosine').flatten()

    max_sim = cos_sim.max()
    matching_indices = np.where(cos_sim == max_sim)[0]

    if len(query.strip().split()) == 1 and query.strip().isalpha():
        name = query.strip()
        return random.choice(set_response).replace("[name]", name)

    THRESHOLD = 0.5

    if max_sim >= THRESHOLD:
        # Get the intent type for the matched phrase

        matched_type = question_type.iloc[matching_indices[0]]  # Extract the first matching type

        # Choose appropriate response based on intent type
        if matched_type == 'set': #set the name
            name=extract_name(query)
            return random.choice(set_response).replace("[name]", name)
        elif matched_type == 'get': #retrieve the name
            if name=="":
                return "I do not know your name yet. What is your name?"
            return random.choice(get_response) + " " + name
        elif matched_type == 'change': #change the name
            pname=name
            name= extract_name(query)
            return random.choice(change_response).replace("[name]", name).replace("[pname]", pname)
    else:
        # Return an error message if no matching phrase meets the threshold
        return random.choice(name_error)


    #return "Response based on identity management model"
