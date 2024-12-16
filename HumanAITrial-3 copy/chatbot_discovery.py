import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
from preprocessing import lemmatisation
from sklearn.metrics import pairwise_distances

data_path = 'Datasets/discovery.csv'
error_message = [
    "Type either 'current' or 'general' to know more about my functionalities",
    "type current or general please",
    "To know more about the chatot type either 'genral' or 'current'",
    "I am looking for either 'general' or 'current'"
]
previous=None
discovery=["I can tell you all about the chatbot functionalities such as the small talk, identity management, "
           "question answerting and restaurant booking\n"
           "\tWhen you ask me what I can do you get the option to know more about my functionalities generally or "
           "more about what you are currently doing"]
restaurant_booking=["I can handle a table booking transaction for a restaurant. I have the following functionalities:\n"
                    "\tbook a table: To book a table try typing in 'book a table'\n"
                    "\tmodify a booking: I can modify the details of an existing booking. Try typing in 'modify "
                    "booking'.\n"
                    "\tCancel booking: I can cancel a booking you have made. Try typing in 'cancel booking'. "]
name_management=["In name management I can do the following things:\n"
                 "\tRemember your name: I can remember your name once you have provided it. Try typing in 'my name "
                 "is..'\n"
                 "\tchange your name: If you would like me to call you something else I can change your name. Try "
                 "typing 'change my name to ..\n"
                 "\trecall name: If you ask me your name I can tell it you. Try typing 'What is my name'. "]
question_answering=["If you ask me a question I will best try to answer it. Try typing 'how does a dredge work'."]
small_talk=["I can have an informal conversation with you. Try typing 'how are you'."]
general=[
        (
            "I can help you with the following things:\n"
            "\tIdentity management: I can remember your name if you tell me. Try typing in 'my name is ...'\n"
            "\tQuestion Answering: Ask me any question and I will answer it. Try typing 'what are stocks and bonds'\n"
            "\tRestaurant booking: I can book a table at a restaurant. Try typing 'book a table'\n"
            "\tDiscoverability: I can tell you all about a specific task in more detail if you type 'what can you do' "
            "\twhile doing one of the tasks. I can tell you more details."
        )]

THRESHOLD = 0.6
section= None #the area in which the response should be
asked= False #has the user been asked general or current
intent_history = [None, None] #keeps track of current intent and previous one
df = pd.read_csv(data_path)
functionality=["small talk: 'How are you'","identity management: 'my name is...'\n","question Answering: 'what are stocks and bonds'","Restaurant booking:'book a table'"]
def create_discovery_model(): #create the discovery model
    df = pd.read_csv(data_path)
    df['question'] = df['phrase'].apply(lemmatisation)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4))  #uses 4 n grams for context
    transform = tfidf_vectorizer.fit_transform(df['question']).toarray()

    # Save the vectorizer and transformed data
    dump((tfidf_vectorizer, transform), 'models/discovery_model.joblib')

def update_intent_history(new_intent):    #keeps track of the intents for context tracking
    global intent_history
    intent_history.pop(0)  # Remove the oldest intent
    intent_history.append(new_intent)  # Add the new intent

# Load the saved small talk model
def load_discovery_model():
    discovery_vectorizer, discovery_transform = load('models/discovery_model.joblib')
    return discovery_vectorizer, discovery_transform

def know(user_query, previous_intent):
    global asked, section
    if not asked:
        if previous_intent =="unknown" or previous_intent is None:
            section ='general'
            asked=True
            return None
        asked = True
        return (f"If you would like to know more about the chatbot in general, type 'general'.\n"
                f"\t Otherwise, if you would like to know about what you are currently doing ({previous_intent}), type 'current'.")
    else:
        if user_query.lower() == 'general':
            section = 'general'
            asked = False
        elif user_query.lower() == 'current':
            section = 'current'
            asked = False
        else:
            return "Please type 'general' or 'current' to continue."




# Function to handle small talk response
def chatbot_discovery(query,previous_intent):
    global previous, section
    previous = previous_intent
    update_intent_history(previous_intent)
    two_behind_intent = intent_history[0]


    if section is None and previous_intent is not None:
        response = know(query, previous_intent)
        if response:  # Return the response if context setup
            return response


    discovery_vectorizer, discovery_transform = load_discovery_model()
    processed_query = lemmatisation(query)
    query_transform = discovery_vectorizer.transform([processed_query]).toarray()

    cos_sim = 1 - pairwise_distances(query_transform, discovery_transform, metric='cosine').flatten()

    max_sim = cos_sim.max()
    matching_indices = np.where(cos_sim == max_sim)[0]

    if max_sim >= THRESHOLD:

        if previous_intent is None or section=='general' or previous_intent =="unknown":
            response="I have the following functionality:\n"
            for i in functionality:
                response= response + " "+i
            section = None

            return response



        if section=='current': #get the context specific functionality
            intent_map = {
                "discovery": discovery,
                "restaurant_booking": restaurant_booking,
                "name_management": name_management,
                "question_answering": question_answering,
                "small_talk": small_talk
            }
            if two_behind_intent in intent_map:
                section = None
                return random.choice(intent_map[two_behind_intent])
    else:
        return random.choice(error_message)
