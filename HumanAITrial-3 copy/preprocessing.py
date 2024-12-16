from datetime import datetime

import nltk
import pandas as pd
from nltk import WordNetLemmatizer, edit_distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from dateutil import parser
from joblib import dump
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('averaged_perceptron_tagger_eng',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('universal_tagset',quiet=True)
nltk.download('vader_lexicon',quiet=True)
lemmatiser = WordNetLemmatizer()


#generic fall back error messages
error_message = [
    "Sorry, I don't have an answer for that.",
    "I'm not sure how to respond to that",
    "Try rephrasing the question.",
    "Sorry I do not understand.",
    "Unfortunately, I cannot help with that question.",
    "I don't have an answer to that question."
]
#specific name handling error messages
name_error =[
    "Sorry i wasn't able to find your name",
    "Try typing my name is [name] to tell me your name",
    "Try rephrasing the question.",
    "Sorry, I can't help with that. Try typing 'help' to know what to do"
    "Sorry I do not understand.",
    "Unfortunately, I cannot help with that question.",
    "I don't have an answer to that question."

]

#sentiment analysis
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(user_query):
    """
    Analyze the sentiment of a user query.
    Returns: 'positive', 'negative', or 'neutral'.
    """
    sentiment_score = sia.polarity_scores(user_query)['compound']
    if sentiment_score > 0.05:
        return "positive"
    elif sentiment_score < -0.05:
        return "negative"
    else:
        return "neutral"


def sentiment_response(user_query):
    sentiment = analyze_sentiment(user_query)
    mood_message = ""
    if sentiment == "positive": #emoji added to positive
        mood_message = "😊"
    elif sentiment == "negative": #added to negative
        mood_message = "😟"
    return mood_message


def contains_date_time_or_number(user_input):
    #checks if the imputs contains a date time or number of guests pattern
    date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    time_pattern = r"\b(?:\d{1,2}[:.]\d{2}(?:\s*[apAP][mM])?|\d{1,2}\s*[apAP][mM])\b"
    number_pattern = r"\b(\d+)\s*(?:people|persons|guests|seats|tables?)\b"
    simple_number_pattern = r"\b\d+\b"

    if re.search(simple_number_pattern, user_input):
        return True

    # Check for date
    date_match = re.search(date_pattern, user_input)
    if date_match:
        try:
            # Confirm it's a valid date
            parser.parse(date_match.group(), dayfirst=True)
            return True
        except ValueError:
            pass

    # Check for time
    time_match = re.search(time_pattern, user_input)
    if time_match:
        try:
            # Confirm it's a valid time
            parser.parse(time_match.group(), fuzzy=True)
            return True
        except ValueError:
            pass

    # Check for number
    if re.search(number_pattern, user_input):
        return True

    # If none of the patterns matched
    return False



def tokenisation(text):
    token_text=word_tokenize(text)

    text_wt_sw=[word.lower() for word in token_text if not word in stopwords.words('english')]
    if len(text_wt_sw)>=1: #this was done because some of the questions where all stopwords. this meant that they could not be matched
        return text_wt_sw
    else:
        return token_text

def lemmatisation(text): #this function is used to lemmatise the text using the tokenize function
    tokens=[]
    postmap={
        'ADJ': 'a',
        'NOUN':'n',
        'VERB':'v',
        'ADV': 'r'
    }

    post=nltk.pos_tag(tokenisation(text), tagset='universal') #adds the tags to the tokenised words using the tokenisation function
    for token in post:
        word=token[0]
        tag=token[1]
        if tag in postmap.keys():
            tokens.append(lemmatiser.lemmatize(word, postmap[tag])) #appends the lemmatised word to the final list of words
        else:
           tokens.append(lemmatiser.lemmatize(word))
    return " ".join(tokens) #this returns the final list

def tokenisation_q(text): #specific to question answerint
    token_text=word_tokenize(text)

    text_wt_sw=[word.lower() for word in token_text if not word in stopwords.words('english')]
    if len(text_wt_sw)>1: # only words when there is less than not less than or equals to
        return text_wt_sw
    else:
        return token_text

def lemmatisation_q(text): #this function is used to lemmatise the text using the tokenize function
    tokens=[]
    postmap={
        'ADJ': 'a',
        'NOUN':'n',
        'VERB':'v',
        'ADV': 'r'
    }

    post=nltk.pos_tag(tokenisation_q(text), tagset='universal') #adds the tags to the tokenised words using the tokenisation function
    for token in post:
        word=token[0]
        tag=token[1]
        if tag in postmap.keys():
            tokens.append(lemmatiser.lemmatize(word, postmap[tag])) #appends the lemmatised word to the final list of words
        else:
           tokens.append(lemmatiser.lemmatize(word))
    return " ".join(tokens) #this returns the final list




