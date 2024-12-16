from datetime import datetime

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from IdentityManagement import identity_management,create_identity_model
from QuestionAnswering import QuestionAnwering,create_question_model
from SmallTalk import talk_response,create_smalltalk_model
from chatbot_discovery import chatbot_discovery,create_discovery_model
from restaurantBooking import restaurant_response, BookingManager,create_restaurant_model
from preprocessing import contains_date_time_or_number, sentiment_response


booking_manager = BookingManager()
chatbot_name = "Chatbot" #name of the chatbot by default

#the code used to create my identity model

def create_intent_model():
    # Data loading and preprocessing steps
    files = [
        ('Datasets/smallTalk.csv', 'small_talk', 'phrase'),
        ('Datasets/COMP3074-CW1-Dataset.csv', 'question_answering', 'Question'),
        ('Datasets/IdentityManagement.csv', 'name_management', 'phrase'),
        ('Datasets/discovery.csv', 'discovery', 'phrase'),
        ('Datasets/RestaurantBooking.csv', 'restaurant_booking', 'phase')
    ]

    intent_data = []
    for file, intent_label, phrase_column in files:
        df = pd.read_csv(file)
        df['intent'] = intent_label
        df = df.rename(columns={phrase_column: 'phrase'})
        df = df.dropna(subset=['phrase'])
        intent_data.append(df[['phrase', 'intent']])

    intent_df = pd.concat(intent_data, ignore_index=True)

    # Train a vectorizer
    vectorizer = TfidfVectorizer()
    intent_vectors = vectorizer.fit_transform(intent_df['phrase']).toarray()
    intent_labels = intent_df['intent'].values

    # Save the model and vectorizer
    dump((vectorizer, intent_vectors, intent_labels), 'models/intent_model.joblib')


def load_intent_model():
    vectorizer, intent_vectors, intent_labels = load('models/intent_model.joblib')
    return vectorizer, intent_vectors, intent_labels

def get_time_greeting():
    current_time = datetime.now().hour
    if 1 <= current_time < 12:    #5 till 11 = morning
        return "Good morning"
    elif 12 <= current_time < 18:  #12 till 15 = afternoon
        return "Good afternoon"
    else:
        return "Good evening"

def detect_intent(user_input, vectorizer, intent_vectors, intent_labels, previous_intent, booking_manager):

    #Detect the intent of user input, considering context and booking state.

    # Check if  in the middle of a modification flow
    if booking_manager.data.get("modifying_booking"):
        return "restaurant_booking"

    # Check if  awaiting booking selection
    if booking_manager.data.get("awaiting_booking_selection"):
        return "restaurant_booking"

    # Check for booking-specific commands first
    cancel_phrases = ["cancel", "delete", "remove"]
    change_phrases = ["change", "modify", "update"]

    # Handle yes/no responses in context
    if user_input.lower() in ["yes", "no"]:
        if not booking_manager.is_active() and not booking_manager.data["confirming_cancellation"]:
            if previous_intent == "restaurant_booking":
                return "restaurant_booking"
            if not booking_manager.is_active() and not booking_manager.data["confirming_cancellation"]:
                return "restaurant_booking"
        return previous_intent if previous_intent else "unknown"

    # Handle single-word responses during name management
    if previous_intent == 'name_management' and len(user_input.strip().split()) == 1:
        return 'name_management'


    # Handle numeric responses during booking operations
    if previous_intent == 'restaurant_booking' and (
            user_input.isdigit() or
            booking_manager.data.get("awaiting_booking_selection") or
            booking_manager.data.get("modifying_booking")
    ):
        return 'restaurant_booking'

    dietary = ['vegan', 'vegetarian', 'halal', 'kosher', 'pescatarian', 'none']
    if previous_intent == 'restaurant_booking' and user_input.lower() in dietary:
        booking_manager.data["dietary_option"] = user_input.lower()
        return 'restaurant_booking'

    # Handle active booking context
    if booking_manager.is_active() and contains_date_time_or_number(user_input):
        return 'restaurant_booking'

    # Regular intent detection
    user_vector = vectorizer.transform([user_input]).toarray()
    similarities = cosine_similarity(user_vector, intent_vectors).flatten()
    max_sim = similarities.max()
    max_index = np.argmax(similarities)

    THRESHOLD = 0.6
    return intent_labels[max_index] if max_sim >= THRESHOLD else "unknown"


def handle_booking_state(booking_manager, current_intent, user_query, previous_intent):
  #handles the booking state

    if not booking_manager.is_active() or current_intent == 'restaurant_booking':
        return False, current_intent, None

    if booking_manager.data["date"] is not None:
        missing_info = booking_manager.get_missing_info()
        print(f"{chatbot_name}: You haven't finished your booking yet - I still need {missing_info}. "
              "Would you like to switch tasks? (yes/no)")

        confirmation = input("You: ").lower()
        if confirmation == "yes":
            booking_manager.store_current_as_pending()
            # Process new query with appropriate intent
            if previous_intent == 'question_answering':
                return False, 'question_answering', QuestionAnwering(user_query)
            elif previous_intent == 'small_talk':
                return False, 'small_talk', talk_response(user_query) + sentiment_response(user_query)
            elif previous_intent == 'discovery':
                return False, 'discovery', chatbot_discovery(user_query, None
                                                             )
            else:
                return False, current_intent, None
        elif confirmation == "no":
            return False, 'restaurant_booking', None
        else:
            print(f"{chatbot_name}: Please answer 'yes' or 'no'")
            return True, current_intent, None

    booking_manager.reset()  # Resets the booking state
    if current_intent == 'question_answering':
        return False, 'question_answering', QuestionAnwering(user_query)
    elif current_intent == 'small_talk':
        return False, 'small_talk', talk_response(user_query) + sentiment_response(user_query)
    elif current_intent == 'discovery':
        return False, 'discovery', chatbot_discovery(user_query, None)
    else:
        return False, current_intent, None


def handle_response(intent, user_query, booking_manager, previous_intent):
    #Generate appropriate response based on intent.
    if intent == 'restaurant_booking':
        if not booking_manager.is_active() and booking_manager.has_pending_booking():
            # Get the details of the pending booking
            pending_details = booking_manager.get_pending_booking_details()
            return (f"You have an unfinished booking:\n{pending_details}\n"
                    "Would you like to continue it? (yes/no)")
        response = restaurant_response(user_query, booking_manager,chatbot_name)
        if response:
            return response + sentiment_response(user_query)
        return ""

    responses = {
        'small_talk': lambda: talk_response(user_query) + sentiment_response(user_query),
        'question_answering': lambda: QuestionAnwering(user_query),
        'name_management': lambda: identity_management(user_query),
        'discovery': lambda: chatbot_discovery(user_query, previous_intent)
    }

    return responses.get(intent, lambda: "I'm not sure how to respond to that. Type 'Help' to know more")()

def main():
    global chatbot_name
    # Initialize models and booking manager
    create_intent_model()
    create_identity_model()
    create_question_model()
    create_smalltalk_model()
    create_discovery_model()
    create_restaurant_model()

    vectorizer, intent_vectors, intent_labels = load_intent_model()
    booking_manager = BookingManager()
    previous_intent = None
    greeting = get_time_greeting()
    print(f"Chatbot: {greeting}! I'm a restaurant booking chatbot.Feel free to ask me anything."
          f"more about me or type 'exit'"
          "when you're ready to leave.\n")
    print(f"Chatbot: Chatbot sounds so boring 😭, Change my name to something more interesting!")
    chatbot_name = input("Type in my new name: ").strip()
    print(f"{chatbot_name}: If you are stuck what to do type 'help' to know more!!")

    while True:
        user_query = input("You: ").strip().capitalize()

        # Handle exit command
        if user_query.lower() in ["exit", "quit", "bye"]:
            print(f"{chatbot_name}: Goodbye!")
            break

        # Detect intent
        current_intent = detect_intent(
            user_query,
            vectorizer,
            intent_vectors,
            intent_labels,
            previous_intent,
            booking_manager
        )
        if not booking_manager.is_active() and booking_manager.has_pending_booking():
            if user_query.lower() in ["yes", "yeah", "yep"]:
                next_step = booking_manager.resume_pending_booking()
                print(f"{chatbot_name}: ", next_step)
                continue
            elif user_query.lower() in ["no", "nope"]:
                booking_manager.reset()
                print(f"{chatbot_name}: Starting a new booking. On what date (dd/mm/yyyy) would you like to book the table?")
                continue

        # Handle booking state
        if booking_manager.is_active() and current_intent != 'restaurant_booking':
            should_continue, new_intent, direct_response = handle_booking_state(
                booking_manager,
                current_intent,
                user_query,
                previous_intent
            )
            if should_continue:
                continue
            if direct_response:
                print(f"{chatbot_name}:", direct_response)
                previous_intent = new_intent
                continue
            current_intent = new_intent

        # Generate and print response
        response = handle_response(
            current_intent,
            user_query,
            booking_manager,
            previous_intent
        )
        if response:
            print(f"{chatbot_name}:", response)

        previous_intent = current_intent


if __name__ == '__main__':
    main()
