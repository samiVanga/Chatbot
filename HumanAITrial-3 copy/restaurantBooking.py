import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from datetime import datetime, timedelta
from joblib import dump, load
from sklearn.metrics import pairwise_distances
from preprocessing import lemmatisation
import re
from dateutil import parser
import random
from IdentityManagement import identity_management, extract_name

chatbot_name="Chatbot" #generic name of the chatbot
class BookingManager: #handles the booking details and states
    def __init__(self):
        self.data = {
            "name": None,
            "date": None,
            "time": None,
            "people": None,
            "dietary": None,
            "booking_active": False,
            "confirming_cancellation": False,
            "confirmed": False,
            "booking_id": None,
            "awaiting_booking_selection": False,
            "modifying_booking": False,
            "awaiting_modification_field": False,
            "awaiting_new_value": False,
            "modification_field": None,
            "last_operation": None
        }
        self.pending_booking = None #if a booking has been interrputed
        self.restaurant_hours = {
            'open': '11:00',
            'close': '23:00'
        }
        self.max_party_size = 20
        self.init_database()

    def init_database(self):  #creates the database
        conn = sqlite3.connect('restaurant_bookings.db')
        cursor = conn.cursor()

        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bookings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        customer_name TEXT NOT NULL,
                        booking_date DATE NOT NULL,
                        booking_time TIME NOT NULL,
                        number_of_people INTEGER NOT NULL,
                        dietary TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        active BOOLEAN DEFAULT 1
                    )
                ''')

        conn.commit()
        conn.close()

    def is_active(self):   #is the booking active?
        return self.data["booking_active"]

    def has_pending_booking(self): #does the booking have a pending booking
        return self.pending_booking is not None


    def resume_pending_booking(self):  #used to resume a pending booking
        if self.has_pending_booking():
            self.data = self.pending_booking.copy()
            self.pending_booking = None
            missing_info = self.get_missing_info()
            next_missing= self.get_next_missing_info()
            if missing_info:
                return (f"Resuming your previous booking. I still need:\n"
                        f" {missing_info}First, can you provide {next_missing}?")
            return "Resuming your previous booking. All details are complete."

    def get_pending_booking_details(self):  #get the details of the pending booking to displau
        """Generate a summary of the pending booking."""
        if self.has_pending_booking():
            booking = self.pending_booking
            return (f"Name: {booking['name']}, "
                    f"Date: {booking['date']}, "
                    f"Time: {booking['time']}, "
                    f"Number of People: {booking['people']}, "
                    f"Dietary requirements: {booking['dietary']}")
        return "No pending booking found."

    def has_confirmed_booking(self):  #has all the booking details been eneterd
        return self.data["confirmed"]

    def get_customer_bookings(self, customer_name):  # get all the booking of the users name
        conn = sqlite3.connect('restaurant_bookings.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, booking_date, booking_time, number_of_people, dietary
            FROM bookings 
            WHERE customer_name = ? AND active = 1
            ORDER BY booking_date, booking_time
        ''', (customer_name,))

        bookings = cursor.fetchall()
        conn.close()
        return bookings

    def get_dietary_preference(self, customer_name):  #gets the dietary prefernces of the user
       #if the customer has two or more booking with the same dietary then is a preference
        conn = sqlite3.connect('restaurant_bookings.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT dietary, COUNT(*) as count
            FROM bookings
            WHERE customer_name = ? AND active = 1
            GROUP BY dietary
            HAVING count >= 2
        ''', (customer_name,))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def validate_booking_selection(self, query):
        try:
            selection = int(query)
            bookings = self.get_customer_bookings(self.data["name"])
            return 1 <= selection <= len(bookings), selection
        except ValueError:
            return False, None

    def handle_cancellation(self, query, bookings):   #cancels a booking
        if len(bookings) == 1:
            self.data["booking_id"] = bookings[0][0]
            self.data["confirming_cancellation"] = True
            return self.format_confirmation_message(bookings[0])

        self.data["awaiting_booking_selection"] = True  #when it is awaiting for the number of booking to cancel
        return self.format_bookings_list(
            bookings) + "\nPlease enter the number of the booking you would like to cancel."

    def format_confirmation_message(self, booking):
        return f"Are you sure you want to cancel this booking?\nDate: {booking[1]}, Time: {booking[2]}, People: {booking[3]}, Dietary requirement: {booking[4]} (yes/no)"

    def format_modification_message(self, booking):
        return (f"Current booking details:\n"
                f"Date: {booking[1]}\n"
                f"Time: {booking[2]}\n"
                f"People: {booking[3]}\n"
                f"Dietary requirement: {booking[4]}\n"
                "What would you like to modify? (date/time/people/dietary)")

    def save_booking_to_db(self):  #when booking complete save to db
        if self.data["name"] and self.data["date"] and self.data["time"] and self.data["people"] and self.data[
            "dietary"]:
            conn = sqlite3.connect('restaurant_bookings.db')
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO bookings (customer_name, booking_date, booking_time, number_of_people,dietary)
                VALUES (?, ?, ?, ?,?)
            ''', (
                self.data["name"],
                self.data["date"].strftime('%Y-%m-%d'),
                self.data["time"].strftime('%H:%M'),
                self.data["people"],
                self.data["dietary"]
            ))

            booking_id = cursor.lastrowid
            conn.commit()
            conn.close()
            self.data["booking_id"] = booking_id
            self.data["booking_active"] = False
            return True
        return False

    def update_booking(self, booking_id, field, value):
       #when modified the booking updates in the SQL
        conn = sqlite3.connect('restaurant_bookings.db')
        cursor = conn.cursor()

        # Map the field names to database column names
        field_mapping = {
            'date': 'booking_date',
            'time': 'booking_time',
            'people': 'number_of_people',
            'dietary': 'dietary'
        }

        # Get the correct database column name
        db_field = field_mapping.get(field)
        if not db_field:
            conn.close()
            return False

        # Format the value based on field type
        if field == "date":
            value = value.strftime('%Y-%m-%d')
        elif field == "time":
            value = value.strftime('%H:%M')

        try:
            cursor.execute(f'''
                UPDATE bookings 
                SET {db_field} = ?
                WHERE id = ?
            ''', (value, booking_id))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error: {e}")
            conn.close()
            return False

    def confirm_booking(self):
        if self.data["name"] and self.data["date"] and self.data["time"] and self.data["people"] and self.data[
            "dietary"]:
            self.data["confirmed"] = True
            self.data["booking_active"] = True
            return self.save_booking_to_db()
        return False

    def get_missing_info(self):
        missing_string = ""
        if not self.data["name"]:
            missing_string += "\tYour name for the booking\n "
        if not self.data["date"]:
            missing_string += "\tThe date of the booking\n"
        if not self.data["time"]:
            missing_string += f"\tThe time for booking on {self.data['date']}\n"
        if not self.data["people"]:
            missing_string += f"\tThe number of people for your booking\n"
        if not self.data["dietary"]:
            missing_string += f"\tThe dietary requirements of your booking\n"
        return missing_string

    def get_next_missing_info(self):
        if not self.data["name"]:
            return "Your name for the booking"
        if not self.data["date"]:
            return "The date of the booking"
        if not self.data["time"]:
            return f"The time for booking on {self.data['date']}"
        if not self.data["people"]:
            return f"The number of people for your booking"
        if not self.data["dietary"]:
            return f"The dietary requirements of your booking"

    def store_current_as_pending(self):
        if self.data["booking_active"] and self.data["date"] is not None:
            self.pending_booking = self.data.copy()
        self.reset()

    def reset(self):
        self.data = {
            "name": None,
            "date": None,
            "time": None,
            "people": None,
            "dietary": None,
            "booking_active": False,
            "confirming_cancellation": False,
            "confirmed": False,
            "booking_id": None,
            "awaiting_booking_selection": False,
            "modifying_booking": False,
            "awaiting_modification_field": False,
            "awaiting_new_value": False,
            "modification_field": None,
            "last_operation": None
        }

    def set_name(self, customer_name):
        self.data["name"] = customer_name
        self.data["booking_active"] = True

    def get_booking_details(self, booking_id):
        conn = sqlite3.connect('restaurant_bookings.db')
        cursor = conn.cursor()

        cursor.execute('''
            SELECT customer_name, booking_date, booking_time, number_of_people, dietary
            FROM bookings 
            WHERE id = ? AND active = 1
        ''', (booking_id,))

        booking = cursor.fetchone()
        conn.close()
        return booking

    def delete_booking(self, booking_id):
        #deletes the booking in the SQL database
        try:
            conn = sqlite3.connect('restaurant_bookings.db')
            cursor = conn.cursor()

            # Use parameterized query to prevent SQL injection
            cursor.execute('''
                UPDATE bookings 
                SET active = 0 
                WHERE id = ? AND active = 1
            ''', (booking_id,))

            # Check if a row was actually updated
            rows_affected = cursor.rowcount

            conn.commit()
            conn.close()

            if rows_affected > 0:
                self.reset()  # Reset booking manager state after successful deletion
                return True
            return False

        except Exception as e:
            print(f"Database error: {e}")
            if conn:
                conn.close()
            return False

    def format_bookings_list(self, bookings): #used to format the list to show all the bookings
        if not bookings:
            return "No active bookings found. To make a booking try typing 'make a booking'."

        formatted = "Your bookings:\n"
        for i, (id, date, time, people, dietary) in enumerate(bookings, 1):
            formatted += f"{i}. Date: {date}, Time: {time}, People: {people}, Dietary requirements: {dietary}\n"
        return formatted

    def parse_input(self, user_input):  #input validation with the error handling
        date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
        time_pattern = r"\b(?:\d{1,2}[:.]\d{2}(?:\s*[apAP][mM])?|\d{1,2}[ \t]?[apAP][mM])\b"
        people_pattern = r"(?i)\b(\d+)\s*(?:people|persons|guests|seats|tables?)\b"

        # Extract date
        date_match = re.search(date_pattern, user_input)
        if date_match:
            raw_date = date_match.group()
            try:
                # Split the date into components
                day, month, year = map(int, re.split(r'[/-]', raw_date))
                if not (1 <= month <= 12):
                    print(f"{chatbot_name}: Invalid month. Please provide a month between 1 and 12.")
                    return False
                if not (1 <= day <= 31):
                    print(f"{chatbot_name}: Invalid day. Please provide a day between 1 and 31.")
                    return False


                # Parse the validated date
                parsed_date = datetime(year, month, day).date()
                current_date = datetime.now().date()
                max_future_date = current_date + timedelta(days=90)

                if parsed_date < current_date:
                    print(f"{chatbot_name}: Sorry, bookings must be for a future date after {current_date}. Please try again.")
                    return False
                if parsed_date > max_future_date:
                    print(f"{chatbot_name}: Sorry, bookings can only be made up to 3 months in advance. Please try again.")
                    return False
                self.data["date"] = parsed_date
                self.data["booking_active"] = True
                return True
            except ValueError:
                pass

        # Extract time
        time_match = re.search(time_pattern, user_input)
        if time_match:
            try:
                parsed_time = parser.parse(time_match.group(), fuzzy=True).time()
                restaurant_open = parser.parse(self.restaurant_hours['open']).time()
                restaurant_close = parser.parse(self.restaurant_hours['close']).time()

                if parsed_time < restaurant_open or parsed_time > restaurant_close:
                    print(
                        f"{chatbot_name}: Sorry, our restaurant is only open between {self.restaurant_hours['open']} and {self.restaurant_hours['close']}. Provide a time within our open hours")
                    return False

                self.data["time"] = parsed_time
                self.data["booking_active"] = True
                return True
            except ValueError:
                pass

        # Extract people
        people_match = re.search(people_pattern, user_input)
        if people_match:
            try:
                num_people = int(people_match.group(1))
                if num_people <= 0:
                    print(
                        f"{chatbot_name}: Please provide a valid number of people. The number of people must be positive, try again")
                    return False
                if num_people > self.max_party_size:
                    print(
                        f"{chatbot_name}: Sorry, we can only accommodate parties up to {self.max_party_size} people. For larger groups, please contact us directly.")
                    return False

                self.data["people"] = num_people
                self.data["booking_active"] = True
                return True
            except ValueError:
                pass

        if user_input.isdigit():
            print(f"{chatbot_name}: please provide the number of people in the correct format. eg 5 people")
            return False

        dietary_r = ['halal', 'vegan', 'vegetarian', 'kosher', 'pescatarian', 'none']
        if user_input.lower() in dietary_r:
            self.data["dietary"] = user_input.lower()
            self.data["booking_active"] = True
            return True

        return None


#code used to create model

def create_restaurant_model():
    data_path = 'Datasets/RestaurantBooking.csv'
    df = pd.read_csv(data_path)
    df['question'] = df['phrase'].apply(lemmatisation)
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4))
    transform = tfidf_vectorizer.fit_transform(df['question']).toarray()
    dump((tfidf_vectorizer, transform), 'models/restaurant_model.joblib')


def load_restaurant_model():
    restaurant_vectorizer, restaurant_transform = load('models/restaurant_model.joblib')
    return restaurant_vectorizer, restaurant_transform


def detect_intent(user_input): #rule basde to detect cancel and modify
    cancel_phrases = ["cancel", "delete", "remove"]
    change_phrases = ["change", "modify", "update"]

    for phrase in cancel_phrases:
        if phrase in user_input.lower():
            return "cancel"
    for phrase in change_phrases:
        if phrase in user_input.lower():
            return "change"
    return "book"


def restaurant_response(query, booking_manager,chat_name):
    from IdentityManagement import name
    global chatbot_name
    chatbot_name = chat_name
    if not booking_manager.data["name"]:
        booking_manager.data["name"] = name


    if booking_manager.data.get("modifying_booking") and booking_manager.data.get("awaiting_modification_field"):
        if query.lower() in ['date', 'time', 'people', 'dietary']:
            booking_manager.data["modification_field"] = query.lower()
            booking_manager.data["awaiting_modification_field"] = False
            booking_manager.data["awaiting_new_value"] = True
            return f"Please enter the new {query.lower()} for your booking:"
        return "Please type either 'date', 'time', 'people' or 'dietary' to modify the specific detail of the booking."

        # Handle new value input for modification
    if booking_manager.data.get("modifying_booking") and booking_manager.data.get("awaiting_new_value"):
        field = booking_manager.data["modification_field"]

        # Parse the new value
        if booking_manager.parse_input(query):
            new_value = None
            if field == 'date' and booking_manager.data["date"]:
                new_value = booking_manager.data["date"]
            elif field == 'time' and booking_manager.data["time"]:
                new_value = booking_manager.data["time"]
            elif field == 'people' and booking_manager.data["people"]:
                new_value = booking_manager.data["people"]
            elif field == 'dietary' and booking_manager.data["dietary"]:
                new_value = booking_manager.data["dietary"]

            if new_value:
                booking_manager.update_booking(booking_manager.data["booking_id"], field, new_value)
                updated_booking = booking_manager.get_booking_details(booking_manager.data["booking_id"])
                booking_manager.reset()
                return (f"Booking updated successfully! New details:\n"
                        f"Date: {updated_booking[1]}\n"
                        f"Time: {updated_booking[2]}\n"
                        f"People: {updated_booking[3]}\n"
                        f"dietary: {updated_booking[4]}")

        return f"Please provide a valid {field} value."

        # Handle modification requests
    if detect_intent(query) == "change":
        if not booking_manager.data["name"]:  # First time asking for name
            print(f"{chatbot_name}: First, please tell me your name so I can find your bookings.")
            input_name = input("You: ").strip()
            input_name_c = extract_name(input_name)
            name = input_name_c
            booking_manager.set_name(input_name_c)
            identity_management(input_name_c)

        bookings = booking_manager.get_customer_bookings(name)
        if not bookings:
            return "You don't have any active bookings to modify. Try typing 'make a booking' to create a booking."

        if len(bookings) == 1:
            booking_id = bookings[0][0]
            booking_details = booking_manager.get_booking_details(booking_id)
            booking_manager.data["booking_id"] = booking_id
            booking_manager.data["modifying_booking"] = True
            booking_manager.data["awaiting_modification_field"] = True
            return booking_manager.format_modification_message(booking_details)

        booking_manager.data["awaiting_booking_selection"] = True
        booking_manager.data["modifying_booking"] = True
        return booking_manager.format_bookings_list(
            bookings) + "\nPlease enter the number of the booking you would like to modify."

        # Handle numeric selection for modification
    if booking_manager.data.get("modifying_booking") and booking_manager.data.get("awaiting_booking_selection"):
        try:
            selection = int(query)
            bookings = booking_manager.get_customer_bookings(name)
            if 1 <= selection <= len(bookings):
                booking_id = bookings[selection - 1][0]
                booking_details = booking_manager.get_booking_details(booking_id)
                booking_manager.data["booking_id"] = booking_id
                booking_manager.data["awaiting_booking_selection"] = False
                booking_manager.data["awaiting_modification_field"] = True
                return booking_manager.format_modification_message(booking_details)
            return f"Invalid selection. Please choose a valid booking number between 1 and {len(bookings)}"
        except ValueError:
            return "Please enter a valid number."

    # Handle  no responses
    if query.lower() in ['no', 'nope']:
        if booking_manager.data["confirming_cancellation"]:
            booking_manager.reset()
            return "Booking cancellation abandoned. Is there anything else I can help you with?"
        if booking_manager.is_active():
            booking_manager.reset()
            return "Booking process cancelled. Is there anything else I can help you with?"
        return "Okay, let me know if you need anything else!"

    if query.lower() in ['yes', 'yeah', 'yep']:
        # If we're not in any active but have existing bookings start new booking
        if not booking_manager.data["confirming_cancellation"]:
            booking_manager.reset()  # Clear any previous state
            booking_manager.set_name(name)  # Set the name for the new booking
            booking_manager.data["booking_active"] = True
            return "On what date (DD/MM/YYYY) would you like to make your booking for?"

    # Handle numeric selection for cancellation
    if booking_manager.data["awaiting_booking_selection"]:
        try:
            selection = int(query)
            bookings = booking_manager.get_customer_bookings(name)
            if 1 <= selection <= len(bookings):
                booking_id = bookings[selection - 1][0]
                booking_manager.data["booking_id"] = booking_id
                booking_manager.data["confirming_cancellation"] = True
                booking_manager.data["awaiting_booking_selection"] = False  # Reset selection
                return f"Are you sure you want to cancel this booking? (yes/no)"
            return f"Invalid selection. Please choose a valid booking number between 1 and {len(bookings)}"
        except ValueError:
            return "Please enter a valid number."

    # Handle confirmation of cancellation
    if booking_manager.data["confirming_cancellation"]:
        if query.lower() == "yes":
            if booking_manager.delete_booking(booking_manager.data["booking_id"]):
                return "Your booking has been cancelled successfully."
            return "There was an error cancelling your booking. Please try again."
        if query.lower() == "no":
            booking_manager.reset()
            return "Booking cancellation abandoned. Is there anything else I can help you with?"

    # Handle initial cancellation request
    if detect_intent(query) == "cancel":
        if not name:
            print(f"{chatbot_name}: First, please tell me your name so I can find your bookings.")
            input_name = input("You: ").strip()
            input_name_c = extract_name(input_name)
            name = input_name_c
            booking_manager.set_name(input_name_c)
            identity_management(input_name_c)

        bookings = booking_manager.get_customer_bookings(name)
        if not bookings:
            return "You don't have any active bookings to cancel. Try typing 'make a booking' to create a booking."

        booking_manager.data["awaiting_booking_selection"] = True
        return booking_manager.format_bookings_list(
            bookings) + "\nPlease enter the number of the booking you'd like to cancel."

    restaurant_vectorizer, restaurant_transform = load_restaurant_model()
    processed_query = lemmatisation(query)
    query_transform = restaurant_vectorizer.transform([processed_query]).toarray()

    cos_sim = 1 - pairwise_distances(query_transform, restaurant_transform, metric='cosine').flatten()
    max_sim = cos_sim.max()
    matching_indices = np.where(cos_sim == max_sim)[0]
    success = booking_manager.parse_input(query)

    # Handle new booking initialization
    if not booking_manager.is_active():
        if not name:
            print(f"{chatbot_name}: First, please tell me your name so I can make a booking.")
            input_name = input("You: ").strip()
            input_name_c = extract_name(input_name)
            name = input_name_c
            booking_manager.set_name(input_name_c)
            identity_management(input_name_c)

        existing_bookings = booking_manager.get_customer_bookings(name)
        if existing_bookings:
            return f"I see you already have {len(existing_bookings)} booking(s):\n" + \
                booking_manager.format_bookings_list(existing_bookings) + \
                "\nWould you like to make another booking? (yes/no)"

        booking_manager.set_name(name)
        if booking_manager.has_pending_booking():
            return

        return f"Thank you {name}! What date (DD/MM/YYYY) would you like to make your booking for?"

    if max_sim >= 0.7:
        df = pd.read_csv('Datasets/RestaurantBooking.csv')
        response = df['Response'].iloc[random.choice(matching_indices)]
        response = response.replace("[date]", str(booking_manager.data["date"])) \
            .replace("[time]", str(booking_manager.data["time"])) \
            .replace("[name]", str(booking_manager.data["name"])) \
            .replace("[people]", str(booking_manager.data["people"]))
        return response

    if not success:
        return None

    #processing regular booking states
    if booking_manager.data["date"] and booking_manager.data["time"] and booking_manager.data["people"] and \
            booking_manager.data["dietary"]:
        if booking_manager.confirm_booking():
            return f"Perfect! Your booking has been saved. Details:\n" + \
                f"Name: {booking_manager.data['name']}\n" + \
                f"Date: {booking_manager.data['date']}\n" + \
                f"Time: {booking_manager.data['time']}\n" + \
                f"Number of people: {booking_manager.data['people']}\n" + \
                f"dietary: {booking_manager.data['dietary']}"

    if booking_manager.data["date"] and not booking_manager.data["time"] and not booking_manager.data["people"]:
        return f"Great! I will check availability on {booking_manager.data['date']}. Next, I need the time you would like to make the booking?"

    if booking_manager.data["time"] and not booking_manager.data["people"]:
        return f"Great! I will check availability at {booking_manager.data['time']}. How many people should I make the booking for?"

    if booking_manager.data["people"] and not booking_manager.data["dietary"]:
        dietary_preference = booking_manager.get_dietary_preference(booking_manager.data['name'])
        preference_hint = f" I notice you usually prefer {dietary_preference}." if dietary_preference else ""
        return (f"Great! I will check availability for {booking_manager.data['people']} people.\n"
                f" Lastly type 'Halal','vegan','vegetarian','kosher','pescatarian' if you have any of the dietary requirements or 'none' if you don't.\n"
                f"\t {preference_hint} ")
    if booking_manager.data["dietary"] and not booking_manager.data["date"]:
        return f"great I will save that. What date would you like the booking on?"

    return None
