# Restaurant Booking Chatbot

This is a modular NLP-based conversational chatbot . It supports **small talk**, **question answering**, **name management**, **restaurant bookings**, and **function discovery**, powered by TF-IDF and cosine similarity intent matching.

---

## Features

- **Small Talk**: Responds to general conversation (e.g., "How are you?")
- **Question Answering**: Answers FAQ-style queries based on a dataset
- **Name Management**: Set, get, and change the user's name
- **Restaurant Booking**: Make, update, and cancel bookings with state tracking and database storage
- **Discoverability**: Users can ask “What can you do?” to see available functions
- **Sentiment Detection**: Responds with emoji-enhanced feedback
- **Error Handling**: Validates dates, times, and incomplete inputs gracefully
- **Modular Architecture**: Each feature is independently scalable

---


---

## How It Works

### Preprocessing
- Tokenisation, lemmatisation, stop-word removal, and sentiment analysis using `nltk`.

### Intent Matching
- Uses **TF-IDF + Cosine Similarity**.
- Modular datasets for each feature (e.g., `SmallTalk.csv`, `IdentityManagement.csv`).
- Easily extendable by adding new datasets.

### Booking System
- Persistent storage with **SQLite**
- Tracks incomplete/resumed bookings
- Validates:
  - Dates (within 90 days)
  - Times (11am–11pm)
  - Party size (< 20 people)

---

## Example Usage

**User:**  
> "Can I book a table?"

**Chatbot:**  
> "Sure! What date would you like to book for?"

---


## Evaluation Summary

-  **Intent Classification Accuracy:** 89%
- **Entity Extraction (Precision/Recall):** Evaluated with 150 annotated queries
- **Avg. Response Time:** Consistently responsive under 100 test queries
- **User Testing CUQ Score:** 8.5 / 10

---

## 🛠 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot.git
   cd chatbot/HumanAITrial-3
   
2. Run the chatbot
   ```bash
   python main.py

