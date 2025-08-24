# College FAQ Chatbot with Guided Flow

This is a chatbot for a college website built with Python, Flask, and NLTK. It combines an AI-based FAQ model with a guided conversation flow to help users find information easily.

## Features
- **Guided Conversation Flow:** Uses buttons to guide users through topics like Admissions, Campus, Placements, etc.
- **Interactive Q&A:** Includes interactive flows for specific queries like Fees and Faculty information.
- **AI-Powered FAQ:** A fallback AI model (trained with `intents.json`) answers free-text questions.
- **Web Interface:** A clean and simple chat interface built with HTML, CSS, and JavaScript.

## How to Run This Project

1.  **Clone the repository:**
    `git clone [Your-Repo-URL-Here]`

2.  **Create a virtual environment:**
    `python -m venv venv`
    `source venv/bin/activate`  (On Windows, use `venv\Scripts\activate`)

3.  **Install the required packages:**
    `pip install -r requirements.txt` 
    *(Note: You will need to create a `requirements.txt` file. See below.)*

4.  **Train the AI model:**
    `python train_model.py`

5.  **Run the Flask application:**
    `python app.py`

6.  Open your web browser and go to `http://127.0.0.1:5000`.