import os
import json
import random
from flask import Flask, render_template, request, jsonify, make_response
from datetime import datetime, timedelta
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy as np

# --- NLTK setup (remains the same) ---
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
custom_words_to_keep = {
    'how', 'are', 'you', 'what', 'is', 'a', 'the', 'hi', 'bye',
    'fees', 'college', 'campus', 'course', 'event', 'hostel', 'library',
    'contact', 'placement', 'faculty', 'sport', 'faq', 'about', 'admission',
    'and', 'or', 'for', 'to', 'in', 'of', 'on', 'with', 'do', 'have', 'i', 's'
}
filtered_stop_words = set(w for w in stopwords.words('english') if w not in custom_words_to_keep)

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in filtered_stop_words and word.isalpha()]
    return ' '.join(tokens)

# --- Configuration ---
INTENTS_FILE = 'intents.json'
FLOW_FILE = 'flow.json' 
CHAT_DIR = "chat_logs"
SESSION_TIMEOUT = timedelta(minutes=10)
MODEL_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'chatbot_model.pkl')

# --- Load All Data and Models ---
vectorizer, label_encoder, model = None, None, None
knowledge_base = {"intents": []}
ml_model_loaded = False


# --- NEW DIAGNOSTIC BLOCK FOR flow.json ---
conversation_flow = {}
print("\n--- [DIAGNOSTIC] STARTING FILE CHECK ---")
if os.path.exists(FLOW_FILE):
    print(f"[DIAGNOSTIC] '{FLOW_FILE}' was found. Reading content...")
    try:
        with open(FLOW_FILE, 'r', encoding='utf-8') as f:
            file_content = f.read()
            print(f"--- START of {FLOW_FILE} content ---")
            print(file_content)
            print(f"--- END of {FLOW_FILE} content ---\n")
            
            print("[DIAGNOSTIC] Attempting to parse JSON...")
            conversation_flow = json.loads(file_content)
            if 'start' in conversation_flow:
                print("[DIAGNOSTIC] SUCCESS: JSON parsed and 'start' key was found!")
            else:
                print("[DIAGNOSTIC] CRITICAL WARNING: JSON parsed, but 'start' key is MISSING!")
    except Exception as e:
        print(f"[DIAGNOSTIC] CRITICAL ERROR: Could not read or parse '{FLOW_FILE}'. Error: {e}")
else:
    print(f"[DIAGNOSTIC] CRITICAL ERROR: The file '{FLOW_FILE}' was NOT FOUND.")
print("--- [DIAGNOSTIC] FILE CHECK COMPLETE ---\n")
# --- END OF DIAGNOSTIC BLOCK ---


try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model = joblib.load(MODEL_PATH)
    with open(INTENTS_FILE, encoding='utf-8') as file:
        knowledge_base = json.load(file)
    print("ML models and intents.json loaded successfully.")
    ml_model_loaded = True
except Exception as e:
    print(f"Warning: Could not load ML models. FAQ will not work. Error: {e}")
    # We allow the app to run without the ML model for flow testing
    ml_model_loaded = False


# --- Centralized Data Structures ---
faculty_links = {
    "BCA": "https://www.iitmjanakpuri.com/faculty/cs.php",
    "MCA": "https://www.iitmjanakpuri.com/faculty/cs.php",
    "BBA": "https://www.iitmjanakpuri.com/faculty/bcom_bba.php",
    "B.COM": "https://www.iitmjanakpuri.com/faculty/bcom_bba.php",
    "BA(JMC)": "https://www.iitmjanakpuri.com/faculty/masscomm_dept.php",
    "MBA": "https://www.iitmjanakpuri.com/faculty/mbadept.php"
}
placement_partners_link = "https://www.iitmjanakpuri.com/placements/partners.php"

course_fees = {
    "BCA": "₹1,50,000 per year",
    "MCA": "₹1,70,000 per year",
    "BBA": "₹1,40,000 per year",
    "MBA": "₹1,80,000 per year",
    "B.COM": "₹1,30,000 per year",
    "BAJMC": "₹1,35,000 per year"
}

# --- Flask setup ---
app = Flask(__name__)
# (The rest of the app.py file is the same as the last version...)
# ... (all helper functions and routes) ...

last_message_time, current_session_log_number, conversations, session_state = {}, {}, {}, {}
os.makedirs(CHAT_DIR, exist_ok=True)

def get_next_global_log_number():
    files = [f for f in os.listdir(CHAT_DIR) if f.startswith("session_log_") and f.endswith(".json")]
    if not files: return 1
    numbers = [int(f.replace("session_log_", "").replace(".json", "")) for f in files if f.replace("session_log_", "").replace(".json", "").isdigit()]
    return max(numbers) + 1 if numbers else 1

def save_conversation_log(session_id):
    if session_id in conversations and conversations[session_id]:
        log_number = current_session_log_number.get(session_id, get_next_global_log_number())
        filepath = os.path.join(CHAT_DIR, f"session_log_{log_number}.json")
        try:
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(conversations[session_id], f, indent=2)
        except Exception as e:
            print(f"Error saving conversation log: {e}")

def find_course_in_input(text):
    course_keywords = ['bca', 'mca', 'bba', 'mba', 'b.com', 'bajmc']
    for keyword in course_keywords:
        if keyword in text.lower():
            if keyword == 'b.com' and 'b.com' in text.lower(): return "B.COM"
            elif keyword != 'b.com': return keyword.upper()
    return None

def get_faq_response(user_input, session_id):
    if not ml_model_loaded:
        return "I'm sorry, the FAQ model is not loaded."
    current_state = session_state.get(session_id, {}).get('state')
    course = find_course_in_input(user_input)
    if current_state == 'awaiting_course_name':
        if course:
            session_state[session_id]['state'] = None
            fee = course_fees.get(course, "not available for that course")
            return f"The fee for the {course} program is {fee}. For a full breakdown, please visit the college website."
        else:
            return "I couldn't find a specific course name. Which program are you interested in (e.g., BCA, BBA, MBA)?"
    processed_input = preprocess_text(user_input)
    if not processed_input.strip():
        return random.choice(next((i['responses'] for i in knowledge_base['intents'] if i['tag'] == 'unknown'), ["Could you please rephrase?"]))
    user_input_vector = vectorizer.transform([processed_input])
    decision_scores = model.decision_function(user_input_vector)[0]
    best_intent_idx = np.argmax(decision_scores)
    confidence = decision_scores[best_intent_idx]
    predicted_tag = label_encoder.inverse_transform([best_intent_idx])[0]
    CONFIDENCE_THRESHOLD = 0.5
    if confidence >= CONFIDENCE_THRESHOLD:
        user_input_lower = user_input.strip().lower()
        if predicted_tag == 'fees' and course:
            fee = course_fees.get(course, "not listed for that program, but you can find it on our website")
            return f"The fee for the {course} program is {fee}."
        if predicted_tag == 'faculty' and course:
            link = faculty_links.get(course)
            return f"You can find faculty info for {course} here: {link}" if link else f"Sorry, I don't have a specific faculty link for {course}."
        if predicted_tag == 'placements' and any(word in user_input_lower for word in ['partners', 'companies', 'recruiters']):
             return f"You can find a list of our placement partners here: {placement_partners_link}"
        if predicted_tag == 'fees' and not course:
            session_state[session_id]['state'] = 'awaiting_course_name'
            return "It seems you're asking about fees. Could you please specify which program (e.g., BCA, MBA)?"
        for intent in knowledge_base['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])
    return random.choice(next((i['responses'] for i in knowledge_base['intents'] if i['tag'] == 'unknown'), ["I'm not sure how to help with that."]))

@app.route("/")
def index():
    session_id = request.cookies.get('session_id') or os.urandom(16).hex()
    resp = make_response(render_template("index.html"))
    resp.set_cookie('session_id', session_id)
    return resp

@app.route("/get", methods=["GET", "POST"])
def get_response_route():
    session_id = request.cookies.get('session_id')
    user_input = request.values.get('msg', '').strip()
    if not session_id or session_id not in last_message_time or (datetime.now() - last_message_time.get(session_id, datetime.now()) > SESSION_TIMEOUT):
        if session_id in conversations: save_conversation_log(session_id)
        if not session_id: session_id = os.urandom(16).hex()
        log_num = get_next_global_log_number()
        current_session_log_number[session_id] = log_num
        conversations[session_id] = []
        session_state[session_id] = {'node': 'start', 'state': None}
    last_message_time[session_id] = datetime.now()
    if user_input.lower() != 'start':
        conversations[session_id].append({"role": "user", "message": user_input, "timestamp": datetime.now().isoformat()})
    if user_input.lower() == 'start':
        session_state[session_id]['node'] = 'start'
        start_node = conversation_flow.get('start')
        # This is the line that crashes
        response_data = {"message": start_node['message'], "options": start_node['options']}
        resp = make_response(jsonify(response_data))
        resp.set_cookie('session_id', session_id)
        return resp
    current_flow_node = session_state[session_id].get('node', 'start')
    node_data = conversation_flow.get(current_flow_node)
    next_node_key = None
    if node_data and 'options' in node_data:
        for option in node_data['options']:
            if option['text'].lower() == user_input.lower():
                next_node_key = option['next_node']
                break
    if next_node_key:
        session_state[session_id]['node'] = next_node_key
        session_state[session_id]['state'] = None
        if next_node_key == 'final_fee_answer':
            course_name = user_input.upper()
            fee = course_fees.get(course_name, "not listed. Please check our website for details.")
            bot_message = f"The fee for the {course_name} program is {fee}."
            session_state[session_id]['node'] = 'post_answer_menu'
            next_options_node = conversation_flow.get('post_answer_menu')
            bot_message += f"\n\n{next_options_node['message']}"
            options = next_options_node.get('options', [])
            response_data = {"message": bot_message, "options": options}
        elif next_node_key == 'final_faculty_answer':
            course_name = user_input.upper()
            link = faculty_links.get(course_name, None)
            if link:
                bot_message = f"You can find the faculty information for the {course_name} program here: {link}"
            else:
                bot_message = f"I'm sorry, I couldn't find a specific faculty link for the {course_name} program."
            session_state[session_id]['node'] = 'post_answer_menu'
            next_options_node = conversation_flow.get('post_answer_menu')
            bot_message += f"\n\n{next_options_node['message']}"
            options = next_options_node.get('options', [])
            response_data = {"message": bot_message, "options": options}
        else:
            node_to_display = conversation_flow.get(next_node_key)
            if 'response' in node_to_display:
                bot_message = node_to_display['response']
                session_state[session_id]['node'] = node_to_display['next_node']
                next_options_node = conversation_flow.get(session_state[session_id]['node'])
                options = next_options_node.get('options', [])
                if 'message' in next_options_node:
                    bot_message += f"\n\n{next_options_node['message']}"
            else:
                bot_message = node_to_display['message']
                options = node_to_display.get('options', [])
            response_data = {"message": bot_message, "options": options}
    else:
        bot_message = get_faq_response(user_input, session_id)
        options = []
        if not session_state[session_id].get('state'):
            main_menu_options = conversation_flow.get('start', {}).get('options', [])
            options = main_menu_options
            session_state[session_id]['node'] = 'start' 
        response_data = {"message": bot_message, "options": options}
    conversations[session_id].append({"role": "bot", "message": response_data.get("message"), "timestamp": datetime.now().isoformat()})
    save_conversation_log(session_id)
    resp = make_response(jsonify(response_data))
    resp.set_cookie('session_id', session_id)
    return resp

if __name__ == "__main__":
    app.run(debug=True)