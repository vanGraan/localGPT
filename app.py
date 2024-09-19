# app.py

import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from werkzeug.utils import secure_filename
from logger import get_logger
from byaldi import RAGMultiModalModel  # Make sure to import RAGMultiModalModel here

# Set the TOKENIZERS_PARALLELISM environment variable to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

logger = get_logger(__name__)

# Configure upload folders
app.config['UPLOAD_FOLDER'] = 'uploaded_documents'
app.config['STATIC_FOLDER'] = 'static'
app.config['SESSION_FOLDER'] = 'sessions'
app.config['INDEX_FOLDER'] = '../'  # Add index folder configuration

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)
os.makedirs(app.config['INDEX_FOLDER'], exist_ok=True)  # Create index folder

# Initialize global variables
RAG_models = {}  # Dictionary to store RAG models per session
logger.info("Application started.")

def load_rag_model_for_session(session_id):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_name = session_id
    index_path = os.path.join(app.config['INDEX_FOLDER'], index_name)

    if os.path.exists(index_path):
        # Load the RAG model
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.route('/', methods=['GET'])
def home():
    """
    Renders the home page.
    """
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """
    Handles the chat interface where users can upload documents, index them, and chat.
    """
    session_id = session['session_id']
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    # Load session data from file
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            chat_history = session_data.get('chat_history', [])
            session_name = session_data.get('session_name', 'Untitled Session')
            indexed_files = session_data.get('indexed_files', [])
    else:
        chat_history = []
        session_name = 'Untitled Session'
        indexed_files = []

    if request.method == 'POST':
        if 'upload' in request.form:
            # Handle file upload
            files = request.files.getlist('files')
            session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            os.makedirs(session_folder, exist_ok=True)
            uploaded_files = []
            for file in files:
                if file and file.filename:
                    filename = secure_filename(os.path.basename(file.filename))
                    file_path = os.path.join(session_folder, filename)
                    file.save(file_path)
                    uploaded_files.append(filename)
                    logger.info(f"File saved: {file_path}")
            try:
                index_name = session_id
                index_path = os.path.join(app.config['INDEX_FOLDER'], index_name)
                flash("Indexing documents. This may take a moment...", "info")
                # Index documents
                RAG = index_documents(session_folder, index_name=index_name, index_path=index_path)
                RAG_models[session_id] = RAG  # Store RAG model for the session
                session['index_name'] = index_name
                session['session_folder'] = session_folder
                # Update indexed_files
                indexed_files = os.listdir(session_folder)
                # Save session data
                session_data = {
                    'session_name': session_name,
                    'chat_history': chat_history,
                    'indexed_files': indexed_files
                }
                with open(session_file, 'w') as f:
                    json.dump(session_data, f)
                flash("Documents indexed successfully.", "success")
                logger.info("Documents indexed successfully.")
            except Exception as e:
                logger.error(f"Error indexing documents: {e}")
                flash("An error occurred while indexing documents.", "danger")
            return redirect(url_for('chat'))
        elif 'send_query' in request.form:
            # Handle chat query
            try:
                query = request.form['query']
                if not query.strip():
                    flash("Please enter a query.", "warning")
                    return redirect(url_for('chat'))

                # Show loading spinner
                flash("Generating response. Please wait...", "info")

                # Get settings
                model_choice = session.get('model', 'qwen')
                resized_height = int(session.get('resized_height', 280))
                resized_width = int(session.get('resized_width', 280))

                RAG = RAG_models.get(session_id)
                if not RAG:
                    # Attempt to load RAG model from index
                    load_rag_model_for_session(session_id)
                    RAG = RAG_models.get(session_id)
                    if not RAG:
                        flash("Please upload and index documents first.", "warning")
                        return redirect(url_for('chat'))

                images = retrieve_documents(RAG, query, session_id)
                response = generate_response(
                    images, query, session_id,
                    resized_height=resized_height,
                    resized_width=resized_width,
                    model_choice=model_choice
                )
                # Save conversation to chat history
                chat_entry = {
                    'user': query,
                    'response': response,
                    'images': images
                }
                chat_history.append(chat_entry)
                # Save session data
                session_data = {
                    'session_name': session_name,
                    'chat_history': chat_history,
                    'indexed_files': indexed_files
                }
                with open(session_file, 'w') as f:
                    json.dump(session_data, f)
                logger.info("Response generated and added to chat history.")
                flash("Response generated.", "success")
                return redirect(url_for('chat'))
            except Exception as e:
                logger.error(f"Error in chat route: {e}")
                flash("An error occurred. Please try again.", "danger")
                return redirect(url_for('chat'))
        elif 'rename_session' in request.form:
            # Handle session renaming
            new_session_name = request.form.get('session_name', 'Untitled Session')
            session_name = new_session_name
            # Save session data
            session_data = {
                'session_name': session_name,
                'chat_history': chat_history,
                'indexed_files': indexed_files
            }
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
            flash("Session name updated.", "success")
            return redirect(url_for('chat'))
        else:
            # If neither 'upload', 'send_query', nor 'rename_session' is in request.form
            flash("Invalid request.", "warning")
            return redirect(url_for('chat'))
    else:
        # Get list of chat sessions
        session_files = os.listdir(app.config['SESSION_FOLDER'])
        chat_sessions = []
        for file in session_files:
            if file.endswith('.json'):
                s_id = file[:-5]
                with open(os.path.join(app.config['SESSION_FOLDER'], file), 'r') as f:
                    data = json.load(f)
                    name = data.get('session_name', 'Untitled Session')
                    chat_sessions.append({'id': s_id, 'name': name})

        # Load settings
        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)

        return render_template('chat.html', chat_history=chat_history, chat_sessions=chat_sessions,
                               current_session=session_id, model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width,
                               session_name=session_name, indexed_files=indexed_files)

@app.route('/switch_session/<session_id>')
def switch_session(session_id):
    """
    Switches to a different chat session.
    """
    session['session_id'] = session_id
    # Attempt to load RAG model for the session
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    flash(f"Switched to session.", "info")
    return redirect(url_for('chat'))

@app.route('/rename_session', methods=['POST'])
def rename_session():
    """
    Renames the current chat session.
    """
    session_id = session['session_id']
    new_session_name = request.form.get('new_session_name', 'Untitled Session')
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    # Load session data
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
    else:
        session_data = {}

    session_data['session_name'] = new_session_name

    # Save session data
    with open(session_file, 'w') as f:
        json.dump(session_data, f)

    flash("Session name updated.", "success")
    return redirect(url_for('chat'))


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """
    Renders and processes the settings page for model selection and image resolution.
    """
    if request.method == 'POST':
        model_choice = request.form.get('model', 'qwen')
        resized_height = request.form.get('resized_height', 280)
        resized_width = request.form.get('resized_width', 280)
        session['model'] = model_choice
        session['resized_height'] = resized_height
        session['resized_width'] = resized_width
        session.modified = True
        logger.info(f"Settings updated: model={model_choice}, resized_height={resized_height}, resized_width={resized_width}")
        flash("Settings updated.", "success")
        return redirect(url_for('chat'))
    else:
        # Load settings
        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        return render_template('settings.html', model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width)

@app.route('/new_session')
def new_session():
    """
    Starts a new chat session.
    """
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    # Assign default session name
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
    # Save session data
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
    session_data = {
        'session_name': session_name,
        'chat_history': [],
        'indexed_files': []
    }
    with open(session_file, 'w') as f:
        json.dump(session_data, f)
    flash("New chat session started.", "success")
    return redirect(url_for('chat'))

@app.route('/delete_session/<session_id>')
def delete_session(session_id):
    """
    Deletes a chat session.
    """
    try:
        # Delete session file
        session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        # Remove uploaded documents and reset RAG model
        global RAG_models
        # Delete session folder
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        # Delete generated images
        session_images_folder = os.path.join('static', 'images', session_id)
        if os.path.exists(session_images_folder):
            import shutil
            shutil.rmtree(session_images_folder)
        # Remove RAG model for session
        RAG_models.pop(session_id, None)
        # If deleting current session, start a new one
        if session.get('session_id') == session_id:
            session['session_id'] = str(uuid.uuid4())
        logger.info(f"Session {session_id} deleted.")
        flash("Session deleted.", "success")
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        flash("An error occurred while deleting the session.", "danger")
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(port=5050, debug=True)
