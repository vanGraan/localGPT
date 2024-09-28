<!-- File: requirements.txt -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/requirements.txt -->
Flask
byaldi
torch
torchvision
google-generativeai
openai
docx2pdf
qwen-vl-utils
vllm>=0.6.1.post1
mistral_common>=1.4.1

<!-- File: make_single_file.py -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/make_single_file.py -->
import os

def consolidate_files(folder_path, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            if filename.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Write the filename and relative path as a comment in Markdown format
                outfile.write(f"<!-- File: {filename} -->\n<!-- Path: {file_path} -->\n")
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read() + '\n\n')  # Add a newline between files

# Example usage
consolidate_files('/Users/prompt/Documents/Github/local_vision', 'consolidated_output.md')

<!-- File: logger.py -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/logger.py -->
# logger.py

import logging

def get_logger(name):
    """
    Creates a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        # File handler
        f_handler = logging.FileHandler('app.log')
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


<!-- File: README.MD -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/README.MD -->
# localGPT-Vision

[![GitHub Stars](https://img.shields.io/github/stars/PromtEngineer/localGPT?style=social)](https://github.com/PromtEngineer/localGPT/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/PromtEngineer/localGPT?style=social)](https://github.com/PromtEngineer/localGPT/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/PromtEngineer/localGPT)](https://github.com/PromtEngineer/localGPT/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/PromtEngineer/localGPT)](https://github.com/PromtEngineer/localGPT/pulls)
[![License](https://img.shields.io/github/license/PromtEngineer/localGPT)](https://github.com/PromtEngineer/localGPT/blob/main/LICENSE)

localGPT-Vision is an end-to-end vision-based Retrieval-Augmented Generation (RAG) system. It allows users to upload and index documents (PDFs and images), ask questions about the content, and receive responses along with relevant document snippets. The retrieval is performed using the [ColPali](https://huggingface.co/blog/manu/colpali) model, and the retrieved pages are passed to a Vision Language Model (VLM) for generating responses. Currently, the code supports three VLMs: Qwen2-VL-7B-Instruct, Google Gemini, and OpenAI GPT-4. The project is built on top of the [Byaldi](https://github.com/AnswerDotAI/byaldi) library.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [System Workflow](#system-workflow)
- [Contributing](#contributing)
- [License](#license)

## Features
- End-to-End Vision-Based RAG: Combines visual document retrieval with language models for comprehensive answers.
- Document Upload and Indexing: Upload PDFs and images, which are then indexed using ColPali for retrieval.
- Chat Interface: Engage in a conversational interface to ask questions about the uploaded documents.
- Session Management: Create, rename, switch between, and delete chat sessions.
- Model Selection: Choose between different Vision Language Models (Qwen2-VL-7B-Instruct, Google Gemini, OpenAI GPT-4).
- Persistent Indexes: Indexes are saved on disk and loaded upon application restart. (TODO: bug fixes needed)

## Architecture
localGPT-Vision is built as an end-to-end vision-based RAG system. T he architecture comprises two main components:

1. Visual Document Retrieval with ColPali:
   - [ColPali](https://huggingface.co/blog/manu/colpali) is a Vision Language Model designed for efficient document retrieval solely using the image representation of document pages.
   - It embeds page images directly, leveraging visual cues like layout, fonts, figures, and tables without relying on OCR or text extraction.
   - During indexing, document pages are converted into image embeddings and stored.
   - During querying, the user query is matched against these embeddings to retrieve the most relevant document pages.
   
   ![ColPali](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)

2. Response Generation with Vision Language Models:
   - The retrieved document images are passed to a Vision Language Model (VLM).
   - Supported models include Qwen2-VL-7B-Instruct, Google Gemini, and OpenAI GPT-4.
   - These models generate responses by understanding both the visual and textual content of the documents.
   - NOTE: The quality of the responses is highly dependent on the VLM used and the resolution of the document images.

This architecture eliminates the need for complex text extraction pipelines and provides a more holistic understanding of documents by considering their visual elements. You don't need any chunking strategies or selection of embeddings model or retrieval strategy used in traditional RAG systems.

## Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.10 or higher
- Git (optional, for cloning the repository)

## Installation
Follow these steps to set up and run the application on your local machine.

1. Clone the Repository
   ```bash
   git clone -b localGPT-Vision --single-branch https://github.com/PromtEngineer/localGPT.git localGPT_Vision
   cd localGPT_Vision
   ```

2. Create a Conda Environment
   ```bash
   conda create -n localgpt-vision python=3.10
   conda activate localgpt-vision
   ```

3a. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3b. Install Transformers from HuggingFace - Dev version
   ```bash
    pip uninstall transformers
    pip install git+https://github.com/huggingface/transformers
   ```

4. Set Environment Variables
   Set your API keys for Google Gemini and OpenAI GPT-4:

   ```bash
   export GENAI_API_KEY='your_genai_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   ```

   On Windows Command Prompt:
   ```cmd
   set GENAI_API_KEY=your_genai_api_key
   set OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the Application
   ```bash
   python app.py
   ```

6. Access the Application
   Open your web browser and navigate to:
   ```
   http://localhost:5000/
   ```

## Usage
### Upload and Index Documents
1. Click on "New Chat" to start a new session.
2. Under "Upload and Index Documents", click "Choose Files" and select your PDF or image files.
3. Click "Upload and Index". The documents will be indexed using ColPali and ready for querying.

### Ask Questions
1. In the "Enter your question here" textbox, type your query related to the uploaded documents.
2. Click "Send". The system will retrieve relevant document pages and generate a response using the selected Vision Language Model.

### Manage Sessions
- Rename Session: Click "Edit Name", enter a new name, and click "Save Name".
- Switch Sessions: Click on a session name in the sidebar to switch to that session.
- Delete Session: Click "Delete" next to a session to remove it permanently.

### Settings
1. Click on "Settings" in the navigation bar.
2. Select the desired language model and image dimensions.
3. Click "Save Settings".

## Project Structure
```
localGPT-Vision/
├── app.py
├── logger.py
├── models/
│   ├── indexer.py
│   ├── retriever.py
│   ├── responder.py
│   ├── model_loader.py
│   └── converters.py
├── sessions/
├── templates/
│   ├── base.html
│   ├── chat.html
│   ├── settings.html
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
├── uploaded_documents/
├── byaldi_indices/
├── requirements.txt
├── .gitignore
└── README.md
```

- `app.py`: Main Flask application.
- `logger.py`: Configures application logging.
- `models/`: Contains modules for indexing, retrieving, and responding.
- `templates/`: HTML templates for rendering views.
- `static/`: Static files like CSS and JavaScript.
- `sessions/`: Stores session data.
- `uploaded_documents/`: Stores uploaded documents.
- `.byaldi/`: Stores the indexes created by Byaldi.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files and directories to be ignored by Git.
- `README.md`: Project documentation.

## System Workflow
1. User Interaction: The user interacts with the web interface to upload documents and ask questions.
2. Document Indexing with ColPali:
   - Uploaded documents are converted to PDFs if necessary.
   - Documents are indexed using ColPali, which creates embeddings based on the visual content of the document pages.
   - The indexes are stored in the byaldi_indices/ directory.
3. Session Management:
   - Each chat session has a unique ID and stores its own index and chat history.
   - Sessions are saved on disk and loaded upon application restart.
4. Query Processing:
   - User queries are sent to the backend.
   - The query is embedded and matched against the visual embeddings of document pages to retrieve relevant pages.
5. Response Generation with Vision Language Models:
   - The retrieved document images and the user query are passed to the selected Vision Language Model (Qwen, Gemini, or GPT-4).
   - The VLM generates a response by understanding both the visual and textual content of the documents.
6. Display Results:
   - The response and relevant document snippets are displayed in the chat interface.

```mermaid
graph TD
    A[User] -->|Uploads Documents| B(Flask App)
    B -->|Saves Files| C[uploaded_documents/]
    B -->|Converts and Indexes with ColPali| D[Indexing Module]
    D -->|Creates Visual Embeddings| E[byaldi_indices/]
    A -->|Asks Question| B
    B -->|Embeds Query and Retrieves Pages| F[Retrieval Module]
    F -->|Retrieves Relevant Pages| E
    F -->|Passes Pages to| G[Vision Language Model]
    G -->|Generates Response| B
    B -->|Displays Response| A
    B -->|Saves Session Data| H[sessions/]
    subgraph Backend
        B
        D
        F
        G
    end
    subgraph Storage
        C
        E
        H
    end
```

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Commit your changes: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)


<!-- File: app.py -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/app.py -->
import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from werkzeug.utils import secure_filename
from logger import get_logger
from byaldi import RAGMultiModalModel

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
app.config['INDEX_FOLDER'] = os.path.join(os.getcwd(), '.byaldi')  # Set to .byaldi folder in current directory

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)

# Initialize global variables
RAG_models = {}  # Dictionary to store RAG models per session
app.config['INITIALIZATION_DONE'] = False  # Flag to track initialization
logger.info("Application started.")

def load_rag_model_for_session(session_id):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_path = os.path.join(app.config['INDEX_FOLDER'], session_id)

    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

def load_existing_indexes():
    """
    Loads all existing indexes from the .byaldi folder when the application starts.
    """
    global RAG_models
    if os.path.exists(app.config['INDEX_FOLDER']):
        for session_id in os.listdir(app.config['INDEX_FOLDER']):
            if os.path.isdir(os.path.join(app.config['INDEX_FOLDER'], session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")

@app.before_request
def initialize_app():
    """
    Initializes the application by loading existing indexes.
    This will run before the first request, but only once.
    """
    if not app.config['INITIALIZATION_DONE']:
        load_existing_indexes()
        app.config['INITIALIZATION_DONE'] = True
        logger.info("Application initialized and indexes loaded.")

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
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
            # Handle file upload and indexing
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
                RAG = index_documents(session_folder, index_name=index_name, index_path=index_path)
                RAG_models[session_id] = RAG
                session['index_name'] = index_name
                session['session_folder'] = session_folder
                indexed_files = os.listdir(session_folder)
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

                flash("Generating response. Please wait...", "info")

                model_choice = session.get('model', 'qwen')
                resized_height = int(session.get('resized_height', 280))
                resized_width = int(session.get('resized_width', 280))

                RAG = RAG_models.get(session_id)
                if not RAG:
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
                chat_entry = {
                    'user': query,
                    'response': response,
                    'images': images
                }
                chat_history.append(chat_entry)
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
            flash("Invalid request.", "warning")
            return redirect(url_for('chat'))
    else:
        session_files = os.listdir(app.config['SESSION_FOLDER'])
        chat_sessions = []
        for file in session_files:
            if file.endswith('.json'):
                s_id = file[:-5]
                with open(os.path.join(app.config['SESSION_FOLDER'], file), 'r') as f:
                    data = json.load(f)
                    name = data.get('session_name', 'Untitled Session')
                    chat_sessions.append({'id': s_id, 'name': name})

        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)

        return render_template('chat.html', chat_history=chat_history, chat_sessions=chat_sessions,
                               current_session=session_id, model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width,
                               session_name=session_name, indexed_files=indexed_files)

@app.route('/switch_session/<session_id>')
def switch_session(session_id):
    session['session_id'] = session_id
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    flash(f"Switched to session.", "info")
    return redirect(url_for('chat'))

@app.route('/rename_session', methods=['POST'])
def rename_session():
    session_id = session['session_id']
    new_session_name = request.form.get('new_session_name', 'Untitled Session')
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
    else:
        session_data = {}

    session_data['session_name'] = new_session_name

    with open(session_file, 'w') as f:
        json.dump(session_data, f)

    flash("Session name updated.", "success")
    return redirect(url_for('chat'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
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
        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        return render_template('settings.html', model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width)

@app.route('/new_session')
def new_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
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
    try:
        session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        global RAG_models
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        session_images_folder = os.path.join('static', 'images', session_id)
        if os.path.exists(session_images_folder):
            import shutil
            shutil.rmtree(session_images_folder)
        RAG_models.pop(session_id, None)
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

<!-- File: consolidated_output.md -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/consolidated_output.md -->
<!-- File: requirements.txt -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/requirements.txt -->
Flask
byaldi
torch
torchvision
google-generativeai
openai
docx2pdf
qwen-vl-utils
vllm>=0.6.1.post1
mistral_common>=1.4.1

<!-- File: make_single_file.py -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/make_single_file.py -->
import os

def consolidate_files(folder_path, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            if filename.startswith('.'):  # Skip hidden files
                continue
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Write the filename and relative path as a comment in Markdown format
                outfile.write(f"<!-- File: {filename} -->\n<!-- Path: {file_path} -->\n")
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read() + '\n\n')  # Add a newline between files

# Example usage
consolidate_files('/Users/prompt/Documents/Github/local_vision', 'consolidated_output.md')

<!-- File: logger.py -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/logger.py -->
# logger.py

import logging

def get_logger(name):
    """
    Creates a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        # File handler
        f_handler = logging.FileHandler('app.log')
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


<!-- File: README.MD -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/README.MD -->
# localGPT-Vision

[![GitHub Stars](https://img.shields.io/github/stars/PromtEngineer/localGPT?style=social)](https://github.com/PromtEngineer/localGPT/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/PromtEngineer/localGPT?style=social)](https://github.com/PromtEngineer/localGPT/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/PromtEngineer/localGPT)](https://github.com/PromtEngineer/localGPT/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/PromtEngineer/localGPT)](https://github.com/PromtEngineer/localGPT/pulls)
[![License](https://img.shields.io/github/license/PromtEngineer/localGPT)](https://github.com/PromtEngineer/localGPT/blob/main/LICENSE)

localGPT-Vision is an end-to-end vision-based Retrieval-Augmented Generation (RAG) system. It allows users to upload and index documents (PDFs and images), ask questions about the content, and receive responses along with relevant document snippets. The retrieval is performed using the [ColPali](https://huggingface.co/blog/manu/colpali) model, and the retrieved pages are passed to a Vision Language Model (VLM) for generating responses. Currently, the code supports three VLMs: Qwen2-VL-7B-Instruct, Google Gemini, and OpenAI GPT-4. The project is built on top of the [Byaldi](https://github.com/AnswerDotAI/byaldi) library.

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [System Workflow](#system-workflow)
- [Contributing](#contributing)
- [License](#license)

## Features
- End-to-End Vision-Based RAG: Combines visual document retrieval with language models for comprehensive answers.
- Document Upload and Indexing: Upload PDFs and images, which are then indexed using ColPali for retrieval.
- Chat Interface: Engage in a conversational interface to ask questions about the uploaded documents.
- Session Management: Create, rename, switch between, and delete chat sessions.
- Model Selection: Choose between different Vision Language Models (Qwen2-VL-7B-Instruct, Google Gemini, OpenAI GPT-4).
- Persistent Indexes: Indexes are saved on disk and loaded upon application restart. (TODO: bug fixes needed)

## Architecture
localGPT-Vision is built as an end-to-end vision-based RAG system. T he architecture comprises two main components:

1. Visual Document Retrieval with ColPali:
   - [ColPali](https://huggingface.co/blog/manu/colpali) is a Vision Language Model designed for efficient document retrieval solely using the image representation of document pages.
   - It embeds page images directly, leveraging visual cues like layout, fonts, figures, and tables without relying on OCR or text extraction.
   - During indexing, document pages are converted into image embeddings and stored.
   - During querying, the user query is matched against these embeddings to retrieve the most relevant document pages.
   
   ![ColPali](https://cdn-uploads.huggingface.co/production/uploads/60f2e021adf471cbdf8bb660/La8vRJ_dtobqs6WQGKTzB.png)

2. Response Generation with Vision Language Models:
   - The retrieved document images are passed to a Vision Language Model (VLM).
   - Supported models include Qwen2-VL-7B-Instruct, Google Gemini, and OpenAI GPT-4.
   - These models generate responses by understanding both the visual and textual content of the documents.
   - NOTE: The quality of the responses is highly dependent on the VLM used and the resolution of the document images.

This architecture eliminates the need for complex text extraction pipelines and provides a more holistic understanding of documents by considering their visual elements. You don't need any chunking strategies or selection of embeddings model or retrieval strategy used in traditional RAG systems.

## Prerequisites
- Anaconda or Miniconda installed on your system
- Python 3.10 or higher
- Git (optional, for cloning the repository)

## Installation
Follow these steps to set up and run the application on your local machine.

1. Clone the Repository
   ```bash
   git clone -b localGPT-Vision --single-branch https://github.com/PromtEngineer/localGPT.git localGPT_Vision
   cd localGPT_Vision
   ```

2. Create a Conda Environment
   ```bash
   conda create -n localgpt-vision python=3.10
   conda activate localgpt-vision
   ```

3a. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3b. Install Transformers from HuggingFace - Dev version
   ```bash
    pip uninstall transformers
    pip install git+https://github.com/huggingface/transformers
   ```

4. Set Environment Variables
   Set your API keys for Google Gemini and OpenAI GPT-4:

   ```bash
   export GENAI_API_KEY='your_genai_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   ```

   On Windows Command Prompt:
   ```cmd
   set GENAI_API_KEY=your_genai_api_key
   set OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the Application
   ```bash
   python app.py
   ```

6. Access the Application
   Open your web browser and navigate to:
   ```
   http://localhost:5000/
   ```

## Usage
### Upload and Index Documents
1. Click on "New Chat" to start a new session.
2. Under "Upload and Index Documents", click "Choose Files" and select your PDF or image files.
3. Click "Upload and Index". The documents will be indexed using ColPali and ready for querying.

### Ask Questions
1. In the "Enter your question here" textbox, type your query related to the uploaded documents.
2. Click "Send". The system will retrieve relevant document pages and generate a response using the selected Vision Language Model.

### Manage Sessions
- Rename Session: Click "Edit Name", enter a new name, and click "Save Name".
- Switch Sessions: Click on a session name in the sidebar to switch to that session.
- Delete Session: Click "Delete" next to a session to remove it permanently.

### Settings
1. Click on "Settings" in the navigation bar.
2. Select the desired language model and image dimensions.
3. Click "Save Settings".

## Project Structure
```
localGPT-Vision/
├── app.py
├── logger.py
├── models/
│   ├── indexer.py
│   ├── retriever.py
│   ├── responder.py
│   ├── model_loader.py
│   └── converters.py
├── sessions/
├── templates/
│   ├── base.html
│   ├── chat.html
│   ├── settings.html
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
├── uploaded_documents/
├── byaldi_indices/
├── requirements.txt
├── .gitignore
└── README.md
```

- `app.py`: Main Flask application.
- `logger.py`: Configures application logging.
- `models/`: Contains modules for indexing, retrieving, and responding.
- `templates/`: HTML templates for rendering views.
- `static/`: Static files like CSS and JavaScript.
- `sessions/`: Stores session data.
- `uploaded_documents/`: Stores uploaded documents.
- `.byaldi/`: Stores the indexes created by Byaldi.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files and directories to be ignored by Git.
- `README.md`: Project documentation.

## System Workflow
1. User Interaction: The user interacts with the web interface to upload documents and ask questions.
2. Document Indexing with ColPali:
   - Uploaded documents are converted to PDFs if necessary.
   - Documents are indexed using ColPali, which creates embeddings based on the visual content of the document pages.
   - The indexes are stored in the byaldi_indices/ directory.
3. Session Management:
   - Each chat session has a unique ID and stores its own index and chat history.
   - Sessions are saved on disk and loaded upon application restart.
4. Query Processing:
   - User queries are sent to the backend.
   - The query is embedded and matched against the visual embeddings of document pages to retrieve relevant pages.
5. Response Generation with Vision Language Models:
   - The retrieved document images and the user query are passed to the selected Vision Language Model (Qwen, Gemini, or GPT-4).
   - The VLM generates a response by understanding both the visual and textual content of the documents.
6. Display Results:
   - The response and relevant document snippets are displayed in the chat interface.

```mermaid
graph TD
    A[User] -->|Uploads Documents| B(Flask App)
    B -->|Saves Files| C[uploaded_documents/]
    B -->|Converts and Indexes with ColPali| D[Indexing Module]
    D -->|Creates Visual Embeddings| E[byaldi_indices/]
    A -->|Asks Question| B
    B -->|Embeds Query and Retrieves Pages| F[Retrieval Module]
    F -->|Retrieves Relevant Pages| E
    F -->|Passes Pages to| G[Vision Language Model]
    G -->|Generates Response| B
    B -->|Displays Response| A
    B -->|Saves Session Data| H[sessions/]
    subgraph Backend
        B
        D
        F
        G
    end
    subgraph Storage
        C
        E
        H
    end
```

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Commit your changes: `git commit -am 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)


<!-- File: app.py -->
<!-- Path: /Users/prompt/Documents/Github/local_vision/app.py -->
import os
import uuid
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from models.indexer import index_documents
from models.retriever import retrieve_documents
from models.responder import generate_response
from werkzeug.utils import secure_filename
from logger import get_logger
from byaldi import RAGMultiModalModel

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
app.config['INDEX_FOLDER'] = os.path.join(os.getcwd(), '.byaldi')  # Set to .byaldi folder in current directory

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FOLDER'], exist_ok=True)

# Initialize global variables
RAG_models = {}  # Dictionary to store RAG models per session
app.config['INITIALIZATION_DONE'] = False  # Flag to track initialization
logger.info("Application started.")

def load_rag_model_for_session(session_id):
    """
    Loads the RAG model for the given session_id from the index on disk.
    """
    index_path = os.path.join(app.config['INDEX_FOLDER'], session_id)

    if os.path.exists(index_path):
        try:
            RAG = RAGMultiModalModel.from_index(index_path)
            RAG_models[session_id] = RAG
            logger.info(f"RAG model for session {session_id} loaded from index.")
        except Exception as e:
            logger.error(f"Error loading RAG model for session {session_id}: {e}")
    else:
        logger.warning(f"No index found for session {session_id}.")

def load_existing_indexes():
    """
    Loads all existing indexes from the .byaldi folder when the application starts.
    """
    global RAG_models
    if os.path.exists(app.config['INDEX_FOLDER']):
        for session_id in os.listdir(app.config['INDEX_FOLDER']):
            if os.path.isdir(os.path.join(app.config['INDEX_FOLDER'], session_id)):
                load_rag_model_for_session(session_id)
    else:
        logger.warning("No .byaldi folder found. No existing indexes to load.")

@app.before_request
def initialize_app():
    """
    Initializes the application by loading existing indexes.
    This will run before the first request, but only once.
    """
    if not app.config['INITIALIZATION_DONE']:
        load_existing_indexes()
        app.config['INITIALIZATION_DONE'] = True
        logger.info("Application initialized and indexes loaded.")

@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('chat'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
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
            # Handle file upload and indexing
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
                RAG = index_documents(session_folder, index_name=index_name, index_path=index_path)
                RAG_models[session_id] = RAG
                session['index_name'] = index_name
                session['session_folder'] = session_folder
                indexed_files = os.listdir(session_folder)
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

                flash("Generating response. Please wait...", "info")

                model_choice = session.get('model', 'qwen')
                resized_height = int(session.get('resized_height', 280))
                resized_width = int(session.get('resized_width', 280))

                RAG = RAG_models.get(session_id)
                if not RAG:
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
                chat_entry = {
                    'user': query,
                    'response': response,
                    'images': images
                }
                chat_history.append(chat_entry)
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
            flash("Invalid request.", "warning")
            return redirect(url_for('chat'))
    else:
        session_files = os.listdir(app.config['SESSION_FOLDER'])
        chat_sessions = []
        for file in session_files:
            if file.endswith('.json'):
                s_id = file[:-5]
                with open(os.path.join(app.config['SESSION_FOLDER'], file), 'r') as f:
                    data = json.load(f)
                    name = data.get('session_name', 'Untitled Session')
                    chat_sessions.append({'id': s_id, 'name': name})

        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)

        return render_template('chat.html', chat_history=chat_history, chat_sessions=chat_sessions,
                               current_session=session_id, model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width,
                               session_name=session_name, indexed_files=indexed_files)

@app.route('/switch_session/<session_id>')
def switch_session(session_id):
    session['session_id'] = session_id
    if session_id not in RAG_models:
        load_rag_model_for_session(session_id)
    flash(f"Switched to session.", "info")
    return redirect(url_for('chat'))

@app.route('/rename_session', methods=['POST'])
def rename_session():
    session_id = session['session_id']
    new_session_name = request.form.get('new_session_name', 'Untitled Session')
    session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")

    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session_data = json.load(f)
    else:
        session_data = {}

    session_data['session_name'] = new_session_name

    with open(session_file, 'w') as f:
        json.dump(session_data, f)

    flash("Session name updated.", "success")
    return redirect(url_for('chat'))

@app.route('/settings', methods=['GET', 'POST'])
def settings():
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
        model_choice = session.get('model', 'qwen')
        resized_height = session.get('resized_height', 280)
        resized_width = session.get('resized_width', 280)
        return render_template('settings.html', model_choice=model_choice,
                               resized_height=resized_height, resized_width=resized_width)

@app.route('/new_session')
def new_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session_files = os.listdir(app.config['SESSION_FOLDER'])
    session_number = len([f for f in session_files if f.endswith('.json')]) + 1
    session_name = f"Session {session_number}"
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
    try:
        session_file = os.path.join(app.config['SESSION_FOLDER'], f"{session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        global RAG_models
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        if os.path.exists(session_folder):
            import shutil
            shutil.rmtree(session_folder)
        session_images_folder = os.path.join('static', 'images', session_id)
        if os.path.exists(session_images_folder):
            import shutil
            shutil.rmtree(session_images_folder)
        RAG_models.pop(session_id, None)
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
