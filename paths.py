import os

# --- BASE DIRECTORY ---
# This locates the root of your project (hoa_project)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- EXPLICIT PATH DEFINITIONS ---

# 1. Path to your configuration file (config.ini)
# This file stores your [FOLDERS] and [SETTINGS]
CONFIG_PATH = os.path.join(BASE_DIR, 'config.ini')

# 2. Path to the folder where raw documents (PDF/Docx) are stored
# Example: D:/Projects/AI/hoa_project/documents
DOCUMENTS_DIR = os.path.join(BASE_DIR, 'documents')

# 3. Path to the folder where the AI "Brains" (Chroma DB) are stored
# Example: D:/Projects/AI/hoa_project/chroma_db
CHROMA_DIR = os.path.join(BASE_DIR, 'chroma_db')

# --- AUTOMATIC DIRECTORY CHECK ---
# This part ensures the folders exist so the app doesn't crash 
# when trying to save files or databases for the first time.
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)

if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR)