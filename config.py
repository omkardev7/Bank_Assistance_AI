# config.py
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Get a configured logger instance"""
    return logging.getLogger(name)

# API Keys
def get_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    return api_key

def get_exa_api_key():
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not found in environment variables")
    return api_key

# File Paths
def get_vector_db_path():
    return "faiss_index_bom"

def get_raw_data_path():
    return "loan_data_raw.txt"

def get_cleaned_data_path():
    return "loan_data_cleaned.txt"

def get_conversation_history_path():
    return "conversations"

# Model Configuration
def get_embedding_model():
    return "sentence-transformers/all-MiniLM-L6-v2"

def get_llm_model():
    return "gemini-2.5-flash"

def get_llm_temperature():
    return 0.2

# Chunking Configuration
def get_chunk_size():
    return 1000

def get_chunk_overlap():
    return 200

# Retrieval Configuration
def get_retrieval_k():
    return 5

# Exa Search Configuration
def get_exa_num_results():
    return 3