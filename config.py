import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # Azure OpenAI settings
    OPENAI_API_TYPE = "azure"
    OPENAI_API_VERSION = "2023-03-15-preview"
    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Model settings
    DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
    
    # Data settings
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    QNA_DIR = os.path.join(DATA_DIR, 'qna')
    
    # Session settings
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour 