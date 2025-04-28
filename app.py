from flask import Flask, render_template, request, jsonify
import os
import logging
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Debug log environment variables (without exposing sensitive data)
logger.info("Verifying environment variables...")
logger.info(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE')}")
logger.info(f"OPENAI_API_VERSION: {os.getenv('OPENAI_API_VERSION')}")
logger.info(f"OPENAI_API_TYPE: {os.getenv('OPENAI_API_TYPE')}")
logger.info(f"OPENAI_DEPLOYMENT_NAME: {os.getenv('OPENAI_DEPLOYMENT_NAME')}")
logger.info("OPENAI_API_KEY: [REDACTED]")

# Validate required environment variables
required_env_vars = [
    'OPENAI_API_BASE',
    'OPENAI_API_KEY',
    'OPENAI_API_VERSION',
    'OPENAI_API_TYPE',
    'OPENAI_DEPLOYMENT_NAME'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Configure Azure OpenAI environment variables
os.environ["OPENAI_API_TYPE"] = os.getenv('OPENAI_API_TYPE')
os.environ["OPENAI_API_VERSION"] = os.getenv('OPENAI_API_VERSION')
os.environ["OPENAI_API_BASE"] = os.getenv('OPENAI_API_BASE')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Configure OpenAI client
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Flask app
app = Flask(__name__)

try:
    # Initialize LLM and embeddings
    deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')
    logger.info(f"Initializing Azure OpenAI with deployment: {deployment_name}")
    
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        temperature=0,
        openai_api_version=os.getenv('OPENAI_API_VERSION')
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1
    )

    # Load and process documents
    logger.info("Loading documents from ./data/qna/")
    loader = DirectoryLoader('./data/qna/', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
    
    if not documents:
        raise ValueError("No documents were loaded. Please check if there are .txt files in the data/qna directory.")
    
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(docs)} chunks")

    # Create vector store
    logger.info("Creating FAISS vector store")
    db = FAISS.from_documents(documents=docs, embedding=embeddings)
    logger.info("FAISS vector store created successfully")

    # Create QnA chain
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:""")

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        return_source_documents=True,
        verbose=False
    )

    # Store chat history in memory (in production, use a proper database)
    chat_history = []

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global chat_history
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        result = qa({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        
        # Update chat history
        chat_history.append((question, answer))
        
        return jsonify({
            'answer': answer,
            'sources': [doc.page_content for doc in result.get('source_documents', [])]
        })
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your question'}), 500

if __name__ == '__main__':
    app.run(debug=True) 