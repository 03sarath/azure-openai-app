import os
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from config import Config

class QAModel:
    def __init__(self):
        # Initialize LLM and embeddings
        self.llm = AzureChatOpenAI(
            deployment_name=Config.DEPLOYMENT_NAME,
            temperature=0,
            openai_api_version=Config.OPENAI_API_VERSION
        )
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            chunk_size=1
        )
        
        # Load and process documents
        self._load_documents()
        
        # Create QA chain
        self._create_qa_chain()
    
    def _load_documents(self):
        """Load and process documents from the data directory."""
        loader = DirectoryLoader(
            Config.QNA_DIR,
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.docs = text_splitter.split_documents(documents)
        
        # Create vector store
        self.db = FAISS.from_documents(documents=self.docs, embedding=self.embeddings)
    
    def _create_qa_chain(self):
        """Create the conversational QA chain."""
        condense_question_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""
        )
        
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.db.as_retriever(),
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            verbose=False
        )
    
    def get_answer(self, question, chat_history=None):
        """Get answer for a question using the QA chain."""
        if chat_history is None:
            chat_history = []
            
        result = self.qa({
            "question": question,
            "chat_history": chat_history
        })
        
        return {
            "answer": result["answer"],
            "sources": result.get("source_documents", [])
        } 