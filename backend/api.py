"""
RAG Backend — Azure OpenAI + FAISS + LangChain
-----------------------------------------------
This Flask app exposes a REST API for the RAG (Retrieval-Augmented Generation)
pipeline that was built step-by-step in the course notebook.

Startup behaviour:
  - If vector_store/ already exists on disk → load it (fast restart).
  - If not → build it from all .txt files in data/qna/ (first run).

Endpoints:
  GET  /health                      health check + status
  GET  /documents                   list stored docs + chunk counts
  POST /documents/add               upload a new .txt file and index it
  DELETE /documents/delete/<name>   remove a doc from the index
  POST /documents/reindex/<name>    re-index an edited file (update flow)
  POST /ask                         ask a question (with chat history)
  POST /history/clear               reset conversation memory
"""

import os
import logging
import shutil
from pathlib import Path

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

# ── Setup ────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths — set via docker-compose environment variables
DATA_DIR         = Path(os.getenv("DATA_DIR",         "./data/qna"))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))

# ── Validate required environment variables before doing anything else ────
_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_api_key  = os.getenv("OPENAI_API_KEY", "")

if not _api_key or _api_key == "your-key-here":
    raise SystemExit("ERROR: OPENAI_API_KEY is not set in your .env file.")

if not _endpoint or _endpoint == "https://your-resource-name.openai.azure.com/":
    raise SystemExit("ERROR: AZURE_OPENAI_ENDPOINT is not set in your .env file.")

if not _endpoint.startswith("https://"):
    raise SystemExit(
        f"ERROR: AZURE_OPENAI_ENDPOINT must start with 'https://'.\n"
        f"  Current value: '{_endpoint}'\n"
        f"  Expected format: 'https://your-resource-name.openai.azure.com/'"
    )

# ── Connect to Azure OpenAI (Step 2 in notebook) ─────────────────────────
logger.info("Connecting to Azure OpenAI...")

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-02-01",
)

logger.info("Azure OpenAI connected")

# Splitter used for every document (same settings as notebook)
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)

# ── Global state ─────────────────────────────────────────────────────────
# These are module-level globals — fine for a single-process course demo.
db: FAISS | None = None
chain = None
chat_history: list = []   # list of HumanMessage / AIMessage objects


# ── Helper: build the QA chain (Step 7 in notebook) ─────────────────────
def build_chain(vector_db: FAISS):
    """Recreate the LCEL chain whenever the vector store changes."""
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using only the context below. "
         "If unsure, say you don't know.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    return (
        {
            "context":      RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            "question":     RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | prompt
        | llm
    )


# ── Helper: document inventory ───────────────────────────────────────────
def get_doc_summary() -> dict[str, int]:
    """Return {filename: chunk_count} for every doc stored in the vector DB."""
    if db is None:
        return {}
    summary: dict[str, int] = {}
    for doc in db.docstore._dict.values():
        name = Path(doc.metadata.get("source", "unknown")).name
        summary[name] = summary.get(name, 0) + 1
    return summary


def find_exact_source(filename: str) -> str | None:
    """Return the exact source path string stored in the DB for a given filename.

    FAISS stores the path that was used when the document was loaded.
    We must use this exact string when filtering — not os.path.abspath().
    """
    if db is None:
        return None
    for doc in db.docstore._dict.values():
        src = doc.metadata.get("source", "")
        if Path(src).name == filename:
            return src
    return None


# ── Load or build vector store on startup ────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

if (VECTOR_STORE_DIR / "index.faiss").exists():
    # Step 5 in notebook: load from disk (fast path on every restart)
    logger.info("Loading vector store from disk...")
    db = FAISS.load_local(
        str(VECTOR_STORE_DIR), embeddings, allow_dangerous_deserialization=True
    )
    logger.info(f"Vector store loaded — {len(db.docstore._dict)} chunks")
else:
    # Steps 3+4 in notebook: load docs and build index (first run only)
    txt_files = list(DATA_DIR.glob("*.txt"))
    if txt_files:
        logger.info(f"Building vector store from {len(txt_files)} file(s)...")
        all_docs = []
        for f in txt_files:
            all_docs.extend(TextLoader(str(f), encoding="utf-8").load())
        chunks = splitter.split_documents(all_docs)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(str(VECTOR_STORE_DIR))
        logger.info(f"Vector store built — {len(chunks)} chunks saved to disk")
    else:
        logger.warning("No .txt files found. Upload documents via the UI to get started.")

if db is not None:
    chain = build_chain(db)
    logger.info("QA chain ready")


# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "vector_store_loaded": db is not None,
        "document_count": len(get_doc_summary()),
        "chat_history_length": len(chat_history),
    })


@app.route("/documents")
def list_documents():
    summary = get_doc_summary()
    return jsonify([
        {"file": name, "chunks": count}
        for name, count in sorted(summary.items())
    ])


@app.route("/documents/add", methods=["POST"])
def add_document():
    """Upload a .txt file and upsert it into the vector store.

    If the filename already exists in the index, old chunks are removed first
    so re-uploading an edited file always replaces the stale content.
    """
    global db, chain

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files are supported"}), 400

    dest = DATA_DIR / file.filename
    file.save(str(dest))
    logger.info(f"Saved '{file.filename}' to disk")

    docs   = TextLoader(str(dest), encoding="utf-8").load()
    chunks = splitter.split_documents(docs)

    if db is None:
        db = FAISS.from_documents(chunks, embeddings)
    else:
        # Remove stale chunks for this file before adding the new version
        exact_source = find_exact_source(file.filename)
        if exact_source:
            all_docs = list(db.docstore._dict.values())
            kept     = [d for d in all_docs if d.metadata.get("source") != exact_source]
            logger.info(f"Replacing existing '{file.filename}' — removed old chunks, indexing new ones")
            if kept:
                db = FAISS.from_documents(kept, embeddings)
                db.add_documents(chunks)
            else:
                db = FAISS.from_documents(chunks, embeddings)
        else:
            db.add_documents(chunks)

    db.save_local(str(VECTOR_STORE_DIR))
    chain = build_chain(db)
    logger.info(f"Upserted '{file.filename}' — {len(chunks)} chunks in index")

    return jsonify({"message": f"Indexed '{file.filename}'", "chunks": len(chunks)})


@app.route("/documents/delete/<filename>", methods=["DELETE"])
def delete_document(filename: str):
    """Remove a document from the vector store (Step 9 — DELETE)."""
    global db, chain

    if db is None:
        return jsonify({"error": "No vector store loaded"}), 404

    exact_source = find_exact_source(filename)
    if exact_source is None:
        return jsonify({"error": f"'{filename}' not found in vector store"}), 404

    all_docs = list(db.docstore._dict.values())
    kept     = [d for d in all_docs if d.metadata.get("source") != exact_source]

    if not kept:
        return jsonify({"error": "Cannot delete — this is the only document in the store"}), 400

    # Rebuild the index without the deleted document
    db = FAISS.from_documents(kept, embeddings)
    db.save_local(str(VECTOR_STORE_DIR))
    chain = build_chain(db)

    # Remove the file from disk as well
    disk_path = DATA_DIR / filename
    if disk_path.exists():
        disk_path.unlink()

    logger.info(f"Deleted '{filename}' — {len(get_doc_summary())} doc(s) remaining")
    return jsonify({"message": f"Deleted '{filename}'", "remaining": len(get_doc_summary())})


@app.route("/documents/reindex/<filename>", methods=["POST"])
def reindex_document(filename: str):
    """Re-index a file that was edited on disk (Step 9 — UPDATE)."""
    global db, chain

    disk_path = DATA_DIR / filename
    if not disk_path.exists():
        return jsonify({"error": f"'{filename}' not found in data folder"}), 404

    # Step 1: remove old chunks for this file
    if db is not None:
        exact_source = find_exact_source(filename)
        all_docs     = list(db.docstore._dict.values())
        kept = (
            [d for d in all_docs if d.metadata.get("source") != exact_source]
            if exact_source else all_docs
        )
    else:
        kept = []

    # Step 2: index the fresh version
    new_docs   = TextLoader(str(disk_path), encoding="utf-8").load()
    new_chunks = splitter.split_documents(new_docs)

    if kept:
        db = FAISS.from_documents(kept, embeddings)
        db.add_documents(new_chunks)
    else:
        db = FAISS.from_documents(new_chunks, embeddings)

    db.save_local(str(VECTOR_STORE_DIR))
    chain = build_chain(db)

    logger.info(f"Re-indexed '{filename}' — {len(new_chunks)} chunks")
    return jsonify({"message": f"Re-indexed '{filename}'", "chunks": len(new_chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    """Ask a question — uses full chat history for multi-turn conversations."""
    global chat_history

    if chain is None:
        return jsonify({"error": "No documents loaded. Upload a .txt file first."}), 503

    data     = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = chain.invoke({
        "question":     question,
        "chat_history": chat_history,
    })
    answer = response.content

    # Maintain conversation memory (same pattern as notebook Step 8)
    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])

    logger.info(f"Q: {question[:60]}...")
    return jsonify({"answer": answer, "question": question})


@app.route("/history/clear", methods=["POST"])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
