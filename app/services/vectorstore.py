import os
import logging
from chromadb import Embeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.services.db_loader import fetch_expense_documents

# 🔧 Configure logging
logging.basicConfig(level=logging.INFO)

# 🌱 Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 🔐 Validate API key
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in environment. Please check your .env file.")

# 🧠 Initialize embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# 📦 Load Chroma vectorstore
from app.services.db_loader import fetch_expense_documents

def get_vector_store():
    docs = fetch_expense_documents()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory="app/chroma_db"
    )
    return vectorstore



# 📄 Embed expense documents into Chroma
def embed_expenses(persist_dir: str = "chroma_index", collection_name: str = "default"):
    docs = fetch_expense_documents()
    if not docs:
        logging.warning("⚠️ No expense documents found to embed.")
        return

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    vectorstore.persist()
    logging.info(f"✅ Embedded {len(docs)} expense documents into Chroma.")

# 🧪 Run embedding manually
if __name__ == "__main__":
    embed_expenses()