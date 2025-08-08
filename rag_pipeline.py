from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# 🧠 Initialize models
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# 📦 Load Chroma index
persist_dir = "chroma_index"
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_dir
)

# 🔍 RAG chain setup
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 💬 Query
query = "Give a summary of my grocery expenses including total spent and frequency"
response = qa_chain.run(query)

# 🧠 Output
print("🧠 Answer:\n", response)
print("✅ Indexed documents:", vectorstore._collection.count())