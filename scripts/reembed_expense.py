import os
import shutil
from dotenv import load_dotenv
from db import SessionLocal
from expense import Expense
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

import expense

# 🌱 Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# 🧠 Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# 🧹 Clear old Chroma index
persist_dir = "chroma_index"
shutil.rmtree(persist_dir, ignore_errors=True)

# 🧾 Load expenses from DB
session = SessionLocal()
docs = []
for exp in session.query(Expense).all():
    doc = Document(
        page_content=f"{exp.date} {exp.category} ₹{float(exp.amount)} {exp.note}",
        metadata={
            "id": exp.id,
            "date": str(exp.date),  # ✅ convert to string
            "category": exp.category,
            "amount": float(exp.amount),
            "note": exp.note
        }
    )
    docs.append(doc)
session.close()

# 💾 Save to Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

print("✅ Re-embedding complete.")
print("📦 Total embedded documents:", vectorstore._collection.count())