import os
import shutil
from dotenv import load_dotenv
from db import SessionLocal
from expense import Expense
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai

# 🌱 Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=api_key)

# 🧠 Initialize models
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# 🧹 Clear Chroma index completely
persist_dir = "chroma_index"
for folder in [persist_dir, ".chroma"]:
    shutil.rmtree(folder, ignore_errors=True)

# 🗃️ Load DB entries
session = SessionLocal()
expenses = session.query(Expense).all()
session.close()

docs = []
for exp in expenses:
    print(f"Embedding: {exp.id} | {exp.date} | ₹{exp.amount} | {exp.category} | {exp.note}")
    doc = Document(
        page_content=f"₹{exp.amount} spent on {exp.category} on {exp.date}. Note: {exp.note}",
        metadata={
            "id": exp.id,
            "date": str(exp.date),
            "category": exp.category,
            "amount": float(exp.amount),
            "note": exp.note
        }
    )
    docs.append(doc)

# 💾 Save to Chroma
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)
print("✅ Embeddings saved to Chroma.")
print("📦 Stored documents (should match DB count):", vectorstore._collection.count())

# 🔍 Semantic search
query = "expenses in July"
results = vectorstore.similarity_search(query, k=10)

# 🧹 Deduplicate by ID + date
unique_docs = {
    f"{doc.metadata['id']}-{doc.metadata['date']}": doc
    for doc in results
}.values()

# 📊 Format summary
total = 0
summary_lines = []
for doc in unique_docs:
    meta = doc.metadata
    summary_lines.append(
        f"- {meta['date']}: ₹{meta['amount']} ({meta['category']}) — {meta['note']}"
    )
    total += meta["amount"]

summary_text = "\n".join(summary_lines)
summary_text += f"\n\n💰 Total expenses: ₹{total:.2f}"

print("\n🧾 Expense Summary:")
print(summary_text)

# 🧠 Gemini-powered summarization
use_llm_summary = True
if use_llm_summary:
    prompt = f"""Based on the following expense records, summarize the user's spending:

{summary_text}

Highlight category-wise totals and any patterns."""
    response = llm.invoke(prompt)
    print("\n🧠 Gemini Summary:")
    print(response.content)