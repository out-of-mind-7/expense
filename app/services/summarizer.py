import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# 🌱 Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

# 🧠 Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# 📦 Chroma vectorstore
vectorstore = Chroma(embedding_function=embedding_model, persist_directory="chroma_index")
retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# 💬 Summarizer function
def summarize_expenses(query: str) -> str:
    # 🔍 Retrieve relevant documents
    results = retriever.get_relevant_documents(query)

    # 🧹 Deduplicate and filter invalid entries
    unique_docs = {
        doc.metadata.get("id"): doc
        for doc in results
        if doc.metadata and "date" in doc.metadata and "amount" in doc.metadata
    }.values()

    if not unique_docs:
        return "I couldn't find any matching expenses. Try a different query. 🤔"

    # 📊 Format context
    context_lines = []
    total = 0
    for doc in unique_docs:
        meta = doc.metadata
        context_lines.append(
            f"- {meta['date']}: ₹{meta['amount']} ({meta['category']}) — {meta['note']}"
        )
        total += float(meta["amount"])

    context = "\n".join(context_lines)
    context += f"\n\n💰 Total expenses: ₹{total:.2f}"

    # 🧠 Ask Gemini to summarize
    prompt = f"""Based on the following expense records, summarize the user's spending:

{context}

Provide a clear summary with category-wise breakdown and any notable patterns."""
    
    response = llm.invoke(prompt)
    return response.content