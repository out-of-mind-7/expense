import os
from dotenv import load_dotenv
from datetime import datetime
import difflib
from collections import defaultdict
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma  # ✅ Updated import
from app.services.db_loader import fetch_expense_documents
from filter_utils import extract_filters_with_gemini
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

# 🌱 Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 🔐 Validate API key
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in environment.")

# 🧠 Initialize embedding model and Gemini LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)


import dateparser

def normalize_date(date_str):
    parsed = dateparser.parse(date_str)
    return parsed.strftime("%Y-%m-%d") if parsed else None

# 🔍 Expense intent detection
def is_expense_query(query: str) -> bool:
    keywords = [
        "expense", "spending", "money", "cost", "purchase",
        "groceries", "grocery", "food", "snacks", "milk",
        "transport", "shopping", "bill", "paid", "buy"
    ]
    return any(word in query.lower() for word in keywords)

def is_summary_query(query: str) -> bool:
    keywords = ["summarize", "summary", "total expenses", "all expenses", "overview"]
    return any(word in query.lower() for word in keywords)
from app.services.db_loader import get_expenses, SessionLocal

from collections import defaultdict

from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict

def summarize_by_category(expenses):
    grouped = defaultdict(Decimal)
    for e in expenses:
        grouped[e.category] += e.amount  # ✅ Keep everything as Decimal

    lines = [f"- {cat}: ₹{grouped[cat].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}" for cat in grouped]
    total = sum(grouped.values())
    lines.append(f"\n💰 Total expenses: ₹{total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
    return "\n".join(lines)

def summarize_all_expenses(filters=None, group_by_category=False):
    with SessionLocal() as session:
        expenses = get_expenses(session)

        filtered = []
        for e in expenses:
            include = True

            if filters:
                # Normalize and match category
                if filters.get("category"):
                    query_cat = filters["category"].lower().rstrip("s")
                    expense_cat = e.category.lower().rstrip("s")
                    if not is_similar(query_cat, expense_cat):
                        include = False

                # Date range filter
                if filters.get("start_date") and filters.get("end_date"):
                    e_date = datetime.strptime(e.date, "%Y-%m-%d")
                    start = datetime.strptime(filters["start_date"], "%Y-%m-%d")
                    end = datetime.strptime(filters["end_date"], "%Y-%m-%d")
                    if not (start <= e_date <= end):
                        include = False

            if include:
                filtered.append(e)

        if not filtered:
            return "🤖 No matching expenses found for your filters."

        if group_by_category:
            return summarize_by_category(filtered)

        # Default summary
        total = sum(e.amount for e in filtered)
        lines = [
            f"- {e.date}: ₹{e.amount:.2f} ({e.category}) — {e.note}"
            for e in filtered
        ]
        lines.append(f"\n💰 Total expenses: ₹{total:.2f}")
        return "\n".join(lines)
    
    
def is_category_summary_query(query: str) -> bool:
    keywords = ["category-wise", "group by category", "expenses by category", "grouped by category"]
    return any(word in query.lower() for word in keywords)
    
    

# 🔁 Fuzzy category matching
def is_similar(a, b, threshold=0.7):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# 📦 Load vector store from documents
def get_vector_store():
    docs = fetch_expense_documents()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="app/chroma_db"
    )
    return vectorstore

def format_expense_context(docs, filters=None) -> str:
    # Deduplicate by ID + date
    unique_docs = {
        f"{doc.metadata['id']}-{doc.metadata['date']}": doc
        for doc in docs
    }.values()

    filtered_docs = []
    for doc in unique_docs:
        meta = doc.metadata
        include = True

    if filters:
        # ✅ Safe category filtering
        if filters.get("category"):
            query_cat = filters["category"].lower()
            doc_cat = meta["category"].lower()
            if not is_similar(query_cat, doc_cat):
                include = False

        # ✅ Date filtering
        if filters.get("start_date") and filters.get("end_date"):
            doc_date = datetime.strptime(meta["date"], "%Y-%m-%d")
            start = datetime.strptime(filters["start_date"], "%Y-%m-%d")
            end = datetime.strptime(filters["end_date"], "%Y-%m-%d")
            if not (start <= doc_date <= end):
                include = False

    if include:
        filtered_docs.append(doc)

    # 🧾 Format summary
    if not filtered_docs:
        return "🤖 No matching expenses found for your filters."

    total = 0
    summary_lines = []
    for doc in filtered_docs:
        meta = doc.metadata
        summary_lines.append(
            f"- {meta['date']}: ₹{meta['amount']} ({meta['category']}) — {meta['note']}"
        )
        total += meta["amount"]

    summary_text = "\n".join(summary_lines)
    summary_text += f"\n\n💰 Total expenses: ₹{total:.2f}"
    return summary_text


# 🧠 Main chatbot response function
def get_rag_response(query: str) -> str:
    filters = extract_filters_with_gemini(query)

    if is_category_summary_query(query):
        with SessionLocal() as session:
            expenses = get_expenses(session)
            # Apply filters manually like in summarize_all_expenses
            filtered = []
            for e in expenses:
                include = True
                if filters:
                    if filters.get("category") and not is_similar(filters["category"], e.category):
                        include = False
                    if filters.get("start_date") and filters.get("end_date"):
                        e_date = datetime.strptime(e.date, "%Y-%m-%d")
                        start = datetime.strptime(filters["start_date"], "%Y-%m-%d")
                        end = datetime.strptime(filters["end_date"], "%Y-%m-%d")
                        if not (start <= e_date <= end):
                            include = False
                if include:
                    filtered.append(e)

            if not filtered:
                return "🤖 No matching expenses found for your filters."

            return summarize_by_category(filtered)

    elif is_summary_query(query):
        return summarize_all_expenses(filters)

    elif is_expense_query(query):
        vectorstore = Chroma(
            persist_directory="chroma_index",
            embedding_function=embeddings
        )
        results = vectorstore.similarity_search(query, k=50)

        if not results:
            return "🤖 I couldn't find any relevant expense records for your query."

        return format_expense_context(results, filters)

    else:
        response = llm.invoke(query)
        return response.content