import os
from langchain_google_genai import ChatGoogleGenerativeAI

from app.services.rag_chatbot import normalize_date
api_key = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)



def extract_filters_with_gemini(query: str) -> dict:
    prompt = f"""
You are a smart assistant. Extract filters from the user's query related to expenses.

Query: "{query}"

Return a JSON object with keys:
- category (string or null)
- start_date (string or null, can be fuzzy like "last week")
- end_date (string or null, can be fuzzy like "today")

Only include values if they are clearly mentioned.
"""

    response = llm.invoke(prompt)
    try:
        import json
        raw_filters = json.loads(response.content)

        # Normalize fuzzy dates
        return {
            "category": raw_filters.get("category"),
            "start_date": normalize_date(raw_filters.get("start_date")),
            "end_date": normalize_date(raw_filters.get("end_date"))
        }
    except Exception:
        return {"category": None, "start_date": None, "end_date": None}