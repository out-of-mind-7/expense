# filter_utils.py

from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

def extract_filters_with_gemini(query: str) -> dict:
    prompt = f"""
You are a smart assistant. Extract filters from the user's query related to expenses.

Query: "{query}"

Return a JSON object with keys:
- category (string or null)
- start_date (YYYY-MM-DD or null)
- end_date (YYYY-MM-DD or null)

Only include values if they are clearly mentioned.
"""
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except Exception:
        return {"category": None, "start_date": None, "end_date": None}