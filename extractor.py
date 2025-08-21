
# extractor.py - Enhanced filter extraction
import os
import json
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
# from app.services.rag_chatbot import normalize_date

logger = logging.getLogger(__name__)

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.1
)

def extract_filters_with_gemini(query: str) -> Dict[str, Any]:
    """Extract expense filters from natural language query"""
    prompt = f"""
    You are an AI assistant that extracts expense filters from natural language queries.
    
    Query: "{query}"
    
    Extract and return a JSON object with these possible filters:
    - category: string (food, transport, groceries, entertainment, shopping, bills, medical, etc.) or null
    - start_date: string in fuzzy format (like "last week", "this month", "January", specific date) or null
    - end_date: string in fuzzy format or null
    - min_amount: number or null
    - max_amount: number or null
    
    Examples:
    - "food expenses last month" → {{"category": "food", "start_date": "last month", "end_date": null}}
    - "expenses over 500 in January" → {{"min_amount": 500, "start_date": "January", "end_date": null}}
    - "transport costs this week" → {{"category": "transport", "start_date": "this week", "end_date": null}}
    
    Return only valid JSON, no explanation:
    """
    
    try:
        response = llm.invoke(prompt)
        raw_filters = json.loads(response.content.strip())
        
        # Normalize the filters
        normalized_filters = {}
        
        if raw_filters.get("category"):
            normalized_filters["category"] = raw_filters["category"].lower().strip()
        
        if raw_filters.get("start_date"):
            normalized_date = normalize_date(raw_filters["start_date"])
            if normalized_date:
                normalized_filters["start_date"] = normalized_date
        
        if raw_filters.get("end_date"):
            normalized_date = normalize_date(raw_filters["end_date"])
            if normalized_date:
                normalized_filters["end_date"] = normalized_date
        
        if raw_filters.get("min_amount") and isinstance(raw_filters["min_amount"], (int, float)):
            normalized_filters["min_amount"] = float(raw_filters["min_amount"])
        
        if raw_filters.get("max_amount") and isinstance(raw_filters["max_amount"], (int, float)):
            normalized_filters["max_amount"] = float(raw_filters["max_amount"])
        
        logger.info(f"Extracted filters: {normalized_filters}")
        return normalized_filters
        
    except Exception as e:
        logger.error(f"Filter extraction error: {e}")
        return {}