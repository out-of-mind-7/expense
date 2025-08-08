from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from app.services.summarizer import summarize_expenses
from app.api import router
from fastapi.routing import APIRoute, APIWebSocketRoute
import logging

from extractor import extract_filters_with_gemini


# 🌱 Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 🚀 Initialize FastAPI app
app = FastAPI(title="Expense Tracker RAG API")

# 🌐 Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 🛠 Include API routes
app.include_router(router, prefix="/api")

from app.services.rag_chatbot import get_rag_response
# ...
@app.post("/query-expenses")
def query_expenses(user_query: str):
    filters = extract_filters_with_gemini(user_query)
    # Use filters["category"], filters["start_date"], filters["end_date"] to query DB

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("✅ WebSocket connection established.")

    try:
        while True:
            query = await websocket.receive_text()
            response = get_rag_response(query)
            await websocket.send_text(response)

    except WebSocketDisconnect:
        logging.info("🔌 Client disconnected from WebSocket.")
    except Exception as e:
        logging.error(f"❌ WebSocket error: {e}")
        await websocket.send_text(f"Error: {str(e)}")
    
    
    
# 🧾 Route diagnostics
for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"{route.path} → {route.methods}")
    elif isinstance(route, APIWebSocketRoute):
        print(f"{route.path} → WebSocket")