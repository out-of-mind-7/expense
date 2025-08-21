# main.py - Updated FastAPI app with SQLAlchemy and WebSocket - FIXED FOR INDIAN RUPEES
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, validator
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import os
import json
import logging
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, Date, DateTime, Text, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.future import select
from dotenv import load_dotenv
from decimal import Decimal, InvalidOperation
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import your semantic system with error handling
try:
    from app.semantic_expense_system import SemanticChatSystem
    logger.info("✅ SemanticChatSystem imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ semantic_expense_system module not found: {e}. AI features will be limited.")
    SemanticChatSystem = None

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("❌ DATABASE_URL environment variable not set!")
    raise ValueError("DATABASE_URL is required")

DATABASE_URL_ASYNC = os.getenv("DATABASE_URL_ASYNC")

if not DATABASE_URL_ASYNC and DATABASE_URL:
    DATABASE_URL_ASYNC = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

logger.info(f"🗄️ Database URL: {DATABASE_URL_ASYNC}")

# Create async engine
async_engine = create_async_engine(DATABASE_URL_ASYNC, echo=False)
async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Database Models
Base = declarative_base()

class Expense(Base):
    __tablename__ = "expenses"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, index=True)
    amount = Column(DECIMAL(10, 2), nullable=False) 
    category = Column(String(100), nullable=False, index=True)
    note = Column(Text, nullable=True)
    
    
    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'amount': float(self.amount) if self.amount else None,
            'category': self.category,
            'note': self.note,
        }

# FIXED: Utility functions for Indian Rupee formatting
def format_inr(amount):
    """Format amount as Indian Rupees with proper comma formatting"""
    if amount is None:
        return "₹0.00"
    
    # Convert to float if it's a Decimal
    if isinstance(amount, Decimal):
        amount = float(amount)
    
    # Format with Indian number system (lakhs, crores)
    if amount >= 10000000:  # 1 crore
        crores = amount / 10000000
        return f"₹{crores:.2f} Cr"
    elif amount >= 100000:  # 1 lakh
        lakhs = amount / 100000
        return f"₹{lakhs:.2f} L"
    elif amount >= 1000:  # Thousands
        return f"₹{amount:,.2f}"
    else:
        return f"₹{amount:.2f}"

def format_inr_detailed(amount):
    """Detailed INR formatting for summaries"""
    if amount is None:
        return "₹0.00"
    
    if isinstance(amount, Decimal):
        amount = float(amount)
    
    # Always show detailed amount in parentheses for large numbers
    formatted = format_inr(amount)
    if amount >= 100000:
        detailed = f"₹{amount:,.2f}"
        return f"{formatted} ({detailed})"
    return formatted

class ExpenseCreate(BaseModel):
    date: date
    amount: str = Field(..., pattern=r'^\d+(\.\d{1,2})?$')    
    category: str = Field(..., min_length=1, max_length=50)
    note: Optional[str] = ""
    
    @validator('amount', pre=True, always=True)
    def validate_amount(cls, v):
        """Convert amount string to Decimal for precise handling"""
        if isinstance(v, str):
            try:
                # Remove any currency symbols and spaces
                clean_amount = v.replace('₹', '').replace(',', '').replace(' ', '')
                clean_amount = ''.join(c for c in clean_amount if c.isdigit() or c == '.')
                decimal_amount = Decimal(clean_amount)
                
                # Ensure positive amount
                if decimal_amount <= 0:
                    raise ValueError('Amount must be positive')
                
                # Ensure max 2 decimal places
                if decimal_amount != decimal_amount.quantize(Decimal('0.01')):
                    raise ValueError('Amount can have at most 2 decimal places')
                
                return str(decimal_amount.quantize(Decimal('0.01')))
                
            except (ValueError, InvalidOperation) as e:
                raise ValueError(f'Invalid amount format: {v}')
        
        return str(v)
    
    @validator('category')
    def validate_category(cls, v):
        """Normalize category"""
        return v.strip().lower()

class ExpenseResponse(BaseModel):
    id: int
    date: date
    amount: Decimal
    category: str
    note: str
    
    class Config:
        from_attributes = True
        json_encoders = {
            # Ensure Decimal is properly serialized
            Decimal: lambda v: float(v)
        }

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The chat message from the user")
    user_id: str = Field(..., min_length=1, description="Unique identifier for the user")

# Enhanced WebSocket Connection Manager (keeping existing code...)
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedError, ConnectionClosed
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[WebSocket, bool] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.connection_states[websocket] = True
            
            if user_id:
                self.user_connections[user_id] = websocket
            
            logger.info(f"🔗 WebSocket connected. Total: {len(self.active_connections)}")
            
            # Send welcome message with INR formatting
            welcome_message = {
                "type": "system_message",
                "data": {
                    "message": "Connected to Expense Chat! Ask me anything about your spending.",
                    "timestamp": datetime.now().isoformat(),
                    "connection_id": user_id or "anonymous"
                }
            }
            await self.send_personal_message(welcome_message, websocket)
            
        except Exception as e:
            logger.error(f"❌ Error during WebSocket connection: {e}")
            self.cleanup_connection(websocket, user_id)
            raise

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Clean disconnect method"""
        self.cleanup_connection(websocket, user_id)
        logger.info(f"🔌 WebSocket disconnected. Total: {len(self.active_connections)}")

    def cleanup_connection(self, websocket: WebSocket, user_id: str = None):
        """Internal cleanup method"""
        try:
            if websocket in self.connection_states:
                self.connection_states[websocket] = False
            
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            if user_id and user_id in self.user_connections:
                if self.user_connections[user_id] == websocket:
                    del self.user_connections[user_id]
            
            if websocket in self.connection_states:
                del self.connection_states[websocket]
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def is_connection_active(self, websocket: WebSocket) -> bool:
        """Check if connection is still active"""
        try:
            return (
                websocket in self.connection_states and 
                self.connection_states[websocket] and
                hasattr(websocket, 'client_state') and
                websocket.client_state.name == 'CONNECTED'
            )
        except:
            return False

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message with proper connection checking"""
        try:
            if websocket not in self.connection_states or not self.connection_states[websocket]:
                logger.warning("⚠️ Attempted to send to inactive connection")
                return False
            
            if hasattr(websocket, 'client_state'):
                if websocket.client_state.name != 'CONNECTED':
                    logger.warning("⚠️ WebSocket not in CONNECTED state")
                    self.cleanup_connection(websocket)
                    return False
            
            message_str = json.dumps(message, default=str)
            await websocket.send_text(message_str)
            logger.debug(f"📤 Sent message: {message.get('type', 'unknown')}")
            return True
            
        except WebSocketDisconnect:
            logger.info("🔌 WebSocket disconnected during send")
            self.cleanup_connection(websocket)
            return False
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(term in error_msg for term in ['closed', 'disconnect', 'connection']):
                logger.info(f"🔌 Connection closed during send: {e}")
            else:
                logger.error(f"❌ Error sending WebSocket message: {e}")
            
            self.cleanup_connection(websocket)
            return False

    async def send_to_user(self, message: dict, user_id: str):
        """Send message to specific user with error handling"""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            success = await self.send_personal_message(message, websocket)
            if not success:
                if user_id in self.user_connections:
                    del self.user_connections[user_id]
            return success
        return False

    async def broadcast(self, message: dict):
        """Broadcast with proper cleanup of failed connections"""
        if not self.active_connections:
            return
        
        connections_copy = self.active_connections.copy()
        failed_connections = []
        
        for connection in connections_copy:
            success = await self.send_personal_message(message, connection)
            if not success:
                failed_connections.append(connection)
        
        for connection in failed_connections:
            self.cleanup_connection(connection)
        
        logger.debug(f"📡 Broadcast sent to {len(connections_copy) - len(failed_connections)}/{len(connections_copy)} connections")

# Global variables
manager = ConnectionManager()
semantic_chat_system = None

# Database dependency
async def get_db():
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"❌ Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

# FIXED: Updated helper functions with INR formatting
async def process_chat_query(query: str, session: AsyncSession, user_id: str = None) -> Dict[str, Any]:
    """Process chat query using semantic AI system with INR formatting"""
    global semantic_chat_system
    
    try:
        logger.info(f"🤖 Processing query: {query[:50]}..." if len(query) > 50 else f"🤖 Processing query: {query}")
        
        if semantic_chat_system:
            response = await semantic_chat_system.process_chat_query(query, session)
            # Ensure response uses INR formatting
            response = ensure_inr_formatting(response)
            return {
                "message": response,
                "source": "ai",
                "enhanced": True
            }
        else:
            response_data = await handle_basic_queries(query, session)
            return {
                "message": response_data["message"],
                "source": "basic",
                "enhanced": False,
                "suggestion": "Connect Google API key for advanced AI features"
            }
                
    except Exception as e:
        logger.error(f"❌ Error processing chat query: {e}")
        return {
            "message": f"I encountered an error processing your request: {str(e)} 😅\n\nPlease try rephrasing your question or check the logs for more details.",
            "source": "error",
            "error": str(e)
        }

def ensure_inr_formatting(text: str) -> str:
    """Ensure all currency mentions use Indian Rupees"""
    import re
    
    # Replace dollar signs with rupee symbols
    text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', r'₹\1', text)
    
    # Replace USD mentions with INR
    text = text.replace('USD', 'INR').replace('dollars', 'rupees').replace('$', '₹')
    
    return text

async def handle_basic_queries(query: str, session: AsyncSession) -> Dict[str, Any]:
    """Enhanced basic query handling with INR formatting"""
    query_lower = query.lower()
    
    expense_keywords = ['expense', 'spend', 'spent', 'money', 'cost', 'budget', 'total', 'category', 'month', 'summary']
    add_keywords = ['add', 'create', 'new', 'record']
    help_keywords = ['help', 'what', 'how', 'commands']
    
    try:
        if any(keyword in query_lower for keyword in add_keywords) and any(keyword in query_lower for keyword in expense_keywords):
            return {
                "message": "💰 To add an expense, use the format:\n'Add ₹50 for groceries in food category'\n\nOr use the API endpoint: POST /api/expenses"
            }
        elif any(keyword in query_lower for keyword in expense_keywords):
            summary = await get_enhanced_expense_summary(session)
            return {
                "message": f"💰 Here's your expense summary:\n\n{summary}\n\n💡 Try asking: 'Show me this month's spending' or 'What's my biggest expense category?'"
            }
        elif any(keyword in query_lower for keyword in help_keywords):
            return {
                "message": """🤖 **Expense Chat Commands:**

• Ask about spending: "How much did I spend this month?"
• Category analysis: "Show me my food expenses"  
• Add expenses: "Add ₹50 for groceries"
• Get summaries: "What's my spending summary?"

📊 **Available via API:**
• GET /api/expenses - View all expenses
• POST /api/expenses - Add new expense
• GET /api/summary - Get spending summary

🔍 **Pro tip:** Connect Google API key for advanced AI features!
💰 **Currency:** All amounts are in Indian Rupees (₹)"""
            }
        else:
            return {
                "message": f"I received: '{query}'\n\n🤖 I can help you track expenses, analyze spending patterns, or provide monthly summaries in Indian Rupees (₹)!\n\nTry asking: 'What did I spend this month?' or type 'help' for more options."
            }
            
    except Exception as e:
        logger.error(f"❌ Error in basic query handling: {e}")
        return {
            "message": f"I had trouble processing that request: {str(e)}"
        }

async def get_enhanced_expense_summary(session: AsyncSession) -> str:
    """Get enhanced expense summary with INR formatting"""
    try:
        # Current month total
        current_month_query = select(func.sum(Expense.amount)).where(
            Expense.date >= func.date_trunc('month', func.current_date())
        )
        result = await session.execute(current_month_query)
        month_total = result.scalar() or 0
        
        # Previous month for comparison
        prev_month_query = select(func.sum(Expense.amount)).where(
            Expense.date >= func.date_trunc('month', func.current_date() - func.interval('1 month')),
            Expense.date < func.date_trunc('month', func.current_date())
        )
        prev_result = await session.execute(prev_month_query)
        prev_month_total = prev_result.scalar() or 0
        
        # Current month by category
        category_query = select(
            Expense.category,
            func.sum(Expense.amount).label('total'),
            func.count(Expense.id).label('count')
        ).where(
            Expense.date >= func.date_trunc('month', func.current_date())
        ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc()).limit(5)
        
        category_result = await session.execute(category_query)
        categories = category_result.all()
        
        # Recent expenses
        recent_query = select(Expense).order_by(
            Expense.date.desc(),
        ).limit(3)
        recent_result = await session.execute(recent_query)
        recent_expenses = recent_result.scalars().all()
        
        if month_total == 0:
            return "📊 **No expenses recorded this month yet!**\n\n💡 Start tracking by adding your first expense."
        
        # Build summary with proper INR formatting
        summary = f"📊 **This Month: {format_inr_detailed(month_total)}**\n"
        
        # Month comparison
        if prev_month_total > 0:
            change = ((month_total - prev_month_total) / prev_month_total) * 100
            trend = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            summary += f"{trend} {change:+.1f}% vs last month ({format_inr_detailed(prev_month_total)})\n"
        
        summary += "\n**🏷️ Top Categories:**\n"
        for cat in categories:
            percentage = (cat.total / month_total * 100) if month_total > 0 else 0
            summary += f"• {cat.category.title()}: {format_inr_detailed(cat.total)} ({percentage:.1f}%) - {cat.count} transactions\n"
        
        if recent_expenses:
            summary += "\n**🕒 Recent Expenses:**\n"
            for exp in recent_expenses:
                summary += f"• {format_inr(exp.amount)} - {exp.category} ({exp.date})\n"
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"❌ Error getting expense summary: {e}")
        return f"Unable to fetch expense data: {str(e)}"




# Startup and shutdown functions
async def init_semantic_chat_system():
    """Initialize semantic AI system"""
    global semantic_chat_system
    
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("⚠️ No Google API key found - AI features will be limited")
            return
        
        if SemanticChatSystem is None:
            logger.warning("⚠️ SemanticChatSystem not available")
            return
            
        semantic_chat_system = SemanticChatSystem(api_key)
        logger.info("🤖 Semantic AI system initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Semantic AI system initialization failed: {e}")

async def verify_database():
    """Verify database connection and existing data"""
    try:
        async with async_session() as session:
            await session.execute(select(1))
            
            result = await session.execute(select(func.count(Expense.id)))
            count = result.scalar()
            logger.info(f"✅ Database connected. Found {count} existing expenses.")
            
    except Exception as e:
        logger.error(f"❌ Database verification failed: {e}")
        raise

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up application...")
    try:
        await verify_database()
        await init_semantic_chat_system()
        logger.info("✅ Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    logger.info("🔌 Shutting down application...")

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Expense Chatbot API (INR)",
    description="AI chatbot with SQLAlchemy, WebSocket support for expense tracking",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000", 
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# API Routes
@app.get("/")
async def root():
    return {
        "message": "🚀 Semantic Expense Chatbot API is running! (INR Edition)",
        "version": "2.1.0",
        "currency": "Indian Rupees (₹)",
        "status": "healthy",
        "features": ["SQLAlchemy", "WebSocket", "AI Chat", "Expense Tracking", "INR Support"],
        "endpoints": {
            "websocket": "/ws",
            "user_websocket": "/ws/{user_id}",
            "api": "/api/expenses",
            "chat": "/api/chat",
            "docs": "/docs",
            "health": "/health"
        },
        "websocket_connections": len(manager.active_connections)
    }

@app.post("/api/expenses", response_model=dict)
async def create_expense(expense: ExpenseCreate, session: AsyncSession = Depends(get_db)):
    """Create expense with precise decimal handling in INR"""
    try:
        decimal_amount = Decimal(expense.amount).quantize(Decimal('0.01'))
        
        db_expense = Expense(
            date=expense.date,
            amount=decimal_amount,
            category=expense.category,
            note=expense.note or ""
        )
        
        session.add(db_expense)
        await session.commit()
        await session.refresh(db_expense)
        
        logger.info(f"✅ Saved expense: {format_inr(db_expense.amount)}")
        
        return {
            "success": True,
            "id": db_expense.id,
            "message": f"💰 Expense added: {format_inr(float(db_expense.amount))} for {db_expense.category}",
            "data": db_expense.to_dict()
        }
        
    except Exception as e:
        await session.rollback()
        logger.error(f"❌ Error creating expense: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create expense: {str(e)}")
    
@app.get("/api/expenses")
async def get_expenses(
    limit: int = 50, 
    offset: int = 0, 
    category: Optional[str] = None,
    session: AsyncSession = Depends(get_db)
):
    """Get expenses with pagination and filtering - INR formatted"""
    try:
        query = select(Expense).order_by(Expense.date.desc())
        
        if category:
            query = query.where(Expense.category.ilike(f"%{category}%"))
            
        query = query.offset(offset).limit(limit)
        
        result = await session.execute(query)
        expenses = result.scalars().all()
        
        return {
            "success": True,
            "expenses": [
                {
                    "id": exp.id,
                    "date": exp.date.isoformat(),
                    "amount": float(exp.amount),
                    "amount_formatted": format_inr(exp.amount),
                    "category": exp.category,
                    "note": exp.note or "",
                }
                for exp in expenses
            ],
            "count": len(expenses),
            "limit": limit,
            "offset": offset,
            "filter": {"category": category} if category else None,
            "currency": "INR"
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching expenses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/summary")
async def get_summary(session: AsyncSession = Depends(get_db)):
    """Get expense summary with INR formatting"""
    try:
        summary = await get_enhanced_expense_summary(session)
        return {
            "success": True,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "currency": "INR"
        }
    except Exception as e:
        logger.error(f"❌ Error fetching summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
async def get_categories(session: AsyncSession = Depends(get_db)):
    """Get all expense categories with INR formatting"""
    try:
        query = select(
            Expense.category, 
            func.count(Expense.id).label('count'),
            func.sum(Expense.amount).label('total')
        ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc())
        
        result = await session.execute(query)
        categories = result.all()
        
        return {
            "success": True,
            "categories": [
                {
                    "name": cat.category,
                    "count": cat.count,
                    "total": float(cat.total),
                    "total_formatted": format_inr(cat.total)
                }
                for cat in categories
            ],
            "currency": "INR"
        }
    except Exception as e:
        logger.error(f"❌ Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage, session: AsyncSession = Depends(get_db)):
    """Alternative REST endpoint for chat with INR support"""
    try:
        response_data = await process_chat_query(message.message, session, message.user_id)
        return {
            "success": True,
            "response": response_data["message"],
            "query": message.message,
            "timestamp": datetime.now().isoformat(),
            "source": response_data.get("source", "unknown"),
            "enhanced": response_data.get("enhanced", False),
            "currency": "INR"
        }
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints (keeping existing implementation but ensuring INR formatting in responses)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint with INR support"""
    user_id = None
    
    try:
        await manager.connect(websocket)
        logger.info("🔗 New WebSocket connection established")
        
        while True:
            try:
                if not manager.is_connection_active(websocket):
                    logger.info("🔌 Connection no longer active, breaking loop")
                    break
                
                try:
                    data = await websocket.receive_text()
                    logger.debug(f"📥 Received WebSocket data: {data[:100]}...")
                except (ConnectionClosedError, WebSocketDisconnect):
                    logger.info("🔌 Connection closed by client")
                    break
                
                try:
                    message = json.loads(data)
                    if not isinstance(message, dict):
                        raise ValueError("Message must be a JSON object")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": "Invalid JSON format. Please send properly formatted JSON.",
                            "error": "json_decode_error",
                            "timestamp": datetime.now().isoformat()
                        }
                    }, websocket)
                    continue
                
                message_type = message.get("type", "chat_message")
                message_data = message.get("data", {})
                
                logger.info(f"🔄 Processing WebSocket message type: {message_type}")
                
                if message_type == "chat_message":
                    user_message = message_data.get("message", "").strip()
                    user_id = message_data.get("user_id")
                    
                    if not user_message:
                        await manager.send_personal_message({
                            "type": "error",
                            "data": {"message": "Empty message received"}
                        }, websocket)
                        continue
                    
                    async with async_session() as session:
                        response_data = await process_chat_query(user_message, session, user_id)
                    
                    # Send AI response with INR formatting
                    response_message = {
                        "type": "ai_response",
                        "data": {
                            "message": response_data["message"],
                            "original_query": user_message,
                            "timestamp": datetime.now().isoformat(),
                            "user_id": user_id,
                            "source": response_data.get("source", "unknown"),
                            "enhanced": response_data.get("enhanced", False),
                            "currency": "INR"
                        }
                    }
                    
                    await manager.send_personal_message(response_message, websocket)
                
                elif message_type == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                            "connections": len(manager.active_connections),
                            "currency": "INR"
                        }
                    }, websocket)
                
                elif message_type == "user_info":
                    user_id = message_data.get("user_id")
                    if user_id and user_id not in manager.user_connections:
                        manager.user_connections[user_id] = websocket
                    
                    await manager.send_personal_message({
                        "type": "user_info_received",
                        "data": {
                            "user_id": user_id,
                            "timestamp": datetime.now().isoformat(),
                            "currency": "INR"
                        }
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": f"Unknown message type: {message_type}",
                            "supported_types": ["chat_message", "ping", "user_info"]
                        }
                    }, websocket)
                
            except (WebSocketDisconnect, ConnectionClosedError):
                logger.info("🔌 WebSocket disconnect in message loop")
                break
                
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket message: {e}")
                try:
                    await manager.send_personal_message({
                        "type": "error",
                        "data": {
                            "message": f"Error processing message: {str(e)}",
                            "error": "processing_error",
                            "timestamp": datetime.now().isoformat()
                        }
                    }, websocket)
                except:
                    logger.error("❌ Could not send error message, connection likely dead")
                    break
                
    except (WebSocketDisconnect, ConnectionClosedError):
        logger.info("🔌 Client disconnected normally")
        
    except Exception as e:
        logger.error(f"❌ Unexpected WebSocket error: {e}")
        
    finally:
        manager.disconnect(websocket, user_id)

@app.websocket("/ws/{user_id}")
async def websocket_user_endpoint(websocket: WebSocket, user_id: str):
    """User-specific WebSocket endpoint with INR support"""
    try:
        await manager.connect(websocket, user_id)
        logger.info(f"🔗 User-specific WebSocket connected: {user_id}")
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "chat_message":
                user_message = message["data"]["message"]
                
                async with async_session() as session:
                    response_data = await process_chat_query(user_message, session, user_id)
                
                await manager.send_to_user({
                    "type": "ai_response", 
                    "data": {
                        "message": response_data["message"],
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "source": response_data.get("source", "unknown"),
                        "currency": "INR"
                    }
                }, user_id)
                
    except (WebSocketDisconnect, ConnectionClosedError):
        manager.disconnect(websocket, user_id)
        logger.info(f"🔌 User {user_id} disconnected")

# Health check with INR info
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint with INR support"""
    try:
        async with async_session() as session:
            await session.execute(select(1))
        
        ai_status = "available" if semantic_chat_system else "limited"
        
        return {
            "status": "healthy",
            "database": "connected",
            "ai_system": ai_status,
            "websocket_connections": len(manager.active_connections),
            "version": "2.1.0",
            "currency": "Indian Rupees (₹)",
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": os.sys.version,
                "database_url_set": bool(DATABASE_URL),
                "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY"))
            }
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "currency": "Indian Rupees (₹)"
        }

# Debug endpoint for development
@app.get("/debug/websocket-status")
async def websocket_debug():
    """Debug endpoint to check WebSocket status"""
    return {
        "active_connections": len(manager.active_connections),
        "user_connections": list(manager.user_connections.keys()),
        "timestamp": datetime.now().isoformat(),
        "currency": "INR"
    }

# Additional endpoint to get expense statistics in INR
@app.get("/api/stats")
async def get_expense_stats(session: AsyncSession = Depends(get_db)):
    """Get detailed expense statistics in INR"""
    try:
        # Total expenses
        total_query = select(func.sum(Expense.amount), func.count(Expense.id))
        total_result = await session.execute(total_query)
        total_amount, total_count = total_result.first()
        
        # Monthly totals
        monthly_query = select(
            func.date_trunc('month', Expense.date).label('month'),
            func.sum(Expense.amount).label('total')
        ).group_by(func.date_trunc('month', Expense.date)).order_by(func.date_trunc('month', Expense.date).desc()).limit(12)
        
        monthly_result = await session.execute(monthly_query)
        monthly_data = monthly_result.all()
        
        # Average per category
        avg_query = select(
            Expense.category,
            func.avg(Expense.amount).label('avg_amount'),
            func.sum(Expense.amount).label('total_amount'),
            func.count(Expense.id).label('count')
        ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc())
        
        avg_result = await session.execute(avg_query)
        category_stats = avg_result.all()
        
        return {
            "success": True,
            "currency": "INR",
            "total_stats": {
                "total_amount": float(total_amount or 0),
                "total_amount_formatted": format_inr_detailed(total_amount or 0),
                "total_transactions": total_count or 0,
                "average_per_transaction": float((total_amount or 0) / (total_count or 1)),
                "average_per_transaction_formatted": format_inr((total_amount or 0) / (total_count or 1))
            },
            "monthly_breakdown": [
                {
                    "month": month.strftime("%Y-%m") if month else "Unknown",
                    "total": float(total),
                    "total_formatted": format_inr_detailed(total)
                }
                for month, total in monthly_data
            ],
            "category_stats": [
                {
                    "category": cat.category,
                    "average": float(cat.avg_amount),
                    "average_formatted": format_inr(cat.avg_amount),
                    "total": float(cat.total_amount),
                    "total_formatted": format_inr_detailed(cat.total_amount),
                    "count": cat.count
                }
                for cat in category_stats
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"🚀 Starting server on {host}:{port} with INR support")
    
    uvicorn.run(
        "main:app", 
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )