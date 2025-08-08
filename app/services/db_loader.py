from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain.schema import Document
from dotenv import load_dotenv
import os

from expense import Expense  # Your SQLAlchemy model

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Create engine and session
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def fetch_expense_documents():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, date, category, amount, note FROM expenses"))
        docs = []

        for row in result:
            expense_id, date, category, amount, note = row
            content = (
                f"On {date}, ₹{amount} was spent on {category}. "
                f"Note: {note}"
            )
            docs.append(Document(page_content=content, metadata={"id": expense_id}))
    
    return docs

def get_expenses(session, category=None, start_date=None, end_date=None):
    query = session.query(Expense)

    if category:
        query = query.filter(Expense.category.ilike(f"%{category}%"))
    if start_date:
        query = query.filter(Expense.date >= start_date)
    if end_date:
        query = query.filter(Expense.date <= end_date)

    return query.all()