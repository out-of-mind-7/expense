from fastapi import APIRouter
from langchain.schema import Document
from app.models.schemas import ExpenseEntry, AddResponse, QueryRequest, QueryResponse
from app.services.summarizer import summarize_expenses, vectorstore
from db import SessionLocal
from expense import Expense

router = APIRouter()

# 🔍 Summarize expenses
@router.post("/summarize", response_model=QueryResponse)
def summarize(request: QueryRequest):
    result = summarize_expenses(request.query)
    return QueryResponse(result=result)

# ➕ Add expense to DB and embed
@router.post("/add-expense", response_model=AddResponse)
def add_expense(entry: ExpenseEntry):
    session = SessionLocal()
    try:
        # 1. Save to DB
        expense = Expense(
            date=entry.date,
            category=entry.category,
            amount=entry.amount,
            note=entry.description
        )
        session.add(expense)
        session.commit()
        session.refresh(expense)

        # 2. Embed with metadata
        doc = Document(
            page_content=f"{expense.date} {expense.category} ₹{expense.amount} {expense.note}",
            metadata={
                "id": expense.id,
                "date": expense.date,
                "category": expense.category,
                "amount": float(expense.amount),
                "note": expense.note
                }
            )
        vectorstore.add_documents([doc])

        return AddResponse(message="✅ Expense embedded and saved to DB successfully.")
    except Exception as e:
        session.rollback()
        return AddResponse(message=f"❌ Failed to add expense: {str(e)}")
    finally:
        session.close()

@router.get("/ping")
def ping():
    return {"message": "pong"}