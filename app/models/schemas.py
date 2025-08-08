from pydantic import BaseModel

# 📮 For summarization endpoint
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

# 🧾 For adding a new expense
class ExpenseEntry(BaseModel):
    date: str           # e.g. "2025-07-10"
    category: str       # e.g. "Groceries"
    amount: float       # e.g. 85.0
    description: str    # e.g. "Fruits and vegetables"

class AddResponse(BaseModel):
    message: str
