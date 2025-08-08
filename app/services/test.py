from db_loader import get_expenses, SessionLocal

with SessionLocal() as session:
    expenses = get_expenses(session, "Groceries", "2025-07-01", "2025-07-31")
    for e in expenses:
        print(e.date, e.category, e.amount, e.description)