from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base  # ✅ Make sure this matches your file structure

DATABASE_URL = "postgresql://postgres:root@localhost:5432/expense_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
