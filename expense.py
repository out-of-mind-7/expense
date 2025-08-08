from sqlalchemy import Column, Integer, Date, Text, Numeric , String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Expense(Base):
    __tablename__ = "expenses"
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    category = Column(String(50))    
    amount = Column(Numeric)
    note = Column(Text)
