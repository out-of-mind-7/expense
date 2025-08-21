# semantic_expense_system.py - Pure Semantic Search (No Keywords)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, extract
import logging
import os
from typing import List, Tuple, Dict
from datetime import datetime, date

logger = logging.getLogger(__name__)

# Import your models
try:
    from main import Expense
except ImportError:
    from sqlalchemy import Column, Integer, String, DECIMAL, Date, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    
    Base = declarative_base()
    
    class Expense(Base):
        __tablename__ = "expenses"
        id = Column(Integer, primary_key=True)
        date = Column(Date, nullable=False)
        amount = Column(DECIMAL(10, 2), nullable=False)
        category = Column(String(100), nullable=False)
        note = Column(Text, nullable=True)
        created_at = Column(DateTime, nullable=True)

class SemanticChatSystem:
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Define semantic concepts for expense-related queries
        self.expense_concepts = {
            "current_spending": "current month spending this month expenses now present",
            "past_spending": "last month previous month past spending historical expenses",
            "comparison": "compare comparison versus vs difference change trend",
            "food_expenses": "food restaurant dining eating meals groceries cooking",
            "transport_expenses": "transport transportation travel gas fuel car uber taxi bus",
            "shopping_expenses": "shopping clothes clothing retail store purchase buy",
            "entertainment": "entertainment movies cinema games fun leisure activities",
            "bills_utilities": "bills utilities electricity water internet phone rent",
            "highest_expenses": "highest biggest largest most expensive maximum top greatest",
            "spending_patterns": "patterns trends behavior analysis habits spending behavior",
            "total_amount": "total sum amount how much spent overall aggregate",
            "categories": "categories breakdown classification types groups",
            "recent_expenses": "recent latest new last few recent transactions",
            "budget_advice": "budget advice tips suggestions help manage financial planning",
            "savings": "savings save money reduce expenses cut costs economy",
            "weekly_spending": "weekly week this week last week seven days",
            "daily_spending": "daily day today yesterday per day average daily"
        }
        
        # Cache for concept embeddings
        self.concept_embeddings = {}
        self.expense_threshold = 0.25  # Lower threshold for more sensitive detection
        self.intent_threshold = 0.35   # Higher threshold for specific intents
        
        self._initialize_concept_embeddings()
    
    def _initialize_concept_embeddings(self):
        """Pre-compute embeddings for all expense concepts"""
        try:
            for concept_key, concept_text in self.expense_concepts.items():
                embedding = self.embedding_model.embed_query(concept_text)
                self.concept_embeddings[concept_key] = np.array(embedding)
            logger.info(f"✅ Initialized {len(self.concept_embeddings)} semantic concept embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize concept embeddings: {e}")
            self.concept_embeddings = {}
    
    def get_semantic_intent(self, query: str) -> Tuple[str, float, bool]:
        """
        Determine query intent using pure semantic similarity
        Returns: (intent, confidence_score, is_expense_related)
        """
        if not self.concept_embeddings:
            logger.error("No concept embeddings available - cannot perform semantic analysis")
            return "unknown", 0.0, False
        
        try:
            # Get query embedding
            query_embedding = np.array(self.embedding_model.embed_query(query))
            
            # Calculate similarities with all concepts
            similarities = {}
            for concept_key, concept_embedding in self.concept_embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    concept_embedding.reshape(1, -1)
                )[0][0]
                similarities[concept_key] = similarity
            
            # Find best matching intent
            best_intent = max(similarities.items(), key=lambda x: x[1])
            intent_name, intent_score = best_intent
            
            # Determine if it's expense-related (any concept above threshold)
            max_similarity = max(similarities.values())
            is_expense_related = max_similarity >= self.expense_threshold
            
            # Only return specific intent if above intent threshold
            if intent_score >= self.intent_threshold:
                final_intent = intent_name
            else:
                final_intent = "general_expense" if is_expense_related else "non_expense"
            
            logger.info(f"Query: '{query}' | Intent: {final_intent} | Score: {intent_score:.3f} | Expense: {is_expense_related}")
            
            return final_intent, intent_score, is_expense_related
            
        except Exception as e:
            logger.error(f"Error in semantic intent detection: {e}")
            return "error", 0.0, False
    
    async def process_chat_query(self, query: str, session: AsyncSession) -> str:
        """Process chat query using pure semantic understanding"""
        try:
            # Get semantic intent
            intent, confidence, is_expense_related = self.get_semantic_intent(query)
            
            if not is_expense_related:
                return await self._handle_non_expense_query(query)
            
            # Route to appropriate expense handler based on semantic intent
            intent_handlers = {
                "current_spending": self._get_current_month_summary,
                "past_spending": self._get_last_month_summary,
                "comparison": self._get_comparison_summary,
                "food_expenses": lambda s: self._get_category_summary(s, "Food"),
                "transport_expenses": lambda s: self._get_category_summary(s, "Transport"),
                "shopping_expenses": lambda s: self._get_category_summary(s, "Shopping"),
                "entertainment": lambda s: self._get_category_summary(s, "Entertainment"),
                "bills_utilities": lambda s: self._get_category_summary(s, "Bills"),
                "highest_expenses": self._get_highest_expenses,
                "spending_patterns": self._get_spending_patterns,
                "total_amount": self._get_total_summary,
                "categories": self._get_category_breakdown,
                "recent_expenses": self._get_recent_expenses,
                "budget_advice": self._get_budget_advice,
                "savings": self._get_savings_advice,
                "weekly_spending": self._get_weekly_summary,
                "daily_spending": self._get_daily_summary,
                "general_expense": self._get_comprehensive_summary
            }
            
            # Get appropriate handler
            handler = intent_handlers.get(intent, self._get_comprehensive_summary)
            expense_data = await handler(session)
            
            # Generate contextual AI response
            return await self._generate_ai_response(query, expense_data, intent, confidence)
            
        except Exception as e:
            logger.error(f"Error processing chat query: {e}")
            return "I encountered an error while analyzing your question. Please try rephrasing it. 😅"
    
    async def _handle_non_expense_query(self, query: str) -> str:
        """Handle non-expense related queries"""
        try:
            prompt = f"""You are a helpful AI assistant specializing in personal finance and expense tracking. 
The user asked: "{query}"

This doesn't seem to be about expense tracking specifically. Provide a helpful, friendly response while gently guiding them back to expense-related topics if appropriate.

Use emojis and keep the tone conversational."""
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error handling non-expense query: {e}")
            return "I'm here to help with your expense tracking! Ask me about your spending patterns, monthly summaries, or budget advice. 💰"
    
    async def _generate_ai_response(self, query: str, expense_data: str, intent: str, confidence: float) -> str:
        """Generate contextual AI response based on semantic intent"""
        try:
            prompt = f"""You are an intelligent expense tracking assistant with semantic understanding capabilities.

User's original question: "{query}"
Detected intent: {intent} (confidence: {confidence:.2f})
Expense data analysis:
{expense_data}

Instructions:
- Provide a natural, conversational response that directly addresses the user's question
- Use the expense data provided to give specific insights
- Include relevant emojis to make the response engaging
- If the confidence is high (>0.5), be more specific about addressing their intent
- If confidence is moderate (0.25-0.5), provide broader expense insights
- Always end with a helpful suggestion or next step
- Keep responses concise but informative

Response:"""
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"{expense_data}\n\n💡 Let me know if you'd like more specific insights about your expenses!"
    
    # Expense Summary Methods (SQLAlchemy implementations)
    async def _get_current_month_summary(self, session: AsyncSession) -> str:
        """Get current month summary"""
        try:
            # Current month expenses by category
            current_month_query = select(
                Expense.category,
                func.sum(Expense.amount).label('total'),
                func.count(Expense.id).label('count')
            ).where(
                Expense.date >= func.date_trunc('month', func.current_date())
            ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc())
            
            result = await session.execute(current_month_query)
            current_month = result.all()
            
            # Total for current month
            month_total_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.date >= func.date_trunc('month', func.current_date())
            )
            result = await session.execute(month_total_query)
            month_total = result.scalar()
            
            if month_total == 0:
                return "📊 You haven't recorded any expenses this month yet! Start tracking to see your spending patterns."
            
            summary = f"📊 **THIS MONTH'S SPENDING**: ${month_total:.2f}\n\n🏷️ **Breakdown by Category**:\n"
            
            for row in current_month:
                percentage = (row.total / month_total * 100) if month_total > 0 else 0
                summary += f"• **{row.category}**: ${row.total:.2f} ({percentage:.1f}%) - {row.count} transactions\n"
            
            if current_month:
                top_category = current_month[0].category
                top_amount = current_month[0].total
                summary += f"\n💡 **Insight**: Your biggest spending category is **{top_category}** at ${top_amount:.2f}"
            
            return summary
        except Exception as e:
            logger.error(f"Error in current month summary: {e}")
            return "Unable to fetch current month data."
    
    async def _get_last_month_summary(self, session: AsyncSession) -> str:
        """Get last month summary"""
        try:
            last_month_query = select(
                Expense.category,
                func.sum(Expense.amount).label('total'),
                func.count(Expense.id).label('count')
            ).where(
                and_(
                    Expense.date >= func.date_trunc('month', func.current_date()) - func.text("INTERVAL '1 month'"),
                    Expense.date < func.date_trunc('month', func.current_date())
                )
            ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc())
            
            result = await session.execute(last_month_query)
            last_month = result.all()
            
            month_total_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                and_(
                    Expense.date >= func.date_trunc('month', func.current_date()) - func.text("INTERVAL '1 month'"),
                    Expense.date < func.date_trunc('month', func.current_date())
                )
            )
            result = await session.execute(month_total_query)
            month_total = result.scalar()
            
            if month_total == 0:
                return "📊 No expenses recorded for last month!"
            
            summary = f"📊 **LAST MONTH'S SPENDING**: ${month_total:.2f}\n\n🏷️ **Breakdown**:\n"
            
            for row in last_month:
                percentage = (row.total / month_total * 100)
                summary += f"• **{row.category}**: ${row.total:.2f} ({percentage:.1f}%)\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error in last month summary: {e}")
            return "Unable to fetch last month data."
    
    async def _get_comparison_summary(self, session: AsyncSession) -> str:
        """Compare this month vs last month"""
        try:
            # Current month
            current_total_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.date >= func.date_trunc('month', func.current_date())
            )
            result = await session.execute(current_total_query)
            current_total = result.scalar()
            
            # Last month
            last_total_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                and_(
                    Expense.date >= func.date_trunc('month', func.current_date()) - func.text("INTERVAL '1 month'"),
                    Expense.date < func.date_trunc('month', func.current_date())
                )
            )
            result = await session.execute(last_total_query)
            last_total = result.scalar()
            
            if current_total == 0 and last_total == 0:
                return "📊 No data to compare yet! Start tracking your expenses."
            
            difference = current_total - last_total
            percent_change = (difference / last_total * 100) if last_total > 0 else 0
            
            trend_emoji = "📈" if difference > 0 else "📉" if difference < 0 else "➡️"
            trend_word = "more" if difference > 0 else "less" if difference < 0 else "same"
            
            summary = f"""📊 **MONTH COMPARISON**
            
💰 **This Month**: ${current_total:.2f}
💰 **Last Month**: ${last_total:.2f}
{trend_emoji} **Change**: ${abs(difference):.2f} {trend_word} ({abs(percent_change):.1f}%)"""
            
            if difference > 0:
                summary += "\n\n💡 **Insight**: You're spending more this month. Consider reviewing your budget!"
            elif difference < 0:
                summary += "\n\n💡 **Insight**: Great job! You're spending less this month. Keep it up!"
            else:
                summary += "\n\n💡 **Insight**: Your spending is consistent month-to-month."
            
            return summary
        except Exception as e:
            logger.error(f"Error in comparison summary: {e}")
            return "Unable to fetch comparison data."
    
    async def _get_category_summary(self, session: AsyncSession, category: str) -> str:
        """Get summary for a specific category"""
        try:
            # This month in category
            this_month_query = select(
                func.coalesce(func.sum(Expense.amount), 0).label('total'),
                func.count(Expense.id).label('count')
            ).where(
                and_(
                    Expense.category.ilike(f'%{category}%'),
                    Expense.date >= func.date_trunc('month', func.current_date())
                )
            )
            result = await session.execute(this_month_query)
            this_month = result.first()
            
            recent_query = select(Expense.date, Expense.amount, Expense.note).where(
                Expense.category.ilike(f'%{category}%')
            ).order_by(Expense.date.desc()).limit(5)
            
            result = await session.execute(recent_query)
            recent = result.all()
            
            if this_month.total == 0:
                return f"📊 No **{category}** expenses recorded this month!"
            
            avg_per_transaction = (this_month.total / this_month.count) if this_month.count > 0 else 0
            
            summary = f"""📊 **{category.upper()} EXPENSES THIS MONTH**

💰 **Total**: ${this_month.total:.2f}
📝 **Transactions**: {this_month.count}
📈 **Average per transaction**: ${avg_per_transaction:.2f}

🕒 **Recent {category} Expenses**:
"""
            
            for expense in recent:
                note_preview = f" - {expense.note[:30]}..." if expense.note else ""
                summary += f"• {expense.date}: ${expense.amount:.2f}{note_preview}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error in category summary: {e}")
            return f"Unable to fetch {category} data."
    
    async def _get_highest_expenses(self, session: AsyncSession) -> str:
        """Get highest expenses"""
        try:
            highest_query = select(
                Expense.date, Expense.amount, Expense.category, Expense.note
            ).order_by(Expense.amount.desc()).limit(10)
            
            result = await session.execute(highest_query)
            highest = result.all()
            
            if not highest:
                return "📊 No expenses recorded yet!"
            
            summary = "💰 **YOUR HIGHEST EXPENSES**:\n\n"
            
            for i, expense in enumerate(highest, 1):
                note_preview = f" - {expense.note[:30]}..." if expense.note else ""
                summary += f"{i}. **${expense.amount:.2f}** ({expense.category}) on {expense.date}{note_preview}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error in highest expenses: {e}")
            return "Unable to fetch highest expenses."
    
    async def _get_spending_patterns(self, session: AsyncSession) -> str:
        """Analyze spending patterns"""
        try:
            # Most common category
            top_category_query = select(
                Expense.category,
                func.count(Expense.id).label('frequency'),
                func.sum(Expense.amount).label('total')
            ).group_by(Expense.category).order_by(func.count(Expense.id).desc()).limit(1)
            
            result = await session.execute(top_category_query)
            top_category = result.first()
            
            # Average expense amount
            avg_query = select(func.avg(Expense.amount))
            result = await session.execute(avg_query)
            avg_amount = result.scalar() or 0
            
            summary = f"""📈 **YOUR SPENDING PATTERNS**

📊 **Average Expense**: ${avg_amount:.2f}
"""
            
            if top_category:
                summary += f"🏆 **Most Frequent Category**: {top_category.category} ({top_category.frequency} transactions, ${top_category.total:.2f} total)\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error in spending patterns: {e}")
            return "Unable to fetch spending patterns."
    
    async def _get_total_summary(self, session: AsyncSession) -> str:
        """Get total spending summary"""
        try:
            total_query = select(func.sum(Expense.amount))
            result = await session.execute(total_query)
            total = result.scalar() or 0
            
            count_query = select(func.count(Expense.id))
            result = await session.execute(count_query)
            count = result.scalar() or 0
            
            return f"📊 **TOTAL SPENDING**: ${total:.2f} across {count} transactions"
        except Exception as e:
            logger.error(f"Error in total summary: {e}")
            return "Unable to fetch total data."
    
    async def _get_category_breakdown(self, session: AsyncSession) -> str:
        """Get category breakdown"""
        try:
            categories_query = select(
                Expense.category,
                func.sum(Expense.amount).label('total'),
                func.count(Expense.id).label('count')
            ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc())
            
            result = await session.execute(categories_query)
            categories = result.all()
            
            if not categories:
                return "📊 No expense categories found!"
            
            total_amount = sum(cat.total for cat in categories)
            summary = f"🏷️ **EXPENSE CATEGORIES** (Total: ${total_amount:.2f}):\n\n"
            
            for cat in categories:
                percentage = (cat.total / total_amount * 100) if total_amount > 0 else 0
                summary += f"• **{cat.category}**: ${cat.total:.2f} ({percentage:.1f}%) - {cat.count} transactions\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error in category breakdown: {e}")
            return "Unable to fetch category data."
    
    async def _get_recent_expenses(self, session: AsyncSession) -> str:
        """Get recent expenses"""
        try:
            recent_query = select(
                Expense.date, Expense.amount, Expense.category, Expense.note
            ).order_by(Expense.date.desc()).limit(10)
            
            result = await session.execute(recent_query)
            recent = result.all()
            
            if not recent:
                return "📊 No recent expenses found!"
            
            summary = "🕒 **RECENT EXPENSES**:\n\n"
            
            for expense in recent:
                note_preview = f" - {expense.note[:30]}..." if expense.note else ""
                summary += f"• {expense.date}: **${expense.amount:.2f}** ({expense.category}){note_preview}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error in recent expenses: {e}")
            return "Unable to fetch recent expenses."
    
    async def _get_budget_advice(self, session: AsyncSession) -> str:
        """Get budget advice based on spending patterns"""
        try:
            # Get current month total
            current_total_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.date >= func.date_trunc('month', func.current_date())
            )
            result = await session.execute(current_total_query)
            current_total = result.scalar()
            
            # Get top spending category
            top_category_query = select(
                Expense.category,
                func.sum(Expense.amount).label('total')
            ).where(
                Expense.date >= func.date_trunc('month', func.current_date())
            ).group_by(Expense.category).order_by(func.sum(Expense.amount).desc()).limit(1)
            
            result = await session.execute(top_category_query)
            top_category = result.first()
            
            summary = f"💡 **BUDGET INSIGHTS** (This month: ${current_total:.2f}):\n\n"
            
            if top_category:
                summary += f"🎯 Your biggest expense category is **{top_category.category}** (${top_category.total:.2f})\n\n"
            
            summary += "📝 **Suggestions**:\n"
            summary += "• Track daily expenses to identify spending patterns\n"
            summary += "• Set category-wise budget limits\n"
            summary += "• Review and reduce discretionary spending\n"
            summary += "• Consider the 50/30/20 rule: 50% needs, 30% wants, 20% savings"
            
            return summary
        except Exception as e:
            logger.error(f"Error in budget advice: {e}")
            return "Unable to generate budget advice."
    
    async def _get_savings_advice(self, session: AsyncSession) -> str:
        """Get savings advice"""
        return "💰 **SAVINGS TIPS**:\n\n• Track all expenses to identify unnecessary spending\n• Set up automatic savings transfers\n• Use the envelope budgeting method\n• Review subscriptions and cancel unused ones\n• Compare prices before major purchases"
    
    async def _get_weekly_summary(self, session: AsyncSession) -> str:
        """Get weekly spending summary"""
        try:
            weekly_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.date >= func.current_date() - func.text("INTERVAL '7 days'")
            )
            result = await session.execute(weekly_query)
            weekly_total = result.scalar()
            
            return f"📅 **THIS WEEK'S SPENDING**: ${weekly_total:.2f}"
        except Exception as e:
            logger.error(f"Error in weekly summary: {e}")
            return "Unable to fetch weekly data."
    
    async def _get_daily_summary(self, session: AsyncSession) -> str:
        """Get daily spending summary"""
        try:
            daily_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.date == func.current_date()
            )
            result = await session.execute(daily_query)
            daily_total = result.scalar()
            
            return f"📆 **TODAY'S SPENDING**: ${daily_total:.2f}"
        except Exception as e:
            logger.error(f"Error in daily summary: {e}")
            return "Unable to fetch daily data."
    
    async def _get_comprehensive_summary(self, session: AsyncSession) -> str:
        """Get comprehensive expense summary"""
        try:
            # Current month total
            month_total_query = select(func.coalesce(func.sum(Expense.amount), 0)).where(
                Expense.date >= func.date_trunc('month', func.current_date())
            )
            result = await session.execute(month_total_query)
            month_total = result.scalar()
            
            # Total expenses count
            count_query = select(func.count(Expense.id))
            result = await session.execute(count_query)
            total_count = result.scalar()
            
            # Average expense
            avg_query = select(func.avg(Expense.amount))
            result = await session.execute(avg_query)
            avg_expense = result.scalar() or 0
            
            summary = f"""📊 **EXPENSE OVERVIEW**

💰 **This Month**: ${month_total:.2f}
📝 **Total Transactions**: {total_count}
📈 **Average Expense**: ${avg_expense:.2f}

💡 Ask me about specific categories, comparisons, or spending patterns for more insights!"""
            
            return summary
        except Exception as e:
            logger.error(f"Error in comprehensive summary: {e}")
            return "Unable to fetch comprehensive data."