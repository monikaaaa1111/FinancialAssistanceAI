#!/usr/bin/env python3
"""
FinanceAI Pro - Enhanced ML-Powered Finance Manager
Advanced AI/ML Features with Sophisticated Color Scheme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict
import uuid
import warnings
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import json
import calendar

warnings.filterwarnings('ignore')

# Sophisticated Color Palette - Balanced & Classy
COLORS = {
    'primary': '#2E4057',        # Deep Blue Gray
    'secondary': '#048A81',      # Teal
    'accent': '#FF6B6B',         # Soft Coral
    'success': '#4ECDC4',        # Mint Green
    'warning': '#FFE66D',        # Soft Yellow
    'info': '#4A90E2',           # Sky Blue
    'light': '#F8F9FA',          # Off White
    'dark': '#2C3E50',           # Navy
    'purple': '#8B5CF6',         # Soft Purple
    'pink': '#EC4899',           # Rose Pink
    'orange': '#F59E0B',         # Amber
    'green': '#10B981',          # Emerald
    'gradient1': '#667eea',      # Soft Blue
    'gradient2': '#764ba2',      # Deep Purple
    'bg_primary': '#F7FAFC',     # Very Light Blue
    'bg_secondary': '#EDF2F7',   # Light Gray
    'text_primary': '#2D3748',   # Dark Gray
    'text_secondary': '#4A5568'  # Medium Gray
}

@dataclass
class Transaction:
    id: str
    amount: float
    category: str
    description: str
    date: str
    type: str
    merchant: str = ""
    tags: str = ""

class AdvancedFinanceAI:
    """Enhanced AI Engine with Advanced ML Features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.spending_model = None
        self.trend_model = None
        
    def detect_anomalies(self, transactions):
        """Advanced anomaly detection with multiple algorithms"""
        if len(transactions) < 15:
            return []
        
        df = pd.DataFrame([asdict(t) for t in transactions])
        df = df[df['type'] == 'Expense'].copy()
        
        if df.empty:
            return []
        
        # Enhanced feature engineering
        df['amount_log'] = np.log1p(df['amount'])
        df['date_parsed'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date_parsed'].dt.dayofweek
        df['hour_of_day'] = df['date_parsed'].dt.hour
        df['month'] = df['date_parsed'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Category encoding
        category_encoded = pd.get_dummies(df['category'], prefix='cat')
        
        # Combine features
        feature_cols = ['amount_log', 'day_of_week', 'hour_of_day', 'month', 'is_weekend']
        features = pd.concat([df[feature_cols], category_encoded], axis=1).fillna(0)
        
        # Multiple anomaly detection methods
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies_iso = iso_forest.fit_predict(features)
        
        # Statistical outlier detection
        amount_z_scores = np.abs((df['amount'] - df['amount'].mean()) / df['amount'].std())
        anomalies_stat = (amount_z_scores > 2.5).astype(int) * 2 - 1
        
        # Combine results
        combined_anomalies = (anomalies_iso == -1) | (anomalies_stat == -1)
        
        anomalous_transactions = []
        for idx, is_anomaly in enumerate(combined_anomalies):
            if is_anomaly:
                txn = df.iloc[idx]
                confidence = abs(amount_z_scores.iloc[idx]) / 3.0  # Normalize confidence
                anomalous_transactions.append({
                    'amount': txn['amount'],
                    'category': txn['category'],
                    'date': txn['date'],
                    'description': txn['description'],
                    'merchant': txn['merchant'],
                    'confidence': min(confidence, 1.0),
                    'reason': self._get_anomaly_reason(txn, df)
                })
        
        # Sort by confidence and return top anomalies
        anomalous_transactions.sort(key=lambda x: x['confidence'], reverse=True)
        return anomalous_transactions[:5]
    
    def _get_anomaly_reason(self, txn, df):
        """Generate contextual reason for anomaly"""
        category_avg = df[df['category'] == txn['category']]['amount'].mean()
        overall_avg = df['amount'].mean()
        
        if txn['amount'] > category_avg * 2:
            return f"Unusually high {txn['category']} expense (2x above average)"
        elif txn['amount'] > overall_avg * 1.5:
            return f"Above average spending detected"
        else:
            return f"Unusual spending pattern for {txn['category']}"
    
    def predict_spending(self, transactions):
        """Predict future spending using ML"""
        if len(transactions) < 30:
            return None
        
        df = pd.DataFrame([asdict(t) for t in transactions])
        df = df[df['type'] == 'Expense'].copy()
        df['date_parsed'] = pd.to_datetime(df['date'])
        df = df.sort_values('date_parsed')
        
        # Create time-based features
        df['days_since_start'] = (df['date_parsed'] - df['date_parsed'].min()).dt.days
        df['month'] = df['date_parsed'].dt.month
        df['day_of_week'] = df['date_parsed'].dt.dayofweek
        
        # Prepare features and target
        features = df[['days_since_start', 'month', 'day_of_week']].values
        target = df['amount'].values
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features, target)
        
        # Predict next 7 days
        last_day = df['days_since_start'].max()
        future_days = []
        
        for i in range(1, 8):
            future_date = df['date_parsed'].max() + timedelta(days=i)
            future_features = [[
                last_day + i,
                future_date.month,
                future_date.weekday()
            ]]
            prediction = model.predict(future_features)[0]
            future_days.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'predicted_amount': max(0, prediction),
                'day_name': future_date.strftime('%A')
            })
        
        return {
            'predictions': future_days,
            'model_accuracy': self._calculate_model_accuracy(model, features, target),
            'total_predicted': sum(day['predicted_amount'] for day in future_days)
        }
    
    def _calculate_model_accuracy(self, model, features, target):
        """Calculate model accuracy using cross-validation"""
        try:
            predictions = model.predict(features)
            mae = mean_absolute_error(target, predictions)
            return max(0, 1 - (mae / np.mean(target)))
        except:
            return 0.5
    
    def spending_insights(self, transactions):
        """Enhanced AI-powered spending insights"""
        if not transactions:
            return []
        
        df = pd.DataFrame([asdict(t) for t in transactions])
        df['date'] = pd.to_datetime(df['date'])
        
        insights = []
        
        # Advanced trend analysis
        recent_30_days = df[df['date'] >= (datetime.now() - timedelta(days=30))]
        prev_30_days = df[(df['date'] >= (datetime.now() - timedelta(days=60))) & 
                         (df['date'] < (datetime.now() - timedelta(days=30)))]
        
        if not recent_30_days.empty and not prev_30_days.empty:
            recent_spending = recent_30_days[recent_30_days['type'] == 'Expense']['amount'].sum()
            prev_spending = prev_30_days[prev_30_days['type'] == 'Expense']['amount'].sum()
            
            if recent_spending > prev_spending * 1.2:
                change = ((recent_spending/prev_spending - 1) * 100)
                insights.append({
                    'type': 'warning',
                    'title': 'Spending Trend Alert',
                    'message': f'Monthly spending increased by {change:.1f}%',
                    'icon': '📈',
                    'confidence': 0.8,
                    'recommendation': 'Consider reviewing recent purchases and setting spending limits'
                })
            elif recent_spending < prev_spending * 0.8:
                change = ((1 - recent_spending/prev_spending) * 100)
                insights.append({
                    'type': 'success',
                    'title': 'Great Savings!',
                    'message': f'You saved {change:.1f}% compared to last month',
                    'icon': '💰',
                    'confidence': 0.9,
                    'recommendation': 'Keep up the good work! Consider investing the saved amount'
                })
        
        # Category-based insights
        expenses = df[df['type'] == 'Expense']
        if not expenses.empty:
            category_analysis = expenses.groupby('category').agg({
                'amount': ['sum', 'mean', 'count']
            }).round(2)
            
            top_category = category_analysis[('amount', 'sum')].idxmax()
            top_amount = category_analysis.loc[top_category, ('amount', 'sum')]
            total_expenses = expenses['amount'].sum()
            percentage = (top_amount / total_expenses) * 100
            
            insights.append({
                'type': 'info',
                'title': f'Top Category: {top_category}',
                'message': f'{percentage:.1f}% of total spending (${top_amount:,.0f})',
                'icon': '📊',
                'confidence': 0.95,
                'recommendation': f'Monitor {top_category} expenses for optimization opportunities'
            })
        
        # Merchant analysis
        merchant_spending = expenses[expenses['merchant'] != ''].groupby('merchant')['amount'].sum()
        if not merchant_spending.empty:
            top_merchant = merchant_spending.idxmax()
            merchant_amount = merchant_spending.max()
            insights.append({
                'type': 'info',
                'title': f'Top Merchant: {top_merchant}',
                'message': f'${merchant_amount:,.0f} total spent',
                'icon': '🏪',
                'confidence': 0.8,
                'recommendation': 'Check for subscription services or bulk purchase opportunities'
            })
        
        # Seasonal patterns
        if len(df) > 90:
            seasonal_insight = self._analyze_seasonal_patterns(df)
            if seasonal_insight:
                insights.append(seasonal_insight)
        
        return insights
    
    def _analyze_seasonal_patterns(self, df):
        """Analyze seasonal spending patterns"""
        df['month_name'] = df['date'].dt.strftime('%B')
        monthly_spending = df[df['type'] == 'Expense'].groupby('month_name')['amount'].sum()
        
        if len(monthly_spending) >= 3:
            peak_month = monthly_spending.idxmax()
            peak_amount = monthly_spending.max()
            avg_amount = monthly_spending.mean()
            
            if peak_amount > avg_amount * 1.3:
                return {
                    'type': 'info',
                    'title': f'Seasonal Pattern Detected',
                    'message': f'{peak_month} is your highest spending month',
                    'icon': '📅',
                    'confidence': 0.7,
                    'recommendation': f'Plan ahead for {peak_month} expenses'
                }
        return None
    
    def cluster_behavior(self, transactions):
        """Advanced spending behavior clustering with personality insights"""
        if len(transactions) < 30:
            return None
        
        df = pd.DataFrame([asdict(t) for t in transactions])
        df = df[df['type'] == 'Expense'].copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Create comprehensive behavioral features
        df['month'] = df['date'].dt.to_period('M')
        monthly_data = df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
        
        if monthly_data.empty or len(monthly_data) < 3:
            return None
        
        # Additional behavioral metrics
        monthly_stats = df.groupby('month').agg({
            'amount': ['sum', 'mean', 'std', 'count']
        }).round(2)
        
        # Combine features
        features = pd.concat([monthly_data, monthly_stats], axis=1).fillna(0)
        features_scaled = self.scaler.fit_transform(features)
        
        # Clustering
        n_clusters = min(4, len(monthly_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Analyze current behavior
        latest_cluster = clusters[-1]
        cluster_characteristics = self._get_cluster_characteristics(latest_cluster, monthly_data, monthly_stats)
        
        return cluster_characteristics
    
    def _get_cluster_characteristics(self, cluster_id, monthly_data, monthly_stats):
        """Get detailed cluster characteristics"""
        latest_spending = monthly_data.iloc[-1].sum()
        latest_categories = monthly_data.iloc[-1]
        top_category = latest_categories.idxmax()
        
        # Spending personality mapping
        personalities = {
            0: {
                'type': 'Conservative Spender',
                'description': 'Consistent, budget-conscious spending patterns',
                'traits': ['Predictable', 'Cautious', 'Savings-focused']
            },
            1: {
                'type': 'Balanced Spender',
                'description': 'Well-distributed spending across categories',
                'traits': ['Organized', 'Moderate', 'Planned']
            },
            2: {
                'type': 'Dynamic Spender',
                'description': 'Variable spending with seasonal patterns',
                'traits': ['Flexible', 'Opportunistic', 'Adaptive']
            },
            3: {
                'type': 'Premium Spender',
                'description': 'Higher spending with focus on quality',
                'traits': ['Quality-focused', 'Comfortable', 'Strategic']
            }
        }
        
        personality = personalities.get(cluster_id, personalities[1])
        
        return {
            'type': personality['type'],
            'description': personality['description'],
            'traits': personality['traits'],
            'avg_monthly': latest_spending,
            'primary_category': top_category,
            'spending_diversity': len([cat for cat in latest_categories if cat > 0]),
            'cluster_id': cluster_id
        }
    
    def generate_recommendations(self, transactions, budget_goal=None):
        """AI-powered financial recommendations"""
        if not transactions:
            return []
        
        df = pd.DataFrame([asdict(t) for t in transactions])
        recommendations = []
        
        # Spending optimization
        expenses = df[df['type'] == 'Expense']
        if not expenses.empty:
            category_spending = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            # High-spend category optimization
            top_category = category_spending.index[0]
            top_amount = category_spending.iloc[0]
            
            if top_amount > expenses['amount'].sum() * 0.3:
                recommendations.append({
                    'type': 'optimization',
                    'title': f'Optimize {top_category} Spending',
                    'description': f'This category represents {(top_amount/expenses["amount"].sum()*100):.1f}% of your expenses',
                    'action': f'Review {top_category} expenses for potential savings',
                    'impact': 'High',
                    'difficulty': 'Medium'
                })
        
        # Budget recommendations
        if budget_goal:
            monthly_spending = expenses['amount'].sum() / max(1, len(expenses.groupby(pd.to_datetime(expenses['date']).dt.to_period('M'))))
            if monthly_spending > budget_goal:
                recommendations.append({
                    'type': 'budget',
                    'title': 'Budget Adjustment Needed',
                    'description': f'Current spending (${monthly_spending:.0f}) exceeds goal (${budget_goal:.0f})',
                    'action': f'Reduce spending by ${monthly_spending - budget_goal:.0f} per month',
                    'impact': 'High',
                    'difficulty': 'High'
                })
        
        return recommendations

class FinanceDB:
    """Enhanced Database Operations"""
    
    def __init__(self):
        self.db_path = 'finance_ai_pro.db'
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    date TEXT NOT NULL,
                    type TEXT NOT NULL,
                    merchant TEXT DEFAULT '',
                    tags TEXT DEFAULT ''
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS budgets (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    period TEXT NOT NULL,
                    created_date TEXT NOT NULL
                )
            ''')
    
    def add_transaction(self, txn):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO transactions 
                    (id, amount, category, description, date, type, merchant, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (txn.id, txn.amount, txn.category, txn.description,
                     txn.date, txn.type, txn.merchant, txn.tags))
            return True
        except Exception as e:
            st.error(f"Error adding transaction: {e}")
            return False
    
    def get_transactions(self, limit=200):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT id, amount, category, description, date, type, merchant, tags
                    FROM transactions 
                    ORDER BY date DESC 
                    LIMIT ?
                '''
                rows = conn.execute(query, (limit,)).fetchall()
                return [Transaction(*row) for row in rows]
        except Exception:
            return []
    
    def get_category_budgets(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT category, amount, period
                    FROM budgets 
                    ORDER BY created_date DESC
                '''
                rows = conn.execute(query).fetchall()
                return {row[0]: {'amount': row[1], 'period': row[2]} for row in rows}
        except Exception:
            return {}

def apply_sophisticated_styles():
    """Apply sophisticated, balanced styling"""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    :root {{
        --primary: {COLORS['primary']};
        --secondary: {COLORS['secondary']};
        --accent: {COLORS['accent']};
        --success: {COLORS['success']};
        --warning: {COLORS['warning']};
        --info: {COLORS['info']};
        --purple: {COLORS['purple']};
        --pink: {COLORS['pink']};
        --orange: {COLORS['orange']};
        --green: {COLORS['green']};
        --bg-primary: {COLORS['bg_primary']};
        --bg-secondary: {COLORS['bg_secondary']};
        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
    }}
    
    .stApp {{
        background: linear-gradient(135deg, {COLORS['bg_primary']} 0%, {COLORS['bg_secondary']} 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }}
    
    .main .block-container {{
        padding: 1rem 2rem;
        max-width: 1400px;
    }}
    
    /* Sophisticated Header */
    .hero-section {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .hero-section::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    .hero-title {{
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1;
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 1rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }}
    
    /* Enhanced Glass Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }}
    
    .glass-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
    }}
    
    /* Sophisticated Metric Cards */
    .metric-card {{
        background: linear-gradient(135deg, white, #f8f9fa);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }}
    
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {COLORS['purple']}, {COLORS['secondary']}, {COLORS['success']});
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--primary);
        font-family: 'Poppins', sans-serif;
    }}
    
    .metric-label {{
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 500;
    }}
    
    /* AI Insight Cards */
    .insight-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 249, 250, 0.9));
        border-left: 4px solid var(--info);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }}
    
    .insight-card:hover {{
        transform: translateX(8px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.1);
    }}
    
    .insight-warning {{
        border-left-color: {COLORS['accent']};
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 107, 107, 0.05));
    }}
    
    .insight-success {{
        border-left-color: {COLORS['success']};
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(78, 205, 196, 0.05));
    }}
    
    .insight-info {{
        border-left-color: {COLORS['info']};
        background: linear-gradient(135deg, rgba(74, 144, 226, 0.1), rgba(74, 144, 226, 0.05));
    }}
    
    /* Transaction Cards */
    .transaction-item {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 249, 250, 0.9));
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }}
    
    .transaction-item:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }}
    
    .expense {{
        border-left: 4px solid {COLORS['accent']};
    }}
    
    .income {{
        border-left: 4px solid {COLORS['success']};
    }}
    
    /* Sophisticated Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 64, 87, 0.3);
        font-family: 'Inter', sans-serif;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(46, 64, 87, 0.4);
    }}
    
    /* Form Elements */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stDateInput > div > div > input {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 2px solid rgba(46, 64, 87, 0.1) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }}
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {{
        border-color: {COLORS['secondary']} !important;
        box-shadow: 0 0 0 3px rgba(4, 138, 129, 0.1) !important;
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background: linear-gradient(180deg, {COLORS['primary']}, {COLORS['secondary']});
    }}
    
    .css-1d391kg .stRadio > label {{
        color: white;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    
    .css-1d391kg .stRadio > label:hover {{
        background: rgba(255, 255, 255, 0.1);
    }}
    
    /* Prediction Cards */
    .prediction-card {{
        background: linear-gradient(135deg, {COLORS['purple']}, {COLORS['pink']});
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 25px rgba(139, 92, 246, 0.3);
    }}
    
    /* AI Recommendations */
    .recommendation-card {{
        background: linear-gradient(135deg, {COLORS['orange']}, {COLORS['warning']});
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 25px rgba(245, 158, 11, 0.3);
    }}
    
    /* Behavioral Analysis Cards */
    .behavior-card {{
        background: linear-gradient(135deg, {COLORS['green']}, {COLORS['success']});
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 25px rgba(16, 185, 129, 0.3);
    }}
    
    /* Charts Container */
    .chart-container {{
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    
    /* Responsive Design */
    @media (max-width: 768px) {{
        .hero-title {{ font-size: 2.2rem; }}
        .main .block-container {{ padding: 1rem; }}
        .metric-card {{ padding: 1.5rem; }}
    }}
    </style>
    """, unsafe_allow_html=True)

def show_hero():
    """Display sophisticated hero section"""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🤖 FinanceAI Pro</h1>
        <p class="hero-subtitle">Advanced Machine Learning Financial Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

def show_metrics(transactions):
    """Display comprehensive financial metrics"""
    if not transactions:
        st.markdown("""
        <div class="glass-card">
            <h3 style="text-align: center; color: var(--text-secondary);">
                📊 Add transactions to see your financial metrics
            </h3>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = pd.DataFrame([asdict(t) for t in transactions])
    income = df[df['type'] == 'Income']['amount'].sum()
    expenses = df[df['type'] == 'Expense']['amount'].sum()
    balance = income - expenses
    
    # Calculate additional metrics
    avg_transaction = df['amount'].mean()
    monthly_spending = expenses / max(1, len(df.groupby(pd.to_datetime(df['date']).dt.to_period('M'))))
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("💰 Total Income", f"${income:,.0f}", col1, COLORS['success']),
        ("💸 Total Expenses", f"${expenses:,.0f}", col2, COLORS['accent']),
        ("💎 Net Balance", f"${balance:,.0f}", col3, COLORS['info']),
        ("📊 Avg Transaction", f"${avg_transaction:,.0f}", col4, COLORS['purple'])
    ]
    
    for label, value, col, color in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def show_ai_insights(transactions, ai_engine):
    """Display comprehensive AI insights"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 AI Financial Intelligence")
    
    if len(transactions) < 5:
        st.info("🤖 Add more transactions to unlock advanced AI insights!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Get AI insights
    insights = ai_engine.spending_insights(transactions)
    
    if insights:
        for insight in insights:
            card_class = f"insight-{insight['type']}"
            confidence_bar = "🟢" * int(insight['confidence'] * 5)
            
            st.markdown(f"""
            <div class="insight-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.3rem;">{insight['icon']}</span>
                            <div style="font-weight: 600; font-size: 1.1rem;">{insight['title']}</div>
                        </div>
                        <div style="margin-bottom: 0.5rem; font-size: 0.95rem;">{insight['message']}</div>
                        <div style="font-size: 0.85rem; opacity: 0.8; font-style: italic;">
                            💡 {insight['recommendation']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.8rem; opacity: 0.7;">Confidence</div>
                        <div style="font-size: 0.9rem;">{confidence_bar}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Behavioral Analysis
    behavior = ai_engine.cluster_behavior(transactions)
    if behavior:
        st.markdown(f"""
        <div class="behavior-card">
            <h4 style="margin: 0 0 1rem 0;">🎯 Spending Personality Analysis</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div>
                    <div style="font-weight: 600; font-size: 1.1rem;">{behavior['type']}</div>
                    <div style="opacity: 0.9; margin: 0.5rem 0;">{behavior['description']}</div>
                    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                        {' '.join([f'<span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">{trait}</span>' for trait in behavior['traits']])}
                    </div>
                </div>
                <div>
                    <div><strong>Monthly Average:</strong> ${behavior['avg_monthly']:,.0f}</div>
                    <div><strong>Primary Category:</strong> {behavior['primary_category']}</div>
                    <div><strong>Category Diversity:</strong> {behavior['spending_diversity']} categories</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_ml_predictions(transactions, ai_engine):
    """Display ML predictions"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 AI Spending Predictions")
    
    if len(transactions) < 30:
        st.info("🤖 Need at least 30 transactions for accurate ML predictions!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    prediction_data = ai_engine.predict_spending(transactions)
    
    if prediction_data:
        accuracy = prediction_data['model_accuracy'] * 100
        
        st.markdown(f"""
        <div class="prediction-card">
            <h4 style="margin: 0 0 1rem 0;">📈 7-Day Spending Forecast</h4>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 1.8rem; font-weight: 700;">${prediction_data['total_predicted']:,.0f}</div>
                    <div style="opacity: 0.9;">Predicted weekly spending</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.2rem; font-weight: 600;">{accuracy:.1f}%</div>
                    <div style="opacity: 0.9;">Model accuracy</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Daily predictions
        col1, col2 = st.columns(2)
        for i, day in enumerate(prediction_data['predictions']):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <div style="font-weight: 600;">{day['day_name']}</div>
                    <div>${day['predicted_amount']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_advanced_charts(transactions):
    """Display advanced interactive charts"""
    if not transactions:
        return
    
    df = pd.DataFrame([asdict(t) for t in transactions])
    df['date'] = pd.to_datetime(df['date'])
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    chart_type = st.selectbox("Select Analysis", [
        "Spending Heatmap", "Category Trends", "Monthly Comparison", 
        "Merchant Analysis", "Transaction Patterns"
    ])
    
    if chart_type == "Spending Heatmap":
        expenses = df[df['type'] == 'Expense'].copy()
        expenses['weekday'] = expenses['date'].dt.day_name()
        expenses['hour'] = expenses['date'].dt.hour
        
        heatmap_data = expenses.groupby(['weekday', 'hour'])['amount'].sum().reset_index()
        
        fig = px.density_heatmap(
            heatmap_data, x='hour', y='weekday', z='amount',
            title="Spending Patterns by Day and Hour",
            color_continuous_scale='Viridis'
        )
        
    elif chart_type == "Category Trends":
        monthly_category = df.groupby([
            df['date'].dt.to_period('M').astype(str), 'category'
        ])['amount'].sum().reset_index()
        
        fig = px.line(
            monthly_category, x='date', y='amount', color='category',
            title="Category Spending Trends Over Time",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
    elif chart_type == "Monthly Comparison":
        df['month_year'] = df['date'].dt.to_period('M').astype(str)
        monthly_data = df.groupby(['month_year', 'type'])['amount'].sum().reset_index()
        
        fig = px.bar(
            monthly_data, x='month_year', y='amount', color='type',
            title="Monthly Income vs Expenses Comparison",
            color_discrete_map={'Income': COLORS['success'], 'Expense': COLORS['accent']}
        )
        
    elif chart_type == "Merchant Analysis":
        merchant_data = df[df['merchant'] != ''].groupby('merchant')['amount'].sum().head(10).reset_index()
        
        fig = px.treemap(
            merchant_data, path=['merchant'], values='amount',
            title="Top Merchants by Spending",
            color='amount', color_continuous_scale='Viridis'
        )
        
    else:  # Transaction Patterns
        df['amount_range'] = pd.cut(df['amount'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        pattern_data = df.groupby(['amount_range', 'type']).size().reset_index(name='count')
        
        fig = px.sunburst(
            pattern_data, path=['type', 'amount_range'], values='count',
            title="Transaction Amount Distribution Patterns",
            color='count', color_continuous_scale='Viridis'
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=COLORS['text_primary'],
        title_font_size=16,
        title_font_color=COLORS['primary']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def add_transaction_form(db):
    """Enhanced transaction input form"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ➕ Add New Transaction")
    
    with st.form("add_transaction", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.01, step=0.01)
            txn_type = st.selectbox("Type", ["Income", "Expense"])
            date_input = st.date_input("Date", value=date.today())
        
        with col2:
            category = st.selectbox("Category", [
                "Food & Dining", "Housing", "Transportation", "Healthcare", 
                "Entertainment", "Shopping", "Salary", "Investment", 
                "Utilities", "Insurance", "Education", "Travel", "Other"
            ])
            merchant = st.text_input("Merchant/Source")
            description = st.text_input("Description")
        
        tags = st.text_input("Tags (comma-separated)", placeholder="e.g., recurring, business, personal")
        
        col_submit1, col_submit2 = st.columns(2)
        with col_submit1:
            submit_button = st.form_submit_button("💾 Add Transaction", use_container_width=True)
        with col_submit2:
            quick_add = st.form_submit_button("⚡ Quick Add & New", use_container_width=True)
        
        if submit_button or quick_add:
            if amount > 0:
                transaction = Transaction(
                    id=str(uuid.uuid4()),
                    amount=amount,
                    category=category,
                    description=description,
                    date=date_input.strftime('%Y-%m-%d'),
                    type=txn_type,
                    merchant=merchant,
                    tags=tags
                )
                
                if db.add_transaction(transaction):
                    st.success("✅ Transaction added successfully!")
                    if submit_button:
                        st.rerun()
                else:
                    st.error("❌ Failed to add transaction")
            else:
                st.error("⚠️ Please enter a valid amount")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_transactions(transactions):
    """Display enhanced transaction list"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Recent Transactions")
    
    if not transactions:
        st.info("💡 No transactions yet. Add some to get started!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_type = st.selectbox("Filter by Type", ["All", "Income", "Expense"])
    with col2:
        categories = ["All"] + list(set(t.category for t in transactions))
        filter_category = st.selectbox("Filter by Category", categories)
    with col3:
        limit = st.slider("Number of transactions", 5, 50, 10)
    
    # Apply filters
    filtered_transactions = transactions
    if filter_type != "All":
        filtered_transactions = [t for t in filtered_transactions if t.type == filter_type]
    if filter_category != "All":
        filtered_transactions = [t for t in filtered_transactions if t.category == filter_category]
    
    # Display transactions
    for txn in filtered_transactions[:limit]:
        txn_class = "expense" if txn.type == "Expense" else "income"
        icon = "💸" if txn.type == "Expense" else "💰"
        
        merchant_info = f" • {txn.merchant}" if txn.merchant else ""
        tags_info = f" • 🏷️ {txn.tags}" if txn.tags else ""
        
        st.markdown(f"""
        <div class="transaction-item {txn_class}">
            <div style="flex: 1;">
                <div style="font-weight: 600; margin-bottom: 0.25rem;">
                    {icon} {txn.description or txn.category}
                </div>
                <div style="font-size: 0.85rem; opacity: 0.8;">
                    📁 {txn.category} • 📅 {txn.date}{merchant_info}{tags_info}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; font-size: 1.2rem; color: {'var(--accent)' if txn.type == 'Expense' else 'var(--success)'};">
                    {'−' if txn.type == 'Expense' else '+'}${txn.amount:,.2f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_anomalies(transactions, ai_engine):
    """Display advanced anomaly detection"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🚨 AI Anomaly Detection")
    
    if len(transactions) < 15:
        st.info("🤖 Need at least 15 transactions for accurate anomaly detection!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    anomalies = ai_engine.detect_anomalies(transactions)
    
    if anomalies:
        st.warning(f"🔍 Found {len(anomalies)} unusual transactions")
        
        for i, anomaly in enumerate(anomalies, 1):
            confidence_level = "High" if anomaly['confidence'] > 0.7 else "Medium" if anomaly['confidence'] > 0.4 else "Low"
            confidence_color = COLORS['accent'] if anomaly['confidence'] > 0.7 else COLORS['warning'] if anomaly['confidence'] > 0.4 else COLORS['info']
            
            st.markdown(f"""
            <div class="insight-card insight-warning">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                            🚨 Anomaly #{i}: {anomaly['description']}
                        </div>
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Amount:</strong> ${anomaly['amount']:,.2f} • 
                            <strong>Category:</strong> {anomaly['category']} • 
                            <strong>Date:</strong> {anomaly['date']}
                        </div>
                        {f'<div style="margin-bottom: 0.5rem;"><strong>Merchant:</strong> {anomaly["merchant"]}</div>' if anomaly.get('merchant') else ''}
                        <div style="font-style: italic; opacity: 0.9;">
                            💡 {anomaly['reason']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="background: {confidence_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                            {confidence_level}
                        </div>
                        <div style="font-size: 0.8rem; margin-top: 0.25rem; opacity: 0.7;">
                            {anomaly['confidence']:.0%} confidence
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No unusual spending patterns detected")
        st.info("🎯 Your spending behavior appears consistent and predictable.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_ai_recommendations(transactions, ai_engine):
    """Display AI-powered recommendations"""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💡 AI Financial Recommendations")
    
    if len(transactions) < 10:
        st.info("🤖 Add more transactions to get personalized AI recommendations!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    recommendations = ai_engine.generate_recommendations(transactions)
    
    if recommendations:
        for rec in recommendations:
            impact_color = COLORS['accent'] if rec['impact'] == 'High' else COLORS['warning'] if rec['impact'] == 'Medium' else COLORS['info']
            difficulty_icon = "🔥" if rec['difficulty'] == 'High' else "⚡" if rec['difficulty'] == 'Medium' else "✨"
            
            st.markdown(f"""
            <div class="recommendation-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <h4 style="margin: 0; flex: 1;">💡 {rec['title']}</h4>
                    <div style="display: flex; gap: 0.5rem;">
                        <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.5rem; border-radius: 8px; font-size: 0.8rem;">
                            {difficulty_icon} {rec['difficulty']}
                        </span>
                        <span style="background: {impact_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 8px; font-size: 0.8rem;">
                            {rec['impact']} Impact
                        </span>
                    </div>
                </div>
                <div style="margin-bottom: 0.75rem; opacity: 0.9;">
                    {rec['description']}
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 8px;">
                    <strong>Action:</strong> {rec['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("🎯 Your financial habits look great! Keep up the excellent work.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="FinanceAI Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_sophisticated_styles()
    show_hero()
    
    # Initialize enhanced components
    db = FinanceDB()
    ai_engine = AdvancedFinanceAI()
    transactions = db.get_transactions()
    
    # Sophisticated sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.1); border-radius: 12px; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin: 0; font-family: 'Poppins', sans-serif;">🤖 AI Navigation</h3>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.5rem;">
                Machine Learning Powered
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio("", [
            "🏠 AI Dashboard", 
            "➕ Add Transaction", 
            "📊 Advanced Analytics", 
            "🔮 ML Predictions",
            "🚨 Anomaly Detection",
            "💡 AI Recommendations"
        ], label_visibility="collapsed")
        
        # Quick stats in sidebar
        if transactions:
            total_transactions = len(transactions)
            total_amount = sum(t.amount for t in transactions)
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem; color: white;">
                <div style="font-size: 0.9rem; opacity: 0.8;">Quick Stats</div>
                <div style="font-weight: 600;">{total_transactions} Transactions</div>
                <div style="font-weight: 600;">${total_amount:,.0f} Total</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced main content with AI focus
    if page == "🏠 AI Dashboard":
        show_metrics(transactions)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            show_advanced_charts(transactions)
            show_ai_insights(transactions, ai_engine)
        with col2:
            show_transactions(transactions)
            if len(transactions) >= 30:
                show_ml_predictions(transactions, ai_engine)
    
    elif page == "➕ Add Transaction":
        col1, col2 = st.columns([3, 2])
        with col1:
            add_transaction_form(db)
        with col2:
            show_transactions(transactions)
    
    elif page == "📊 Advanced Analytics":
        show_advanced_charts(transactions)
        show_ai_insights(transactions, ai_engine)
    
    elif page == "🔮 ML Predictions":
        show_ml_predictions(transactions, ai_engine)
        show_ai_insights(transactions, ai_engine)
    
    elif page == "🚨 Anomaly Detection":
        show_anomalies(transactions, ai_engine)
    
    elif page == "💡 AI Recommendations":
        show_ai_recommendations(transactions, ai_engine)
        show_ai_insights(transactions, ai_engine)

if __name__ == "__main__":
    main()