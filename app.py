
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Blinkit Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 24px;
        font-weight: bold;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 14px;
    }
    div[data-testid="metric-container"] {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 10px;
    }
    .stSelectbox label, .stMultiSelect label {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
    }
    .sidebar .stDateInput label {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    orders = pd.read_csv('blinkit_orders.csv')
    order_items = pd.read_csv('blinkit_order_items.csv')
    inventory = pd.read_csv('blinkit_inventory.csv')
    customers = pd.read_csv('blinkit_customers.csv')
    feedback = pd.read_csv('blinkit_customer_feedback.csv')
    
    # Convert date columns to datetime
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['promised_delivery_time'] = pd.to_datetime(orders['promised_delivery_time'])
    orders['actual_delivery_time'] = pd.to_datetime(orders['actual_delivery_time'])
    customers['registration_date'] = pd.to_datetime(customers['registration_date'])
    feedback['feedback_date'] = pd.to_datetime(feedback['feedback_date'])
    
    # Calculate delivery time difference in minutes
    orders['delivery_time_diff'] = (orders['actual_delivery_time'] - orders['promised_delivery_time']).dt.total_seconds() / 60
    
    # Merge dataframes
    df = pd.merge(orders, order_items, on='order_id')
    df = pd.merge(df, inventory, on='product_id')
    df = pd.merge(df, customers, on='customer_id')
    df = pd.merge(df, feedback[['order_id', 'rating', 'sentiment', 'feedback_category']], on='order_id', how='left')
    
    return df

st.set_page_config(page_title="Blinkit Sales Dashboard", layout="wide")

st.title("Blinkit Business Intelligence Dashboard")

df = load_data()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🎯 Dashboard Filters")
    st.markdown("---")

    # Date range selector
    st.markdown("### 📅 Date Range")
    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()
    date_range = st.date_input(
        "Select Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown("---")
    
    # Customer segment selector
    st.markdown("### 👥 Customer Segments")
    selected_segment = st.multiselect(
        "Choose Segments",
        options=sorted(df["customer_segment"].unique()),
        default=sorted(df["customer_segment"].unique())
    )
    
    st.markdown("---")
    
    # Payment method selector
    st.markdown("### 💳 Payment Methods")
    selected_payment = st.multiselect(
        "Choose Methods",
        options=sorted(df["payment_method"].unique()),
        default=sorted(df["payment_method"].unique())
    )

# Filter data based on selections
mask = (
    (df["customer_segment"].isin(selected_segment)) &
    (df["payment_method"].isin(selected_payment)) &
    (df["order_date"].dt.date >= date_range[0]) &
    (df["order_date"].dt.date <= date_range[1])
)
df_selection = df[mask]

# --- Main Page ---

# Key Performance Indicators (KPIs)
st.markdown("### 📊 Key Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

# Total Revenue
total_revenue = int(df_selection["order_total"].sum())
with col1:
    st.metric(
        label="Total Revenue",
        value=f"₹{total_revenue:,.0f}",
        delta=f"{((total_revenue/df['order_total'].sum())*100):.1f}% of total",
        delta_color="normal"
    )

# Average Order Value
avg_order_value = round(df_selection.groupby("order_id")["order_total"].first().mean(), 2)
with col2:
    st.metric(
        label="Average Order Value",
        value=f"₹{avg_order_value:,.2f}",
        delta=f"{((avg_order_value/df.groupby('order_id')['order_total'].first().mean())-1)*100:.1f}% vs overall",
        delta_color="normal"
    )

# Customer Satisfaction
avg_rating = round(df_selection["rating"].mean(), 2)
with col3:
    st.metric(
        label="Customer Rating",
        value=f"{avg_rating:.2f}/5",
        delta=f"{((avg_rating/df['rating'].mean())-1)*100:.1f}% vs overall",
        delta_color="normal"
    )

# On-Time Delivery Rate
on_time_rate = (df_selection["delivery_status"] == "On Time").mean() * 100
with col4:
    st.metric(
        label="On-Time Delivery",
        value=f"{on_time_rate:.1f}%",
        delta=f"{(on_time_rate - ((df['delivery_status'] == 'On Time').mean() * 100)):.1f}% vs overall",
        delta_color="normal"
    )

st.markdown("---")

# Create two columns for charts
st.markdown("### 💰 Revenue Analysis")
col1, col2 = st.columns(2)

# Sales by Customer Segment
with col1:
    sales_by_segment = df_selection.groupby("customer_segment")["order_total"].sum().sort_values(ascending=False)
    fig_sales_by_segment = px.bar(
        sales_by_segment,
        x=sales_by_segment.index,
        y=sales_by_segment.values,
        title="Revenue Distribution by Customer Segment",
        labels={"y": "Revenue (₹)", "x": "Customer Segment"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_sales_by_segment.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_sales_by_segment, use_container_width=True)

# Payment Method Distribution
with col2:
    payment_dist = df_selection.groupby("payment_method").agg({
        "order_total": "sum",
        "order_id": "count"
    }).reset_index()
    payment_dist.columns = ["Payment Method", "Total Revenue", "Number of Orders"]
    
    fig_payment = px.pie(
        payment_dist,
        values="Total Revenue",
        names="Payment Method",
        title="Revenue Share by Payment Method",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.5
    )
    fig_payment.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_payment.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_payment, use_container_width=True)


# Time Series Analysis
st.header("Revenue Over Time")
df_selection['month_year'] = df_selection['order_date'].dt.to_period('M')
monthly_sales = df_selection.groupby('month_year')['order_total'].sum().reset_index()
monthly_sales['month_year'] = monthly_sales['month_year'].dt.to_timestamp()

fig_monthly_sales = px.line(
    monthly_sales,
    x="month_year",
    y="order_total",
    title="Monthly Revenue",
    labels={"order_total": "Total Revenue (₹)", "month_year": "Month"},
    template="plotly_white"
)
st.plotly_chart(fig_monthly_sales, use_container_width=True)


# Time Series Forecasting
st.header("Demand Forecasting")
st.write("This section provides a sales forecast for the next 12 months.")

# Prepare data for forecasting
monthly_sales_ts = monthly_sales.set_index('month_year')['order_total']

# Fit SARIMA model
# These are example parameters, for a real-world scenario, you'd perform grid search
model = SARIMAX(monthly_sales_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

# Forecast
forecast = results.get_forecast(steps=12)
forecast_df = forecast.summary_frame()

# Plot forecast
fig_forecast = px.line(monthly_sales, x='month_year', y='order_total', title='Sales Forecast')
fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast')
fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', mode='lines', line_color='rgba(255, 255, 255, 0)', name='Lower CI')
fig_forecast.add_scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', mode='lines', line_color='rgba(255, 255, 255, 0)', name='Upper CI')
st.plotly_chart(fig_forecast, use_container_width=True)


# Customer Experience Analytics
st.markdown("### 📊 Customer Experience Analytics")
st.markdown("---")

# Create three columns for charts
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    delivery_performance = df_selection.groupby("delivery_status")["order_id"].count().reset_index()
    total_orders = delivery_performance["order_id"].sum()
    delivery_performance["percentage"] = (delivery_performance["order_id"] / total_orders) * 100
    
    fig_delivery = go.Figure()
    colors = {"On Time": "#00CC96", "Slightly Delayed": "#FFA15A", "Significantly Delayed": "#EF553B"}
    
    fig_delivery.add_trace(go.Bar(
        x=delivery_performance["delivery_status"],
        y=delivery_performance["order_id"],
        text=[f"{p:.1f}%" for p in delivery_performance["percentage"]],
        textposition="auto",
        marker_color=[colors[status] for status in delivery_performance["delivery_status"]],
        name="Orders"
    ))
    
    fig_delivery.update_layout(
        title={
            'text': "Delivery Performance Analysis",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Delivery Status",
        yaxis_title="Number of Orders",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_delivery, use_container_width=True)
    
    # Add insights below the chart
    st.markdown("""
    📈 **Key Observations:**
    - On-time delivery rate shows strong operational efficiency
    - Slightly delayed orders require attention in specific areas
    - Significant delays are minimal but need immediate action
    """)

with col2:
    rating_dist = df_selection["rating"].value_counts().sort_index()
    total_ratings = rating_dist.sum()
    avg_rating = (rating_dist * rating_dist.index).sum() / total_ratings

    fig_ratings = go.Figure()
    fig_ratings.add_trace(go.Bar(
        x=rating_dist.index,
        y=rating_dist.values,
        text=[f"{(v/total_ratings)*100:.1f}%" for v in rating_dist.values],
        textposition="auto",
        marker_color="#636EFA",
        name="Ratings"
    ))

    fig_ratings.update_layout(
        title={
            'text': f"Rating Distribution<br><sup>Average Rating: {avg_rating:.2f}/5</sup>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_ratings, use_container_width=True)

with col3:
    sentiment_dist = df_selection["sentiment"].value_counts()
    colors = {"Positive": "#00CC96", "Neutral": "#636EFA", "Negative": "#EF553B"}
    
    fig_sentiment = go.Figure(data=[go.Pie(
        labels=sentiment_dist.index,
        values=sentiment_dist.values,
        hole=0.6,
        marker_colors=[colors[sentiment] for sentiment in sentiment_dist.index],
        textinfo="percent+label",
        textposition="outside"
    )])

    fig_sentiment.update_layout(
        title={
            'text': "Customer Sentiment",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        annotations=[dict(text=f"Total<br>Reviews", x=0.5, y=0.5, font_size=12, showarrow=False)]
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Regional Performance Analysis
st.markdown("### 📍 Regional Performance Analysis")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    # Calculate metrics per area
    area_metrics = df_selection.groupby("area").agg({
        "order_total": ["sum", "mean"],
        "order_id": "count",
        "rating": "mean"
    }).reset_index()
    
    area_metrics.columns = ["Area", "Total Revenue", "Avg Order Value", "Number of Orders", "Avg Rating"]
    area_metrics = area_metrics.sort_values("Total Revenue", ascending=False)
    area_metrics["Total Revenue"] = area_metrics["Total Revenue"].round(2)
    area_metrics["Avg Order Value"] = area_metrics["Avg Order Value"].round(2)
    
    # Create figure with secondary y-axis
    fig_area = go.Figure()
    
    # Add bars for revenue
    fig_area.add_trace(
        go.Bar(
            x=area_metrics["Area"].head(10),
            y=area_metrics["Total Revenue"].head(10),
            name="Revenue",
            marker_color="#636EFA",
            text=[f"₹{x:,.0f}" for x in area_metrics["Total Revenue"].head(10)],
            textposition="auto",
        )
    )
    # Add line for average rating
    fig_area.add_trace(
        go.Scatter(
            x=area_metrics["Area"].head(10),
            y=area_metrics["Avg Rating"].head(10),
            name="Avg Rating",
            line=dict(color="#EF553B", width=3),
            yaxis="y2"
        )
    )
    
    fig_area.update_layout(
        title={
            'text': "Top 10 Areas by Revenue & Customer Satisfaction",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Area",
        yaxis_title="Revenue (₹)",
        yaxis2=dict(
            title="Average Rating",
            overlaying="y",
            side="right",
            range=[1, 5]
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_area, use_container_width=True)

with col2:
    avg_delivery_time = df_selection.groupby("area")["delivery_time_diff"].mean().reset_index()
    avg_delivery_time = avg_delivery_time.sort_values("delivery_time_diff", ascending=True)

    fig_delivery_time = px.bar(
        avg_delivery_time.head(10),
        x="area",
        y="delivery_time_diff",
        title="Average Delivery Time by Area (minutes)",
        labels={"delivery_time_diff": "Average Delivery Time (min)", "area": "Area"},
        template="plotly_white"
    )
    st.plotly_chart(fig_delivery_time, use_container_width=True)


# Business Insights and Recommendations
st.header("Business Insights & Growth Opportunities")

# Create two columns for different types of insights
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Performance Insights")
    st.markdown("""
    - **Revenue Patterns**
        - Track monthly revenue trends to identify seasonal patterns
        - Monitor growth rate in different customer segments
        - Analyze impact of payment methods on order values
    
    - **Customer Behavior**
        - Premium segment shows highest average order value
        - Digital payments (UPI/Cards) are increasingly preferred
        - Customer ratings indicate overall satisfaction level
    
    - **Operational Efficiency**
        - Monitor delivery performance across areas
        - Track inventory levels against demand patterns
        - Analyze delivery partner performance
    """)

with col2:
    st.subheader("🎯 Growth Opportunities")
    st.markdown("""
    - **Customer Experience**
        - Improve delivery times in areas with delays
        - Address common issues from customer feedback
        - Implement loyalty programs for regular customers
    
    - **Market Expansion**
        - Focus on high-performing areas for deeper penetration
        - Identify underserved areas with growth potential
        - Optimize store locations based on demand
    
    - **Operational Improvements**
        - Optimize inventory based on demand forecasts
        - Enhance delivery partner allocation
        - Implement dynamic pricing during peak hours
    """)

# Risk Factors and Mitigation
st.subheader("⚠️ Risk Factors and Mitigation Strategies")
st.markdown("""
- **Delivery Delays**
    - Implement better route optimization
    - Add more delivery partners in high-demand areas
    - Set realistic delivery time promises

- **Inventory Management**
    - Monitor stock levels regularly
    - Implement automated reordering system
    - Maintain safety stock for popular items

- **Customer Retention**
    - Address negative feedback promptly
    - Implement a customer win-back program
    - Regular analysis of churn patterns
""")

# Footer with dashboard info
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p><small>Last updated: {}</small></p>
<p><small>Data timeframe: {} to {}</small></p>
</div>
""".format(
    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    df["order_date"].min().strftime("%Y-%m-%d"),
    df["order_date"].max().strftime("%Y-%m-%d")
), unsafe_allow_html=True)
