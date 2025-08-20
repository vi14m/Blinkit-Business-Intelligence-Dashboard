# Blinkit Business Intelligence Dashboard

An interactive dashboard for analyzing Blinkit operations, customer feedback, delivery performance, ratings, payment methods, and revenue. Built with Python, Streamlit, Plotly, and Pandas.

## Features
- Visualize sales, customer ratings, and feedback sentiment
- Segment users into new vs. returning cohorts
- Analyze payment preferences and operational efficiency
- Regional performance analysis with dual-axis charts
- Actionable insights for business decisions

## Demo
See the live dashboard and code on [GitHub](https://github.com/vi14m/Blinkit-Business-Intelligence-Dashboard).

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/vi14m/Blinkit-Business-Intelligence-Dashboard.git
   cd Blinkit-Business-Intelligence-Dashboard
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the dashboard:**
   ```sh
   streamlit run app.py
   ```

## Project Structure
```
├── app.py
├── blinkit_customers.csv
├── blinkit_orders.csv
├── blinkit_order_items.csv
├── blinkit_inventory.csv
├── blinkit_customer_feedback.csv
├── .venv/
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.8+
- Streamlit
- Plotly
- Pandas

## License
MIT

## Author
[vi14m](https://github.com/vi14m)
