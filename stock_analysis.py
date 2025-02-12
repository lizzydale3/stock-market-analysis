import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def fetch_stock_data(ticker, start_date="2023-01-01", end_date="2024-01-01"):
    """
    Fetches historical stock data from Yahoo Finance.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError("No data found for the given ticker.")
        return stock_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_moving_averages(stock_data, short_window=50, long_window=200):
    """
    Adds moving average indicators to the stock dataframe.
    """
    stock_data[f"{short_window}_MA"] = stock_data["Close"].rolling(window=short_window).mean()
    stock_data[f"{long_window}_MA"] = stock_data["Close"].rolling(window=long_window).mean()
    return stock_data

def train_ml_model(stock_data):
    """
    Uses Linear Regression to predict future stock prices.
    """
    stock_data = stock_data.dropna()  # Remove missing values

    # Prepare features (X) and target variable (y)
    stock_data["Day"] = range(len(stock_data))  # Convert dates into numerical values
    X = stock_data[["Day"]]  # Feature: Days
    y = stock_data["Close"]  # Target: Closing Price

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate error
    error = mean_absolute_error(y_test, y_pred)
    print(f"Model Mean Absolute Error: {error:.2f}")

    return model, stock_data

def plot_stock_data(stock_data, model, ticker):
    """
    Plots actual stock prices and ML predictions.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data["Close"], label="Actual Price", linewidth=2)
    plt.plot(stock_data.index, model.predict(stock_data[["Day"]]), label="Predicted Price", linestyle="dashed", color="red")

    plt.legend()
    plt.title(f"{ticker} Stock Price & ML Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g., AAPL, TSLA, GOOG): ").upper()
    
    stock_data = fetch_stock_data(ticker)
    if stock_data is not None:
        stock_data = calculate_moving_averages(stock_data)
        model, stock_data = train_ml_model(stock_data)
        plot_stock_data(stock_data, model, ticker)
