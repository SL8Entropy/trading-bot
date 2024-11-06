import os
import pandas as pd
import numpy as np

# Load the CSV data
# Get the directory of the current Python file
directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(directory, 'data.csv')
data = pd.read_csv(file_path)


# Helper functions for calculations
'''
def calculate_macd(close_prices, short_window=12, long_window=26, signal_window=9):
    short_ema = close_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = close_prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal
'''
def calculate_rsi(close_prices, window):
    delta = close_prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_kd(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    stochastic_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stochastic_d = stochastic_k.rolling(window=d_window).mean()
    return stochastic_k, stochastic_d

# Calculate indicators for each row
data['RSI_7'] = calculate_rsi(data['close'], 7)
data['RSI_14'] = calculate_rsi(data['close'], 14)
data['RSI_21'] = calculate_rsi(data['close'], 21)
data['Stochastic %K'], data['Stochastic %D'] = calculate_stochastic_kd(data['high'], data['low'], data['close'])

# Save to new CSV with indicators
output_file = os.path.join(directory, 'data_with_indicators.csv')
data.to_csv(output_file, index=False)
print(f"Indicators added and saved to {output_file}")


