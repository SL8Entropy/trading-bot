import asyncio
import json
import time
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import sys
from datetime import datetime
import traceback
import numpy as np

# Get the directory of the current Python file
try:
    directory = os.path.dirname(os.path.abspath(__file__))
except NameError:
    directory = os.getcwd()

model_file_path = os.path.join(directory, 'random_forest_model.joblib')
# Define log file path
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
base_name = f"trade_logs_{ts}"
log_file_path = os.path.join(directory, base_name + '.jsonl')

print("Logging to:", log_file_path)
fail_file_path = os.path.join(directory, 'fail_count.txt')

def log_event(event: str, info: dict, log_time: datetime = None):
    """
    Append a JSON entry with a specific timestamp, event type, and info dict to the log file.
    If log_time is None, it uses the current system time.
    """
    if log_time:
        timestamp_str = log_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "datetime": timestamp_str,
        "event": event,
    }
    entry.update(info)
    try:
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        print(f"Logging failed: {e}")

# Path for the data used to train/load the model
csv_file_path_for_model = os.path.join(directory, 'data_with_indicators.csv')

# --- Model Loading/Training Block (No Changes) ---
feature_columns = ['close', 'RSI_7', 'RSI_14', 'RSI_21', 'Stochastic %K', 'Stochastic %D']
if os.path.exists(model_file_path):
    print("Loading pre-trained model...")
    try:
        X = pd.DataFrame(columns=feature_columns)
        model = joblib.load(model_file_path)
        print("Model loaded from file.")
        log_event("model_loaded", {"model_path": model_file_path})
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
else:
    # This block is for generating and training the model if it's missing.
    # It remains unchanged but is crucial for the script to run.
    print("Training new ML model...")
    log_event("model_training_start", {})
    try:
        data_with_indicators_list = pd.read_csv(csv_file_path_for_model)
        X_full = data_with_indicators_list[feature_columns]
        X = X_full.iloc[0:-1].copy()
        Y = data_with_indicators_list['close'].iloc[1:].values.ravel()
        X.fillna(0, inplace=True)
        if len(X) > len(Y):
            X = X.iloc[:len(Y)]
        print(f"Training model with {len(X)} samples using {len(X.columns)} features.")
        model = RandomForestRegressor()
        model.fit(X, Y)
        joblib.dump(model, model_file_path)
        print("Model trained and saved to file.")
        log_event("model_trained", {"model_path": model_file_path})
    except Exception as e:
        print(f"Error training model: {e}")
        traceback.print_exc()
        sys.exit(1)


# --- Trading Parameters ---
startAmount = 10
Lowamount = 48
Highamount = 52
symbol = "R_100"
barrier = 0.01
interval = 60
periods = [14, 7, 21]
maxFailAmount = 100

# --- Balance and User Info Simulation ---
loginid = "VRTC11012957"
currency = "USD"
current_balance = 10000.0
payout_rate = 0.90

try:
    with open(fail_file_path, 'r') as f:
        failAmount = int(f.read().strip())
except (FileNotFoundError, ValueError):
    failAmount = 0

full_data_df = pd.read_csv(os.path.join(directory, '8_hour_data.csv'))
full_data_df['timestamp'] = pd.to_datetime(full_data_df['timestamp'])
full_data_df.set_index('timestamp', inplace=True)


async def trade(symbol, interval, direction, current_data_index, full_data_df, log_time):
    global failAmount
    global current_balance
    amount = startAmount * (2 ** failAmount)

    print(f"Making Simulated Trade: {symbol}, Interval: {interval}, Direction: {direction}, amount: {amount}")
    log_event("trade_attempt", {"symbol": symbol, "interval": interval, "direction": direction, "amount": amount, "failAmount": failAmount}, log_time)

    entry_price = float(full_data_df.iloc[current_data_index]['close'])
    trade_time = full_data_df.index[current_data_index]
    target_time = trade_time + pd.Timedelta(seconds=interval)
    outcome_price = None
    status = None

    try:
        future_data = full_data_df[full_data_df.index >= target_time]
        if not future_data.empty:
            outcome_row = future_data.iloc[0]
            outcome_price = float(outcome_row['close'])
            outcome_time = outcome_row.name
            print(f"Outcome timestamp found: {outcome_time}, Outcome price: {outcome_price}")
        else:
            print("No subsequent data available to determine trade outcome. Assuming lost.")
            log_event("trade_outcome_error", {"error": "No subsequent data for outcome."}, log_time)
            status = 'lost'
    except Exception as e:
        print(f"Error finding outcome timestamp: {e}. Assuming lost.")
        log_event("trade_outcome_error", {"error": str(e), "traceback": traceback.format_exc()}, log_time)
        status = 'lost'

    if outcome_price is not None:
        if direction == "CALL":
            status = 'won' if outcome_price > (entry_price + barrier) else 'lost'
        elif direction == "PUT":
            status = 'won' if outcome_price < (entry_price - barrier) else 'lost'

    if status == 'won':
        print("Simulated Trade won!")
        failAmount = 0
        profit = amount * payout_rate
        current_balance += profit
        print(f"Profit: ${profit:.2f}, New Balance: ${current_balance:.2f}")
        log_event("trade_result", {"status": "won", "entry_price": entry_price, "outcome_price": outcome_price, "failAmount": failAmount}, log_time)
    elif status == 'lost':
        print("Simulated Trade lost.")
        failAmount += 1
        current_balance -= amount
        print(f"Loss: ${amount:.2f}, New Balance: ${current_balance:.2f}")
        print("Number of times failed in a row:", failAmount)
        log_event("trade_result", {"status": "lost", "entry_price": entry_price, "outcome_price": outcome_price, "failAmount": failAmount}, log_time)
    else:
        print("Simulated trade status is unknown.")
        log_event("trade_result", {"status": "unknown", "entry_price": entry_price, "outcome_price": outcome_price, "failAmount": failAmount}, log_time)

    log_event("post_trade_balance", {
        "balance": round(current_balance, 2),
        "currency": currency,
        "loginid": loginid
    }, log_time)

    with open(fail_file_path, 'w') as f:
        f.write(str(failAmount))

    if failAmount >= maxFailAmount:
        print("Failed too many times in a row. Simulating automatic shutdown.")
        log_event("max_failures_reached", {"failAmount": failAmount}, log_time)
        sys.exit(1)

async def fetch_historical_data(symbol, count, current_data_index, full_data_df, log_time):
    actual_count = count
    if current_data_index < count - 1:
        actual_count = current_data_index + 1
    start_idx = max(0, current_data_index - actual_count + 1)
    end_idx = current_data_index + 1
    relevant_data = full_data_df.iloc[start_idx:end_idx]
    prices = relevant_data['close'].tolist()
    times = [int(t.timestamp()) for t in relevant_data.index]
    data = {"history": {"prices": [str(p) for p in prices], "times": times}}
    log_event("historical_data_fetched_simulated", {"symbol": symbol, "requested_count": count, "actual_count": len(prices)}, log_time)
    return data

def calculate_rsi(data, period):
    ticks = data.get('history', {}).get('prices', [])
    if len(ticks) < period + 1:
        raise ValueError(f"Not enough data for RSI period {period}.")
    closes = np.array(list(map(float, ticks)))
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = (avg_gain / avg_loss) if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(data, period=14):
    ticks = data.get('history', {}).get('prices', [])
    if len(ticks) < period:
        raise ValueError(f"Not enough data for Stochastic period {period}.")
    closes = np.array(list(map(float, ticks[-period:])))
    low, high = np.min(closes), np.max(closes)
    k_value = ((closes[-1] - low) / (high - low)) * 100 if (high - low) != 0 else 0.0
    return k_value, k_value

async def update_rsi_and_indicators(periods, data, log_time):
    rsi_values = {}
    for period in periods:
        rsi_values[period] = calculate_rsi(data, period)
    stoch_k, stoch_d = calculate_stochastic(data)
    log_event("indicators_calculated", {"rsi": rsi_values, "stoch_k": stoch_k, "stoch_d": stoch_d}, log_time)
    return rsi_values, stoch_k, stoch_d

def enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d, data, log_time):
    prices = data.get('history', {}).get('prices', [])
    if not prices:
        log_event("strategy_error", {"error": "no price data"}, log_time)
        return None
    price = float(prices[-1])
    print(f"Current Price = {price}")
    rsi_7, rsi_14, rsi_21 = rsi_values[7], rsi_values[14], rsi_values[21]
    latest_data_list = [price, rsi_7, rsi_14, rsi_21, stoch_k, stoch_d]
    latest_data = pd.DataFrame([latest_data_list], columns=X.columns)
    Y_pred = model.predict(latest_data)[0]
    print(f"RSI (7,14,21): {rsi_7:.2f}, {rsi_14:.2f}, {rsi_21:.2f}")
    print(f"Stochastic (K,D): {stoch_k:.2f}, {stoch_d:.2f}")
    print(f"Predicted price after 1 minute = {Y_pred:.2f}")
    log_event("prediction", {"price": price, "rsi_7": rsi_7, "rsi_14": rsi_14, "rsi_21": rsi_21, "stoch_k": stoch_k, "stoch_d": stoch_d, "predicted_price": Y_pred}, log_time)
    decision = None
    if Y_pred > price and rsi_7 < Lowamount and rsi_14 < Lowamount and rsi_21 < Lowamount + 5 and stoch_k < Lowamount and stoch_d < Lowamount:
        print("Decision: Betting UP (CALL)")
        decision = "CALL"
    elif Y_pred < price and rsi_7 > Highamount and rsi_14 > Highamount and rsi_21 > Highamount - 5 and stoch_k > Highamount and stoch_d > Highamount:
        print("Decision: Betting DOWN (PUT)")
        decision = "PUT"
    else:
        print("Decision: Not betting")
    log_event("strategy_decision", {"decision": decision}, log_time)
    return decision

async def reset_bot():
    print("Resetting bot due to critical error... (Simulated exit for backtesting)")
    log_event("bot_reset_triggered", {})
    sys.exit(1)

async def main():
    try:
        start_index = 71
        print(f"Starting backtest from data point index: {start_index} (total data points: {len(full_data_df)})")
        
        for i in range(start_index, len(full_data_df)):
            current_timestamp = full_data_df.index[i]
            
            # --- START OF FIX ---
            # Log the initial balance only ONCE on the first iteration, using the first historical timestamp
            if i == start_index:
                log_event("initial_balance", {"balance": current_balance, "currency": currency, "loginid": loginid}, current_timestamp)
            # --- END OF FIX ---

            print(f"\n--- Processing data point {i+1}/{len(full_data_df)} (Timestamp: {current_timestamp}) ---")
            
            data = await fetch_historical_data(symbol, 71, i, full_data_df, current_timestamp)
            
            try:
                rsi_values, stoch_k, stoch_d = await update_rsi_and_indicators(periods, data, current_timestamp)
            except ValueError as e:
                print(f"Skipping trade at index {i}: {e}")
                log_event("skip_trade", {"reason": str(e), "index": i}, current_timestamp)
                continue

            direction = enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d, data, current_timestamp)
            
            if direction:
                await trade(symbol, interval, direction, i, full_data_df, current_timestamp)
            else:
                print(f"Parameters not met. Fail count remains: {failAmount}")
                log_event("no_trade", {"failAmount": failAmount, "index": i}, current_timestamp)
            
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        log_event("interrupted", {})
    except Exception as e:
        print(f"A critical error occurred in main loop: {e}")
        log_event("main_error", {"error": str(e), "traceback": traceback.format_exc()})
        await reset_bot()

print(f"Trading in {symbol} (Backtesting Mode)")
log_event("script_start", {"symbol": symbol, "mode": "backtesting"})
asyncio.run(main())