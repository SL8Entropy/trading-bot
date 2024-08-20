import asyncio
import websockets
import json
from deriv_api import DerivAPI  # Importing DerivAPI
from datetime import datetime, timedelta

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"  # Your API token
failAmount = 0
startAmount = 100
symbol = "R_100"
Lowamount = 30
Highamount = 70
barrier = "0.1"
interval = 120  # in seconds
periods = [14, 7, 21]
min_data_points = max(periods) + 1
initial_balance = 10000.0
balance = initial_balance

# Simulate a trade based on historical data
def simulate_trade(direction, start_index, end_index, closes):
    global failAmount
    global startAmount
    global balance
    amount = startAmount * (2 ** failAmount)
    
    print(f"Simulating Trade: {symbol}, Direction: {direction}, amount: {amount}")
    
    entry_price = closes[start_index]
    exit_price = closes[end_index]
    
    if direction == "CALL":
        if exit_price > entry_price:
            print("Trade won!")
            failAmount = 0
            balance += amount
        else:
            print("Trade lost.")
            failAmount += 1
            balance -= amount
    elif direction == "PUT":
        if exit_price < entry_price:
            print("Trade won!")
            failAmount = 0
            balance += amount
        else:
            print("Trade lost.")
            failAmount += 1
            balance -= amount

    print(f"Current balance: ${balance:.2f}\n")

async def fetch_historical_data(symbol, count, start, end):
    async with websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}') as websocket:
        request = {
            "ticks_history": symbol,
            "start": start,
            "end": end,
            "count": count,
            "style": "ticks"
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        data = json.loads(response)
        if 'error' in data:
            raise Exception(f"Error fetching historical data: {data['error']}")
        return data

def calculate_rsi(data, period):
    ticks = data.get('history', {}).get('prices', [])

    if len(ticks) < period:
        raise ValueError("Not enough data points to calculate RSI.")

    closes = list(map(float, ticks))
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))

    rsi_values = [rsi]

    for i in range(period, len(closes)-1):
        gain = gains[i]
        loss = losses[i]
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    return rsi_values

def calculate_stochastic(data, period=14):
    ticks = data.get('history', {}).get('prices', [])

    if len(ticks) < period:
        raise ValueError("Not enough data points to calculate Stochastic Oscillator.")

    closes = list(map(float, ticks))
    stoch_k = []
    
    for i in range(period - 1, len(closes)):
        low = min(closes[i - period + 1:i + 1])
        high = max(closes[i - period + 1:i + 1])
        k_value = ((closes[i] - low) / (high - low)) * 100
        stoch_k.append(k_value)
    
    stoch_d = sum(stoch_k[-3:]) / 3  # Simple moving average of %K

    return stoch_k[-1], stoch_d

async def update_rsi_and_indicators(symbol, periods, start, end):
    count = max(periods) + 1
    data = await fetch_historical_data(symbol, count, start, end)
    
    rsi_values = {}
    for period in periods:
        rsi_values[period] = calculate_rsi(data, period)
    
    stoch_k, stoch_d = calculate_stochastic(data)
    
    return rsi_values, stoch_k, stoch_d, data

def enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d):
    rsi_14 = rsi_values[14][-1]
    rsi_7 = rsi_values[7][-1]
    rsi_21 = rsi_values[21][-1]

    # Print RSI and Stochastic values for debugging
    print(f"RSI 7: {rsi_7}, RSI 14: {rsi_14}, RSI 21: {rsi_21}")
    print(f"Stochastic K: {stoch_k}, Stochastic D: {stoch_d}")

    if (rsi_7 < Lowamount and rsi_14 < Lowamount and rsi_21 < Lowamount + 5 and stoch_k < 20 and stoch_d < 20):
        print("Conditions met for CALL.")
        return "CALL"
    elif (rsi_7 > Highamount and rsi_14 > Highamount and rsi_21 > Highamount - 5 and stoch_k > 80 and stoch_d > 80):
        print("Conditions met for PUT.")
        return "PUT"
    else:
        print("Conditions not met.")
        return None


async def backtest(symbol, periods, interval, initial_balance):
    global balance
    balance = initial_balance

    # Calculate timestamps for the last 7 days
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=7)).timestamp())
    
    for i in range(min_data_points, len(data.get('history', {}).get('prices', [])) - interval):
        start_i = start + i * interval
        end_i = start_i + interval

        rsi_values, stoch_k, stoch_d, data = await update_rsi_and_indicators(symbol, periods, start_i, end_i)
        
        direction = enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d)
        
        if direction:
            simulate_trade(direction, i, i + interval, closes)
        else:
            print("Parameters not met. Moving to next data point.")

            
async def main():
    await backtest(symbol, periods, interval, initial_balance)

print(f"Starting backtest with ${initial_balance} initial balance.")
asyncio.run(main())
