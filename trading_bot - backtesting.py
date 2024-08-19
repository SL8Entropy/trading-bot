import asyncio
import websockets
import json
import time
from deriv_api import DerivAPI

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"

money = 10000
failAmount = 0
startAmount = 100
symbol = "R_100"
Lowamount = 30
Highamount = 70
barrier = "0.1"
interval = 120  # Trade duration in seconds
check_interval = 25  # Time between market checks in ticks
periods = [14, 7, 21]
min_data_points = max(periods) + 1


async def fetch_historical_data(symbol, count, start="latest"):
    async with websockets.connect(
            f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}'
    ) as websocket:
        request = {
            "ticks_history": symbol,
            "end": start,
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
    gains, losses = [], []

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

    for i in range(period, len(closes) - 1):
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
        raise ValueError(
            "Not enough data points to calculate Stochastic Oscillator.")

    closes = list(map(float, ticks))
    stoch_k = []

    for i in range(period - 1, len(closes)):
        low = min(closes[i - period + 1:i + 1])
        high = max(closes[i - period + 1:i + 1])
        k_value = ((closes[i] - low) / (high - low)) * 100
        stoch_k.append(k_value)

    stoch_d = sum(stoch_k[-3:]) / 3  # Simple moving average of %K

    return stoch_k[-1], stoch_d



async def update_indicators(symbol, periods):
    count = max(periods) + 100
    while True:
        try:
            data = await fetch_historical_data(symbol, count)
            rsi_values = {}
            for period in periods:
                rsi_values[period] = calculate_rsi(data, period)
            stoch_k, stoch_d = calculate_stochastic(data)

            return rsi_values, stoch_k, stoch_d
        except ValueError as e:
            print(
                f"Not enough data points: {e}. Increasing count and retrying..."
            )
            count += 100
            await asyncio.sleep(5)


def enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d):
    rsi_14 = rsi_values[14][-1]
    rsi_7 = rsi_values[7][-1]
    rsi_21 = rsi_values[21][-1]

    if (rsi_7 < Lowamount and rsi_14 < Lowamount and rsi_21 < Lowamount + 5
            and stoch_k < 20 and stoch_d < 20):
        return "CALL"
    elif (rsi_7 > Highamount and rsi_14 > Highamount
          and rsi_21 > Highamount - 5 and stoch_k > 80 and stoch_d > 80):
        return "PUT"
    else:
        return None


async def backtest_strategy():
    global failAmount, money
    try:
        api = DerivAPI(app_id=app_id)
        authorize = await api.authorize(app_token)
        print("Authorize response:", authorize)

        # Fetch historical data for the last week
        num_intervals = (7 * 24 * 60 * 60) // check_interval
        historical_data = await fetch_historical_data(symbol, num_intervals)

        prices = historical_data['history']['prices']

        start = 0
        while start < len(prices) - interval:
            # Update indicators at each check interval
            sliced_data = {
                'history': {
                    'prices': prices[start:start + interval]
                }
            }
            rsi_values, stoch_k, stoch_d= await update_indicators(
                symbol, periods)

            direction = enhanced_triple_rebound_strategy(
                rsi_values, stoch_k, stoch_d)

            if direction:
                money -= startAmount * (2**failAmount)
                entry_price = float(prices[start])
                exit_price = float(prices[start + interval])
                print(
                    f"Simulated Trade: {symbol}, Interval: {interval}, Direction: {direction}"
                )
                print(f"Entry Price: {entry_price}, Exit Price: {exit_price}")

                # Determine if the trade is successful
                if direction == "CALL" and exit_price > entry_price:
                    print(f"Simulated CALL trade successful!")
                    money += startAmount * (2**(failAmount + 1))
                    failAmount = 0
                elif direction == "PUT" and exit_price < entry_price:
                    print(f"Simulated PUT trade successful!")
                    money += startAmount * (2**(failAmount + 1))
                    failAmount = 0
                else:
                    print(f"Simulated trade failed.")
                    failAmount += 1

                # Skip the next `interval` worth of market indices
                start += interval
            else:
                print(
                    f"Parameters not met for interval starting at price index {start}."
                )

            print(f"Current money: {money}")
            print(f"Fail amount: {failAmount}")
            print(f"simulated market time elapsed: {start/5} seconds")
            start += check_interval  # Move to the next check interval
            #await asyncio.sleep(check_interval)  # Simulate waiting for 5 seconds before the next check

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


print(f"Backtesting strategy for {symbol}")
asyncio.run(backtest_strategy())
