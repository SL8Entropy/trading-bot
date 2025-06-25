import asyncio
import websockets
import json
import time
import os
from deriv_api import DerivAPI
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Get the directory of the current Python file
directory = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(directory, 'random_forest_model.joblib')
# Define log file path
log_file_path = os.path.join(directory, 'trade_logs.jsonl')

def log_event(event: str, info: dict):
    """
    Append a JSON entry with timestamp, event type, and info dict to the log file.
    """
    entry = {
        "timestamp": time.time(),
        "event": event,
    }
    # Copy info so we don't modify caller's dict
    entry.update(info)
    try:
        with open(log_file_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as e:
        # If logging fails, still print error
        print(f"Logging failed: {e}")

# Load or train the model
if os.path.exists(model_file_path):
    csv_file_path = os.path.join(directory, 'data_with_indicators.csv')
    # Load data
    data_with_indicators_list = pd.read_csv(csv_file_path)
    # Drop the first 4 columns (e.g., date/time and maybe open/high/low as in your original)
    data_with_indicators_list = data_with_indicators_list.iloc[:, 4:]

    X = data_with_indicators_list[0:5]
    # Load the model from the file
    model = joblib.load(model_file_path)
    print("Model loaded from file.")
    log_event("model_loaded", {"model_path": model_file_path})
else:
    print("training ml model")
    log_event("model_training_start", {})
    csv_file_path = os.path.join(directory, 'data_with_indicators.csv')
    # Load data
    data_with_indicators_list = pd.read_csv(csv_file_path)
    # Drop the first 4 columns (e.g., date/time and maybe open/high/low)
    data_with_indicators_list = data_with_indicators_list.iloc[:, 4:]

    X = data_with_indicators_list[0:-1]
    Y = data_with_indicators_list[1:].iloc[:, 0].values.ravel()

    # Train model
    model = RandomForestRegressor()
    model.fit(X, Y)
    joblib.dump(model, model_file_path)
    print("Model trained and saved to file.")
    log_event("model_trained", {"model_path": model_file_path})

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"
failAmount = 0
startAmount = 10
Lowamount = 48  # for rsi and stochastic indicators
Highamount = 52  # for rsi and stochastic indicators
symbol = "R_100"
barrier = "0.01"
interval = 60  # in seconds. model predicts price after 1 minute.
periods = [14, 7, 21]
min_data_points = max(periods) + 1
maxFailAmount = 100

async def trade(api, symbol, interval, direction):
    """
    Attempt to make a trade and always accept the current ask_price.
    If a buy is rejected due to price movement, it will retry with a fresh proposal.
    """
    global failAmount
    global startAmount

    RETRY_DELAY = 1  # seconds to wait before retrying after a price-move error

    amount = startAmount * (2 ** failAmount)
    time_elapsed = 0

    print(f"Making Trade: {symbol}, Interval: {interval}, Direction: {direction}, amount: {amount}")
    log_event("trade_attempt", {
        "symbol": symbol,
        "interval": interval,
        "direction": direction,
        "amount": amount,
        "failAmount": failAmount
    })

    # Determine barrier string
    if direction == "CALL":
        bar = "+" + barrier
    else:
        bar = "-" + barrier

    try:
        # Loop until buy succeeds or a non-price-movement error occurs
        while True:
            # 1) Fetch a new proposal
            try:
                proposal = await api.proposal({
                    "proposal": 1,
                    "amount": amount,
                    "barrier": bar,
                    "basis": "payout",
                    "contract_type": direction,
                    "currency": "USD",
                    "duration": interval,
                    "duration_unit": "s",
                    "symbol": symbol
                })
            except Exception as e:
                # If proposal itself fails, log and abort this trade
                print(f"Proposal request failed: {e}")
                log_event("proposal_error", {"error": str(e)})
                raise

            # Extract proposal_id and ask_price
            proposal_id = proposal.get('proposal', {}).get('id')
            ask_price = proposal.get('proposal', {}).get('ask_price')
            if not proposal_id or ask_price is None:
                msg = f"Invalid proposal response: {proposal}"
                print(msg)
                log_event("proposal_invalid", {"proposal": proposal})
                raise Exception("Failed to get valid proposal ID or ask_price")

            # Log proposal summary
            prop = proposal.get('proposal', {})
            proposal_summary = {
                "proposal_id": proposal_id,
                "ask_price": ask_price,
                "spot": prop.get("spot"),
                "spot_time": prop.get("spot_time"),
                "date_start": prop.get("date_start"),
                "date_expiry": prop.get("date_expiry")
            }
            print(f"Proposal fetched: ask_price = {ask_price}")
            log_event("proposal_fetched", proposal_summary)

            # 2) Attempt to buy at the current ask_price
            try:
                buy = await api.buy({"buy": proposal_id, "price": ask_price})
                print("Buy response:", buy)

                # Clean buy_data for logging (omit longcode)
                buy_data = buy.get("buy", {}).copy()
                buy_data.pop("longcode", None)
                log_event("buy_response", {"buy": buy_data, "proposal_id": proposal_id})
                break  # exit retry loop on successful buy
            except ResponseError as e:
                err_msg = str(e)
                # If the error indicates price movement, retry with a fresh proposal
                if "moved too much" in err_msg or "changed from" in err_msg:
                    print(f"Buy failed due to price movement: {err_msg}. Retrying...")
                    log_event("buy_price_moved", {"error": err_msg, "proposal_id": proposal_id})
                    await asyncio.sleep(RETRY_DELAY)
                    continue  # fetch a new proposal and try again
                else:
                    # Non-price-movement error: abort and propagate
                    print(f"Buy failed with non-price error: {err_msg}")
                    log_event("buy_error", {"error": err_msg, "proposal_id": proposal_id})
                    raise
            except Exception as e:
                # Other exceptions: abort
                print(f"Buy raised exception: {e}")
                log_event("buy_exception", {"error": str(e), "proposal_id": proposal_id})
                raise

        # At this point, buy succeeded and buy_data is available
        contract_id = buy_data.get("contract_id")
        if not contract_id:
            raise Exception("Failed to get contract ID after buy")

        print(f"Trade made. Amount = {amount}, direction = {direction}, duration = {interval}")
        log_event("trade_made", {
            "symbol": symbol,
            "amount": amount,
            "direction": direction,
            "duration": interval,
            "contract_id": contract_id
        })

        # Poll contract status as before
        while True:
            poc = await api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": contract_id})
            poc_data = poc.get("proposal_open_contract", {})
            poc_summary = {
                "contract_id": poc_data.get("contract_id"),
                "status": poc_data.get("status"),
                "is_sold": poc_data.get("is_sold"),
                "current_spot": poc_data.get("current_spot"),
                "current_spot_time": poc_data.get("current_spot_time"),
                "profit": poc_data.get("profit"),
                "profit_percentage": poc_data.get("profit_percentage"),
            }
            print("Proposal open contract:", poc_summary)
            print(f"Trade ongoing. Time elapsed = {time_elapsed}")
            log_event("proposal_open_contract_poll", {"time_elapsed": time_elapsed, **poc_summary})
            time_elapsed += interval + 5/5

            if poc_data.get("is_sold"):
                status = poc_data.get("status")
                if status == 'won':
                    print("Trade won!")
                    failAmount = 0
                    log_event("trade_result", {
                        "status": "won",
                        "contract_id": contract_id,
                        "failAmount": failAmount
                    })
                elif status == 'lost':
                    print("Trade lost.")
                    failAmount += 1
                    print("Number of times failed in a row:", failAmount)
                    log_event("trade_result", {
                        "status": "lost",
                        "contract_id": contract_id,
                        "failAmount": failAmount
                    })
                else:
                    print("Trade status is unknown.")
                    log_event("trade_result", {
                        "status": status,
                        "contract_id": contract_id,
                        "failAmount": failAmount
                    })
                break

            await asyncio.sleep(interval + 5/5)

        # Handle excessive failures if desired
        if failAmount >= maxFailAmount:
            time_left = 0
            print("Failed too many times in a row. Automatic shutdown soon.")
            log_event("max_failures_reached", {"failAmount": failAmount})
            time_remaining = 35
            while True:
                time_remaining -= 5
                print(f"Time until automatic shutdown: {time_remaining}")
                log_event("shutdown_countdown", {"time_remaining": time_remaining})
                time.sleep(5)
                if time_remaining <= 0:
                    exit(1)

    except Exception as e:
        print(f"An error occurred in trade: {e}")
        log_event("trade_error", {"error": str(e)})
        # Re-authorize and recover API if needed
        try:
            api = DerivAPI(app_id=app_id)
            authorize = await api.authorize(app_token)
            print("Re-authorize response:", authorize)
            auth_info = {"loginid": authorize.get("authorize", {}).get("loginid")}
            log_event("re_authorize", auth_info)
        except Exception as auth_e:
            print(f"Re-authorization failed: {auth_e}")
            log_event("re_authorize_error", {"error": str(auth_e)})


async def fetch_historical_data(symbol, count):
    async with websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}') as websocket:
        request = {
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "ticks"
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        data = json.loads(response)
        if 'error' in data:
            raise Exception(f"Error fetching historical data: {data['error']}")
        log_event("historical_data_fetched", {"symbol": symbol, "count": count, "data_summary": {
            "num_ticks": len(data.get('history', {}).get('prices', []))
        }})
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
        # Avoid division by zero if high == low
        if high - low == 0:
            k_value = 0.0
        else:
            k_value = ((closes[i] - low) / (high - low)) * 100
        stoch_k.append(k_value)
    stoch_d = sum(stoch_k[-3:]) / 3  # Simple moving average of %K
    return stoch_k[-1], stoch_d

async def update_rsi_and_indicators(periods, data):
    # This function returns latest RSI values dict and stoch values
    count = None 
    while True:
        try:
            rsi_values = {}
            for period in periods:
                rsi_values[period] = calculate_rsi(data, period)
            stoch_k, stoch_d = calculate_stochastic(data)
            log_event("indicators_calculated", {
                "rsi_lengths": {str(k): len(v) for k, v in rsi_values.items()},
                "stoch_k": stoch_k,
                "stoch_d": stoch_d
            })
            return rsi_values, stoch_k, stoch_d
        except ValueError as e:
            print(f"Not enough data points: {e}. Increasing count and retrying...")
            log_event("indicator_error", {"error": str(e)})
            await asyncio.sleep(5)

def enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d, data):
    prices = data.get('history', {}).get('prices', [])
    if not prices:
        print("No price data available for strategy.")
        log_event("strategy_error", {"error": "no price data"})
        return None
    # Use the latest price as current price:
    price = float(prices[-1])
    print(f"price = {price}")
    rsi_14 = rsi_values[14][-1]
    rsi_7 = rsi_values[7][-1]
    rsi_21 = rsi_values[21][-1]

    latest_data_list = [price, rsi_7, rsi_14, rsi_21, stoch_k, stoch_d]
    latest_data = pd.DataFrame([latest_data_list], columns=X.columns)

    # Use [0] to access the first element if Y_pred is a single prediction
    Y_pred = model.predict(latest_data)[0]
    print(f"rsi values (7,14,21): {rsi_7}, {rsi_14}, {rsi_21} ")
    print(f"stochk = {stoch_k}, stochd = {stoch_d}")
    print(f"Predicted price after 1 minute = {Y_pred}")

    decision = None
    # Logging prediction details
    log_event("prediction", {
        "price": price,
        "rsi_7": rsi_7,
        "rsi_14": rsi_14,
        "rsi_21": rsi_21,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "predicted_price": Y_pred,
        "Lowamount": Lowamount,
        "Highamount": Highamount
    })

    if Y_pred > price and rsi_7 < Lowamount and rsi_14 < Lowamount and rsi_21 < Lowamount + 5 and stoch_k < Lowamount and stoch_d < Lowamount:
        print("Betting UP")
        decision = "CALL"
    elif Y_pred < price and rsi_7 > Highamount and rsi_14 > Highamount and rsi_21 > Highamount - 5 and stoch_k > Highamount and stoch_d > Highamount:
        print("Betting DOWN")
        decision = "PUT"
    else:
        print("Not betting")
        decision = None

    log_event("strategy_decision", {"decision": decision})
    return decision

async def main():
    global failAmount
    try:
        api = DerivAPI(app_id=app_id)
        authorize = await api.authorize(app_token)
        print("Authorize response:", authorize)
        log_event("authorize", {"response": authorize})

        while True:
            data = await fetch_historical_data(symbol, 71)
            rsi_values, stoch_k, stoch_d = await update_rsi_and_indicators(periods, data)

            direction = enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d, data)
            if direction:
                await trade(api, symbol, interval, direction)
            else:
                print("Parameters not met. Waiting 5 seconds then rechecking")
                print("failamount = " + str(failAmount))
                log_event("no_trade", {"failAmount": failAmount})
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
        log_event("interrupted", {})
    except Exception as e:
        print(f"An error occurred: {e}")
        log_event("main_error", {"error": str(e)})

print(f"Trading in {symbol}")
log_event("script_start", {"symbol": symbol})
asyncio.run(main())
