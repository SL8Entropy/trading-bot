import asyncio
import websockets
import json
import time
from deriv_api import DerivAPI

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"
failAmount = 0
startAmount = 10
symbol = "R_100"
Lowamount = 35
Highamount = 65
barrier = "0.01"
interval = 15  # in seconds 
periods = [14, 7, 21]
min_data_points = max(periods) + 1

async def trade(api, symbol, interval, direction):
    global failAmount
    global startAmount
    amount = startAmount * (2 ** failAmount)
    time_elapsed = 0
    print(f"Making Trade: {symbol}, Interval: {interval}, Direction: {direction}, amount: {amount}")
    
    if direction == "CALL":
        bar = "+" + barrier
    else:
        bar = "-" + barrier

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
        print("Proposal response:", proposal)

        proposal_id = proposal.get('proposal', {}).get('id')

        if not proposal_id:
            raise Exception("Failed to get proposal ID")

        buy = await api.buy({"buy": proposal_id, "price": 100})
        print("Buy response:", buy)
        print(f"Trade made. Amount = {amount}, direction = {direction}, duration = {interval}")
        contract_id = buy.get('buy', {}).get('contract_id')

        if not contract_id:
            raise Exception("Failed to get contract ID")

        while True:
            poc = await api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": contract_id})
            print("Proposal open contract:", poc)
            print(f"Trade ongoing, Please wait. Time elapsed = {time_elapsed}")
            time_elapsed += interval/3
            is_sold = poc.get('proposal_open_contract', {}).get('is_sold')
            if is_sold:
                contract_status = poc.get('proposal_open_contract', {}).get('status')
                if contract_status == 'won':
                    print("Trade won!")
                    failAmount = 0
                elif contract_status == 'lost':
                    print("Trade lost.")
                    failAmount += 1
                    print("Number of times failed in a row: " + str(failAmount))
                else:
                    print("Trade status is unknown.")
                break

            await asyncio.sleep(interval/3 + 5)
        if failAmount >= 4:
            time_left = 0
            print("Failed too many times in a row. This is usually due to market conditions not being normal. Please try again another day.")
            while True:
                time_remaining = 30 - time_left
                print(f"Time until automatic shutdown: {time_remaining}")
                time_left += 5
                time.sleep(5)
            exit(1)
    except Exception as e:
        print(f"An error occurred in trade: {e}")
        api = DerivAPI(app_id=app_id)
        authorize = await api.authorize(app_token)
        print("Authorize response:", authorize)

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

async def update_rsi_and_indicators(symbol, periods):
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
            print(f"Not enough data points: {e}. Increasing count and retrying...")
            count += 100
            await asyncio.sleep(5)

def enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d):
    rsi_14 = rsi_values[14][-1]
    rsi_7 = rsi_values[7][-1]
    rsi_21 = rsi_values[21][-1]
    print(f"rsi 7: {rsi_7}, rsi 14: {rsi_14}, rsi: 21: {rsi_21}, stoch k : {stoch_k}, stoch d: {stoch_d}")  
    if (rsi_7 < Lowamount and rsi_14 < Lowamount and rsi_21 < Lowamount + 5 and stoch_k < Lowamount and stoch_d < Lowamount):
        return "CALL"
    elif (rsi_7 > Highamount and rsi_14 > Highamount and rsi_21 > Highamount - 5 and stoch_k > Highamount and stoch_d > Highamount):
        return "PUT"
    else:
        return None

async def main():
    global failAmount
    try:
        api = DerivAPI(app_id=app_id)
        authorize = await api.authorize(app_token)
        print("Authorize response:", authorize)

        while True:
            rsi_values, stoch_k, stoch_d = await update_rsi_and_indicators(symbol, periods)

            direction = enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d)
            if direction:
                await trade(api, symbol, interval, direction)
            else:
                print("Parameters not met. Waiting 5 seconds then rechecking")
                print("failamount = " + str(failAmount))
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Trading in {symbol}")
asyncio.run(main())