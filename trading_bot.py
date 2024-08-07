import asyncio
import websockets
import json
import time
from deriv_api import DerivAPI

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"
failAmount = 0
startAmount = 100
symbol = "R_100"
print(f"Trading in {symbol}")
interval = 180 #in seconds
periods = [14, 7, 21]
min_data_points = max(periods) + 1
####to make terminak more clear, i have put 4 hashes #### at every over the top print function
async def trade(api, symbol, interval, direction):
    global failAmount
    global startAmount
    amount = startAmount*(2**failAmount)
    time = 0
    print(f"Making Trade: {symbol}, Interval: {interval}, Direction: {direction}, amount: {amount}")
    
    try:
        proposal = await api.proposal({
            "proposal": 1,
            "amount": amount,
            "barrier": "+0.1",
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
            print(f"Trade ongoing, Please wait. Time elapsed = {time}")
            time += 30
            # Check if the contract has expired or if the trade is sold
            is_sold = poc.get('proposal_open_contract', {}).get('is_sold')
            if is_sold:
                contract_status = poc.get('proposal_open_contract', {}).get('status')
                if contract_status == 'won':
                    print("Trade won!")
                    failAmount=0
                elif contract_status == 'lost':
                    print("Trade lost.")
                    failAmount+=1
                    print("number of times failed in a row"+str(failAmount))####
                else:
                    print("Trade status is unknown.")
                break
            
                    
            await asyncio.sleep(30)
        if failAmount>=4:
            time = 0
            print("Failed too many times in a row. This is usually due to market conditions not being normal. Please try again another day.")
            while True:
                timeLeft = 30-time
                print(f"Time until automatic shutdown {timeLeft}")
                time+=5
                time.sleep(5)
    except Exception as e:
        print(f"An error occurred in trade: {e}")

    except Exception as e:
        print(f"An error occurred in trade: {e}")

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

async def update_rsi(symbol, periods):
    count = max(periods) + 100
    while True:
        try:
            data = await fetch_historical_data(symbol, count)
            ####print(f"Fetched historical data: {data}")
            rsi_values = {}
            for period in periods:
                rsi_values[period] = calculate_rsi(data, period)
            ####print(f"Calculated RSI values: {rsi_values}")
            return rsi_values
        except ValueError as e:
            print(f"Not enough data points: {e}. Increasing count and retrying...")
            count += 100
            await asyncio.sleep(5)

def triple_rebound_strategy(rsi_values):
    try:
        rsi_14 = rsi_values[14][-1]
        rsi_7 = rsi_values[7][-1]
        rsi_21 = rsi_values[21][-1]
        '''
        count_below_30 = sum(rsi < 30 for rsi in [rsi_14, rsi_7, rsi_21])
        count_above_70 = sum(rsi > 70 for rsi in [rsi_14, rsi_7, rsi_21])

        if count_below_30 >= 2:
            return "CALL"
        elif count_above_70 >= 2:
            return "PUT"
        else:
            return None
        '''
        
        if rsi_7 <30 and rsi_14<30 and rsi_21<35:
            return "CALL"
        if rsi_7 > 70 and rsi_14>70 and rsi_21>65:
            return "PUT"   
        else:
            return None    
    except IndexError as e:
        print(f"Index error in triple rebound strategy: {e}")
        return None

async def main():
    global failAmount
    try:
        api = DerivAPI(app_id=app_id)
        authorize = await api.authorize(app_token)
        print("Authorize response:", authorize)

        while True:
            rsi_values = await update_rsi(symbol, periods)
            ####print(f"Latest RSI values: {rsi_values}")

            direction = triple_rebound_strategy(rsi_values)
            if direction:
                await trade(api, symbol, interval, direction)
            else:
                print("Parameters not met. Waiting 1 second then rechecking")
                print("failamount = "+str(failAmount))####
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

asyncio.run(main())