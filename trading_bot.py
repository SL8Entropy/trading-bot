@ -10,28 +10,25 @@ app_token = "AP3ri2UNkUqqoCf"
failAmount = 0
startAmount = 100
symbol = "R_100"
#Lowamount and Highamount respectively
#30,70 for medium trading. reccomended barrier: 0.1
#20,80 for safe trading, but not many trades per second. reccomended barrier:0.2
#40,60 for fast trading, but not much chance for success. reccomended barrier:0.01
Lowamount = 30
Highamount = 70
barrier = "0.1"
interval = 120 #in seconds
interval = 120  # in seconds
periods = [14, 7, 21]
min_data_points = max(periods) + 1
####to make terminak more clear, i have put 4 hashes #### at every over the top print function

async def trade(api, symbol, interval, direction):
    global failAmount
    global startAmount
    amount = startAmount*(2**failAmount)
    time = 0
    amount = startAmount * (2 ** failAmount)
    time_elapsed = 0
    print(f"Making Trade: {symbol}, Interval: {interval}, Direction: {direction}, amount: {amount}")
    
    if direction == "CALL":
        bar = "+"+barrier
        bar = "+" + barrier
    else:
        bar = "-"+barrier
    
        bar = "-" + barrier

    try:
        proposal = await api.proposal({
            "proposal": 1,
@ -62,40 +59,35 @@ async def trade(api, symbol, interval, direction):
        while True:
            poc = await api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": contract_id})
            print("Proposal open contract:", poc)
            print(f"Trade ongoing, Please wait. Time elapsed = {time}")
            time += 30
            # Check if the contract has expired or if the trade is sold
            print(f"Trade ongoing, Please wait. Time elapsed = {time_elapsed}")
            time_elapsed += 30
            is_sold = poc.get('proposal_open_contract', {}).get('is_sold')
            if is_sold:
                contract_status = poc.get('proposal_open_contract', {}).get('status')
                if contract_status == 'won':
                    print("Trade won!")
                    failAmount=0
                    failAmount = 0
                elif contract_status == 'lost':
                    print("Trade lost.")
                    failAmount+=1
                    print("number of times failed in a row"+str(failAmount))####
                    failAmount += 1
                    print("Number of times failed in a row: " + str(failAmount))
                else:
                    print("Trade status is unknown.")
                break
            
                    

            await asyncio.sleep(31)
        if failAmount>=4:
            time = 0
        if failAmount >= 4:
            time_left = 0
            print("Failed too many times in a row. This is usually due to market conditions not being normal. Please try again another day.")
            while True:
                timeLeft = 30-time
                print(f"Time until automatic shutdown {timeLeft}")
                time+=5
                time_remaining = 30 - time_left
                print(f"Time until automatic shutdown: {time_remaining}")
                time_left += 5
                time.sleep(5)
            exit(1)
    except Exception as e:
        print(f"An error occurred in trade: {e}")

    except Exception as e:
        print(f"An error occurred in trade: {e}")

async def fetch_historical_data(symbol, count):
    async with websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}') as websocket:
        request = {
@ -148,48 +140,77 @@ def calculate_rsi(data, period):
        rsi_values.append(rsi)
    return rsi_values

async def update_rsi(symbol, periods):
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

def calculate_ema(prices, period):
    ema = []
    k = 2 / (period + 1)
    for i in range(len(prices)):
        if i == 0:
            ema.append(prices[i])
        else:
            ema.append((prices[i] * k) + (ema[i - 1] * (1 - k)))
    return ema

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    ticks = data.get('history', {}).get('prices', [])
    closes = list(map(float, ticks))

    ema_short = calculate_ema(closes, short_period)
    ema_long = calculate_ema(closes, long_period)

    macd_line = [ema_short[i] - ema_long[i] for i in range(len(ema_short))]
    signal_line = calculate_ema(macd_line, signal_period)
    macd_histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]

    return macd_line[-1], signal_line[-1], macd_histogram[-1]

async def update_rsi_and_indicators(symbol, periods):
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
            stoch_k, stoch_d = calculate_stochastic(data)
            macd_line, signal_line, macd_histogram = calculate_macd(data)

            return rsi_values, stoch_k, stoch_d, macd_line, signal_line, macd_histogram
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
        global Lowamount,Highamount
        
        if rsi_7 <Lowamount and rsi_14<Lowamount and rsi_21<Lowamount+5:
            return "CALL"
        if rsi_7 > Highamount and rsi_14>Highamount and rsi_21>Highamount-5:
            return "PUT"   
        else:
            return None    
    except IndexError as e:
        print(f"Index error in triple rebound strategy: {e}")
def enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d, macd_line, signal_line, macd_histogram):
    rsi_14 = rsi_values[14][-1]
    rsi_7 = rsi_values[7][-1]
    rsi_21 = rsi_values[21][-1]

    if (rsi_7 < Lowamount and rsi_14 < Lowamount and rsi_21 < Lowamount + 5 and stoch_k < 20 and stoch_d < 20 and
        macd_line > signal_line and macd_histogram > 0):
        return "CALL"
    elif (rsi_7 > Highamount and rsi_14 > Highamount and rsi_21 > Highamount - 5 and stoch_k > 80 and stoch_d > 80 and
          macd_line < signal_line and macd_histogram < 0):
        return "PUT"
    else:
        return None

async def main():
@ -200,22 +221,20 @@ async def main():
        print("Authorize response:", authorize)

        while True:
            rsi_values = await update_rsi(symbol, periods)
            ####print(f"Latest RSI values: {rsi_values}")
            rsi_values, stoch_k, stoch_d, macd_line, signal_line, macd_histogram = await update_rsi_and_indicators(symbol, periods)

            direction = triple_rebound_strategy(rsi_values)
            direction = enhanced_triple_rebound_strategy(rsi_values, stoch_k, stoch_d, macd_line, signal_line, macd_histogram)
            if direction:
                await trade(api, symbol, interval, direction)
            else:
                print("Parameters not met. Waiting 5 seconds then rechecking")
                print("failamount = "+str(failAmount))####
                print("failamount = " + str(failAmount))
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        

print(f"Trading in {symbol}")
asyncio.run(main())
asyncio.run(main())