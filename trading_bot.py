from deriv_api import DerivAPI
import asyncio
import requests
import time

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"


# Example usage
symbol = "R_100"  # Replace with the correct symbol
interval = 60     # 60 seconds (1 minute)
period = 14       # Williams %R period

async def trade(symbol, interval, direction):
    print("Making Trade" +symbol+str(interval)+ direction)
    try:
        # Initialize the Deriv API connection
        api = DerivAPI(app_id=app_id)

        # Authorize with your API token
        authorize = await api.authorize(app_token)
        print("Authorize response:", authorize)

        # Propose an Up/Down contract
        proposal = await api.proposal({
            "proposal": 1,
            "amount": 1,
            "barrier": "+0.1",  # Adjust barrier based on your strategy
            "basis": "payout",
            "contract_type": direction,  # Use "CALL" for Up, "PUT" for Down
            "currency": "USD",
            "duration": interval,  # Duration in seconds
            "duration_unit": "s",
            "symbol": symbol  # Replace with the desired symbol (R_100 = volality 100 index)
        })
        print("Proposal response:", proposal)

        proposal_id = proposal.get('proposal').get('id')

        # Buy the proposed contract
        buy = await api.buy({"buy": proposal_id, "price": 100})
        print("Buy response:", buy)

        contract_id = buy.get('buy').get('contract_id')
        '''
        # Optionally, subscribe to contract updates
        source_poc = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        source_poc.subscribe(lambda poc: print("Subscribed POC:", poc))
        '''
        # Wait for the contract to be sold or expired
        while True:
            poc = await api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": contract_id})
            print("Proposal open contract:", poc)

            if poc.get('proposal_open_contract').get('is_sold'):
                break

            # If you want to sell the contract before expiry, implement the logic here
            # sell = await api.sell({"sell": contract_id, "price": 40})
            # print("Sell response:", sell)

            await asyncio.sleep(1)
        '''
        # Get the profit table
        profit_table = await api.profit_table({"profit_table": 1, "description": 1, "sort": "ASC"})
        print("Profit table:", profit_table)

        # Get the statement
        statement = await api.statement({"statement": 1, "description": 1, "limit": 100, "offset": 25})
        print("Statement:", statement)
        '''
    except Exception as e:
        print("An error occurred:", e)

        # example make a trade up
        #asyncio.run(trade(symbol,interval,"CALL"))


def fetch_historical_data(symbol, interval, count):
    """
    Fetch historical price data from the Deriv API.
    """
    url = "https://api.deriv.com/v1/ticks"
    params = {
        "symbol": symbol,
        "interval": interval,
        "count": count
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Check for request errors
    return response.json()






## DO the calculate and update for second indicator
def calculate_williams_r(data, period):
    """
    Calculate Williams %R based on historical price data.
    """
    ticks = data.get('ticks', [])
    if len(ticks) < period:
        raise ValueError("Not enough data points to calculate Williams %R.")

    closes = [tick['close'] for tick in ticks]
    highs = [tick['high'] for tick in ticks]
    lows = [tick['low'] for tick in ticks]

    williams_r_values = []

    for i in range(period-1, len(ticks)):
        highest_high = max(highs[i-period+1:i+1])
        lowest_low = min(lows[i-period+1:i+1])
        current_close = closes[i]
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        williams_r_values.append(williams_r)

    return williams_r_values

def update_williams_r(symbol, interval, period):
    """
    Fetch historical data and calculate Williams %R.
    """
    data = fetch_historical_data(symbol, interval, count=period + 100)  # Fetch extra data to ensure full period calculation
    return calculate_williams_r(data, period)









try:
    while True:
        williams_r_values = update_williams_r(symbol, interval, period)
        print(f"Latest Williams %R: {williams_r_values[-1]}")
        #########################################Place update for second indicator here
        #########################################Place trading strategy here
        time.sleep(interval)  # Wait for the next interval
except KeyboardInterrupt:
    print("Process interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")

