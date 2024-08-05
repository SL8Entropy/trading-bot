from deriv_api import DerivAPI
import asyncio

app_id = 63226
app_token = "AP3ri2UNkUqqoCf"

async def main():
    print("Bot Activated")
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
            "contract_type": "CALL",  # Use "CALL" for Up, "PUT" for Down
            "currency": "USD",
            "duration": 30,  # Duration in seconds
            "duration_unit": "s",
            "symbol": "R_100"  # Replace with the desired symbol (R_100 = volality 100 index)
        })
        print("Proposal response:", proposal)

        proposal_id = proposal.get('proposal').get('id')

        # Buy the proposed contract
        buy = await api.buy({"buy": proposal_id, "price": 100})
        print("Buy response:", buy)

        contract_id = buy.get('buy').get('contract_id')

        # Optionally, subscribe to contract updates
        source_poc = await api.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})
        source_poc.subscribe(lambda poc: print("Subscribed POC:", poc))

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

        # Get the profit table
        profit_table = await api.profit_table({"profit_table": 1, "description": 1, "sort": "ASC"})
        print("Profit table:", profit_table)

        # Get the statement
        statement = await api.statement({"statement": 1, "description": 1, "limit": 100, "offset": 25})
        print("Statement:", statement)

    except Exception as e:
        print("An error occurred:", e)

# Run the main function
asyncio.run(main())