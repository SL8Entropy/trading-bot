import websockets
import json
import pandas as pd
import datetime
import os
import asyncio

# Get the directory of the current Python file
directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(directory, 'data.csv')

# Your Deriv API token and app ID
api_token = 'AP3ri2UNkUqqoCf'
app_id = 63226

# Request parameters
symbol = 'R_100'  # Volatility 100 Index
granularity = 60  # 1-minute data
count = 100  # Amount of data pulled per request
data_list = []

# Define the start and end date for the data collection range
start_date = datetime.datetime.now() - datetime.timedelta(days=30)
end_date = datetime.datetime.now()
increment = datetime.timedelta(days=5)  # Set increment period


async def fetch_data(symbol, count):
    async with websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}') as websocket:
        
        # Authorize WebSocket connection
        auth_request = {
            "authorize": api_token
        }
        await websocket.send(json.dumps(auth_request))
        auth_response = await websocket.recv()
        auth_data = json.loads(auth_response)
        if 'error' in auth_data:
            raise Exception(f"Authorization error: {auth_data['error']}")

        # Now request historical data
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


async def main():
    current_start = start_date
    x = 0
    try:
        while current_start < end_date:
            current_end = min(current_start + increment, end_date)

            # Fetch and append data
            data = await fetch_data(symbol, count)
            if data and 'history' in data:
                history = data['history']
                data_list.extend([
                    {'timestamp': t, 'price': p} for t, p in zip(history['times'], history['prices'])
                ])
            else:
                print("No data received in response.")
            
            # Update the date range and delay to respect rate limits
            current_start = current_end
            await asyncio.sleep(2)
            x += 1
            if x >= 5:
                break
            
    except Exception as e:
        print(f"An error occurred: {e}")

    # Save data to CSV
    if data_list:
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.to_csv(csv_file_path, index=False)
        print("Data saved to volatility_100_data.csv")
    else:
        print("No data collected.")


# Run the main function
asyncio.run(main())
