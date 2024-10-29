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
granularity = 60  # in seconds
count = 5000  # Amount of data points per request
data_list = []

# Define the start and end date for the data collection range
start_date = datetime.datetime.now() - datetime.timedelta(days =90)
end_date = datetime.datetime.now()
increment = datetime.timedelta(days=5)  # Increment range


async def fetch_data(symbol, start, end, granularity, count):
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

        # Format start and end times as timestamps for the API
        start_timestamp = int(start.timestamp())
        end_timestamp = int(end.timestamp())

        # Request historical data with defined start, end, and granularity
        request = {
            "ticks_history": symbol,
            "start": start_timestamp,
            "end": end_timestamp,
            "granularity": granularity,  # Set to 60 seconds (1 minute) or other desired intervals
            "count": count,
            "style": "candles"
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
            # Define the range for the current request
            current_end = min(current_start + increment, end_date)

            # Fetch and append data
            data = await fetch_data(symbol, current_start, current_end, granularity, count)
            if data and 'candles' in data:
                candles = data['candles']
                data_list.extend([
                    {'timestamp': candle['epoch'], 'open': candle['open'], 'high': candle['high'],
                     'low': candle['low'], 'close': candle['close']} for candle in candles
                ])
            else:
                print("No data received in response.")
            
            # Update the start date for the next range
            current_start = current_end
            await asyncio.sleep(2)
            x += 1
            if x >= 5:  # Limit requests as a safeguard
                break
            
    except Exception as e:
        print(f"An error occurred: {e}")

    # Save data to CSV (append if file exists)
    if data_list:
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Append to the existing CSV file if it exists
        df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
        print("Data appended to data.csv")
    else:
        print("No data collected.")



# Run the main function
asyncio.run(main())
