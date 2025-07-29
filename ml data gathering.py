import websockets
import json
import pandas as pd
import datetime
import os
import asyncio

directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(directory, '8_hour_data.csv')

# → START ADDITION
print(f"Output CSV will be written to:\n    {csv_file_path}")
resp = input("Is this the correct file name and location? [Y/n]: ").strip().lower()
if resp not in ('y', 'yes', ''):
    print("Exiting. Please fix the file name and rerun.")
    import sys; sys.exit(1)
# ← END ADDITION
print("working...")




api_token = 'AP3ri2UNkUqqoCf'
app_id = 63226

symbol = 'R_100'
granularity = 60  # seconds(60,120,180,etc)
dataCount = 5000  # max per request
data_list = []

# Adjust the start date for the desired range
start_date = datetime.datetime.now() - datetime.timedelta(days=15)
end_date = start_date + datetime.timedelta(hours=8)
increment = datetime.timedelta(days=1)  # Increment range

async def fetch_data(symbol, start, end, granularity, count):
    async with websockets.connect(f'wss://ws.binaryws.com/websockets/v3?app_id={app_id}') as websocket:
        await websocket.send(json.dumps({"authorize": api_token}))
        auth_response = await websocket.recv()
        auth_data = json.loads(auth_response)
        if 'error' in auth_data:
            raise Exception(f"Authorization error: {auth_data['error']}")

        start_timestamp = int(start.timestamp())
        end_timestamp = int(end.timestamp())
        request = {
            "ticks_history": symbol,
            "start": start_timestamp,
            "end": end_timestamp,
            "granularity": granularity,
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
    total_data_points = 0
    count =1
    try:
        while current_start < end_date:
            current_end = min(current_start + increment, end_date)
            data = await fetch_data(symbol, current_start, current_end, granularity, dataCount)
            if data and 'candles' in data:
                candles = data['candles']
                data_list.extend([
                    {'timestamp': candle['epoch'], 'open': candle['open'], 'high': candle['high'],
                     'low': candle['low'], 'close': candle['close']} for candle in candles
                ])
                total_data_points += len(candles)
            else:
                print("No data received in response.")

            current_start = current_end
            await asyncio.sleep(2)
            print(f"fetching. please wait. time elapsed = {count*2}")
            count+=1

    except Exception as e:
        print(f"An error occurred: {e}")

    if data_list:
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))
        print(f"{total_data_points} data points saved to data.csv")
    else:
        print("No data collected.")

asyncio.run(main())
