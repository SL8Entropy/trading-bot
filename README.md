Overview of the Trading Bot

This trading bot utilizes the Deriv API and a Scikit-learn machine learning model to predict the behavior of markets such as the Volatility Index 100.

Steps to Run the Bot:

1. Data Gathering:

Run ml_data_gathering.py to collect price points at one-minute intervals for the past 30 days.



2. Indicator Calculation:

Run rsi_and_stochastic_calcer.py to calculate key indicators (RSI and stochastic values) for each price point.



3. Trading Bot Setup:

Open trading_bot_using_machine_learning.py.

Replace the placeholders for your API key and account details with your credentials.




How It Works:

The bot uses the gathered data to train a machine learning model.

Trades are executed based on predictions, with the Martingale strategy employed to manage risks and maximize returns.



---

Performance & Testing

Test Period: From 20th November 2024 to 28th November 2024, running for 3 hours daily.

Average Profit: $50/hour with an initial investment of $1,000 and a minimum trade value of $1.

Risk Management: A built-in stop-loss mechanism minimizes steep losses. However, the bot operates with a low-risk, slow-recovery approach. This means recovering from occasional losses might take time.


Under normal market conditions, the bot provides consistent returns on investment.

---
Proof Of Working

https://youtu.be/nJqub27-saw?si=YSeVKotHBznjODYd
---

Planned Updates

To enhance performance and reliability, future updates will include:

A pre-check feature to assess market conditions before initiating trades.

The bot will evaluate prediction accuracy over the previous 5 minutes to ensure the market is in an ideal state for trading.



---

This setup ensures steady and low-risk trading, with room for further optimization in market adaptabili
ty.
