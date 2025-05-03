# --- Q-Learning Based Trading Strategy ---
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('Honda_Data.csv')  # Replace with your actual file path
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df = df[['Date', 'Close']]
df.dropna(inplace=True)

# Q-Learning Agent
class QTrader:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # State: 'up' or 'down'
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, price_diff):
        return 'up' if price_diff > 0 else 'down'

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # [buy, hold, sell]
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

# Initialize trader and capital
trader = QTrader()
capital = 10000
shares = 0
trade_log = []

# Simulate trading
for i in range(1, len(df) - 1):
    today_price = df.iloc[i]['Close']
    tomorrow_price = df.iloc[i + 1]['Close']
    price_diff = tomorrow_price - today_price
    state = trader.get_state(price_diff)
    action = trader.choose_action(state)

    reward = 0
    action_taken = "Hold"

    if action == 0 and capital >= today_price:
        shares += 1
        capital -= today_price
        action_taken = "Buy"
    elif action == 2 and shares > 0:
        shares -= 1
        capital += today_price
        reward = price_diff
        action_taken = "Sell"

    next_state = trader.get_state(df.iloc[i + 1]['Close'] - df.iloc[i]['Close'])
    trader.update(state, action, reward, next_state)

    trade_log.append((df.iloc[i]['Date'].date(), action_taken, capital + shares * today_price))

# Final results
final_value = capital + shares * df.iloc[-1]['Close']
print(f"\n💰 Final Portfolio Value with Q-Learning: ${final_value:.2f}")
print(f"📈 Total Trades Executed: {len(trade_log)}")

# Display last 10 trades
print("\n📝 Last 10 Trade Actions:")
for date, action, value in trade_log[-10:]:
    print(f"{date} - Action: {action}, Portfolio Value: ${value:.2f}")
